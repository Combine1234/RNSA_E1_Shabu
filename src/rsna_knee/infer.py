from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

import numpy as np

from .cache import RawStudySource
from .constants import ID_COLUMN, TARGETS
from .dataset import select_tta_window
from .models import load_study_model


class TorchRawDataset:
    def __init__(self, source: RawStudySource):
        self.source = source

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        import torch

        study_uid, images, valid = self.source[index]
        return study_uid, torch.from_numpy(images).float().div_(255.0), torch.from_numpy(valid)


def _run_device(models, images, valid, offsets, device: str):
    import torch

    if not models:
        return {}
    images = images.to(device, non_blocking=True)
    valid = valid.to(device, non_blocking=True)
    sums: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        for family, model in models:
            for offset in offsets:
                window_images, window_valid = select_tta_window(images, valid, offset)
                prediction = torch.sigmoid(model(window_images, window_valid)).float()
                sums[family] = sums.get(family, torch.zeros_like(prediction)) + prediction
                counts[family] = counts.get(family, 0) + 1
    return {family: (value / counts[family]).cpu().numpy() for family, value in sums.items()}


def _load_groups(checkpoint_groups: dict[str, list[str | Path]]):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("submission inference requires a Kaggle GPU runtime")
    devices = [f"cuda:{index}" for index in range(max(1, torch.cuda.device_count()))]
    by_device: dict[str, list[tuple[str, object]]] = {device: [] for device in devices}
    counter = 0
    for family, paths in checkpoint_groups.items():
        if not paths:
            raise ValueError(f"checkpoint group {family!r} is empty")
        for path in paths:
            device = devices[counter % len(devices)]
            model, _ = load_study_model(path, device)
            by_device[device].append((family, model))
            counter += 1
    return by_device


def predict_groups(
    source: RawStudySource,
    checkpoint_groups: dict[str, list[str | Path]],
    output_dir: str | Path,
    offsets: Iterable[int] = (0,),
    batch_size: int = 2,
    num_workers: int = 4,
    runtime_limit_seconds: int = 27000,
    hard_stop_seconds: int = 30600,
    checkpoint_every: int = 50,
) -> dict[str, Path]:
    import pandas as pd
    import torch

    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    offsets = tuple(int(offset) for offset in offsets)
    if not offsets:
        raise ValueError("at least one TTA offset is required")
    by_device = _load_groups(checkpoint_groups)
    loader = torch.utils.data.DataLoader(
        TorchRawDataset(source),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    ids: list[str] = []
    unrecoverable: list[str] = []
    family_rows: dict[str, list[np.ndarray]] = {family: [] for family in checkpoint_groups}
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=len(by_device)) as executor:
        for batch_ids, images, valid in loader:
            batch_id_values = [str(value) for value in batch_ids]
            recoverable = valid.reshape(len(batch_id_values), -1).any(dim=1).cpu().numpy()
            unrecoverable.extend([uid for uid, available in zip(batch_id_values, recoverable) if not available])
            elapsed = time.monotonic() - started
            if elapsed >= hard_stop_seconds:
                raise TimeoutError(f"hard runtime guard reached after {len(ids)} of {len(source)} studies")
            futures = [
                executor.submit(_run_device, models, images, valid, offsets, device)
                for device, models in by_device.items()
                if models
            ]
            device_results = [future.result() for future in futures]
            batch_sums: dict[str, np.ndarray] = {}
            device_counts: dict[str, int] = {}
            for result, models in zip(device_results, [models for models in by_device.values() if models]):
                per_device_family_counts: dict[str, int] = {}
                for family, _ in models:
                    per_device_family_counts[family] = per_device_family_counts.get(family, 0) + 1
                for family, values in result.items():
                    count = per_device_family_counts[family]
                    batch_sums[family] = batch_sums.get(family, np.zeros_like(values)) + values * count
                    device_counts[family] = device_counts.get(family, 0) + count
            for family in checkpoint_groups:
                family_rows[family].append(batch_sums[family] / device_counts[family])
            ids.extend(batch_id_values)
            if len(ids) >= 32:
                forecast = (time.monotonic() - started) / len(ids) * len(source)
                if forecast > runtime_limit_seconds:
                    print(f"WARNING projected runtime {forecast / 3600:.2f}h exceeds promotion limit")
            if checkpoint_every and len(ids) % checkpoint_every < len(batch_ids):
                partial = {family: np.concatenate(rows) for family, rows in family_rows.items()}
                np.savez(output_dir / "partial_predictions.npz", ids=np.asarray(ids), **partial)

    outputs: dict[str, Path] = {}
    for family, rows in family_rows.items():
        values = np.concatenate(rows)
        frame = pd.DataFrame({ID_COLUMN: ids})
        for index, target in enumerate(TARGETS):
            frame[target] = values[:, index]
        path = output_dir / f"predictions_{family}.csv"
        frame.to_csv(path, index=False)
        outputs[family] = path
    summary = {
        "studies": len(ids),
        "families": {family: len(paths) for family, paths in checkpoint_groups.items()},
        "offsets": list(offsets),
        "elapsed_seconds": time.monotonic() - started,
        "unrecoverable_studies": unrecoverable,
        "outputs": {key: str(value) for key, value in outputs.items()},
    }
    (output_dir / "inference_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return outputs


def benchmark_inference(
    source: RawStudySource,
    checkpoint_groups: dict[str, list[str | Path]],
    sample_studies: int = 32,
    test_studies: int = 1322,
    offsets: Iterable[int] = (-1, 0, 1),
    safety_factor: float = 1.3,
) -> dict[str, float]:
    subset = RawStudySource.__new__(RawStudySource)
    subset.study_ids = source.study_ids[:sample_studies]
    subset.grouped = source.grouped
    subset.dicom_root = source.dicom_root
    subset.spec = source.spec
    import tempfile

    with tempfile.TemporaryDirectory() as folder:
        started = time.monotonic()
        predict_groups(
            subset,
            checkpoint_groups,
            folder,
            offsets=offsets,
            runtime_limit_seconds=10**9,
            hard_stop_seconds=10**9,
            checkpoint_every=0,
        )
        total_elapsed = time.monotonic() - started
        summary = json.loads((Path(folder) / "inference_summary.json").read_text(encoding="utf-8"))
        sample_elapsed = float(summary["elapsed_seconds"])
    one_time_overhead = max(0.0, total_elapsed - sample_elapsed)
    projected = one_time_overhead + sample_elapsed / max(len(subset), 1) * test_studies * safety_factor
    return {
        "sample_studies": len(subset),
        "sample_seconds": sample_elapsed,
        "model_load_and_setup_seconds": one_time_overhead,
        "safety_factor": safety_factor,
        "projected_test_studies": test_studies,
        "projected_seconds": projected,
        "projected_hours": projected / 3600,
        "passes_runtime_gate": projected <= 27000,
    }


def verify_repeatability(
    source: RawStudySource,
    checkpoint_groups: dict[str, list[str | Path]],
    sample_studies: int = 4,
    offsets: Iterable[int] = (0,),
    tolerance: float = 1e-5,
) -> dict[str, float | bool]:
    import pandas as pd
    import tempfile

    subset = RawStudySource.__new__(RawStudySource)
    subset.study_ids = source.study_ids[:sample_studies]
    subset.grouped = source.grouped
    subset.dicom_root = source.dicom_root
    subset.spec = source.spec
    maximum = 0.0
    with tempfile.TemporaryDirectory() as first_folder, tempfile.TemporaryDirectory() as second_folder:
        first = predict_groups(
            subset, checkpoint_groups, first_folder, offsets=offsets,
            runtime_limit_seconds=10**9, hard_stop_seconds=10**9, checkpoint_every=0,
        )
        second = predict_groups(
            subset, checkpoint_groups, second_folder, offsets=offsets,
            runtime_limit_seconds=10**9, hard_stop_seconds=10**9, checkpoint_every=0,
        )
        for family in checkpoint_groups:
            left = pd.read_csv(first[family])
            right = pd.read_csv(second[family])
            if not left[ID_COLUMN].astype(str).equals(right[ID_COLUMN].astype(str)):
                raise RuntimeError(f"repeatability ID mismatch for {family}")
            maximum = max(
                maximum,
                float(np.max(np.abs(left[list(TARGETS)].to_numpy() - right[list(TARGETS)].to_numpy()))),
            )
    return {"maximum_absolute_difference": maximum, "tolerance": tolerance, "passes": maximum <= tolerance}


def apply_no_data_fallback(submission_path: str | Path, summary_path: str | Path) -> int:
    import pandas as pd

    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    missing = {str(value) for value in summary.get("unrecoverable_studies", [])}
    if not missing:
        return 0
    frame = pd.read_csv(submission_path)
    mask = frame[ID_COLUMN].astype(str).isin(missing)
    frame.loc[mask, list(TARGETS)] = 0.5
    frame.to_csv(submission_path, index=False)
    return int(mask.sum())
