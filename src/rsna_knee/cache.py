from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .constants import ID_COLUMN, SLOTS
from .dicom import (
    centered_consecutive_indices,
    decode_series,
    geometric_position,
    physical_center_crop,
    read_series,
    robust_uint8,
)


@dataclass(frozen=True)
class CacheSpec:
    image_size: int = 336
    crop_mm: float = 150.0
    slices_per_slot: int = 11
    edge_fraction: float = 0.02
    percentile_low: float = 0.5
    percentile_high: float = 99.5


def _series_paths(root: Path, study_uid: str, series_uid: str) -> list[Path]:
    folder = root / study_uid / series_uid
    return sorted(folder.glob("*.dcm")) if folder.is_dir() else []


def _series_quality(paths: list[Path]) -> tuple[int, float, int]:
    headers = read_series(paths, stop_before_pixels=True)
    positions = [geometric_position(dataset) for _, dataset in headers]
    positions = [value for value in positions if value is not None and np.isfinite(value)]
    span = float(max(positions) - min(positions)) if len(positions) >= 2 else 0.0
    areas = []
    for _, dataset in headers:
        try:
            areas.append(int(dataset.Rows) * int(dataset.Columns))
        except Exception:
            pass
    return len(headers), span, max(areas, default=0)


def choose_series_for_slots(study_series, dicom_root: str | Path) -> dict[tuple[str, int], list[Path]]:
    root = Path(dicom_root)
    selected: dict[tuple[str, int], list[Path]] = {}
    for plane, fluid in SLOTS:
        candidates = study_series[
            (study_series["Anatomical_Plane"].astype(str).str.casefold() == plane.casefold())
            & (study_series["Fluid_Sensitive"].fillna(0).astype(int) == fluid)
        ]
        ranked: list[tuple[tuple[int, float, int], str, list[Path]]] = []
        for row in candidates.itertuples(index=False):
            uid = str(getattr(row, "SeriesInstanceUID"))
            paths = _series_paths(root, str(getattr(row, ID_COLUMN)), uid)
            if paths:
                ranked.append((_series_quality(paths), uid, paths))
        if ranked:
            # Prefer physical coverage, then valid slice count and matrix detail; UID breaks ties.
            ranked.sort(key=lambda item: (-item[0][1], -item[0][0], -item[0][2], item[1]))
            selected[(plane, fluid)] = ranked[0][2]
    return selected


def build_study_array(study_series, dicom_root: str | Path, spec: CacheSpec) -> tuple[np.ndarray, np.ndarray]:
    images = np.zeros(
        (len(SLOTS), spec.slices_per_slot, spec.image_size, spec.image_size),
        dtype=np.uint8,
    )
    valid = np.zeros((len(SLOTS), spec.slices_per_slot), dtype=bool)
    selected = choose_series_for_slots(study_series, dicom_root)
    for slot_index, (plane, fluid) in enumerate(SLOTS):
        paths = selected.get((plane, fluid), [])
        if not paths:
            continue
        decoded = decode_series(paths, plane)
        if decoded.pixels.size == 0:
            continue
        normalized = robust_uint8(decoded.pixels, spec.percentile_low, spec.percentile_high)
        indices = centered_consecutive_indices(len(normalized), spec.slices_per_slot)
        for output_index, source_index in enumerate(indices):
            dataset = decoded.datasets[int(source_index)]
            spacing = getattr(dataset, "PixelSpacing", None)
            images[slot_index, output_index] = physical_center_crop(
                normalized[int(source_index)], spacing, spec.crop_mm, spec.image_size
            )
            valid[slot_index, output_index] = True
    return images, valid


def build_cache_shard(
    studies,
    series,
    dicom_root: str | Path,
    output: str | Path,
    spec: CacheSpec,
) -> dict[str, Any]:
    output = Path(output)
    if output.suffix:
        output = output.with_suffix("")
    output.parent.mkdir(parents=True, exist_ok=True)
    study_ids = studies[ID_COLUMN].astype(str).tolist()
    images_path = output.with_name(f"{output.name}_images.npy")
    valid_path = output.with_name(f"{output.name}_valid.npy")
    ids_path = output.with_name(f"{output.name}_ids.npy")
    images = np.lib.format.open_memmap(
        images_path,
        mode="w+",
        shape=(len(study_ids), len(SLOTS), spec.slices_per_slot, spec.image_size, spec.image_size),
        dtype=np.uint8,
    )
    valid = np.lib.format.open_memmap(
        valid_path,
        mode="w+",
        dtype=np.bool_,
        shape=(len(study_ids), len(SLOTS), spec.slices_per_slot),
    )
    images[:] = 0
    valid[:] = False
    failures: dict[str, str] = {}
    grouped = {str(key): frame for key, frame in series.groupby(ID_COLUMN, sort=False)}
    for index, study_uid in enumerate(study_ids):
        frame = grouped.get(study_uid)
        if frame is None:
            failures[study_uid] = "no series metadata"
            continue
        try:
            images[index], valid[index] = build_study_array(frame, dicom_root, spec)
            if not valid[index].any():
                failures[study_uid] = "no decodable selected series"
        except Exception as exc:  # a single corrupt study must not lose a multi-hour cache run
            failures[study_uid] = f"{type(exc).__name__}: {exc}"
        if (index + 1) % 50 == 0 or index + 1 == len(study_ids):
            print(
                f"cache progress {index + 1}/{len(study_ids)}; "
                f"valid={int(valid[: index + 1].any(axis=(1, 2)).sum())}; failures={len(failures)}",
                flush=True,
            )
    images.flush()
    valid.flush()
    np.save(ids_path, np.asarray(study_ids, dtype=str), allow_pickle=False)
    metadata = {
        "images": str(images_path),
        "valid": str(valid_path),
        "ids": str(ids_path),
        "studies": len(study_ids),
        "valid_studies": int(valid.any(axis=(1, 2)).sum()),
        "failures": failures,
        "spec": asdict(spec),
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


class CacheIndex:
    def __init__(self, paths: Iterable[str | Path]):
        self.paths = [Path(path) for path in paths]
        self.lookup: dict[str, tuple[int, int]] = {}
        for shard_index, path in enumerate(self.paths):
            if not path.name.endswith("_images.npy"):
                raise ValueError(f"cache path must end in _images.npy: {path}")
            ids_path = path.with_name(path.name.replace("_images.npy", "_ids.npy"))
            for row, study_uid in enumerate(np.load(ids_path, allow_pickle=False).astype(str)):
                if study_uid in self.lookup:
                    raise ValueError(f"duplicate cached study: {study_uid}")
                self.lookup[study_uid] = (shard_index, row)

    @lru_cache(maxsize=2)
    def _load(self, shard_index: int):
        images_path = self.paths[shard_index]
        valid_path = images_path.with_name(images_path.name.replace("_images.npy", "_valid.npy"))
        return np.load(images_path, mmap_mode="r"), np.load(valid_path, mmap_mode="r")

    def get(self, study_uid: str) -> tuple[np.ndarray, np.ndarray]:
        shard_index, row = self.lookup[str(study_uid)]
        images, valid = self._load(shard_index)
        return np.asarray(images[row]), np.asarray(valid[row])

    def __contains__(self, study_uid: str) -> bool:
        return str(study_uid) in self.lookup


class RawStudySource:
    """On-demand hidden-test-safe preprocessing used by the final notebook."""

    def __init__(self, studies, series, dicom_root: str | Path, spec: CacheSpec):
        self.study_ids = studies[ID_COLUMN].astype(str).tolist()
        self.grouped = {str(key): frame for key, frame in series.groupby(ID_COLUMN, sort=False)}
        self.dicom_root = Path(dicom_root)
        self.spec = spec

    def __len__(self) -> int:
        return len(self.study_ids)

    def __getitem__(self, index: int) -> tuple[str, np.ndarray, np.ndarray]:
        study_uid = self.study_ids[index]
        frame = self.grouped.get(study_uid)
        if frame is None:
            images = np.zeros(
                (len(SLOTS), self.spec.slices_per_slot, self.spec.image_size, self.spec.image_size),
                dtype=np.uint8,
            )
            valid = np.zeros((len(SLOTS), self.spec.slices_per_slot), dtype=bool)
        else:
            try:
                images, valid = build_study_array(frame, self.dicom_root, self.spec)
            except Exception:
                images = np.zeros(
                    (len(SLOTS), self.spec.slices_per_slot, self.spec.image_size, self.spec.image_size),
                    dtype=np.uint8,
                )
                valid = np.zeros((len(SLOTS), self.spec.slices_per_slot), dtype=bool)
        return study_uid, images, valid
