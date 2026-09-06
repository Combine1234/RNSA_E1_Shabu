from __future__ import annotations

from pathlib import Path

import numpy as np

from .cache import CacheIndex
from .constants import ID_COLUMN, TARGETS


class BlockShuffleSampler:
    """Shuffle contiguous row blocks while preserving efficient mmap reads."""

    def __init__(self, length: int, block_size: int = 64, seed: int = 20260905):
        self.length = int(length)
        self.block_size = max(1, int(block_size))
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return self.length

    def __iter__(self):
        blocks = [
            list(range(start, min(start + self.block_size, self.length)))
            for start in range(0, self.length, self.block_size)
        ]
        rng = np.random.default_rng(self.seed + self.epoch)
        rng.shuffle(blocks)
        self.epoch += 1
        return iter([index for block in blocks for index in block])


def _center_scale(images, scale: float):
    import torch

    height, width = images.shape[-2:]
    new_height = max(1, int(round(height * scale)))
    new_width = max(1, int(round(width * scale)))
    resized = torch.nn.functional.interpolate(
        images, size=(new_height, new_width), mode="bilinear", align_corners=False
    )
    if new_height >= height and new_width >= width:
        y0 = (new_height - height) // 2
        x0 = (new_width - width) // 2
        return resized[..., y0 : y0 + height, x0 : x0 + width]
    pad_y = height - new_height
    pad_x = width - new_width
    return torch.nn.functional.pad(
        resized,
        (pad_x // 2, pad_x - pad_x // 2, pad_y // 2, pad_y - pad_y // 2),
    )


def select_tta_window(images, valid, offset: int, model_slices: int = 3):
    total = images.shape[-3]
    start = (total - model_slices) // 2 + int(offset)
    start = min(max(start, 0), max(total - model_slices, 0))
    return images[..., start : start + model_slices, :, :], valid[..., start : start + model_slices]


class CachedStudyDataset:
    def __init__(
        self,
        labels,
        cache_paths: list[str | Path],
        indices: np.ndarray | None = None,
        training: bool = False,
        seed: int = 20260905,
        gold_weight: float = 4.0,
    ):
        self.labels = labels.reset_index(drop=True)
        self.cache = CacheIndex(cache_paths)
        self.indices = np.arange(len(self.labels)) if indices is None else np.asarray(indices)
        self.training = training
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self._rng_worker: int | None = None
        self.gold_weight = gold_weight

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        import torch

        worker = torch.utils.data.get_worker_info()
        worker_id = -1 if worker is None else int(worker.id)
        if self._rng_worker != worker_id:
            worker_seed = self.seed if worker is None else int(worker.seed)
            self.rng = np.random.default_rng(worker_seed % (2**32))
            self._rng_worker = worker_id

        row = self.labels.iloc[int(self.indices[item])]
        study_uid = str(row[ID_COLUMN])
        images, valid = self.cache.get(study_uid)
        offset = int(self.rng.integers(-2, 3)) if self.training else 0
        images, valid = select_tta_window(images, valid, offset)
        image_tensor = torch.from_numpy(images.copy()).float().div_(255.0)
        if self.training:
            image_tensor = _center_scale(image_tensor, float(self.rng.uniform(0.94, 1.06)))
            gain = float(self.rng.uniform(0.9, 1.1))
            bias = float(self.rng.uniform(-0.04, 0.04))
            image_tensor.mul_(gain).add_(bias).clamp_(0.0, 1.0)
        targets = np.asarray([float(row[target]) for target in TARGETS], dtype=np.float32)
        confidence = np.asarray([float(row[f"{target}__confidence"]) for target in TARGETS], dtype=np.float32)
        gold = np.asarray([bool(row[f"{target}__gold"]) for target in TARGETS])
        weights = confidence * np.where(gold, self.gold_weight, 1.0)
        return {
            "id": study_uid,
            "images": image_tensor,
            "valid": torch.from_numpy(valid.copy()),
            "targets": torch.from_numpy(targets),
            "weights": torch.from_numpy(weights.astype(np.float32)),
            "gold": torch.from_numpy(gold),
        }
