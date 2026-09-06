from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


def _value(dataset: Any, name: str, default: Any = None) -> Any:
    if isinstance(dataset, dict):
        return dataset.get(name, default)
    return getattr(dataset, name, default)


def _float_vector(value: Any, expected: int) -> np.ndarray | None:
    if value is None:
        return None
    try:
        result = np.asarray([float(item) for item in value], dtype=np.float64)
    except (TypeError, ValueError):
        return None
    return result if result.shape == (expected,) else None


def geometric_position(dataset: Any) -> float | None:
    orientation = _float_vector(_value(dataset, "ImageOrientationPatient"), 6)
    position = _float_vector(_value(dataset, "ImagePositionPatient"), 3)
    if orientation is None or position is None:
        return None
    normal = np.cross(orientation[:3], orientation[3:])
    norm = np.linalg.norm(normal)
    if norm < 1e-8:
        return None
    return float(np.dot(position, normal / norm))


def slice_sort_key(dataset: Any, path: str | Path = "") -> tuple[int, float | str]:
    position = geometric_position(dataset)
    if position is not None and np.isfinite(position):
        return (0, position)
    for priority, name in ((1, "SliceLocation"), (2, "InstanceNumber")):
        try:
            value = float(_value(dataset, name))
            if np.isfinite(value):
                return (priority, value)
        except (TypeError, ValueError):
            pass
    return (3, str(path))


def infer_laterality(dataset: Any) -> str | None:
    for name in ("ImageLaterality", "Laterality"):
        value = str(_value(dataset, name, "")).strip().upper()
        if value in {"L", "R"}:
            return value
    return None


def patient_center_x(dataset: Any) -> float | None:
    """Patient-left/right coordinate of the image centre, not its top-left corner."""
    orientation = _float_vector(_value(dataset, "ImageOrientationPatient"), 6)
    position = _float_vector(_value(dataset, "ImagePositionPatient"), 3)
    spacing = _float_vector(_value(dataset, "PixelSpacing"), 2)
    try:
        rows = int(_value(dataset, "Rows"))
        columns = int(_value(dataset, "Columns"))
    except (TypeError, ValueError):
        return None
    if orientation is None or position is None or spacing is None or rows <= 0 or columns <= 0:
        return None
    centre = (
        position
        + orientation[:3] * spacing[1] * (columns - 1) / 2.0
        + orientation[3:] * spacing[0] * (rows - 1) / 2.0
    )
    return float(centre[0]) if np.isfinite(centre[0]) else None


def infer_position_laterality(dataset: Any, minimum_offset_mm: float = 20.0) -> str | None:
    centre_x = patient_center_x(dataset)
    if centre_x is None or abs(centre_x) < minimum_offset_mm:
        return None
    # DICOM patient coordinates increase toward the patient's left.
    return "L" if centre_x > 0 else "R"


def should_flip_horizontal(dataset: Any, plane: str) -> bool:
    """Canonicalize medial side to image-right where L/R is the horizontal axis."""
    if plane.casefold() == "sagittal":
        return False
    laterality = infer_laterality(dataset) or infer_position_laterality(dataset)
    orientation = _float_vector(_value(dataset, "ImageOrientationPatient"), 6)
    if laterality is None or orientation is None:
        return False
    horizontal = orientation[:3]
    if abs(horizontal[0]) < max(abs(horizontal[1]), abs(horizontal[2])):
        return False
    medial_patient_x = 1.0 if laterality == "R" else -1.0
    medial_is_image_right = medial_patient_x * horizontal[0] > 0
    return not medial_is_image_right


def should_reverse_slice_order(dataset: Any, plane: str) -> bool:
    """Canonicalize sagittal channel order from lateral toward medial."""
    if plane.casefold() != "sagittal":
        return False
    laterality = infer_laterality(dataset) or infer_position_laterality(dataset)
    orientation = _float_vector(_value(dataset, "ImageOrientationPatient"), 6)
    if laterality is None or orientation is None:
        return False
    normal = np.cross(orientation[:3], orientation[3:])
    if abs(normal[0]) < max(abs(normal[1]), abs(normal[2])):
        return False
    # Ascending geometric position follows the normal. Lateral-to-medial is
    # +patient-x for a right knee and -patient-x for a left knee.
    desired_x = 1.0 if laterality == "R" else -1.0
    return bool(desired_x * normal[0] < 0)


def robust_uint8(volume: np.ndarray, low: float = 0.5, high: float = 99.5) -> np.ndarray:
    volume = np.asarray(volume, dtype=np.float32)
    finite = volume[np.isfinite(volume)]
    if finite.size == 0:
        return np.zeros(volume.shape, dtype=np.uint8)
    lo, hi = np.percentile(finite, [low, high])
    if hi <= lo:
        return np.zeros(volume.shape, dtype=np.uint8)
    scaled = np.clip((volume - lo) / (hi - lo), 0.0, 1.0)
    return np.rint(scaled * 255.0).astype(np.uint8)


def physical_center_crop(
    image: np.ndarray,
    pixel_spacing: Sequence[float] | None,
    crop_mm: float,
    output_size: int,
) -> np.ndarray:
    import cv2

    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError("physical_center_crop expects a 2D image")
    if pixel_spacing is None or len(pixel_spacing) < 2:
        spacing_y = spacing_x = crop_mm / min(image.shape)
    else:
        spacing_y, spacing_x = (max(float(pixel_spacing[0]), 1e-6), max(float(pixel_spacing[1]), 1e-6))
    crop_h = max(1, int(round(crop_mm / spacing_y)))
    crop_w = max(1, int(round(crop_mm / spacing_x)))
    pad_y = max(0, crop_h - image.shape[0])
    pad_x = max(0, crop_w - image.shape[1])
    if pad_y or pad_x:
        image = np.pad(
            image,
            ((pad_y // 2, pad_y - pad_y // 2), (pad_x // 2, pad_x - pad_x // 2)),
            mode="constant",
        )
    y0 = max(0, (image.shape[0] - crop_h) // 2)
    x0 = max(0, (image.shape[1] - crop_w) // 2)
    crop = image[y0 : y0 + crop_h, x0 : x0 + crop_w]
    return cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_AREA)


@dataclass
class DecodedSeries:
    pixels: np.ndarray
    datasets: list[Any]
    valid_files: list[str]


def read_series(paths: Iterable[str | Path], stop_before_pixels: bool = False) -> list[tuple[Path, Any]]:
    try:
        import pydicom
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pydicom is required for DICOM decoding") from exc
    result: list[tuple[Path, Any]] = []
    for path in paths:
        path = Path(path)
        try:
            dataset = pydicom.dcmread(path, stop_before_pixels=stop_before_pixels, force=True)
            result.append((path, dataset))
        except Exception:
            continue
    result.sort(key=lambda item: slice_sort_key(item[1], item[0]))
    return result


def decode_series(paths: Iterable[str | Path], plane: str) -> DecodedSeries:
    pairs = read_series(paths, stop_before_pixels=False)
    pixels: list[np.ndarray] = []
    datasets: list[Any] = []
    valid_files: list[str] = []
    for path, dataset in pairs:
        try:
            array = dataset.pixel_array.astype(np.float32)
            array = array * float(_value(dataset, "RescaleSlope", 1.0)) + float(
                _value(dataset, "RescaleIntercept", 0.0)
            )
            if str(_value(dataset, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
                array = array.max() + array.min() - array
            if should_flip_horizontal(dataset, plane):
                array = array[:, ::-1]
            pixels.append(array)
            datasets.append(dataset)
            valid_files.append(str(path))
        except Exception:
            continue
    if not pixels:
        return DecodedSeries(np.empty((0, 0, 0), dtype=np.float32), [], [])
    shape_counts: dict[tuple[int, ...], int] = {}
    for item in pixels:
        shape_counts[item.shape] = shape_counts.get(item.shape, 0) + 1
    common_shape = max(shape_counts, key=lambda shape: (shape_counts[shape], shape))
    keep = [i for i, item in enumerate(pixels) if item.shape == common_shape]
    if keep and should_reverse_slice_order(datasets[keep[len(keep) // 2]], plane):
        keep = keep[::-1]
    return DecodedSeries(
        np.stack([pixels[i] for i in keep]),
        [datasets[i] for i in keep],
        [valid_files[i] for i in keep],
    )


def evenly_spaced_indices(length: int, count: int, edge_fraction: float = 0.02) -> np.ndarray:
    if length <= 0 or count <= 0:
        return np.empty(0, dtype=np.int64)
    lo = int(np.floor((length - 1) * edge_fraction))
    hi = int(np.ceil((length - 1) * (1.0 - edge_fraction)))
    if lo > hi:
        lo, hi = 0, length - 1
    return np.rint(np.linspace(lo, hi, count)).astype(np.int64)


def centered_consecutive_indices(length: int, count: int) -> np.ndarray:
    """Return a centered run of real adjacent slices, edge-padding only if needed."""
    if length <= 0 or count <= 0:
        return np.empty(0, dtype=np.int64)
    start = (length - count) // 2
    return np.clip(np.arange(start, start + count), 0, length - 1).astype(np.int64)
