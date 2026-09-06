from collections import Counter
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pydicom

exec(Path(next(Path("/kaggle/input").glob("*/kaggle_support.py"))).read_text(), globals())
REPO = bootstrap_repo()
from rsna_knee.constants import ID_COLUMN
from rsna_knee.dicom import (
    geometric_position,
    infer_laterality,
    infer_position_laterality,
    patient_center_x,
    should_flip_horizontal,
)

ROOT = competition_root()
series = pd.read_csv(ROOT / "train_series.csv")
counts = Counter()
examples = []

for row in series.itertuples(index=False):
    plane = str(getattr(row, "Anatomical_Plane"))
    study_uid = str(getattr(row, ID_COLUMN))
    series_uid = str(getattr(row, "SeriesInstanceUID"))
    paths = sorted((ROOT / "train_series" / study_uid / series_uid).glob("*.dcm"))
    if not paths:
        counts[(plane, "missing_file")] += 1
        continue
    try:
        dataset = pydicom.dcmread(paths[len(paths) // 2], stop_before_pixels=True, force=True)
        orientation = np.asarray([float(value) for value in dataset.ImageOrientationPatient])
        normal = np.cross(orientation[:3], orientation[3:])
        laterality = infer_laterality(dataset) or "unknown"
        dominant = int(np.argmax(np.abs(normal)))
        sign = "+" if normal[dominant] >= 0 else "-"
        counts[(plane, "rows")] += 1
        counts[(plane, f"laterality_{laterality}")] += 1
        counts[(plane, f"normal_axis_{dominant}_{sign}")] += 1
        counts[(plane, f"horizontal_flip_{should_flip_horizontal(dataset, plane)}")] += 1
        if plane.casefold() == "sagittal" and laterality in {"L", "R"} and dominant == 0:
            desired_x = 1.0 if laterality == "R" else -1.0
            reverse = bool(desired_x * normal[0] < 0)
            counts[(plane, f"slice_reverse_{reverse}")] += 1
        position = np.asarray([float(value) for value in dataset.ImagePositionPatient])
        if abs(position[0]) >= 20.0:
            position_laterality = "R" if position[0] < 0 else "L"
            counts[(plane, f"position_laterality_{position_laterality}")] += 1
            if laterality in {"L", "R"}:
                counts[(plane, f"position_agrees_{position_laterality == laterality}")] += 1
        centre_laterality = infer_position_laterality(dataset)
        if centre_laterality:
            counts[(plane, f"centre_laterality_{centre_laterality}")] += 1
            if laterality in {"L", "R"}:
                counts[(plane, f"centre_agrees_{centre_laterality == laterality}")] += 1
        if len(examples) < 24:
            examples.append({
                "study": study_uid,
                "series": series_uid,
                "plane": plane,
                "laterality": laterality,
                "orientation": orientation.tolist(),
                "normal": normal.tolist(),
                "geometric_position": geometric_position(dataset),
                "patient_center_x": patient_center_x(dataset),
            })
    except Exception as exc:
        counts[(plane, f"error_{type(exc).__name__}")] += 1

payload = {
    "series_rows": int(len(series)),
    "counts": {f"{plane}|{key}": value for (plane, key), value in sorted(counts.items())},
    "examples": examples,
}
Path("/kaggle/working/orientation_audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps(payload["counts"], indent=2))
