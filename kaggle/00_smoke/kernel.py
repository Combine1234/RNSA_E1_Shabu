from pathlib import Path
import importlib.metadata
import json

import numpy as np
import pandas as pd

exec(Path(next(Path("/kaggle/input").glob("*/kaggle_support.py"))).read_text(), globals())
REPO = bootstrap_repo()
from rsna_knee.cache import CacheSpec, build_study_array
from rsna_knee.config import load_config
from rsna_knee.constants import ID_COLUMN, TARGETS

ROOT = competition_root()
train = pd.read_csv(ROOT / "train.csv")
test = pd.read_csv(ROOT / "test.csv")
series = pd.read_csv(ROOT / "train_series.csv")
expected_train = [ID_COLUMN, "Report", *TARGETS]
expected_series = [ID_COLUMN, "SeriesInstanceUID", "Fluid_Sensitive", "Fat_Suppression", "Anatomical_Plane"]
missing_train = sorted(set(expected_train) - set(train.columns))
missing_series = sorted(set(expected_series) - set(series.columns))
if missing_train or missing_series:
    raise ValueError({"missing_train": missing_train, "missing_series": missing_series})
if test.columns.tolist() != [ID_COLUMN]:
    raise ValueError(f"unexpected test schema: {test.columns.tolist()}")

cache_config = load_config(REPO / "configs/base.yaml")["cache"]
spec = CacheSpec(**{key: cache_config[key] for key in CacheSpec.__dataclass_fields__ if key in cache_config})
indices = np.unique(np.linspace(0, len(train) - 1, 3, dtype=int))
studies = []
for index in indices:
    uid = str(train.iloc[index][ID_COLUMN])
    image, valid = build_study_array(series[series[ID_COLUMN].astype(str) == uid], ROOT / "train_series", spec)
    studies.append({
        ID_COLUMN: uid,
        "shape": list(image.shape),
        "valid_slots": int(valid.any(axis=1).sum()),
        "valid_slices": int(valid.sum()),
    })
    if not valid.any():
        raise RuntimeError(f"smoke decode produced no pixels for {uid}")

versions = {}
for package in ("numpy", "pandas", "pydicom", "opencv-python", "torch", "timm"):
    try:
        versions[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        versions[package] = None
audit = {
    "train_rows": len(train),
    "series_rows": len(series),
    "visible_test_rows": len(test),
    "gold_studies": int(train[list(TARGETS)].notna().all(axis=1).sum()),
    "versions": versions,
    "studies": studies,
    "passes": True,
}
Path("/kaggle/working/smoke_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
print(json.dumps(audit, indent=2))
