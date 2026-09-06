from pathlib import Path
import json

import numpy as np
import pandas as pd

exec(Path(next(Path("/kaggle/input").glob("*/kaggle_support.py"))).read_text(), globals())
REPO = bootstrap_repo()
from rsna_knee.cache import CacheSpec, RawStudySource
from rsna_knee.config import load_config
from rsna_knee.constants import ID_COLUMN
from rsna_knee.infer import benchmark_inference, verify_repeatability

ROOT = competition_root()
config = load_config(REPO / "configs/base.yaml")
cache_config = config["cache"]
spec = CacheSpec(**{key: cache_config[key] for key in CacheSpec.__dataclass_fields__ if key in cache_config})
train = pd.read_csv(ROOT / "train.csv")
series = pd.read_csv(ROOT / "train_series.csv")

# Cover the range of series counts instead of benchmarking only the first IDs.
counts = series.groupby(ID_COLUMN).size().reindex(train[ID_COLUMN]).fillna(0).to_numpy()
ordered = np.argsort(counts, kind="stable")
sample_count = min(int(config["inference"]["benchmark_sample_studies"]), len(ordered))
sample_indices = ordered[np.unique(np.linspace(0, len(ordered) - 1, sample_count, dtype=int))]
sample = train.iloc[sample_indices].reset_index(drop=True)
sample_ids = set(sample[ID_COLUMN].astype(str))
sample_series = series[series[ID_COLUMN].astype(str).isin(sample_ids)].copy()
source = RawStudySource(sample, sample_series, ROOT / "train_series", spec)
family_order = ("dino", "coatnet")
weights_path = one_file(["/kaggle/input/**/blend_weights.json"], "blend_weights.json")
weights_config = json.loads(weights_path.read_text(encoding="utf-8"))
active_families = {
    family
    for index, family in enumerate(family_order)
    if any(float(target["weights"][index]) > 0.0 for target in weights_config["targets"].values())
}
if not active_families:
    raise RuntimeError("blend configuration does not select any model family")
all_groups = {
    "dino": [str(path) for path in files(
        ["/kaggle/input/**/rsna-knee-03-train-dino/fold_*.pt"], "DINO checkpoints"
    )],
    "coatnet": [str(path) for path in files(
        ["/kaggle/input/**/rsna-knee-04-train-coatnet/fold_*.pt"], "CoAtNet checkpoints"
    )],
}
groups = {family: all_groups[family] for family in family_order if family in active_families}
if any(len(paths) != 5 for paths in groups.values()):
    raise FileNotFoundError(f"expected five checkpoints per family, found {groups}")
print(f"benchmarking promoted model families only: {list(groups)}")

test_studies = int(config["inference"]["test_studies"])
requested = list(config["inference"]["tta_offsets"])
result = benchmark_inference(source, groups, len(source), test_studies, requested)
result["requested_offsets"] = requested
result["selected_offsets"] = requested
if not result["passes_runtime_gate"]:
    fallback = benchmark_inference(source, groups, len(source), test_studies, [0])
    fallback["requested_offsets"] = requested
    fallback["selected_offsets"] = [0]
    fallback["fallback_from_three_pass"] = result
    result = fallback
if not result["passes_runtime_gate"]:
    raise RuntimeError(f"even center-only inference fails the runtime gate: {result}")
repeatability = verify_repeatability(source, groups, sample_studies=4, offsets=result["selected_offsets"])
result["repeatability"] = repeatability
if not repeatability["passes"]:
    raise RuntimeError(f"inference repeatability failed: {repeatability}")
Path("/kaggle/working/benchmark.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
