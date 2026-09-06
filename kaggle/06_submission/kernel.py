from pathlib import Path
import json
import pandas as pd

exec(Path(next(Path("/kaggle/input").glob("*/kaggle_support.py"))).read_text(), globals())
REPO = bootstrap_repo()
from rsna_knee.blend import apply_blend
from rsna_knee.cache import CacheSpec, RawStudySource
from rsna_knee.config import load_config
from rsna_knee.infer import apply_no_data_fallback, predict_groups
from rsna_knee.validate import validate_submission

ROOT = competition_root()
config = load_config(REPO / "configs/base.yaml")
cache_config = config["cache"]
spec = CacheSpec(**{key: cache_config[key] for key in CacheSpec.__dataclass_fields__ if key in cache_config})
test = pd.read_csv(ROOT / "test.csv")
series = pd.read_csv(ROOT / "test_series.csv")
source = RawStudySource(test, series, ROOT / "test_series", spec)
weights = one_file(["/kaggle/input/**/blend_weights.json"], "blend_weights.json")
weights_config = json.loads(weights.read_text(encoding="utf-8"))
family_order = ("dino", "coatnet")
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
print(f"running promoted model families only: {list(groups)}")
benchmark_path = one_file(["/kaggle/input/**/benchmark.json"], "benchmark.json")
benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
projected = float(benchmark["projected_seconds"]) * len(test) / float(benchmark["projected_test_studies"])
if not benchmark.get("passes_runtime_gate") or projected > config["inference"]["runtime_limit_seconds"]:
    raise RuntimeError(f"benchmark did not promote this candidate: projected {projected / 3600:.2f}h")
predictions = predict_groups(
    source, groups, "/kaggle/working", offsets=benchmark["selected_offsets"],
    runtime_limit_seconds=config["inference"]["runtime_limit_seconds"],
    hard_stop_seconds=config["inference"]["hard_stop_seconds"],
)
submission = Path("/kaggle/working/submission.csv")
fallback_prediction = next(iter(predictions.values()))
ordered_predictions = [predictions.get(family, fallback_prediction) for family in family_order]
apply_blend(ordered_predictions, weights, submission)
fallback_count = apply_no_data_fallback(submission, "/kaggle/working/inference_summary.json")
print(f"all-missing studies forced to 0.5: {fallback_count}")
print(validate_submission(submission, ROOT / "test.csv", require_variation=len(test) > 3))
