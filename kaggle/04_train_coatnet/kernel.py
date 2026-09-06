from pathlib import Path

exec(Path(next(Path("/kaggle/input").glob("*/kaggle_support.py"))).read_text(), globals())
REPO = bootstrap_repo()
from rsna_knee.config import load_config
from rsna_knee.provenance import write_artifact_manifest
from rsna_knee.train import train_frozen_feature_folds

config = load_config(REPO / "configs/base.yaml")
require_json_gate(["/kaggle/input/**/label_audit.json"], "passes_0_89_gate", "label audit")
require_json_gate(
    ["/kaggle/input/**/raptor_preflight.json"],
    "selected_checkpoint_compatible",
    "Raptor preflight",
)
require_cache_coverage(
    ["/kaggle/input/**/cache_*_of_02.json"],
    float(config["cache"]["minimum_valid_study_fraction"]),
)
labels = one_file(["/kaggle/input/**/labels_with_folds.csv"], "labels_with_folds.csv")
caches = files(["/kaggle/input/**/*_images.npy"], "pixel cache shards")
raptor_roots = [Path("/kaggle/input/raptor-knee-widedense")]
raptor_roots.extend(Path("/kaggle/input/datasets").glob("*/raptor-knee-widedense"))
raptor = sorted({
    path for suffix in ("*.pt", "*.pth", "*.bin")
    for root in raptor_roots if root.exists()
    for path in root.glob(f"**/{suffix}")
    if path.is_file()
})
initial = next((path for path in raptor if "swa" in path.stem.casefold()), raptor[0] if raptor else None)
print(train_frozen_feature_folds(
    labels,
    caches,
    "/kaggle/working",
    config["model_b"],
    seed=config["seed"] + 100,
    gold_weight=config["labels"]["gold_weight"],
    backbone_checkpoint=initial,
    backbone_checkpoint_license="CC0-1.0",
))
artifacts = sorted(Path("/kaggle/working").glob("fold_*.pt"))
write_artifact_manifest(artifacts, "/kaggle/working/artifact_manifest.json", {
    "family": "coatnet",
    "training_mode": "frozen_encoder_single_pass_features",
    "requested_raptor_checkpoint": str(initial) if initial else None,
})
