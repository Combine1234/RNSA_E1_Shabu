from pathlib import Path
import glob

exec(Path(next(Path("/kaggle/input").glob("*/kaggle_support.py"))).read_text(), globals())
REPO = bootstrap_repo()
from rsna_knee.config import load_config
from rsna_knee.provenance import write_artifact_manifest
from rsna_knee.train import train_fold

config = load_config(REPO / "configs/base.yaml")
require_json_gate(["/kaggle/input/**/label_audit.json"], "passes_0_89_gate", "label audit")
require_cache_coverage(
    ["/kaggle/input/**/cache_*_of_02.json"],
    float(config["cache"]["minimum_valid_study_fraction"]),
)
labels = one_file(["/kaggle/input/**/labels_with_folds.csv"], "labels_with_folds.csv")
caches = files(["/kaggle/input/**/*_images.npy"], "pixel cache shards")
for fold in range(5):
    print(train_fold(labels, caches, "/kaggle/working", config["model_a"], fold,
                     seed=config["seed"], gold_weight=config["labels"]["gold_weight"]))
artifacts = sorted(Path("/kaggle/working").glob("fold_*.pt"))
write_artifact_manifest(artifacts, "/kaggle/working/artifact_manifest.json", {"family": "dino"})
