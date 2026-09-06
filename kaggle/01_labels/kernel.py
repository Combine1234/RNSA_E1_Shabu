from pathlib import Path
import json
import numpy as np

exec(Path(next(Path("/kaggle/input").glob("*/kaggle_support.py"))).read_text(), globals())
REPO = bootstrap_repo()

from rsna_knee.constants import TARGETS
from rsna_knee.config import load_config
from rsna_knee.folds import grouped_multilabel_folds
from rsna_knee.labels import (
    load_and_align_label_sources,
    mask_suspected_gold_leakage,
    robust_label_ensemble,
    write_label_bundle,
)
from rsna_knee.metrics import bootstrap_macro_auc, macro_auc

ROOT = competition_root()
SOURCES = discover_label_sources()
config = load_config(REPO / "configs/base.yaml")
synovitis_weight = float(config["labels"]["synovitis_effusion_weight"])
minimum_auc = float(config["labels"]["minimum_gold_auc"])
rank_normalize = bool(config["labels"]["rank_normalize_sources"])
train, source_values, source_confidence, source_addressed, gold = load_and_align_label_sources(ROOT / "train.csv", SOURCES)
audit_confidence, audit_addressed, leakage = mask_suspected_gold_leakage(
    source_values, source_confidence, source_addressed, gold
)
teacher = robust_label_ensemble(
    source_values, gold=None, synovitis_effusion_weight=synovitis_weight,
    source_confidence=audit_confidence, source_addressed=audit_addressed,
    rank_normalize_sources=rank_normalize,
)
gold_mask = np.isfinite(gold)
audit = bootstrap_macro_auc(gold, teacher.probabilities, gold_mask, repeats=2000)
print("label sources:", [str(path) for path in SOURCES])
print("gold macro AUC / 95% bootstrap interval:", audit)
bundle = robust_label_ensemble(
    source_values, gold=gold, synovitis_effusion_weight=synovitis_weight,
    source_confidence=source_confidence, source_addressed=source_addressed,
    rank_normalize_sources=rank_normalize,
)
output = Path("/kaggle/working/labels_with_folds.csv")
write_label_bundle(train, bundle, output)

import pandas as pd
frame = pd.read_csv(output)
gold_study = np.isfinite(gold).all(axis=1)
fold_targets = np.column_stack([
    frame[list(TARGETS)].to_numpy(),
    gold_study.astype(np.float32),
    np.nan_to_num(gold, nan=0.0),
])
frame["fold"] = grouped_multilabel_folds(
    fold_targets, frame["report_hash"].to_numpy(), n_folds=5, seed=config["seed"]
)
frame.to_csv(output, index=False)
gold_fold_counts = {
    str(int(fold)): int(count)
    for fold, count in frame.loc[gold_study, "fold"].value_counts().sort_index().items()
}
gold_positive_counts = {
    str(fold): {
        target: int(gold[(frame["fold"].to_numpy() == fold) & gold_study, index].sum())
        for index, target in enumerate(TARGETS)
    }
    for fold in range(5)
}
Path("/kaggle/working/label_audit.json").write_text(json.dumps({
    "sources": [str(path) for path in SOURCES],
    "rank_normalize_sources": rank_normalize,
    "gold_leakage_audit": leakage,
    "gold_auc": audit[0], "ci_low": audit[1], "ci_high": audit[2],
    "minimum_gold_auc": minimum_auc,
    "passes_0_89_gate": bool(audit[0] >= minimum_auc),
    "fold_counts": {
        str(int(fold)): int(count) for fold, count in frame["fold"].value_counts().sort_index().items()
    },
    "gold_fold_counts": gold_fold_counts,
    "gold_positive_counts": gold_positive_counts,
}, indent=2))
