from pathlib import Path
import json
import numpy as np

exec(Path(next(Path("/kaggle/input").glob("*/kaggle_support.py"))).read_text(), globals())
REPO = bootstrap_repo()

from rsna_knee.config import load_config
from rsna_knee.constants import TARGETS
from rsna_knee.folds import grouped_multilabel_folds
from rsna_knee.labels import (
    load_and_align_label_sources,
    mask_suspected_gold_leakage,
    robust_label_ensemble,
    write_label_bundle,
)
from rsna_knee.metrics import bootstrap_macro_auc, bootstrap_macro_auc_delta

ROOT = competition_root()
public_sources = discover_label_sources()
open_llm = one_file(["/kaggle/input/**/open_llm_labels.csv"], "open_llm_labels.csv")
SOURCES = [*public_sources, open_llm]
config = load_config(REPO / "configs/base.yaml")
open_llm_audit_path = one_file(["/kaggle/input/**/open_llm_audit.json"], "open LLM audit")
open_llm_audit = json.loads(open_llm_audit_path.read_text(encoding="utf-8"))
maximum_parse_failure_rate = float(config["open_llm"]["maximum_parse_failure_rate"])
llm_parse_gate_passed = bool(
    float(open_llm_audit.get("parse_failure_rate", 1.0)) <= maximum_parse_failure_rate
)
if not llm_parse_gate_passed:
    print(f"WARNING rejecting open LLM candidate due to parse audit: {open_llm_audit}")
synovitis_weight = float(config["labels"]["synovitis_effusion_weight"])
minimum_auc = float(config["labels"]["minimum_gold_auc"])
rank_normalize = bool(config["labels"]["rank_normalize_sources"])
minimum_llm_gain = float(config["labels"]["minimum_llm_gold_gain"])
train, values, confidence, addressed, gold = load_and_align_label_sources(ROOT / "train.csv", SOURCES)
audit_confidence, audit_addressed, leakage = mask_suspected_gold_leakage(values, confidence, addressed, gold)
gold_mask = np.isfinite(gold)
public_count = len(public_sources)
public_teacher = robust_label_ensemble(
    values[:public_count], gold=None, synovitis_effusion_weight=synovitis_weight,
    source_confidence=audit_confidence[:public_count], source_addressed=audit_addressed[:public_count],
    rank_normalize_sources=rank_normalize,
)
llm_teacher = robust_label_ensemble(
    values, gold=None, synovitis_effusion_weight=synovitis_weight,
    source_confidence=audit_confidence, source_addressed=audit_addressed,
    rank_normalize_sources=rank_normalize,
)
public_audit = bootstrap_macro_auc(gold, public_teacher.probabilities, gold_mask, repeats=2000)
llm_audit = bootstrap_macro_auc(gold, llm_teacher.probabilities, gold_mask, repeats=2000)
llm_delta = bootstrap_macro_auc_delta(
    gold, llm_teacher.probabilities, public_teacher.probabilities, gold_mask, repeats=2000
)
use_llm = bool(llm_parse_gate_passed and llm_delta[0] >= minimum_llm_gain and llm_delta[1] > 0.0)
selected_count = len(SOURCES) if use_llm else public_count
audit = llm_audit if use_llm else public_audit
bundle = robust_label_ensemble(
    values[:selected_count], gold=gold, synovitis_effusion_weight=synovitis_weight,
    source_confidence=confidence[:selected_count], source_addressed=addressed[:selected_count],
    rank_normalize_sources=rank_normalize,
)
output = Path("/kaggle/working/labels_with_folds.csv")
write_label_bundle(train, bundle, output)

import pandas as pd
frame = pd.read_csv(output)
gold_study = np.isfinite(gold).all(axis=1)
fold_targets = np.column_stack([frame[list(TARGETS)].to_numpy(), gold_study.astype(np.float32), np.nan_to_num(gold)])
frame["fold"] = grouped_multilabel_folds(
    fold_targets, frame["report_hash"].to_numpy(), n_folds=5, seed=config["seed"]
)
frame.to_csv(output, index=False)
payload = {
    "candidate_sources": [str(path) for path in SOURCES],
    "selected_sources": [str(path) for path in SOURCES[:selected_count]],
    "rank_normalize_sources": rank_normalize,
    "open_llm_audit": open_llm_audit,
    "open_llm_parse_gate_passed": llm_parse_gate_passed,
    "public_only_gold_audit": {"point": public_audit[0], "ci_low": public_audit[1], "ci_high": public_audit[2]},
    "with_llm_gold_audit": {"point": llm_audit[0], "ci_low": llm_audit[1], "ci_high": llm_audit[2]},
    "llm_paired_gold_delta": {"point": llm_delta[0], "ci_low": llm_delta[1], "ci_high": llm_delta[2]},
    "minimum_llm_gold_gain": minimum_llm_gain,
    "llm_promoted": use_llm,
    "gold_leakage_audit": leakage,
    "gold_auc": audit[0], "ci_low": audit[1], "ci_high": audit[2],
    "minimum_gold_auc": minimum_auc,
    "passes_0_89_gate": bool(audit[0] >= minimum_auc),
    "fold_counts": {str(int(k)): int(v) for k, v in frame["fold"].value_counts().sort_index().items()},
    "gold_fold_counts": {
        str(int(k)): int(v) for k, v in frame.loc[gold_study, "fold"].value_counts().sort_index().items()
    },
    "gold_positive_counts": {
        str(fold): {
            target: int(gold[(frame["fold"].to_numpy() == fold) & gold_study, index].sum())
            for index, target in enumerate(TARGETS)
        }
        for fold in range(5)
    },
}
Path("/kaggle/working/label_audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps(payload, indent=2))
if not payload["passes_0_89_gate"]:
    raise RuntimeError("label ensemble still fails 0.89 after adding the open-source LLM; do not train")
