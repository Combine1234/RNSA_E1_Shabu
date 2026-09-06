#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from rsna_knee.labels import (
    load_and_align_label_sources,
    mask_suspected_gold_leakage,
    robust_label_ensemble,
    write_label_bundle,
)
from rsna_knee.metrics import macro_auc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--source", action="append", required=True, help="Repeat for at least three CSV/Parquet sources")
    parser.add_argument("--output", required=True)
    parser.add_argument("--synovitis-effusion-weight", type=float, default=0.55)
    parser.add_argument("--minimum-gold-auc", type=float, default=0.89)
    args = parser.parse_args()
    train, sources, source_confidence, source_addressed, gold = load_and_align_label_sources(args.train, args.source)
    audit_confidence, audit_addressed, leakage = mask_suspected_gold_leakage(
        sources, source_confidence, source_addressed, gold
    )
    print(f"gold leakage audit: {leakage}")
    teacher = robust_label_ensemble(
        sources,
        gold=None,
        synovitis_effusion_weight=args.synovitis_effusion_weight,
        source_confidence=audit_confidence,
        source_addressed=audit_addressed,
    )
    gold_auc, _ = macro_auc(gold, teacher.probabilities, np.isfinite(gold))
    print(f"teacher macro AUC on 58-study gold audit: {gold_auc:.4f}")
    if gold_auc < args.minimum_gold_auc:
        print("WARNING label ensemble is below the promotion threshold; add/revise a free label source before training")
    bundle = robust_label_ensemble(
        sources,
        gold,
        synovitis_effusion_weight=args.synovitis_effusion_weight,
        source_confidence=source_confidence,
        source_addressed=source_addressed,
    )
    write_label_bundle(train, bundle, args.output)
    print(f"wrote {len(train)} studies to {Path(args.output)}")


if __name__ == "__main__":
    main()
