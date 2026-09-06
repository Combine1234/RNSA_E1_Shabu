#!/usr/bin/env python3
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from rsna_knee.constants import TARGETS
from rsna_knee.folds import grouped_multilabel_folds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260905)
    args = parser.parse_args()
    frame = pd.read_parquet(args.labels) if args.labels.endswith((".parquet", ".pq")) else pd.read_csv(args.labels)
    targets = frame[list(TARGETS)].to_numpy(dtype=np.float64)
    frame["fold"] = grouped_multilabel_folds(targets, frame["report_hash"].to_numpy(), args.folds, args.seed)
    if args.output.endswith((".parquet", ".pq")):
        frame.to_parquet(args.output, index=False)
    else:
        frame.to_csv(args.output, index=False)
    print(frame["fold"].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()

