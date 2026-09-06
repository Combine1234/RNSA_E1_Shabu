#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob

from rsna_knee.config import load_config
from rsna_knee.train import train_fold


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--cache", action="append", required=True, help="Glob; repeat for multiple cache datasets")
    parser.add_argument("--family", choices=("model_a", "model_b"), required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--backbone-checkpoint")
    args = parser.parse_args()
    config = load_config(args.config)
    cache_paths = sorted({path for pattern in args.cache for path in glob.glob(pattern)})
    if not cache_paths:
        raise FileNotFoundError("no cache shards matched")
    result = train_fold(
        args.labels,
        cache_paths,
        args.output,
        config[args.family],
        args.fold,
        seed=int(config["seed"]),
        gold_weight=float(config["labels"]["gold_weight"]),
        backbone_checkpoint=args.backbone_checkpoint,
    )
    print(result)


if __name__ == "__main__":
    main()

