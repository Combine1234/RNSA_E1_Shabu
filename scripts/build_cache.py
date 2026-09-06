#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from rsna_knee.cache import CacheSpec, build_cache_shard
from rsna_knee.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--studies", required=True)
    parser.add_argument("--series", required=True)
    parser.add_argument("--dicom-root", required=True)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=2)
    args = parser.parse_args()
    config = load_config(args.config)["cache"]
    spec = CacheSpec(**{key: config[key] for key in CacheSpec.__dataclass_fields__ if key in config})
    studies = pd.read_csv(args.studies)
    series = pd.read_csv(args.series)
    indices = np.array_split(np.arange(len(studies)), args.shard_count)[args.shard_index]
    selected = studies.iloc[indices].reset_index(drop=True)
    output = Path(args.output)
    if output.suffix:
        output = output.with_suffix("")
    if output.name == "cache" or output.is_dir():
        output = output / f"cache_{args.shard_index:02d}_of_{args.shard_count:02d}"
    metadata = build_cache_shard(selected, series, args.dicom_root, output, spec)
    print(metadata)


if __name__ == "__main__":
    main()
