#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import pandas as pd

from rsna_knee.blend import apply_blend
from rsna_knee.cache import CacheSpec, RawStudySource
from rsna_knee.config import load_config
from rsna_knee.infer import apply_no_data_fallback, benchmark_inference, predict_groups
from rsna_knee.validate import validate_submission


def parse_groups(values: list[str]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--group values must be FAMILY=GLOB")
        family, pattern = value.split("=", 1)
        paths = sorted(glob.glob(pattern))
        if not paths:
            raise FileNotFoundError(f"no checkpoints matched {pattern}")
        result[family] = paths
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", required=True)
    parser.add_argument("--series", required=True)
    parser.add_argument("--dicom-root", required=True)
    parser.add_argument("--group", action="append", required=True)
    parser.add_argument("--blend-weights", required=True)
    parser.add_argument("--output-dir", default="/kaggle/working")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    cache_config = config["cache"]
    spec = CacheSpec(**{key: cache_config[key] for key in CacheSpec.__dataclass_fields__ if key in cache_config})
    test = pd.read_csv(args.test)
    series = pd.read_csv(args.series)
    source = RawStudySource(test, series, args.dicom_root, spec)
    groups = parse_groups(args.group)
    if args.benchmark:
        result = benchmark_inference(
            source,
            groups,
            sample_studies=min(int(config["inference"]["benchmark_sample_studies"]), len(source)),
            test_studies=int(config["inference"]["test_studies"]),
            offsets=config["inference"]["tta_offsets"],
        )
        print(json.dumps(result, indent=2))
        if not result["passes_runtime_gate"]:
            raise RuntimeError("three-pass TTA failed the 7.5-hour runtime promotion gate")
        return
    output_dir = Path(args.output_dir)
    predictions = predict_groups(
        source,
        groups,
        output_dir,
        offsets=config["inference"]["tta_offsets"],
        runtime_limit_seconds=int(config["inference"]["runtime_limit_seconds"]),
        hard_stop_seconds=int(config["inference"]["hard_stop_seconds"]),
    )
    ordered_paths = [predictions[family] for family in groups]
    submission = output_dir / "submission.csv"
    apply_blend(ordered_paths, args.blend_weights, submission)
    apply_no_data_fallback(submission, output_dir / "inference_summary.json")
    print(validate_submission(submission, args.test, require_variation=len(test) > 3))


if __name__ == "__main__":
    main()
