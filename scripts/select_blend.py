#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from rsna_knee.blend import fit_blend


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--minimum-gain", type=float, default=0.002)
    parser.add_argument("--maximum-gold-drop", type=float, default=0.01)
    args = parser.parse_args()
    print(json.dumps(fit_blend(
        args.oof, args.output, args.step, args.minimum_gain, args.maximum_gold_drop
    ), indent=2))


if __name__ == "__main__":
    main()
