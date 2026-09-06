#!/usr/bin/env python3
from __future__ import annotations

import argparse

from rsna_knee.validate import validate_submission


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission")
    parser.add_argument("--test", required=True)
    parser.add_argument("--allow-constant", action="store_true")
    args = parser.parse_args()
    print(validate_submission(args.submission, args.test, not args.allow_constant))


if __name__ == "__main__":
    main()

