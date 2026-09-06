#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob

import pandas as pd

from rsna_knee.constants import ID_COLUMN


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = sorted({path for pattern in args.input for path in glob.glob(pattern)})
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    if frame[ID_COLUMN].astype(str).duplicated().any():
        raise ValueError("OOF folds contain duplicate study IDs")
    frame.sort_values(ID_COLUMN).to_csv(args.output, index=False)
    print(f"combined {len(paths)} folds and {len(frame)} studies")


if __name__ == "__main__":
    main()

