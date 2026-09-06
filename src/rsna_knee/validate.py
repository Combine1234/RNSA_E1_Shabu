from __future__ import annotations

from pathlib import Path

import numpy as np

from .constants import ID_COLUMN, TARGETS


def validate_submission(submission_path: str | Path, test_path: str | Path, require_variation: bool = True) -> dict:
    import pandas as pd

    submission = pd.read_csv(submission_path)
    test = pd.read_csv(test_path)
    expected_columns = [ID_COLUMN, *TARGETS]
    if submission.columns.tolist() != expected_columns:
        raise ValueError(f"columns must exactly equal {expected_columns}")
    ids = submission[ID_COLUMN].astype(str)
    expected_ids = test[ID_COLUMN].astype(str)
    if len(submission) != len(test) or not ids.equals(expected_ids):
        raise ValueError("submission IDs and order must exactly match test.csv")
    if ids.duplicated().any():
        raise ValueError("submission contains duplicate StudyInstanceUID values")
    values = submission[list(TARGETS)].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("submission contains NaN or infinite predictions")
    if np.any((values < 0) | (values > 1)):
        raise ValueError("submission probabilities must lie in [0, 1]")
    constant = [target for i, target in enumerate(TARGETS) if np.ptp(values[:, i]) == 0]
    if require_variation and len(submission) > 3 and constant:
        raise ValueError(f"constant target predictions detected: {constant}")
    return {
        "rows": len(submission),
        "targets": len(TARGETS),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "constant_targets": constant,
    }

