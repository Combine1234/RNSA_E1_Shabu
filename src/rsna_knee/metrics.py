from __future__ import annotations

import numpy as np


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Return 0..1 percentile ranks with deterministic average handling of ties."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("average_ranks expects one dimension")
    n = len(values)
    if n == 0:
        return values.copy()
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranked = np.empty(n, dtype=np.float64)
    start = 0
    while start < n:
        end = start + 1
        while end < n and sorted_values[end] == sorted_values[start]:
            end += 1
        ranked[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranked / max(n - 1, 1)


def binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=np.float64)
    valid = np.isfinite(y_true) & np.isfinite(y_score)
    y_true = y_true[valid].astype(np.int8)
    y_score = y_score[valid]
    positives = y_true == 1
    negatives = y_true == 0
    n_pos = int(positives.sum())
    n_neg = int(negatives.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = average_ranks(y_score) * max(len(y_score) - 1, 1) + 1.0
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def macro_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if y_true.shape != y_score.shape or y_true.ndim != 2:
        raise ValueError("y_true and y_score must have identical [N, T] shapes")
    if mask is None:
        mask = np.ones_like(y_true, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != y_true.shape:
            raise ValueError("mask shape does not match targets")
    aucs = np.array(
        [binary_auc(y_true[mask[:, t], t], y_score[mask[:, t], t]) for t in range(y_true.shape[1])],
        dtype=np.float64,
    )
    value = float(np.nanmean(aucs)) if np.isfinite(aucs).any() else float("nan")
    return value, aucs


def bootstrap_macro_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    mask: np.ndarray,
    repeats: int = 1000,
    seed: int = 20260905,
) -> tuple[float, float, float]:
    point, _ = macro_auc(y_true, y_score, mask)
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    n = len(y_true)
    for _ in range(repeats):
        idx = rng.integers(0, n, size=n)
        value, _ = macro_auc(y_true[idx], y_score[idx], mask[idx])
        if np.isfinite(value):
            samples.append(value)
    if not samples:
        return point, float("nan"), float("nan")
    low, high = np.quantile(samples, [0.025, 0.975])
    return point, float(low), float(high)


def bootstrap_macro_auc_delta(
    y_true: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray,
    repeats: int = 2000,
    seed: int = 20260905,
) -> tuple[float, float, float]:
    """Paired study bootstrap for candidate minus reference macro AUC."""
    candidate_point, _ = macro_auc(y_true, candidate, mask)
    reference_point, _ = macro_auc(y_true, reference, mask)
    point = candidate_point - reference_point
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    n = len(y_true)
    for _ in range(repeats):
        indices = rng.integers(0, n, size=n)
        left, _ = macro_auc(y_true[indices], candidate[indices], mask[indices])
        right, _ = macro_auc(y_true[indices], reference[indices], mask[indices])
        if np.isfinite(left) and np.isfinite(right):
            samples.append(left - right)
    if not samples:
        return point, float("nan"), float("nan")
    low, high = np.quantile(samples, [0.025, 0.975])
    return point, float(low), float(high)
