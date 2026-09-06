from __future__ import annotations

from collections import defaultdict

import numpy as np


def grouped_multilabel_folds(
    targets: np.ndarray,
    groups: np.ndarray,
    n_folds: int = 5,
    seed: int = 20260905,
) -> np.ndarray:
    """Greedy deterministic group assignment balancing soft target prevalence."""
    targets = np.asarray(targets, dtype=np.float64)
    groups = np.asarray(groups).astype(str)
    if targets.ndim != 2 or len(targets) != len(groups):
        raise ValueError("targets must be [N,T] and groups [N]")
    if n_folds < 2:
        raise ValueError("n_folds must be at least two")

    members: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        members[group].append(index)
    if len(members) < n_folds:
        raise ValueError("number of unique groups must be at least n_folds")
    rng = np.random.default_rng(seed)
    group_rows = []
    for group, indices in members.items():
        values = np.nan_to_num(targets[indices], nan=0.5).mean(axis=0)
        rarity = float(np.abs(values - 0.5).sum())
        group_rows.append((group, indices, values, rarity, float(rng.random())))
    group_rows.sort(key=lambda row: (-len(row[1]), -row[3], row[4], row[0]))

    target_totals = np.nan_to_num(targets, nan=0.5).sum(axis=0)
    expected_sums = target_totals / n_folds
    expected_count = len(targets) / n_folds
    fold_sums = np.zeros((n_folds, targets.shape[1]), dtype=np.float64)
    fold_counts = np.zeros(n_folds, dtype=np.int64)
    assignment = np.empty(len(targets), dtype=np.int64)

    for row_index, (group, indices, values, _, _) in enumerate(group_rows):
        # Seed every fold once.  Pure greedy minimisation can otherwise keep an
        # empty fold forever because an already-populated fold has a good mean.
        if row_index < n_folds:
            chosen = row_index
            assignment[indices] = chosen
            fold_sums[chosen] += values * len(indices)
            fold_counts[chosen] += len(indices)
            continue
        costs = []
        for fold in range(n_folds):
            candidate_sums = fold_sums.copy()
            candidate_counts = fold_counts.copy()
            candidate_sums[fold] += values * len(indices)
            candidate_counts[fold] += len(indices)
            target_balance = np.var(
                candidate_sums / np.maximum(expected_sums[None, :], 1.0), axis=0
            ).mean()
            size_balance = np.var(candidate_counts / max(expected_count, 1.0))
            costs.append(target_balance + 0.5 * size_balance)
        chosen = int(np.argmin(costs))
        assignment[indices] = chosen
        fold_sums[chosen] += values * len(indices)
        fold_counts[chosen] += len(indices)
    return assignment
