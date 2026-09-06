from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .constants import ID_COLUMN, TARGETS
from .metrics import average_ranks, binary_auc, bootstrap_macro_auc_delta, macro_auc


def simplex_grid(n_models: int, step: float) -> Iterable[np.ndarray]:
    units = int(round(1.0 / step))
    if units <= 0 or not np.isclose(units * step, 1.0):
        raise ValueError("grid step must divide one exactly")
    for cuts in itertools.combinations_with_replacement(range(units + 1), n_models - 1):
        boundaries = (0, *cuts, units)
        parts = np.diff(boundaries)
        for permutation in set(itertools.permutations(parts)):
            if sum(permutation) == units:
                yield np.asarray(permutation, dtype=np.float64) / units


def rank_fuse(predictions: np.ndarray, weights: np.ndarray) -> np.ndarray:
    predictions = np.asarray(predictions, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if predictions.ndim != 2 or predictions.shape[1] != len(weights):
        raise ValueError("predictions must be [N, models]")
    ranks = np.column_stack([average_ranks(predictions[:, i]) for i in range(predictions.shape[1])])
    return ranks @ (weights / weights.sum())


def select_target_weights(
    predictions: np.ndarray,
    truth: np.ndarray,
    addressed: np.ndarray,
    gold: np.ndarray,
    step: float = 0.05,
    minimum_gain: float = 0.002,
    maximum_gold_drop: float = 0.01,
) -> tuple[np.ndarray, dict[str, float]]:
    model_scores = np.asarray([binary_auc(truth[addressed], predictions[addressed, i]) for i in range(predictions.shape[1])])
    if not np.isfinite(model_scores).any():
        fallback = np.eye(predictions.shape[1])[0]
        return fallback, {
            "best_single_pseudo_auc": float("nan"),
            "best_single_gold_auc": float("nan"),
            "candidate_pseudo_auc": float("nan"),
            "candidate_gold_auc": float("nan"),
            "promoted": False,
        }
    best_model = int(np.nanargmax(model_scores))
    best_weights = np.eye(predictions.shape[1])[best_model]
    best_pseudo = float(model_scores[best_model])
    best_gold = binary_auc(truth[gold], predictions[gold, best_model])
    best_objective = best_pseudo + (0.25 * best_gold if np.isfinite(best_gold) else 0.0)
    candidate = best_weights
    candidate_pseudo = best_pseudo
    candidate_gold = best_gold
    for weights in simplex_grid(predictions.shape[1], step):
        fused = rank_fuse(predictions, weights)
        pseudo = binary_auc(truth[addressed], fused[addressed])
        gold_score = binary_auc(truth[gold], fused[gold])
        objective = pseudo + (0.25 * gold_score if np.isfinite(gold_score) else 0.0)
        if objective > best_objective:
            best_objective = objective
            candidate = weights
            candidate_pseudo = pseudo
            candidate_gold = gold_score
    gold_ok = not np.isfinite(best_gold) or not np.isfinite(candidate_gold) or candidate_gold >= best_gold - maximum_gold_drop
    gain_ok = candidate_pseudo >= best_pseudo + minimum_gain
    promoted = bool(gold_ok and gain_ok)
    chosen = candidate if promoted else best_weights
    return chosen, {
        "best_single_pseudo_auc": best_pseudo,
        "best_single_gold_auc": best_gold,
        "candidate_pseudo_auc": float(candidate_pseudo),
        "candidate_gold_auc": float(candidate_gold),
        "promoted": promoted,
    }


def fit_blend(
    oof_paths: list[str | Path],
    output: str | Path,
    step: float = 0.05,
    minimum_gain: float = 0.002,
    maximum_gold_drop: float = 0.01,
    bootstrap_repeats: int = 2000,
) -> dict:
    import pandas as pd

    if len(oof_paths) < 2:
        raise ValueError("at least two OOF files are required")
    frames = [pd.read_csv(path) for path in oof_paths]
    ids = frames[0][ID_COLUMN].astype(str)
    aligned = [frame.assign(**{ID_COLUMN: frame[ID_COLUMN].astype(str)}).set_index(ID_COLUMN).reindex(ids) for frame in frames]
    if any(frame.filter(like="pred__").isna().any().any() for frame in aligned):
        raise ValueError("OOF files do not contain identical study sets")
    result = {"models": [str(path) for path in oof_paths], "targets": {}}
    n, n_targets, n_models = len(ids), len(TARGETS), len(aligned)
    prediction_cube = np.empty((n, n_targets, n_models), dtype=np.float64)
    truths = np.empty((n, n_targets), dtype=bool)
    addressed_mask = np.empty((n, n_targets), dtype=bool)
    gold_mask = np.empty((n, n_targets), dtype=bool)
    candidate_predictions = np.empty((n, n_targets), dtype=np.float64)
    for target_index, target in enumerate(TARGETS):
        predictions = np.column_stack([frame[f"pred__{target}"].to_numpy() for frame in aligned])
        truth = aligned[0][f"true__{target}"].to_numpy() > 0.5
        addressed = aligned[0][f"weight__{target}"].to_numpy() > 0
        gold = aligned[0][f"gold__{target}"].to_numpy().astype(bool)
        for other in aligned[1:]:
            if not np.allclose(other[f"true__{target}"], aligned[0][f"true__{target}"], equal_nan=True):
                raise ValueError(f"OOF truth mismatch for {target}")
            if not np.array_equal(other[f"gold__{target}"].to_numpy().astype(bool), gold):
                raise ValueError(f"OOF gold mask mismatch for {target}")
        weights, diagnostics = select_target_weights(
            predictions,
            truth,
            addressed,
            gold,
            step=step,
            minimum_gain=minimum_gain,
            maximum_gold_drop=maximum_gold_drop,
        )
        result["targets"][target] = {"weights": weights.tolist(), **diagnostics}
        prediction_cube[:, target_index] = predictions
        truths[:, target_index] = truth
        addressed_mask[:, target_index] = addressed
        gold_mask[:, target_index] = gold
        candidate_predictions[:, target_index] = rank_fuse(predictions, weights)

    model_pseudo = [macro_auc(truths, prediction_cube[:, :, i], addressed_mask)[0] for i in range(n_models)]
    finite = np.isfinite(model_pseudo)
    if not np.any(finite):
        raise ValueError("no OOF target has both positive and negative addressed examples")
    best_single = int(np.nanargmax(model_pseudo))
    candidate_pseudo = macro_auc(truths, candidate_predictions, addressed_mask)[0]
    gold_rows = gold_mask.any(axis=1)
    model_gold = [
        macro_auc(truths[gold_rows], prediction_cube[gold_rows, :, i], gold_mask[gold_rows])[0]
        for i in range(n_models)
    ]
    best_gold_single = int(np.nanargmax(model_gold))
    gold_delta = bootstrap_macro_auc_delta(
        truths[gold_rows],
        candidate_predictions[gold_rows],
        prediction_cube[gold_rows, :, best_gold_single],
        gold_mask[gold_rows],
        repeats=bootstrap_repeats,
    )
    macro_ok = bool(candidate_pseudo >= model_pseudo[best_single] + minimum_gain)
    bootstrap_ok = bool(np.isfinite(gold_delta[1]) and gold_delta[1] > 0.0)
    promoted = macro_ok and bootstrap_ok
    if not promoted:
        fallback = np.eye(n_models)[best_single].tolist()
        for target in TARGETS:
            result["targets"][target]["weights"] = fallback
    result["promotion"] = {
        "model_pseudo_macro_auc": model_pseudo,
        "model_gold_macro_auc": model_gold,
        "best_single_model": best_single,
        "best_gold_single_model": best_gold_single,
        "candidate_pseudo_macro_auc": candidate_pseudo,
        "minimum_required_gain": minimum_gain,
        "gold_paired_bootstrap_delta": {"point": gold_delta[0], "ci_low": gold_delta[1], "ci_high": gold_delta[2]},
        "macro_gate_passed": macro_ok,
        "gold_bootstrap_gate_passed": bootstrap_ok,
        "ensemble_promoted": promoted,
        "fallback": None if promoted else "best_single_model_for_all_targets",
    }
    Path(output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def apply_blend(prediction_paths: list[str | Path], weights_path: str | Path, output: str | Path) -> None:
    import pandas as pd

    config = json.loads(Path(weights_path).read_text(encoding="utf-8"))
    frames = [pd.read_csv(path) for path in prediction_paths]
    ids = frames[0][ID_COLUMN].astype(str)
    aligned = [frame.assign(**{ID_COLUMN: frame[ID_COLUMN].astype(str)}).set_index(ID_COLUMN).reindex(ids) for frame in frames]
    result = pd.DataFrame({ID_COLUMN: ids})
    for target in TARGETS:
        predictions = np.column_stack([frame[target].to_numpy() for frame in aligned])
        weights = np.asarray(config["targets"][target]["weights"], dtype=np.float64)
        result[target] = rank_fuse(predictions, weights)
    result.to_csv(output, index=False)
