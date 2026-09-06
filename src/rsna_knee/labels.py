from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .constants import ID_COLUMN, TARGETS


UNKNOWN_VALUE = 0.5
_EXCLUDED_COLUMN_MARKERS = (
    "confidence", "conf", "addressed", "status", "verdict", "reason", "explanation", "gold", "mask", "fold"
)


@dataclass(frozen=True)
class LabelBundle:
    probabilities: np.ndarray
    confidence: np.ndarray
    addressed: np.ndarray
    gold: np.ndarray


def mask_suspected_gold_leakage(
    sources: np.ndarray,
    source_confidence: np.ndarray,
    source_addressed: np.ndarray,
    gold: np.ndarray,
    exact_threshold: float = 0.995,
    coverage_threshold: float = 0.95,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float | int | bool]]]:
    """Remove sources that copied official labels from the gold-only audit."""
    gold_mask = np.isfinite(gold)
    confidence = np.asarray(source_confidence, dtype=np.float64).copy()
    addressed = np.asarray(source_addressed, dtype=bool).copy()
    diagnostics = []
    for index, values in enumerate(np.asarray(sources, dtype=np.float64)):
        compared = gold_mask & addressed[index] & np.isfinite(values)
        coverage = float(compared.sum() / max(gold_mask.sum(), 1))
        exact = float(np.isclose(values[compared], gold[compared]).mean()) if compared.any() else 0.0
        suspected = bool(coverage >= coverage_threshold and exact >= exact_threshold)
        diagnostics.append({
            "source_index": index,
            "gold_coverage": coverage,
            "gold_exact_match": exact,
            "suspected_gold_copy": suspected,
        })
        if suspected:
            addressed[index, gold_mask] = False
            confidence[index, gold_mask] = 0.0
    return confidence, addressed, diagnostics


def normalize_report(text: object) -> str:
    value = "" if text is None else str(text)
    value = value.casefold()
    value = re.sub(r"\s+", " ", value).strip()
    return value


def report_hash(text: object) -> str:
    import hashlib

    return hashlib.sha256(normalize_report(text).encode("utf-8")).hexdigest()[:20]


def _normalized_column(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def target_column_candidates(columns: Iterable[object], target: str) -> list[str]:
    """Find common public-label naming variants while rejecting audit columns."""
    target_key = _normalized_column(target)
    accepted = {
        target_key,
        f"pseudo{target_key}",
        f"soft{target_key}",
        f"llm{target_key}",
        f"predicted{target_key}",
        f"label{target_key}",
        f"{target_key}score",
        f"{target_key}prob",
        f"{target_key}probability",
        f"{target_key}soft",
    }
    result = []
    for column in columns:
        key = _normalized_column(column)
        if any(marker in key for marker in _EXCLUDED_COLUMN_MARKERS):
            continue
        if key in accepted:
            result.append(str(column))
    return result


def resolve_target_column(frame, target: str) -> str | None:
    """Prefer the candidate containing the most usable report-derived values."""
    import pandas as pd

    candidates = target_column_candidates(frame.columns, target)
    ranked = []
    for column in candidates:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
        finite = np.isfinite(values)
        informative = finite & ~np.isclose(values, UNKNOWN_VALUE)
        exact = int(column == target)
        ranked.append((int(informative.sum()), int(finite.sum()), exact, column))
    return max(ranked)[-1] if ranked else None


def _resolve_aux_column(columns: Iterable[object], target: str, suffixes: tuple[str, ...]) -> str | None:
    target_key = _normalized_column(target)
    accepted = {f"{target_key}{suffix}" for suffix in suffixes}
    accepted |= {f"{suffix}{target_key}" for suffix in suffixes}
    matches = [str(column) for column in columns if _normalized_column(column) in accepted]
    return sorted(matches)[0] if matches else None


def _source_masks(frame, target: str, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, str | None]]:
    import pandas as pd

    verdict_column = _resolve_aux_column(frame.columns, target, ("verdict", "status"))
    addressed_column = _resolve_aux_column(frame.columns, target, ("addressed", "mask"))
    confidence_column = _resolve_aux_column(frame.columns, target, ("confidence", "conf"))
    finite = np.isfinite(values)
    addressed = finite & ~np.isclose(values, UNKNOWN_VALUE)
    if verdict_column is not None:
        verdict = frame[verdict_column].fillna("").astype(str).str.strip().str.upper()
        unknown = verdict.isin({"", "UNK", "UNKNOWN", "NOT ADDRESSED", "NOT_ADDRESSED", "N/A", "NA"})
        addressed = finite & ~unknown.to_numpy()
    elif addressed_column is not None:
        raw = frame[addressed_column]
        numeric = pd.to_numeric(raw, errors="coerce")
        if numeric.notna().any():
            addressed = finite & (numeric.fillna(0).to_numpy() > 0)
        else:
            text = raw.fillna("").astype(str).str.strip().str.casefold()
            addressed = finite & text.isin({"true", "yes", "y", "addressed"}).to_numpy()
    if confidence_column is None:
        confidence = addressed.astype(np.float64)
    else:
        confidence = pd.to_numeric(frame[confidence_column], errors="coerce").to_numpy(dtype=np.float64)
        confidence = np.where(np.isfinite(confidence), np.clip(confidence, 0.0, 1.0), 0.0)
        confidence = np.where(addressed, confidence, 0.0)
    return addressed, confidence, {
        "verdict": verdict_column,
        "addressed": addressed_column,
        "confidence": confidence_column,
    }


def robust_label_ensemble(
    sources: np.ndarray,
    gold: np.ndarray | None = None,
    unknown_value: float = UNKNOWN_VALUE,
    synovitis_effusion_weight: float = 0.55,
    source_confidence: np.ndarray | None = None,
    source_addressed: np.ndarray | None = None,
    rank_normalize_sources: bool = False,
) -> LabelBundle:
    """Combine [sources, studies, targets] probabilities without forcing unknowns negative."""
    sources = np.asarray(sources, dtype=np.float64)
    if sources.ndim != 3 or sources.shape[2] != len(TARGETS):
        raise ValueError(f"sources must be [S, N, {len(TARGETS)}]")
    if np.any((sources[np.isfinite(sources)] < 0) | (sources[np.isfinite(sources)] > 1)):
        raise ValueError("label probabilities must lie in [0, 1]")

    default_addressed = np.isfinite(sources) & ~np.isclose(sources, unknown_value)
    if source_addressed is None:
        addressed_by_source = default_addressed
    else:
        source_addressed = np.asarray(source_addressed, dtype=bool)
        if source_addressed.shape != sources.shape:
            raise ValueError("source_addressed shape does not match sources")
        addressed_by_source = source_addressed & np.isfinite(sources)
    if source_confidence is None:
        source_confidence = addressed_by_source.astype(np.float64)
    else:
        source_confidence = np.asarray(source_confidence, dtype=np.float64)
        if source_confidence.shape != sources.shape:
            raise ValueError("source_confidence shape does not match sources")
        source_confidence = np.where(
            addressed_by_source & np.isfinite(source_confidence), np.clip(source_confidence, 0.0, 1.0), np.nan
        )
    working = np.where(addressed_by_source, sources, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        probabilities = np.nanmedian(working, axis=0)
    addressed = addressed_by_source.any(axis=0)
    probabilities = np.where(addressed, probabilities, unknown_value)
    raw_probabilities = probabilities.copy()

    if rank_normalize_sources:
        from .metrics import average_ranks

        ranked = np.full_like(working, np.nan)
        for source_index in range(working.shape[0]):
            for target_index in range(working.shape[2]):
                known = np.isfinite(working[source_index, :, target_index])
                if known.any():
                    ranked[source_index, known, target_index] = average_ranks(
                        working[source_index, known, target_index]
                    )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            rank_median = np.nanmedian(ranked, axis=0)
        calibrated = np.full_like(probabilities, unknown_value)
        for target_index in range(probabilities.shape[1]):
            known = addressed[:, target_index] & np.isfinite(rank_median[:, target_index])
            if not known.any():
                continue
            target_mean = float(np.clip(raw_probabilities[known, target_index].mean(), 1e-4, 1.0 - 1e-4))
            logits = np.log(
                np.clip(rank_median[known, target_index], 1e-4, 1.0 - 1e-4)
                / np.clip(1.0 - rank_median[known, target_index], 1e-4, 1.0)
            )
            low, high = -20.0, 20.0
            for _ in range(60):
                midpoint = 0.5 * (low + high)
                mean = float((1.0 / (1.0 + np.exp(-(logits + midpoint)))).mean())
                if mean < target_mean:
                    low = midpoint
                else:
                    high = midpoint
            calibrated[known, target_index] = 1.0 / (1.0 + np.exp(-(logits + 0.5 * (low + high))))
        probabilities = calibrated

    # Agreement and coverage jointly express how much a target should contribute to BCE.
    coverage = addressed_by_source.mean(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        deviation = np.nanmedian(np.abs(working - raw_probabilities[None, :, :]), axis=0)
        source_quality = np.nanmedian(source_confidence, axis=0)
    deviation = np.nan_to_num(deviation, nan=0.5)
    source_quality = np.nan_to_num(source_quality, nan=0.0)
    confidence = np.clip(coverage * source_quality * (1.0 - 2.0 * deviation), 0.0, 1.0)
    confidence = np.where(addressed, np.maximum(confidence, 0.05), 0.0)

    effusion_idx = TARGETS.index("Effusion")
    synovitis_idx = TARGETS.index("Synovitis")
    missing_synovitis = ~addressed[:, synovitis_idx]
    effusion_known = addressed[:, effusion_idx]
    impute = missing_synovitis & effusion_known
    probabilities[impute, synovitis_idx] = (
        synovitis_effusion_weight * probabilities[impute, effusion_idx]
        + (1.0 - synovitis_effusion_weight) * unknown_value
    )
    confidence[impute, synovitis_idx] = 0.5 * confidence[impute, effusion_idx]
    addressed[impute, synovitis_idx] = True

    if gold is None:
        gold_mask = np.zeros_like(probabilities, dtype=bool)
    else:
        gold = np.asarray(gold, dtype=np.float64)
        if gold.shape != probabilities.shape:
            raise ValueError("gold shape does not match ensemble")
        gold_mask = np.isfinite(gold)
        probabilities = np.where(gold_mask, gold, probabilities)
        confidence = np.where(gold_mask, 1.0, confidence)
        addressed = addressed | gold_mask

    return LabelBundle(
        probabilities=probabilities.astype(np.float32),
        confidence=confidence.astype(np.float32),
        addressed=addressed,
        gold=gold_mask,
    )


def load_and_align_label_sources(
    train_csv: str | Path,
    source_paths: Iterable[str | Path],
):
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pandas is required for label file processing") from exc

    try:
        train = pd.read_csv(train_csv, encoding="utf-8")
    except UnicodeDecodeError:
        train = pd.read_csv(train_csv, encoding="latin-1")
    required = {ID_COLUMN, *TARGETS}
    missing_train = required - set(train.columns)
    if missing_train:
        raise ValueError(f"train CSV is missing columns: {sorted(missing_train)}")
    ids = train[ID_COLUMN].astype(str)
    if ids.duplicated().any():
        raise ValueError("training StudyInstanceUID values must be unique")

    arrays: list[np.ndarray] = []
    confidence_arrays: list[np.ndarray] = []
    addressed_arrays: list[np.ndarray] = []
    for path in source_paths:
        path = Path(path)
        frame = pd.read_parquet(path) if path.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(path)
        if ID_COLUMN not in frame:
            raise ValueError(f"{path} has no {ID_COLUMN}")
        frame[ID_COLUMN] = frame[ID_COLUMN].astype(str)
        frame = frame.drop_duplicates(ID_COLUMN).set_index(ID_COLUMN).reindex(ids)
        values = np.full((len(ids), len(TARGETS)), np.nan, dtype=np.float64)
        confidence = np.zeros_like(values)
        addressed = np.zeros_like(values, dtype=bool)
        mapping: dict[str, str | None] = {}
        for index, target in enumerate(TARGETS):
            column = resolve_target_column(frame, target)
            mapping[target] = column
            if column is not None:
                values[:, index] = pd.to_numeric(frame[column], errors="coerce").to_numpy()
                addressed[:, index], confidence[:, index], auxiliary = _source_masks(frame, target, values[:, index])
                mapping[f"{target}__aux"] = str(auxiliary)
        missing = [target for target, column in mapping.items() if column is None]
        if missing:
            raise ValueError(f"{path} has no resolvable label columns for {missing}")
        print(f"label source {path}: {mapping}")
        arrays.append(values)
        confidence_arrays.append(confidence)
        addressed_arrays.append(addressed)
    if len(arrays) < 3:
        raise ValueError("at least three independently generated label sources are required")

    gold = train[list(TARGETS)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    return train, np.stack(arrays), np.stack(confidence_arrays), np.stack(addressed_arrays), gold


def write_label_bundle(train, bundle: LabelBundle, output: str | Path) -> None:
    import pandas as pd

    output = Path(output)
    frame = pd.DataFrame({ID_COLUMN: train[ID_COLUMN].astype(str)})
    for index, target in enumerate(TARGETS):
        frame[target] = bundle.probabilities[:, index]
        frame[f"{target}__confidence"] = bundle.confidence[:, index]
        frame[f"{target}__addressed"] = bundle.addressed[:, index].astype(np.uint8)
        frame[f"{target}__gold"] = bundle.gold[:, index].astype(np.uint8)
    reports = train["Report"] if "Report" in train else pd.Series([""] * len(train))
    frame["report_hash"] = reports.map(report_hash)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() in {".parquet", ".pq"}:
        frame.to_parquet(output, index=False)
    else:
        frame.to_csv(output, index=False)
