from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path


def bootstrap_repo() -> Path:
    embedded = os.environ.get("RSNA_KNEE_EMBEDDED_REPO")
    if embedded:
        root = Path(embedded)
        if (root / "src/rsna_knee/__init__.py").exists():
            sys.path.insert(0, str(root))
            sys.path.insert(0, str(root / "src"))
            return root
    candidates = sorted({
        *glob.glob("/kaggle/input/*/src/rsna_knee/__init__.py"),
        *glob.glob("/kaggle/input/datasets/*/*/**/src/rsna_knee/__init__.py", recursive=True),
    })
    if not candidates:
        raise FileNotFoundError("attach the private rsna-knee-code dataset to this notebook")
    root = Path(candidates[0]).parents[2]
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "src"))
    return root


def competition_root() -> Path:
    candidates = [
        Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
        Path("/kaggle/input/rsna-knee-abnormality-detection"),
    ]
    for path in candidates:
        if (path / "train.csv").exists() or (path / "test.csv").exists():
            return path
    discovered = sorted(Path("/kaggle/input").glob("**/rsna-knee-abnormality-detection"))
    if discovered:
        return discovered[0]
    raise FileNotFoundError("RSNA competition input was not attached")


def one_file(patterns: list[str], description: str) -> Path:
    matches = sorted({Path(path) for pattern in patterns for path in glob.glob(pattern, recursive=True)})
    files = [path for path in matches if path.is_file()]
    if len(files) != 1:
        raise FileNotFoundError(f"expected exactly one {description}, found {files}")
    return files[0]


def files(patterns: list[str], description: str) -> list[Path]:
    matches = sorted({Path(path) for pattern in patterns for path in glob.glob(pattern, recursive=True)})
    result = [path for path in matches if path.is_file()]
    if not result:
        raise FileNotFoundError(f"no {description} found for {patterns}")
    return result


def require_json_gate(patterns: list[str], key: str, description: str) -> dict:
    path = one_file(patterns, description)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get(key) is not True:
        raise RuntimeError(f"{description} gate {key!r} did not pass: {payload}")
    return payload


def require_cache_coverage(patterns: list[str], minimum_fraction: float) -> list[dict]:
    paths = files(patterns, "cache manifests")
    if len(paths) != 2:
        raise FileNotFoundError(f"expected two cache manifests, found {paths}")
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    for path, payload in zip(paths, payloads):
        studies = int(payload.get("studies", 0))
        valid = int(payload.get("valid_studies", 0))
        fraction = valid / max(studies, 1)
        if fraction < minimum_fraction:
            raise RuntimeError(
                f"cache coverage {fraction:.4f} from {path} is below {minimum_fraction:.4f}: {payload}"
            )
    return payloads


def discover_label_sources(excluded_roots: tuple[str, ...] = ("competitions",)) -> list[Path]:
    import pandas as pd

    from rsna_knee.constants import ID_COLUMN, TARGETS
    from rsna_knee.labels import target_column_candidates

    preferred_slugs = (
        "rsna-knee-llm-labels",
        "rsna-knee-llm-report-labels-sol56",
        "rsna-knee-llm-report-labels",
    )
    selected: list[Path] = []
    for slug in preferred_slugs:
        best: tuple[int, int, Path] | None = None
        roots = [Path("/kaggle/input") / slug]
        roots.extend(Path("/kaggle/input/datasets").glob(f"*/{slug}"))
        candidates = (path for root in roots if root.exists() for path in root.rglob("*"))
        for path in candidates:
            if path.suffix.lower() not in {".csv", ".parquet", ".pq"}:
                continue
            try:
                frame = pd.read_parquet(path).head(2) if path.suffix.lower() != ".csv" else pd.read_csv(path, nrows=2)
            except Exception:
                continue
            score = int(ID_COLUMN in frame) + sum(bool(target_column_candidates(frame.columns, target)) for target in TARGETS)
            normalized = [str(column).casefold() for column in frame.columns]
            auxiliary = sum("conf" in column or "verdict" in column or "addressed" in column for column in normalized)
            if score == 13 and (best is None or (score, auxiliary) > best[:2]):
                best = (score, auxiliary, path)
        if best is not None:
            selected.append(best[2])
    unique = list(dict.fromkeys(selected))
    if len(unique) < 3:
        raise FileNotFoundError(f"attach at least three public label datasets; discovered {unique}")
    return unique[:3]
