#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import re
import shutil
import zipfile
from pathlib import Path


KERNELS = (
    "00_smoke",
    "01_labels",
    "01a_llm_probe",
    "01b_open_llm",
    "01c_labels_plus_llm",
    "02a_orientation_audit",
    "02_cache_0",
    "02_cache_1",
    "02b_raptor_preflight",
    "03_train_dino",
    "04_train_coatnet",
    "05_select_blend",
    "05_benchmark",
    "06_submission",
)
DATASET_ITEMS = ("src", "configs", "assets", "tests", "README.md", "LICENSE", "pyproject.toml", "kaggle_support.py")
DENIED_SUFFIXES = {".dcm", ".pt", ".pth", ".ckpt", ".bin", ".npy", ".npz", ".parquet"}
SUPPORT_BOOTSTRAP_LINE = 'exec(Path(next(Path("/kaggle/input").glob("*/kaggle_support.py"))).read_text(), globals())'


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def replace_username(template: Path, username: str) -> dict:
    payload = template.read_text(encoding="utf-8").replace("__KAGGLE_USERNAME__", username)
    return json.loads(payload)


def embedded_code_prelude(repo: Path) -> str:
    """Build a small source-only fallback because Kaggle dataset mounts can race."""
    buffer = io.BytesIO()
    included = ("src", "configs", "assets", "README.md", "LICENSE", "pyproject.toml", "kaggle_support.py")
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for item in included:
            source = repo / item
            paths = sorted(source.rglob("*")) if source.is_dir() else [source]
            for path in paths:
                if path.is_file() and "__pycache__" not in path.parts:
                    archive.write(path, path.relative_to(repo))
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f'''# Self-contained source/config fallback; contains no competition data or model weights.
import base64 as _base64
import io as _io
import os as _os
from pathlib import Path as _EmbeddedPath
import zipfile as _zipfile
_EMBEDDED_REPO = _EmbeddedPath("/tmp/rsna-knee-code-{hashlib.sha256(buffer.getvalue()).hexdigest()[:12]}")
if not (_EMBEDDED_REPO / "src/rsna_knee/__init__.py").exists():
    _EMBEDDED_REPO.mkdir(parents=True, exist_ok=True)
    with _zipfile.ZipFile(_io.BytesIO(_base64.b64decode("{payload}"))) as _archive:
        _archive.extractall(_EMBEDDED_REPO)
_os.environ["RSNA_KNEE_EMBEDDED_REPO"] = str(_EMBEDDED_REPO)
def _rsna_support_path():
    _mounted = sorted(_EmbeddedPath("/kaggle/input").glob("*/kaggle_support.py"))
    return _mounted[0] if _mounted else _EMBEDDED_REPO / "kaggle_support.py"
'''


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a small, raw-data-free Kaggle upload bundle")
    parser.add_argument("--username", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{1,49}", args.username):
        raise ValueError("Kaggle username must contain only lowercase letters, digits and hyphens")
    repo = Path(__file__).resolve().parents[1]
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty bundle directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    dataset = output / "dataset"
    dataset.mkdir()
    for item in DATASET_ITEMS:
        source = repo / item
        destination = dataset / item
        shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__")) if source.is_dir() else shutil.copy2(source, destination)
    metadata = replace_username(repo / "kaggle/dataset-metadata.template.json", args.username)
    (dataset / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    kernels_root = output / "kernels"
    kernels_root.mkdir()
    prelude = embedded_code_prelude(repo)
    for name in KERNELS:
        destination = kernels_root / name
        destination.mkdir()
        source_code = (repo / f"kaggle/{name}/kernel.py").read_text(encoding="utf-8")
        if SUPPORT_BOOTSTRAP_LINE not in source_code:
            raise RuntimeError(f"{name} does not contain the expected support bootstrap line")
        source_code = source_code.replace(
            SUPPORT_BOOTSTRAP_LINE,
            'exec(_rsna_support_path().read_text(encoding="utf-8"), globals())',
            1,
        )
        (destination / "kernel.py").write_text(prelude + "\n" + source_code, encoding="utf-8")
        metadata = replace_username(repo / f"kaggle/{name}/kernel-metadata.template.json", args.username)
        (destination / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    records = []
    for path in sorted(output.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.casefold() in DENIED_SUFFIXES or path.stat().st_size > 50 * 1024 * 1024:
            raise RuntimeError(f"upload bundle contains a denied data/model artifact: {path}")
        records.append({"path": str(path.relative_to(output)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    (output / "bundle_manifest.json").write_text(json.dumps({"files": records}, indent=2), encoding="utf-8")
    print(json.dumps({"bundle": str(output), "files": len(records), "kernels": list(KERNELS)}, indent=2))


if __name__ == "__main__":
    main()
