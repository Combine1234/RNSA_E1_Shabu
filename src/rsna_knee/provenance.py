from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Iterable


def file_sha256(path: str | Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def write_artifact_manifest(
    paths: Iterable[str | Path],
    output: str | Path,
    metadata: dict | None = None,
) -> dict:
    import importlib.metadata

    files = []
    for value in sorted(Path(path) for path in paths):
        files.append({"name": value.name, "bytes": value.stat().st_size, "sha256": file_sha256(value)})
    versions = {"python": platform.python_version()}
    for package in ("torch", "timm", "numpy", "pandas", "pydicom"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    manifest = {"files": files, "versions": versions, "metadata": metadata or {}}
    Path(output).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
