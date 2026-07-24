"""Capture code, configuration, runtime, and Git provenance for experiments."""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy
import torch
import yaml


def capture_provenance(
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    """Return a stable source digest plus descriptive runtime metadata."""

    root = (
        Path(workspace_root).resolve()
        if workspace_root is not None
        else Path(__file__).resolve().parents[2]
    )
    source_files = tuple(_source_files(root))
    file_hashes = {
        str(path.relative_to(root)): _sha256(path.read_bytes())
        for path in source_files
    }
    combined = hashlib.sha256()
    for name, digest in file_hashes.items():
        combined.update(name.encode())
        combined.update(b"\0")
        combined.update(digest.encode())
        combined.update(b"\n")
    git_commit = _git(root, "rev-parse", "HEAD")
    git_status = _git(root, "status", "--porcelain", "--untracked-files=normal")
    return {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "workspace": str(root),
        "source_sha256": combined.hexdigest(),
        "source_file_count": len(file_hashes),
        "source_files": file_hashes,
        "git_commit": git_commit,
        "git_dirty": bool(git_status),
        "git_status": git_status.splitlines() if git_status else [],
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": numpy.__version__,
            "pyyaml": yaml.__version__,
        },
    }


def _source_files(root: Path) -> Iterable[Path]:
    candidates = [
        *root.glob("calmodel/**/*.py"),
        *root.glob("experiments/*.yaml"),
        *root.glob("tests/**/*.py"),
    ]
    for name in ("pyproject.toml", "uv.lock"):
        candidate = root / name
        if candidate.exists():
            candidates.append(candidate)
    return sorted(
        path
        for path in candidates
        if path.is_file() and "__pycache__" not in path.parts
    )


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _git(root: Path, *arguments: str) -> str | None:
    result = subprocess.run(
        ("git", *arguments),
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None
