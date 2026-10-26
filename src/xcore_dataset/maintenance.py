"""Controlled deletion of immutable published revisions.

Only `<dataset_id>/rev-NNNN` directories directly under the published root may be
removed. Raw, staging, jobs and Hub data are never touched. This is intentionally
separate from conversion and never runs implicitly.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

DATASET = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
REVISION = re.compile(r"rev-\d{4,}")


def delete_revision(root, repo_id: str, *, delete_from_hub: bool = False) -> dict:
    if delete_from_hub:
        raise ValueError(
            "Hub deletion is disabled; only local revisions can be removed"
        )
    if not isinstance(repo_id, str) or not repo_id.strip():
        raise ValueError("dataset_repo_id must be a non-empty string")
    parts = Path(repo_id).parts
    if len(parts) != 2 or parts[0] in (".", "..") or parts[1] in (".", ".."):
        raise ValueError("expected exactly <dataset_id>/<revision>")
    if not DATASET.fullmatch(parts[0]) or not REVISION.fullmatch(parts[1]):
        raise ValueError("invalid dataset or revision name")
    root = Path(root).expanduser().resolve()
    target = (root / parts[0] / parts[1]).resolve()
    if target.parent.parent != root:
        raise ValueError("dataset path escapes the published root")
    if any(
        part.is_symlink() for part in (target, *target.parents) if part != root.parent
    ):
        raise ValueError("symlinks are not allowed in published datasets")
    if not target.is_dir() or not (target / "READY.json").is_file():
        raise ValueError(f"published revision not found: {repo_id}")
    shutil.rmtree(target)
    parent = target.parent
    if parent != root and parent.is_dir() and not any(parent.iterdir()):
        parent.rmdir()
    return {
        "success": True,
        "message": f"Deleted {repo_id}",
        "dataset": parts[0],
        "revision": parts[1],
    }
