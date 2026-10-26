"""Resolve a published revision to the requested representation for local pull.

This explicit export does not change collection, save, or session conversion.
It reads the exact raw provenance of an immutable published revision.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import re
import uuid
from pathlib import Path

from .integrity import file_hash, safe_file, verify
from .published import REPRESENTATIONS, detect_representation


def revision_path(root, dataset, revision):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", dataset):
        raise ValueError("invalid dataset ID")
    if not re.fullmatch(r"rev-[0-9]{4,}", revision):
        raise ValueError("invalid revision")
    root = Path(root).expanduser().resolve()
    path = root / "lerobot" / dataset / revision
    safe_file(root / "lerobot", f"{dataset}/{revision}/READY.json")
    verify(path)
    return path


def export_revision(root, dataset, revision, representation, *, max_skew_ms):
    if representation not in REPRESENTATIONS:
        raise ValueError("representation must be pose10 or joints")
    root = Path(root).expanduser().resolve()
    source = revision_path(root, dataset, revision)

    def result(path):
        return {
            "dataset_id": path.parent.name,
            "revision": path.name,
            "representation": representation,
        }

    if detect_representation(source) == representation:
        return result(source)

    # Legacy publications still carry a checksummed quality report with raw
    # episode paths and COMMITTED hashes. Never derive raw by stripping a suffix.
    report = json.loads(safe_file(source, "quality_report.json").read_text())
    commits = report.get("source_commits")
    episodes = report.get("episodes")
    if not commits or not episodes or len(commits) != len(episodes):
        raise ValueError("published revision has no verifiable raw provenance")
    paths = [Path(item["path"]) for item in episodes]
    session = paths[0].parent.parent
    if session.parent != root / "raw" or any(
        path.parent != session / "episodes" for path in paths
    ):
        raise ValueError("source episodes must belong to one session under raw")
    safe_file(root / "raw", f"{session.name}/session.json")
    expected = {item["episode"]: item["sha256"] for item in commits}
    if len(expected) != len(paths) or set(expected) != {p.name for p in paths}:
        raise ValueError("source episode provenance mismatch")
    for path in paths:
        marker = safe_file(
            root / "raw", str(path.relative_to(root / "raw") / "COMMITTED.json")
        )
        if file_hash(marker) != expected[path.name]:
            raise ValueError("raw source changed since the selected revision")
    # The existing converter consumes the entire session. Refuse extra/partial
    # episodes rather than silently exporting different data from the selection.
    if set((session / "episodes").iterdir()) != set(paths):
        raise ValueError(
            "raw session has additional episodes; select its final revision"
        )
    fps = report["fps"]
    fingerprint = hashlib.sha256(
        json.dumps(
            [str(source.relative_to(root)), commits, representation, fps, max_skew_ms],
            sort_keys=True,
        ).encode()
    ).hexdigest()[:12]
    export_id = f"{session.name[:36]}-{representation}-{fingerprint}"
    destination = root / "lerobot" / export_id / "rev-0001"
    (root / "jobs").mkdir(parents=True, exist_ok=True)
    with (root / "jobs" / ".pull-export.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if destination.exists() or destination.is_symlink():
            revision_path(root, export_id, "rev-0001")
            if detect_representation(destination) != representation:
                raise ValueError("existing export representation mismatch")
            return result(destination)
        from .convert import convert

        convert(
            session,
            root / "staging" / ("pull-export-" + uuid.uuid4().hex),
            destination,
            fps=fps,
            max_skew_ms=max_skew_ms,
            representation=representation,
            expected_source_commits=commits,
        )
        return result(destination)
