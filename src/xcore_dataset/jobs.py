"""Durable conversion queue: queued -> running -> ready/failed.

A job left in `running` by a crash is requeued on the next run. Staging from an
interrupted attempt is preserved under a suffixed name and never published.
Published revisions are immutable and are never overwritten.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from .convert import convert
from .integrity import write_json

ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")


def data_root(root) -> Path:
    return Path(root).expanduser().resolve()


def jobs_dir(root) -> Path:
    return data_root(root) / "jobs"


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("format") != "xcore_conversion_job_v1":
        raise ValueError(f"invalid job file: {path}")
    return value


def _save(path: Path, job: dict) -> None:
    write_json(path, job)


def next_revision(root: Path, dataset_id: str) -> str:
    base = root / "lerobot" / dataset_id
    numbers = []
    if base.is_dir():
        for item in base.iterdir():
            match = re.fullmatch(r"rev-(\d{4,})", item.name)
            if match:
                numbers.append(int(match.group(1)))
    for path in jobs_dir(root).glob("*.json"):
        job = _load(path)
        if job.get("dataset_id") == dataset_id and job.get("revision"):
            match = re.fullmatch(r"rev-(\d{4,})", job["revision"])
            if match:
                numbers.append(int(match.group(1)))
    return f"rev-{max(numbers, default=0) + 1:04d}"


def enqueue(
    root,
    session,
    *,
    fps: int,
    max_skew_ms: float,
    dataset_id: str | None = None,
) -> dict:
    root = data_root(root)
    session = Path(session).expanduser()
    if not session.is_absolute():
        session = root / "raw" / session
    session = session.resolve()
    if not session.is_relative_to((root / "raw").resolve()):
        raise ValueError("session must be under the data root's raw directory")
    if not (session / "episodes").is_dir():
        raise ValueError("session has no episodes directory")
    dataset_id = dataset_id or session.name
    if not ID.fullmatch(dataset_id):
        raise ValueError("invalid dataset_id")
    revision = next_revision(root, dataset_id)
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    job_id = f"{stamp}-{dataset_id}"
    jobs_dir(root).mkdir(parents=True, exist_ok=True)
    path = jobs_dir(root) / f"{job_id}.json"
    suffix = 1
    while path.exists():
        path = jobs_dir(root) / f"{job_id}-{suffix:02d}.json"
        suffix += 1
    job = {
        "format": "xcore_conversion_job_v1",
        "job_id": path.stem,
        "state": "queued",
        "attempts": 0,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "session": str(session),
        "dataset_id": dataset_id,
        "revision": revision,
        "fps": fps,
        "max_skew_ms": max_skew_ms,
        "destination": str(root / "lerobot" / dataset_id / revision),
        "error": None,
    }
    _save(path, job)
    return job


def list_jobs(root) -> list[dict]:
    directory = jobs_dir(root)
    if not directory.is_dir():
        return []
    return [
        dict(_load(path), file=str(path)) for path in sorted(directory.glob("*.json"))
    ]


def run_pending(root, *, max_attempts: int = 3) -> list[dict]:
    root = data_root(root)
    directory = jobs_dir(root)
    if not directory.is_dir():
        return []
    for path in sorted(directory.glob("*.json")):
        job = _load(path)
        if job["state"] == "ready" or job["state"] == "failed":
            continue
        if job["state"] == "running":
            job["attempts"] += 1
            if job["attempts"] > max_attempts:
                job["state"], job["error"] = "failed", "too many interrupted attempts"
            else:
                job["state"] = "queued"
            _save(path, job)
            if job["state"] == "failed":
                continue
        if job["state"] != "queued":
            continue
        job["state"] = "running"
        job["attempts"] += 1
        _save(path, job)
        staging = root / "staging" / job["job_id"]
        destination = Path(job["destination"])
        if staging.exists():
            stale = root / "staging" / f"{job['job_id']}.attempt-{job['attempts']:02d}"
            if stale.exists():
                raise FileExistsError(f"stale staging already exists: {stale}")
            staging.rename(stale)
        try:
            if destination.exists():
                raise FileExistsError("destination revision already exists")
            convert(
                job["session"],
                staging,
                destination,
                fps=job["fps"],
                max_skew_ms=job["max_skew_ms"],
            )
        except Exception as exc:  # noqa: BLE001 - recorded for retry/diagnosis
            job["state"] = "failed"
            job["error"] = f"{type(exc).__name__}: {exc}"
        else:
            job["state"] = "ready"
            job["error"] = None
            job["ready"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        _save(path, job)
    return list_jobs(root)


def retry(root, job_id: str) -> dict:
    root = data_root(root)
    path = jobs_dir(root) / f"{job_id}.json"
    job = _load(path)
    if job["state"] != "failed":
        raise ValueError("only failed jobs can be retried")
    job["state"] = "queued"
    job["error"] = None
    _save(path, job)
    return job
