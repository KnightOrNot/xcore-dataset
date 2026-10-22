"""Immutable input/output inventories and atomic publication markers."""

import hashlib
import json
import os
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, allow_nan=False, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def safe_file(root, name):
    root = Path(root).resolve()
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("path escapes dataset")
    p = root / relative
    if any(part.is_symlink() for part in (p, *p.parents) if part != root.parent):
        raise ValueError("symlinks are not allowed in dataset payloads")
    if not p.is_file() or not p.resolve().is_relative_to(root):
        raise ValueError(f"missing or unsafe data file: {name}")
    return p


def inventory(root, exclude=()):
    root = Path(root)
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("dataset contains a symlink")
        if path.is_file() and path.relative_to(root).as_posix() not in exclude:
            result[path.relative_to(root).as_posix()] = {
                "bytes": path.stat().st_size,
                "sha256": file_hash(path),
            }
    return result


def verify(root, marker="READY.json"):
    root = Path(root)
    meta = json.loads(safe_file(root, marker).read_text())
    if not meta.get("files"):
        raise ValueError("empty integrity manifest")
    if inventory(root, (marker,)) != meta["files"]:
        raise ValueError("file inventory/checksum mismatch")
    return meta
