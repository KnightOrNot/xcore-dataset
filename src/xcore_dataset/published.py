"""Read-only inspection of published revisions and their representation.

New revisions carry representation.json. Legacy EE10 revisions are inferred from
the LeRobot feature shape; a reader never guesses a representation it cannot
prove from the payload.
"""

from __future__ import annotations

import json
from pathlib import Path

REPRESENTATIONS = ("pose10", "joints")


def detect_representation(path: Path) -> str:
    path = Path(path)
    marker = path / "representation.json"
    if marker.is_file():
        value = json.loads(marker.read_text(encoding="utf-8"))
        if value.get("format") != "xcore_representation_v1":
            raise ValueError("unsupported representation marker")
        representation = value.get("representation")
        if representation not in REPRESENTATIONS:
            raise ValueError(f"unknown representation: {representation}")
        return representation
    info = path / "meta/info.json"
    if not info.is_file():
        raise ValueError("dataset has neither representation.json nor meta/info.json")
    features = json.loads(info.read_text(encoding="utf-8")).get("features", {})
    shape = features.get("observation.state", {}).get("shape")
    if shape == [10]:
        return "pose10"
    if shape == [7]:
        return "joints"
    raise ValueError(
        f"cannot infer representation from observation.state shape {shape}"
    )


def published_datasets(root: Path) -> list[dict]:
    root = Path(root).expanduser().resolve()
    result = []
    if not root.is_dir():
        return result
    for ready in sorted(root.glob("*/*/READY.json")):
        path = ready.parent
        entry = {
            "dataset_id": path.parent.name,
            "revision": path.name,
            "representation": detect_representation(path),
            "path": str(path),
        }
        report = path / "quality_report.json"
        if report.is_file():
            value = json.loads(report.read_text(encoding="utf-8"))
            entry["simulated"] = value.get("simulated")
            entry["frames"] = value.get("output_frames")
            entry["episodes"] = len(value.get("episodes", []))
        result.append(entry)
    return result
