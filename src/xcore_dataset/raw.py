"""A closed episode is the only legal input to the offline converter."""

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from xcore_adapters.pose import validate_pose10, validate_wrench

from .integrity import inventory, safe_file, verify, write_json

STREAMS = ("cam1_rgb", "cam1_depth", "cam2_rgb", "cam2_depth")
FORMAT = "xcore_ee10_rgbd_v1"


def validate_manifest(meta):
    if meta.get("format") != FORMAT:
        raise ValueError("unsupported raw format")
    if meta.get("position_unit") != "m" or meta.get("rotation") != "columns_0_1":
        raise ValueError("unknown pose units or rotation convention")
    for key in ("base_frame", "tcp", "action_source", "task"):
        if not isinstance(meta.get(key), str) or not meta[key].strip():
            raise ValueError(f"missing manifest {key}")
    if type(meta.get("simulated")) is not bool:
        raise ValueError("manifest must state whether input is simulated")
    for camera in ("cam1", "cam2"):
        c = meta["cameras"][camera]
        scale = c["depth_scale_m_per_unit"]
        if not np.isfinite(scale) or not 0 < scale < 1:
            raise ValueError("invalid depth scale")
        for key in ("serial", "color_intrinsics", "depth_intrinsics", "depth_to_color"):
            if key not in c:
                raise ValueError(f"missing camera metadata {key}")


def validate_row(row):
    validate_pose10(row["state"])
    validate_pose10(row["action"])
    validate_wrench(row["wrench"], row["wrench_valid"])
    for value in (
        row["timestamp_ns"],
        *(row["images"][s]["timestamp_ns"] for s in STREAMS),
    ):
        if type(value) is not int or value < 0:
            raise ValueError("timestamps must be nonnegative integer nanoseconds")
    for s in STREAMS:
        frame = row["images"][s]
        if frame.get("clock_domain") != "remote_monotonic":
            raise ValueError("image clock must be mapped to remote_monotonic")
        uncertainty = frame.get("uncertainty_ns")
        if type(uncertainty) is not int or uncertainty < 0:
            raise ValueError("clock uncertainty must be explicit")
        if "clock_uncertainty_ns" in frame:
            clock = frame["clock_uncertainty_ns"]
            if type(clock) is not int or clock < 0:
                raise ValueError("clock uncertainty must be a nonnegative integer")
        if "age_ns" in frame:
            age = frame["age_ns"]
            if type(age) is not int or age < 0:
                raise ValueError("frame age must be a nonnegative integer")


class EpisodeWriter:
    def __init__(self, path, manifest):
        validate_manifest(manifest)
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=False)
        write_json(self.path / "manifest.json", manifest)
        self.stream = (self.path / "samples.jsonl.partial").open("x", encoding="utf-8")
        self.count, self.last_stamp = 0, -1

    def append(self, row):
        validate_row(row)
        if row["timestamp_ns"] <= self.last_stamp:
            raise ValueError("sample timestamps must increase")
        self.stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        self.last_stamp, self.count = row["timestamp_ns"], self.count + 1

    def close(self):
        if not self.stream.closed:
            self.stream.flush()
            os.fsync(self.stream.fileno())
            self.stream.close()

    def commit(self):
        self.close()
        if not self.count:
            raise ValueError("cannot commit an empty episode")
        samples = self.path / "samples.jsonl.partial"
        for row in map(json.loads, samples.read_text().splitlines()):
            for name, item in row["images"].items():
                read_image(self.path, item, name.endswith("depth"))
        for item in self.path.rglob("*"):
            if item.is_file():
                with item.open("rb") as f:
                    os.fsync(f.fileno())
        directories = sorted(
            (p for p in self.path.rglob("*") if p.is_dir()),
            key=lambda p: len(p.parts),
            reverse=True,
        )
        for directory in [*directories, self.path]:
            fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        samples.rename(self.path / "samples.jsonl")
        write_json(
            self.path / "COMMITTED.json",
            {"format": FORMAT, "frames": self.count, "files": inventory(self.path)},
        )


def read_episode(path):
    root = Path(path)
    seal = verify(root, "COMMITTED.json")
    meta = json.loads((root / "manifest.json").read_text())
    validate_manifest(meta)
    rows = [
        json.loads(line) for line in (root / "samples.jsonl").read_text().splitlines()
    ]
    if not rows or len(rows) != seal["frames"]:
        raise ValueError("episode count mismatch")
    previous = -1
    for row in rows:
        validate_row(row)
        if row["timestamp_ns"] <= previous:
            raise ValueError("sample timestamps must strictly increase")
        previous = row["timestamp_ns"]
    return meta, rows


def read_image(root, item, depth):
    with Image.open(safe_file(root, item["path"])) as image:
        array = np.asarray(image)
    if depth:
        if array.ndim != 2 or array.dtype != np.uint16:
            raise ValueError("depth must be lossless uint16 Z16")
    elif array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
        raise ValueError("color must be uint8 RGB")
    return array
