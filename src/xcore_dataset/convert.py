"""Read sealed raw, align samples, write and read back a real LeRobot dataset."""

import bisect
import importlib.metadata
import json
import math
from pathlib import Path

import numpy as np
from xcore_adapters.pose import POSE_NAMES, WRENCH_NAMES

from .integrity import file_hash, inventory, verify, write_json
from .raw import STREAMS, read_episode, read_image


def nearest(times, target):
    k = bisect.bisect_left(times, target)
    choices = [i for i in (k - 1, k) if 0 <= i < len(times)]
    return min(choices, key=lambda i: (abs(times[i] - target), times[i]))


def convert(source, staging, destination, *, fps, max_skew_ms):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    source, staging, destination = map(Path, (source, staging, destination))
    if (
        type(fps) is not int
        or fps < 1
        or not math.isfinite(max_skew_ms)
        or max_skew_ms <= 0
    ):
        raise ValueError("invalid conversion frequency/tolerance")
    if any(p.exists() or p.is_symlink() for p in (staging, destination)):
        raise FileExistsError("staging and destination must be new directories")
    if any(
        p.resolve().is_relative_to(source.resolve()) for p in (staging, destination)
    ):
        raise ValueError("conversion must not write inside immutable raw input")
    meta, rows = read_episode(source)
    times = [r["timestamp_ns"] for r in rows]
    targets = [
        times[0] + round(i * 1e9 / fps)
        for i in range(math.floor((times[-1] - times[0]) * fps / 1e9) + 1)
    ]
    streams = {}
    for s in STREAMS:
        # One physical image may be referenced by multiple robot samples.
        unique = {}
        for r in rows:
            frame = r["images"][s]
            key = frame["timestamp_ns"]
            if key in unique and unique[key] != frame:
                raise ValueError("conflicting image metadata at same timestamp")
            unique[key] = frame
        streams[s] = [unique[t] for t in sorted(unique)]
    features = {
        "observation.state": dict(dtype="float32", shape=(10,), names=POSE_NAMES),
        "action": dict(dtype="float32", shape=(10,), names=POSE_NAMES),
        "observation.wrench": dict(dtype="float32", shape=(12,), names=WRENCH_NAMES),
        "observation.wrench_valid": dict(
            dtype="bool", shape=(2,), names=["sensor1", "sensor2"]
        ),
    }
    for s in STREAMS:
        a = read_image(source, streams[s][0], s.endswith("depth"))
        depth = s.endswith("depth")
        shape = (*a.shape, 1) if depth else a.shape
        f = dict(
            dtype="image" if depth else "video",
            shape=shape,
            names=["height", "width", "channels"],
        )
        if depth:
            f["info"] = dict(is_depth_map=True, depth_unit="m")
        features["observation.images." + s] = f
    selected, max_error, used = [], {}, {s: set() for s in STREAMS}
    for stamp in targets:
        i = nearest(times, stamp)
        errors = {"robot": abs(times[i] - stamp)}
        images = {}
        for s in STREAMS:
            options = streams[s]
            j = nearest([f["timestamp_ns"] for f in options], stamp)
            item = options[j]
            error = abs(item["timestamp_ns"] - stamp) + item["uncertainty_ns"]
            errors[s] = error
            used[s].add(j)
            images[s] = item
        stamps = [f["timestamp_ns"] for f in images.values()]
        errors["rgbd_pair"] = max(stamps) - min(stamps)
        if any(v > max_skew_ms * 1e6 for v in errors.values()):
            raise ValueError(f"timestamp tolerance exceeded: {errors}")
        for name, error in errors.items():
            max_error[name] = max(max_error.get(name, 0), error / 1e6)
        selected.append((i, images))
    staging.parent.mkdir(parents=True, exist_ok=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if staging.parent.stat().st_dev != destination.parent.stat().st_dev:
        raise ValueError("staging and publication must share a filesystem")
    dataset = LeRobotDataset.create(
        repo_id="local/" + destination.name,
        root=staging,
        fps=fps,
        features=features,
        robot_type="rokae_cr7_ee10",
        image_writer_threads=2,
    )
    try:
        for i, images in selected:
            row = rows[i]
            frame = {
                "observation.state": np.asarray(row["state"], np.float32),
                "action": np.asarray(row["action"], np.float32),
                "observation.wrench": np.asarray(row["wrench"], np.float32),
                "observation.wrench_valid": np.asarray(row["wrench_valid"], bool),
                "task": meta["task"],
            }
            for s, item in images.items():
                depth = s.endswith("depth")
                a = read_image(source, item, depth)
                if depth:
                    camera = meta["cameras"][s.split("_")[0]]
                    a = a.astype(np.float32)[..., None] * np.float32(
                        camera["depth_scale_m_per_unit"]
                    )
                if tuple(a.shape) != tuple(
                    features["observation.images." + s]["shape"]
                ):
                    raise ValueError("stream dimensions changed during episode")
                frame["observation.images." + s] = a
            dataset.add_frame(frame)
        dataset.save_episode(parallel_encoding=False)
    finally:
        dataset.finalize()
    # Native loader, explicit meter depth; no mocked dataset factory.
    loaded = LeRobotDataset(
        "local/" + destination.name, root=staging, depth_output_unit="m"
    )
    if len(loaded) != len(selected):
        raise ValueError("LeRobot readback frame count mismatch")
    for n, (i, images) in enumerate(selected):
        got = loaded[n]
        for key, rawkey in (
            ("observation.state", "state"),
            ("action", "action"),
            ("observation.wrench", "wrench"),
            ("observation.wrench_valid", "wrench_valid"),
        ):
            np.testing.assert_allclose(
                got[key].numpy(), rows[i][rawkey], rtol=1e-6, atol=1e-6
            )
        if got["task"] != meta["task"]:
            raise ValueError("task text lost during conversion")
        for s in STREAMS:
            if s.endswith("depth"):
                scale = meta["cameras"][s.split("_")[0]]["depth_scale_m_per_unit"]
                expected = read_image(source, images[s], True).astype(
                    np.float32
                ) * np.float32(scale)
                np.testing.assert_allclose(
                    got["observation.images." + s].numpy()[0],
                    expected,
                    rtol=1e-6,
                    atol=1e-7,
                )
    del loaded
    write_json(staging / "xcore_manifest.json", meta)
    with (staging / "sampling_audit.jsonl").open("x", encoding="utf-8") as audit:
        for stamp, (i, images) in zip(targets, selected, strict=True):
            audit.write(
                json.dumps(
                    dict(
                        target_timestamp_ns=stamp,
                        source_row=i,
                        state_timestamp_ns=rows[i]["timestamp_ns"],
                        images=images,
                    ),
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )
    write_json(
        staging / "versions.json",
        {
            p: importlib.metadata.version(p)
            for p in (
                "xcore-dataset",
                "xcore-adapters",
                "lerobot",
                "torch",
                "torchcodec",
            )
        },
    )
    write_json(
        staging / "quality_report.json",
        dict(
            source_commit_sha256=file_hash(source / "COMMITTED.json"),
            input_frames=len(rows),
            output_frames=len(selected),
            fps=fps,
            max_error_ms=max_error,
            reused_frames={s: len(selected) - len(used[s]) for s in STREAMS},
            valid_force_samples=np.sum(
                [r["wrench_valid"] for r in rows], axis=0
            ).tolist(),
            simulated=meta["simulated"],
            readback_verified=True,
        ),
    )
    write_json(staging / "READY.json", dict(version=1, files=inventory(staging)))
    verify(staging)
    # Detect a source mutation during conversion before publishing anything.
    verify(source, "COMMITTED.json")
    staging.rename(destination)
    return destination
