"""Read sealed raw, align samples, write and read back a real LeRobot dataset.

`source` may be one sealed episode or a session directory containing
`episodes/episode_*/COMMITTED.json`. A session becomes one LeRobot dataset with
one episode per sealed raw episode.
"""

import bisect
import importlib.metadata
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from xcore_adapters.pose import POSE_NAMES, WRENCH_NAMES

from .integrity import file_hash, inventory, verify, write_json
from .raw import STREAMS, read_episode, read_image


def nearest(times, target):
    k = bisect.bisect_left(times, target)
    choices = [i for i in (k - 1, k) if 0 <= i < len(times)]
    return min(choices, key=lambda i: (abs(times[i] - target), times[i]))


@dataclass
class Episode:
    path: Path
    meta: dict
    rows: list
    times: list
    streams: dict
    selected: list = None
    errors: dict = None
    used: dict = None


def episode_paths(source: Path) -> list[Path]:
    source = Path(source)
    if (source / "episodes").is_dir():
        paths = sorted(p for p in (source / "episodes").iterdir() if p.is_dir())
        if not paths:
            raise ValueError("session has no episode directories")
        for path in paths:
            if not (path / "COMMITTED.json").is_file():
                raise ValueError(f"episode is not sealed: {path.name}")
        return paths
    return [source]


def load_episode(path: Path) -> Episode:
    meta, rows = read_episode(path)
    times = [r["timestamp_ns"] for r in rows]
    streams = {}
    for stream in STREAMS:
        # One physical image may be referenced by multiple robot samples.
        unique = {}
        for row in rows:
            frame = row["images"][stream]
            key = frame["timestamp_ns"]
            if key in unique and unique[key] != frame:
                raise ValueError("conflicting image metadata at same timestamp")
            unique[key] = frame
        streams[stream] = [unique[t] for t in sorted(unique)]
    return Episode(path, meta, rows, times, streams)


def build_features(episodes: list[Episode]) -> dict:
    features = {
        "observation.state": dict(dtype="float32", shape=(10,), names=POSE_NAMES),
        "action": dict(dtype="float32", shape=(10,), names=POSE_NAMES),
        "observation.wrench": dict(dtype="float32", shape=(12,), names=WRENCH_NAMES),
        "observation.wrench_valid": dict(
            dtype="bool", shape=(2,), names=["sensor1", "sensor2"]
        ),
    }
    shapes = {}
    for episode in episodes:
        for stream in STREAMS:
            array = read_image(
                episode.path, episode.streams[stream][0], stream.endswith("depth")
            )
            shape = (*array.shape, 1) if stream.endswith("depth") else array.shape
            if stream in shapes and shapes[stream] != shape:
                raise ValueError(f"{stream} dimensions changed between episodes")
            shapes[stream] = shape
    for stream, shape in shapes.items():
        depth = stream.endswith("depth")
        feature = dict(
            dtype="image" if depth else "video",
            shape=shape,
            names=["height", "width", "channels"],
        )
        if depth:
            feature["info"] = dict(is_depth_map=True, depth_unit="m")
        features["observation.images." + stream] = feature
    return features


def select(episode: Episode, fps: int, max_skew_ms: float):
    times = episode.times
    targets = [
        times[0] + round(i * 1e9 / fps)
        for i in range(math.floor((times[-1] - times[0]) * fps / 1e9) + 1)
    ]
    selected, max_error, used = [], {}, {s: set() for s in STREAMS}
    for stamp in targets:
        i = nearest(times, stamp)
        errors = {"robot": abs(times[i] - stamp)}
        images = {}
        for stream in STREAMS:
            options = episode.streams[stream]
            j = nearest([f["timestamp_ns"] for f in options], stamp)
            item = options[j]
            error = abs(item["timestamp_ns"] - stamp) + item["uncertainty_ns"]
            errors[stream] = error
            used[stream].add(j)
            images[stream] = item
        stamps = [f["timestamp_ns"] for f in images.values()]
        errors["rgbd_pair"] = max(stamps) - min(stamps)
        if any(v > max_skew_ms * 1e6 for v in errors.values()):
            raise ValueError(f"timestamp tolerance exceeded: {errors}")
        for name, error in errors.items():
            max_error[name] = max(max_error.get(name, 0), error / 1e6)
        selected.append((i, images))
    return targets, selected, max_error, used


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
    paths = episode_paths(source)
    episodes = [load_episode(path) for path in paths]
    features = build_features(episodes)
    for episode in episodes:
        (episode.selected, episode.errors, episode.used) = select(
            episode, fps, max_skew_ms
        )[1:]

    staging.parent.mkdir(parents=True, exist_ok=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if staging.parent.stat().st_dev != destination.parent.stat().st_dev:
        raise ValueError("staging and publication must share one filesystem")
    dataset = LeRobotDataset.create(
        repo_id="local/" + destination.name,
        root=staging,
        fps=fps,
        features=features,
        robot_type="rokae_cr7_ee10",
        image_writer_threads=2,
    )
    audits = []
    try:
        for index, episode in enumerate(episodes):
            for i, images in episode.selected:
                row = episode.rows[i]
                frame = {
                    "observation.state": np.asarray(row["state"], np.float32),
                    "action": np.asarray(row["action"], np.float32),
                    "observation.wrench": np.asarray(row["wrench"], np.float32),
                    "observation.wrench_valid": np.asarray(row["wrench_valid"], bool),
                    "task": episode.meta["task"],
                }
                for stream, item in images.items():
                    depth = stream.endswith("depth")
                    array = read_image(episode.path, item, depth)
                    if depth:
                        camera = episode.meta["cameras"][stream.split("_")[0]]
                        array = array.astype(np.float32)[..., None] * np.float32(
                            camera["depth_scale_m_per_unit"]
                        )
                    if tuple(array.shape) != tuple(
                        features["observation.images." + stream]["shape"]
                    ):
                        raise ValueError("stream dimensions changed during episode")
                    frame["observation.images." + stream] = array
                dataset.add_frame(frame)
            dataset.save_episode(parallel_encoding=False)
            audits.append((index, episode))
    finally:
        dataset.finalize()

    loaded = LeRobotDataset(
        "local/" + destination.name, root=staging, depth_output_unit="m"
    )
    expected = sum(len(episode.selected) for episode in episodes)
    if len(loaded) != expected:
        raise ValueError("LeRobot readback frame count mismatch")
    position = 0
    for episode in episodes:
        for i, images in episode.selected:
            got = loaded[position]
            position += 1
            row = episode.rows[i]
            for key, rawkey in (
                ("observation.state", "state"),
                ("action", "action"),
                ("observation.wrench", "wrench"),
                ("observation.wrench_valid", "wrench_valid"),
            ):
                np.testing.assert_allclose(
                    got[key].numpy(), row[rawkey], rtol=1e-6, atol=1e-6
                )
            if got["task"] != episode.meta["task"]:
                raise ValueError("task text lost during conversion")
            for stream in STREAMS:
                if stream.endswith("depth"):
                    camera = episode.meta["cameras"][stream.split("_")[0]]
                    expected_depth = read_image(
                        episode.path, images[stream], True
                    ).astype(np.float32) * np.float32(camera["depth_scale_m_per_unit"])
                    np.testing.assert_allclose(
                        got["observation.images." + stream].numpy()[0],
                        expected_depth,
                        rtol=1e-6,
                        atol=1e-7,
                    )
    del loaded

    if (source / "session.json").is_file():
        session = json.loads((source / "session.json").read_text())
        write_json(
            staging / "xcore_manifest.json",
            dict(session=session, episodes=[e.meta for e in episodes]),
        )
    else:
        write_json(staging / "xcore_manifest.json", episodes[0].meta)
    with (staging / "sampling_audit.jsonl").open("x", encoding="utf-8") as audit:
        for episode_index, episode in audits:
            targets = [
                episode.times[0] + round(i * 1e9 / fps)
                for i in range(
                    math.floor((episode.times[-1] - episode.times[0]) * fps / 1e9) + 1
                )
            ]
            for stamp, (i, images) in zip(targets, episode.selected, strict=True):
                audit.write(
                    json.dumps(
                        dict(
                            episode=episode_index,
                            episode_path=str(episode.path.relative_to(source))
                            if episode.path != source
                            else ".",
                            target_timestamp_ns=stamp,
                            source_row=i,
                            state_timestamp_ns=episode.rows[i]["timestamp_ns"],
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
    max_error = {}
    for episode in episodes:
        for name, value in episode.errors.items():
            max_error[name] = max(max_error.get(name, 0), value)
    write_json(
        staging / "quality_report.json",
        dict(
            source_commits=[
                dict(
                    episode=episode.path.name,
                    sha256=file_hash(episode.path / "COMMITTED.json"),
                )
                for episode in episodes
            ],
            episodes=[
                dict(
                    index=index,
                    path=str(episode.path),
                    input_frames=len(episode.rows),
                    output_frames=len(episode.selected),
                )
                for index, episode in enumerate(episodes)
            ],
            input_frames=sum(len(e.rows) for e in episodes),
            output_frames=expected,
            fps=fps,
            max_error_ms=max_error,
            reused_frames={
                stream: sum(
                    len(episode.selected) - len(episode.used[stream])
                    for episode in episodes
                )
                for stream in STREAMS
            },
            valid_force_samples=np.sum(
                [r["wrench_valid"] for e in episodes for r in e.rows], axis=0
            ).tolist(),
            simulated=any(e.meta["simulated"] for e in episodes),
            readback_verified=True,
        ),
    )
    write_json(staging / "READY.json", dict(version=1, files=inventory(staging)))
    verify(staging)
    for episode in episodes:
        verify(episode.path, "COMMITTED.json")
    staging.rename(destination)
    return destination
