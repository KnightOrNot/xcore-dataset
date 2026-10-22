import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from xcore_dataset.app import create_app
from xcore_dataset.convert import convert
from xcore_dataset.demo import make_demo
from xcore_dataset.integrity import inventory, verify, write_json
from xcore_dataset.raw import read_episode


def test_real_lerobot_dual_rgbd_roundtrip(tmp_path):
    source = make_demo(tmp_path / "raw", frames=4)
    output = convert(
        source, tmp_path / "staging", tmp_path / "published", fps=10, max_skew_ms=50
    )
    verify(output)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset("local/published", root=output, depth_output_unit="m")
    got = ds[0]
    assert got["observation.state"].shape == (10,)
    assert got["action"].shape == (10,)
    assert got["observation.wrench"].shape == (12,)
    assert got["observation.wrench_valid"].tolist() == [False, False]
    assert ds[2]["observation.wrench_valid"].tolist() == [True, True]
    np.testing.assert_allclose(ds[2]["observation.wrench"], np.arange(12) / 10)
    for c in ("cam1", "cam2"):
        assert got["observation.images." + c + "_rgb"].shape == (3, 48, 64)
        depth = got["observation.images." + c + "_depth"].numpy()[0]
        expected_m = 2 if c == "cam2" else 1
        np.testing.assert_allclose(depth[0, :3], [0, 65.535, expected_m], atol=1e-5)
    assert got["task"] == "合成数据：移动夹爪"
    with TestClient(create_app(tmp_path, lelab=False)) as client:
        sample = client.get("/api/sample", params={"dataset": "published", "index": 0})
        assert sample.status_code == 200
        assert sample.json()["depth"]["cam2"]["unit"] == "m"
        image = client.get(
            "/api/depth-preview.png",
            params={"dataset": "published", "camera": "cam2", "index": 0},
        )
        assert image.status_code == 200 and image.content.startswith(b"\x89PNG")
    assert read_episode(source)[0]["simulated"] is True
    with pytest.raises(FileExistsError):
        convert(source, tmp_path / "other", output, fps=10, max_skew_ms=50)
    (output / "quality_report.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        verify(output)


def test_session_with_two_episodes_becomes_one_dataset(tmp_path):
    session = tmp_path / "raw" / "session_test"
    for index in range(2):
        episode = session / "episodes" / f"episode_{index:06d}"
        episode.parent.mkdir(parents=True, exist_ok=True)
        make_demo(episode, frames=3)
    (session / "session.json").write_text(
        json.dumps({"session": "session_test", "task": "test"})
    )
    output = convert(
        session, tmp_path / "staging", tmp_path / "published", fps=10, max_skew_ms=50
    )
    verify(output)
    report = json.loads((output / "quality_report.json").read_text())
    assert report["episodes"] and len(report["episodes"]) == 2
    assert report["output_frames"] == 6
    assert len(report["source_commits"]) == 2
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset("local/published", root=output, depth_output_unit="m")
    assert len(ds) == 6
    assert ds.meta.total_episodes == 2
    assert ds[0]["task"] == "合成数据：移动夹爪"


def test_job_queue_publish_and_crash_recovery(tmp_path):
    from xcore_dataset.jobs import enqueue, list_jobs, run_pending

    root = tmp_path / "data"
    session = root / "raw" / "session_jobs"
    episode = session / "episodes" / "episode_000000"
    episode.parent.mkdir(parents=True)
    make_demo(episode, frames=3)
    job = enqueue(root, session, fps=10, max_skew_ms=50)
    assert job["state"] == "queued" and job["revision"] == "rev-0001"
    jobs = run_pending(root)
    assert jobs[0]["state"] == "ready"
    destination = Path(jobs[0]["destination"])
    verify(destination)
    assert not (root / "staging" / jobs[0]["job_id"]).exists()
    # A second run never republishes or overwrites the ready revision.
    assert run_pending(root)[0]["state"] == "ready"
    # Simulate a crash: a fresh job left in running is requeued and completed.
    job = enqueue(root, session, fps=10, max_skew_ms=50)
    assert job["revision"] == "rev-0002"
    path = root / "jobs" / (job["job_id"] + ".json")
    value = json.loads(path.read_text())
    value["state"] = "running"
    path.write_text(json.dumps(value))
    jobs = run_pending(root)
    done = [j for j in jobs if j["job_id"] == job["job_id"]]
    assert done and done[0]["state"] == "ready" and done[0]["attempts"] == 2
    assert list_jobs(root)[0]["state"] == "ready"


def test_partial_tamper_and_bad_sync_rejected(tmp_path):
    raw = make_demo(tmp_path / "raw", frames=3)
    seal = raw / "COMMITTED.json"
    seal.unlink()
    with pytest.raises(ValueError):
        read_episode(raw)
    rows = (raw / "samples.jsonl").read_text().splitlines()
    row = json.loads(rows[0])
    row["images"]["cam2_depth"]["timestamp_ns"] += 1_000_000_000
    rows[0] = json.dumps(row)
    (raw / "samples.jsonl").write_text("\n".join(rows) + "\n")
    write_json(seal, {"frames": 3, "files": inventory(raw)})
    with pytest.raises(ValueError, match="tolerance"):
        convert(raw, tmp_path / "staging", tmp_path / "out", fps=10, max_skew_ms=50)
    assert not (tmp_path / "out").exists()


def test_converter_does_not_write_into_raw(tmp_path):
    raw = make_demo(tmp_path / "raw", frames=3)
    with pytest.raises(ValueError, match="immutable"):
        convert(raw, raw / "staging", tmp_path / "out", fps=10, max_skew_ms=50)
    assert not (raw / "staging").exists()


def test_browser_never_initializes_hardware_and_blocks_mutation(tmp_path):
    with TestClient(create_app(tmp_path, lelab=True)) as c:
        assert c.get("/api/datasets").json() == {"datasets": []}
        assert c.get("/edit-dataset").status_code == 200
        assert c.post("/delete-dataset", json={}).status_code == 403
        assert c.get("/api/verify", params={"dataset": "../escape"}).status_code == 400
