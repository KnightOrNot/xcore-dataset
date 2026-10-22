import json

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
