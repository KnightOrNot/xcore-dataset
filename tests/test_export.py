import json

import pytest

from xcore_dataset.convert import convert
from xcore_dataset.demo import make_demo
from xcore_dataset.export import export_revision
from xcore_dataset.integrity import inventory, write_json


@pytest.fixture
def publication(tmp_path):
    session = tmp_path / "raw" / "pull-test"
    episode = session / "episodes" / "episode_000000"
    episode.parent.mkdir(parents=True)
    make_demo(episode, frames=3)
    write_json(session / "session.json", {"session": session.name})
    destination = tmp_path / "lerobot" / "pull-test" / "rev-0001"
    convert(
        session, tmp_path / "staging" / "initial", destination, fps=10, max_skew_ms=100
    )
    return tmp_path, session, destination


def test_select_pose_or_joints_with_actual_loader_and_reuse(publication):
    root, _, original = publication
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    pose = export_revision(root, "pull-test", "rev-0001", "pose10", max_skew_ms=100)
    assert pose["dataset_id"] == "pull-test"
    joints = export_revision(root, "pull-test", "rev-0001", "joints", max_skew_ms=100)
    destination = root / "lerobot" / joints["dataset_id"] / joints["revision"]
    ds = LeRobotDataset("local/export", root=destination, depth_output_unit="m")
    assert len(ds) == 3
    assert ds[0]["observation.state"].shape == (7,)
    assert ds[0]["action"].shape == (7,)
    assert ds[0]["observation.state"][-1] == pytest.approx(0.25)
    assert ds[0]["action"][-1] == pytest.approx(0.3)
    assert ds[0]["observation.images.cam1_depth"].shape == (1, 48, 64)
    assert (
        export_revision(root, "pull-test", "rev-0001", "joints", max_skew_ms=100)
        == joints
    )
    assert (original / "READY.json").is_file()
    assert len(list((root / "lerobot").glob("*/*/READY.json"))) == 2


def test_export_rejects_changed_raw_and_traversal(publication):
    root, session, _ = publication
    with pytest.raises(ValueError, match="dataset ID"):
        export_revision(root, "../raw", "rev-0001", "joints", max_skew_ms=100)
    with pytest.raises(ValueError, match="revision"):
        export_revision(root, "pull-test", "..", "joints", max_skew_ms=100)
    marker = session / "episodes/episode_000000/COMMITTED.json"
    marker.write_text(marker.read_text() + "\n")
    with pytest.raises(ValueError, match="raw source changed"):
        export_revision(root, "pull-test", "rev-0001", "joints", max_skew_ms=100)


def test_missing_joint_fields_fail_without_publishing(publication):
    root, session, original = publication
    episode = session / "episodes/episode_000000"
    rows = [
        json.loads(line)
        for line in (episode / "samples.jsonl").read_text().splitlines()
    ]
    for row in rows:
        del row["joints_rad"]
    (episode / "samples.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    (episode / "COMMITTED.json").unlink()
    write_json(
        episode / "COMMITTED.json", {"frames": len(rows), "files": inventory(episode)}
    )
    # Publish a fresh pose revision with genuine provenance but no joint fields.
    convert(
        session,
        root / "staging/no-joints",
        original.parent / "rev-0002",
        fps=10,
        max_skew_ms=100,
    )
    with pytest.raises(ValueError, match="joint"):
        export_revision(root, "pull-test", "rev-0002", "joints", max_skew_ms=100)
    assert len(list((root / "lerobot").glob("*/*/READY.json"))) == 2
