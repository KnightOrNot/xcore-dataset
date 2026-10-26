import json

import pytest
from fastapi.testclient import TestClient

from xcore_dataset.app import create_app
from xcore_dataset.maintenance import delete_revision


def published(root, dataset="ds", revision="rev-0001"):
    path = root / dataset / revision
    path.mkdir(parents=True)
    (path / "READY.json").write_text("{}")
    (path / "representation.json").write_text(
        json.dumps({"format": "xcore_representation_v1", "representation": "pose10"})
    )
    return path


def test_delete_only_published_revision(tmp_path):
    root = tmp_path / "lerobot"
    target = published(root)
    result = delete_revision(root, "ds/rev-0001")
    assert result["success"] is True
    assert not target.exists()
    assert not (root / "ds").exists()


@pytest.mark.parametrize(
    "repo_id",
    ["../ds/rev-0001", "/etc/passwd", "ds", "ds/rev-0001/..", "raw/rev-0001", "ds/tmp"],
)
def test_delete_rejects_unsafe_or_unpublished(tmp_path, repo_id):
    root = tmp_path / "lerobot"
    published(root)
    with pytest.raises(ValueError):
        delete_revision(root, repo_id)


def test_delete_rejects_hub_and_missing_ready(tmp_path):
    root = tmp_path / "lerobot"
    published(root)
    with pytest.raises(ValueError, match="Hub"):
        delete_revision(root, "ds/rev-0001", delete_from_hub=True)
    (root / "ds" / "rev-0001" / "READY.json").unlink()
    with pytest.raises(ValueError, match="not found"):
        delete_revision(root, "ds/rev-0001")


def test_delete_http_route_is_explicitly_enabled(tmp_path):
    root = tmp_path / "lerobot"
    published(root)
    raw = tmp_path / "raw" / "session"
    raw.mkdir(parents=True)
    enabled = TestClient(create_app(root, lelab=False, allow_delete=True))
    response = enabled.post(
        "/delete-dataset",
        json={"dataset_repo_id": "../raw/session", "delete_from_hub": False},
    )
    assert response.status_code == 200 and response.json()["success"] is False
    assert raw.exists()
    response = enabled.post(
        "/delete-dataset",
        json={"dataset_repo_id": "ds/rev-0001", "delete_from_hub": True},
    )
    assert response.json()["success"] is False
    response = enabled.post(
        "/delete-dataset",
        json={"dataset_repo_id": "ds/rev-0001", "delete_from_hub": False},
    )
    assert response.json()["success"] is True
    assert not (root / "ds" / "rev-0001").exists()

    published(root, "other")
    disabled = TestClient(create_app(root, lelab=False, allow_delete=False))
    response = disabled.post(
        "/delete-dataset",
        json={"dataset_repo_id": "other/rev-0001", "delete_from_hub": False},
    )
    assert response.status_code == 403
    assert (root / "other" / "rev-0001").exists()
