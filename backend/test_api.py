"""Tests for the backend API contract.

These tests verify that the backend API behaves correctly for all
frontend interaction scenarios, without requiring the heavy ML pipeline.
"""

import io
import time
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a fresh TestClient with process_video mocked out."""
    with patch("main.process_video") as mock_pv:
        mock_pv.return_value = {
            "filename": "test.mp4",
            "duration": "00:10",
            "resolution": "1920x1080",
            "processed_at": "2025-01-01 12:00:00",
            "detections": [
                {
                    "timestamp": "00:03",
                    "horse_id": "A12",
                    "person_name": "张三",
                    "confidence": "0.95_0.88",
                }
            ],
        }
        from main import app, tasks
        tasks.clear()
        yield TestClient(app)


@pytest.fixture
def mock_process_video():
    with patch("main.process_video") as mock_pv:
        yield mock_pv


# ── Upload endpoint ──────────────────────────────────────────────────

class TestUpload:
    def test_upload_returns_task_id(self, client):
        """POST /api/upload with a video file should return code=200 and task_id."""
        fake_video = io.BytesIO(b"\x00" * 1024)
        resp = client.post(
            "/api/upload",
            files={"video": ("test.mp4", fake_video, "video/mp4")},
        )
        data = resp.json()

        assert resp.status_code == 200
        assert data["code"] == 200
        assert "task_id" in data["data"]
        assert isinstance(data["data"]["task_id"], str)
        assert len(data["data"]["task_id"]) > 0

    def test_upload_without_file_returns_error(self, client):
        """POST /api/upload without a file should return 422."""
        resp = client.post("/api/upload")
        assert resp.status_code == 422

    def test_upload_creates_initial_task_status(self, client):
        """After upload, the task status should be queryable."""
        fake_video = io.BytesIO(b"\x00" * 1024)
        resp = client.post(
            "/api/upload",
            files={"video": ("test.mp4", fake_video, "video/mp4")},
        )
        task_id = resp.json()["data"]["task_id"]

        status_resp = client.get(f"/api/status?task_id={task_id}")
        status_data = status_resp.json()

        assert status_data["code"] == 200
        assert "status" in status_data["data"]
        assert "progress" in status_data["data"]


# ── Status endpoint ──────────────────────────────────────────────────

class TestStatus:
    def test_status_unknown_task_returns_404_code(self, client):
        """GET /api/status with a non-existent task_id should return code=404."""
        resp = client.get("/api/status?task_id=nonexistent-id")
        data = resp.json()

        assert data["code"] == 404

    def test_status_missing_task_id_param(self, client):
        """GET /api/status without task_id parameter should return 422."""
        resp = client.get("/api/status")
        assert resp.status_code == 422


# ── Processing result format (frontend contract) ────────────────────

class TestResultFormat:
    def _upload_and_wait(self, client, timeout=10):
        """Helper: upload a file and wait for processing to complete."""
        fake_video = io.BytesIO(b"\x00" * 1024)
        resp = client.post(
            "/api/upload",
            files={"video": ("test.mp4", fake_video, "video/mp4")},
        )
        task_id = resp.json()["data"]["task_id"]

        deadline = time.time() + timeout
        while time.time() < deadline:
            status_resp = client.get(f"/api/status?task_id={task_id}")
            status_data = status_resp.json()["data"]
            if status_data["status"] in ("completed", "error"):
                return task_id, status_data
            time.sleep(0.1)

        raise TimeoutError("Task did not complete in time")

    def test_completed_task_has_processed_video_url(self, client):
        """Completed tasks must include processed_video_url for frontend."""
        task_id, data = self._upload_and_wait(client)

        assert data["status"] == "completed"
        assert "processed_video_url" in data
        assert data["processed_video_url"].startswith("/videos/")

    def test_completed_task_has_result_with_required_fields(self, client):
        """The result dict must have filename, processed_at, and detections
        because the frontend template directly reads these fields."""
        task_id, data = self._upload_and_wait(client)

        result = data["result"]
        assert "filename" in result
        assert "processed_at" in result
        assert "detections" in result
        assert isinstance(result["detections"], list)

    def test_detection_items_have_required_fields(self, client):
        """Each detection item must have timestamp, horse_id, person_name,
        confidence — the frontend table columns bind to these exact keys."""
        task_id, data = self._upload_and_wait(client)

        detections = data["result"]["detections"]
        assert len(detections) > 0

        for det in detections:
            assert "timestamp" in det, f"Missing 'timestamp' in detection: {det}"
            assert "horse_id" in det, f"Missing 'horse_id' in detection: {det}"
            assert "person_name" in det, f"Missing 'person_name' in detection: {det}"
            assert "confidence" in det, f"Missing 'confidence' in detection: {det}"

    def test_completed_task_has_progress_100(self, client):
        """Frontend shows progress bar; completed tasks should be at 100%."""
        task_id, data = self._upload_and_wait(client)

        assert data["progress"] == 100


# ── Error handling ───────────────────────────────────────────────────

class TestErrorHandling:
    def test_processing_error_sets_error_status(self, client, mock_process_video):
        """When process_video raises an exception, status should be 'error'."""
        mock_process_video.side_effect = RuntimeError("Model load failed")

        fake_video = io.BytesIO(b"\x00" * 1024)
        resp = client.post(
            "/api/upload",
            files={"video": ("test.mp4", fake_video, "video/mp4")},
        )
        task_id = resp.json()["data"]["task_id"]

        deadline = time.time() + 5
        while time.time() < deadline:
            status_resp = client.get(f"/api/status?task_id={task_id}")
            status_data = status_resp.json()["data"]
            if status_data["status"] == "error":
                break
            time.sleep(0.1)

        assert status_data["status"] == "error"
        assert "message" in status_data

    def test_error_status_has_error_message(self, client, mock_process_video):
        """Error status should contain a user-readable message."""
        mock_process_video.side_effect = ValueError("Cannot open video")

        fake_video = io.BytesIO(b"\x00" * 1024)
        resp = client.post(
            "/api/upload",
            files={"video": ("test.mp4", fake_video, "video/mp4")},
        )
        task_id = resp.json()["data"]["task_id"]

        deadline = time.time() + 5
        while time.time() < deadline:
            status_resp = client.get(f"/api/status?task_id={task_id}")
            status_data = status_resp.json()["data"]
            if status_data["status"] == "error":
                break
            time.sleep(0.1)

        assert "Cannot open video" in status_data["message"]


# ── Security: internal fields must not leak to client ────────────────

class TestSecurityNoLeaks:
    def test_status_does_not_expose_file_path(self, client):
        """The server-side file_path should NOT be sent to the frontend.
        Leaking full filesystem paths is a security risk."""
        fake_video = io.BytesIO(b"\x00" * 1024)
        resp = client.post(
            "/api/upload",
            files={"video": ("test.mp4", fake_video, "video/mp4")},
        )
        task_id = resp.json()["data"]["task_id"]

        status_resp = client.get(f"/api/status?task_id={task_id}")
        status_data = status_resp.json()["data"]

        assert "file_path" not in status_data, \
            f"Internal file_path leaked to frontend: {status_data.get('file_path')}"

    def test_error_message_does_not_contain_server_paths(self, client, mock_process_video):
        """Error messages must not leak server filesystem paths."""
        mock_process_video.side_effect = ValueError(
            "Cannot open video: /home/server/data/videos/upload/abc_test.mp4"
        )

        fake_video = io.BytesIO(b"\x00" * 1024)
        resp = client.post(
            "/api/upload",
            files={"video": ("test.mp4", fake_video, "video/mp4")},
        )
        task_id = resp.json()["data"]["task_id"]

        deadline = time.time() + 5
        while time.time() < deadline:
            status_resp = client.get(f"/api/status?task_id={task_id}")
            status_data = status_resp.json()["data"]
            if status_data["status"] == "error":
                break
            time.sleep(0.1)

        msg = status_data["message"]
        assert "/home/" not in msg, \
            f"Server path leaked in error message: {msg}"


# ── Frontend error-status display bug ────────────────────────────────

class TestFrontendContract:
    def test_error_status_is_distinguishable_from_processing(self, client, mock_process_video):
        """The frontend only continues polling for 'processing' and 'uploading'.
        If status is 'error', polling stops. The frontend MUST show error UI,
        not a loading spinner. Verify the backend sends a distinct 'error' status."""
        mock_process_video.side_effect = RuntimeError("GPU OOM")

        fake_video = io.BytesIO(b"\x00" * 1024)
        resp = client.post(
            "/api/upload",
            files={"video": ("test.mp4", fake_video, "video/mp4")},
        )
        task_id = resp.json()["data"]["task_id"]

        deadline = time.time() + 5
        while time.time() < deadline:
            status_resp = client.get(f"/api/status?task_id={task_id}")
            status_data = status_resp.json()["data"]
            if status_data["status"] not in ("uploading", "processing"):
                break
            time.sleep(0.1)

        assert status_data["status"] == "error"
        assert status_data["status"] not in ("uploading", "processing")
