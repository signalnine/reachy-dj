"""Smoke test for the lifted CameraWorker (Task 7).

Requires a live ReachyMini daemon on localhost:8000 with the camera available.
Asserts that we can spin up the worker, capture at least one frame within 2s,
then shut down cleanly.

If the daemon is currently holding the camera, release it first via:
    curl -s -X POST http://localhost:8000/api/media/release
"""

from __future__ import annotations

import time
from typing import Any

import httpx
import numpy as np
import pytest

from reachy_mini import ReachyMini

from reachy_mini_dance_party_app.vision.camera_worker import CameraWorker


DAEMON_URL = "http://localhost:8000/api/daemon/status"


def _daemon_running_or_skip() -> None:
    """Skip if the daemon backend isn't initialized.

    The SDK's `ReachyMini()` connects via WebSocket (port 8443) which only
    accepts when the daemon's backend is up. If motors are dropped the
    backend never initializes and any SDK connect would hang/fail; skip
    rather than spend the timeout.
    """
    resp = httpx.get(DAEMON_URL, timeout=5.0)
    payload: dict[str, Any] = resp.json()
    if payload.get("state") != "running" or payload.get("backend_status") is None:
        pytest.skip(
            f"daemon not in running state (state={payload.get('state')!r}, "
            f"error={payload.get('error')!r}) — SDK client cannot connect"
        )


@pytest.mark.requires_robot
def test_camera_worker_captures_a_frame() -> None:
    _daemon_running_or_skip()
    robot = ReachyMini()
    # No HeadTracker injected — the dance-party app uses its own face_tracker
    # downstream of get_latest_frame(); this test only validates the frame pump.
    worker = CameraWorker(reachy_mini=robot, head_tracker=None)
    worker.set_head_tracking_enabled(False)

    try:
        worker.start()
        # Give the camera pipeline ~2s to warm up and emit a frame.
        time.sleep(2.0)
        frame = worker.get_latest_frame()
    finally:
        worker.stop()

    assert frame is not None, "CameraWorker produced no frame after 2s"
    assert isinstance(frame, np.ndarray), f"frame should be ndarray, got {type(frame)!r}"
    assert frame.ndim == 3, f"frame should be HxWxC, got shape {frame.shape}"
    h, w, c = frame.shape
    assert h > 0 and w > 0 and c > 0, f"frame has zero dimension: shape={frame.shape}"
