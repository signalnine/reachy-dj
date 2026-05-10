"""Smoke test for the lifted MovementManager (Task 6).

Requires a live ReachyMini daemon on localhost:8000 with motors enabled.
Asserts that we can spin up the worker, push a no-op move through it, and
shut it down cleanly without bumping the daemon's `nb_error` counter.
"""

from __future__ import annotations

import time
from typing import Any

import httpx
import numpy as np
import pytest
from numpy.typing import NDArray

from reachy_mini import ReachyMini
from reachy_mini.motion.move import Move
from reachy_mini.utils import create_head_pose

from reachy_mini_dance_party_app.moves import MovementManager


DAEMON_URL = "http://localhost:8000/api/daemon/status"


class _HoldPoseMove(Move):  # type: ignore[misc]
    """No-op move: holds the neutral pose for `duration` seconds.

    We return concrete head/antennas/body_yaw values so the worker actually
    composes and sends a `set_target` per tick; this exercises the full
    primary -> set_target path without producing any visible motion.
    """

    def __init__(self, duration: float = 1.5) -> None:
        self._duration = float(duration)
        self._neutral_head: NDArray[np.float64] = create_head_pose(
            0, 0, 0, 0, 0, 0, degrees=True
        ).astype(np.float64)
        self._neutral_antennas: NDArray[np.float64] = np.array(
            [-0.1745, 0.1745], dtype=np.float64
        )

    @property
    def duration(self) -> float:
        return self._duration

    def evaluate(
        self, t: float
    ) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None, float | None]:
        return self._neutral_head, self._neutral_antennas, 0.0


def _read_nb_error() -> int:
    resp = httpx.get(DAEMON_URL, timeout=5.0)
    resp.raise_for_status()
    payload: dict[str, Any] = resp.json()
    return int(payload["backend_status"]["control_loop_stats"]["nb_error"])


def _daemon_running_or_skip() -> None:
    resp = httpx.get(DAEMON_URL, timeout=5.0)
    payload: dict[str, Any] = resp.json()
    if payload.get("state") != "running" or payload.get("backend_status") is None:
        pytest.skip(
            f"daemon not in running state (state={payload.get('state')!r}, "
            f"error={payload.get('error')!r}) — motor bus down or backend not initialized"
        )


@pytest.mark.requires_robot
def test_movement_manager_runs_no_op_move_without_daemon_errors() -> None:
    _daemon_running_or_skip()
    nb_error_before = _read_nb_error()

    robot = ReachyMini()
    manager = MovementManager(current_robot=robot)

    try:
        manager.start()
        # Give the worker a tick to spin up, then enqueue and let it run.
        time.sleep(0.1)
        manager.queue_move(_HoldPoseMove(duration=1.5))
        time.sleep(2.0)
        # If the worker thread crashed it will be dead by now.
        assert manager._thread is not None and manager._thread.is_alive(), (
            "MovementManager worker thread died during no-op move"
        )
    finally:
        manager.stop()

    nb_error_after = _read_nb_error()
    assert nb_error_after == nb_error_before, (
        f"daemon nb_error increased during smoke test: "
        f"{nb_error_before} -> {nb_error_after}"
    )
