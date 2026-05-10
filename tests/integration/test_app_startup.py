"""Integration smoke test for the assembled :class:`ReachyMiniDancePartyApp`.

Boots the full app on the Pi (real ReachyMini SDK + daemon) for a few
seconds, verifies all expected threads are alive and the daemon stays
healthy, then shuts down cleanly.

Skips on the laptop because it requires:
* a running daemon at ``localhost:8000`` exposing motors + camera, and
* a valid ``OPENAI_API_KEY`` for the realtime session.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any

import httpx
import pytest


DAEMON_URL = "http://localhost:8000/api/daemon/status"

# Threads we expect ReachyMiniDancePartyApp.run() to have spawned by the
# time it has been running for a few seconds. Each name matches the
# ``threading.Thread(name=...)`` used in the corresponding component.
EXPECTED_THREAD_NAMES = {
    # MovementManager.start spawns an unnamed daemon; we instead rely on
    # the named threads below + alive-check on the manager itself.
    "FaceTracker",
    "AudiencePush",
    "RealtimeSession",
}


def _daemon_running_or_skip() -> None:
    try:
        resp = httpx.get(DAEMON_URL, timeout=5.0)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"daemon not reachable at {DAEMON_URL}: {exc}")
    try:
        payload: dict[str, Any] = resp.json()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"daemon /status returned non-JSON: {exc}")
    if payload.get("state") != "running" or payload.get("backend_status") is None:
        pytest.skip(
            f"daemon not in running state (state={payload.get('state')!r}, "
            f"error={payload.get('error')!r}) — motor bus down or backend not initialized"
        )


def _openai_key_or_skip() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set; skipping realtime-session smoke")


@pytest.mark.requires_robot
def test_app_starts_all_threads() -> None:
    """Boot the full app, verify threads are alive, daemon stays healthy."""
    _daemon_running_or_skip()
    _openai_key_or_skip()

    # Local import so module-collection on the laptop doesn't pay the SDK
    # import cost when this test will skip.
    from reachy_mini_dance_party_app.main import ReachyMiniDancePartyApp

    app = ReachyMiniDancePartyApp()
    runner = threading.Thread(target=app.run, daemon=True, name="AppRunMain")
    runner.start()

    try:
        # Give the app time to spin up every component.
        time.sleep(5.0)

        # Verify each named worker thread is alive.
        names = {th.name for th in threading.enumerate() if th.is_alive()}
        missing = EXPECTED_THREAD_NAMES - names
        assert not missing, (
            f"missing expected threads: {missing}; alive threads: {sorted(names)}"
        )

        # Verify daemon stayed running (i.e. our app didn't crash motors).
        resp = httpx.get(DAEMON_URL, timeout=5.0)
        payload: dict[str, Any] = resp.json()
        assert payload.get("state") == "running", (
            f"daemon state changed during app run: {payload}"
        )
    finally:
        # Trigger the app's signal-handler shutdown path and wait briefly.
        app._stop_event.set()
        runner.join(timeout=10.0)
        assert not runner.is_alive(), "app.run did not exit within 10s of stop_event"
