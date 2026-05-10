"""Integration test for PlaybackEngine against the real sounddevice backend.

Skip-aware: skips on hosts that have no audio output device (CI / headless
laptop) and skips when the click-track fixture hasn't been generated yet.
This file is designed to run on the Pi.
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.requires_robot


def _audio_device_or_skip() -> None:
    """Skip the test if no audio output device is available."""
    try:
        import sounddevice as sd
    except OSError as exc:  # PortAudio missing on the host.
        pytest.skip(f"sounddevice unavailable: {exc}")
        return

    try:
        devices = sd.query_devices()
    except Exception as exc:  # PortAudio init may fail on headless hosts.
        pytest.skip(f"sounddevice.query_devices failed: {exc}")
        return

    has_output = any(d.get("max_output_channels", 0) > 0 for d in devices)
    if not has_output:
        pytest.skip("no audio output device available on this host")


def _fixture_or_skip(fixtures_dir: Path) -> Path:
    wav = fixtures_dir / "120bpm_click.wav"
    if not wav.exists():
        pytest.skip(f"fixture not present yet: {wav} (Task 9 generates it)")
    return wav


def test_playback_advances_real_clock(fixtures_dir: Path) -> None:
    _audio_device_or_skip()
    wav = _fixture_or_skip(fixtures_dir)

    from reachy_mini_dance_party_app.music.playback import PlaybackEngine

    engine = PlaybackEngine()
    engine.load(wav)
    engine.start()
    try:
        time.sleep(5.0)
        elapsed = engine.playback_time()
    finally:
        engine.stop()

    assert elapsed >= 4.0, f"playback_time() advanced only {elapsed:.2f}s in 5s"
