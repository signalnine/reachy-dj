"""Tests for the audio Mixer and Ducker (pure-logic, no audio I/O)."""
from __future__ import annotations

import numpy as np
import pytest

from reachy_mini_dance_party_app.music.mixer import Ducker, Mixer


class FakeClock:
    """Stateful fake monotonic clock; advance via `advance()` or set `now`."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = float(start)

    def __call__(self) -> float:
        return self.now

    def advance(self, dt: float) -> None:
        self.now += float(dt)


# ---------- Mixer ----------

def test_mixer_mixes_two_streams_samplewise():
    mixer = Mixer()
    music = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    tts = np.array([0.05, 0.05, 0.05, 0.05], dtype=np.float32)

    out = mixer.mix(music, tts, music_gain=1.0, tts_gain=1.0)

    expected = np.array([0.15, 0.25, 0.35, 0.45], dtype=np.float32)
    assert out.dtype == np.float32
    assert out.shape == music.shape
    np.testing.assert_allclose(out, expected, atol=1e-6)


def test_mixer_applies_gains():
    mixer = Mixer()
    music = np.array([0.4, 0.4, 0.4, 0.4], dtype=np.float32)
    tts = np.array([0.2, 0.2, 0.2, 0.2], dtype=np.float32)

    out = mixer.mix(music, tts, music_gain=0.5, tts_gain=2.0)

    # 0.4*0.5 + 0.2*2.0 = 0.2 + 0.4 = 0.6
    expected = np.full(4, 0.6, dtype=np.float32)
    np.testing.assert_allclose(out, expected, atol=1e-6)


def test_mixer_clips_to_bounds():
    mixer = Mixer()
    music = np.array([1.5, -1.5, 0.8, -0.8, 2.0], dtype=np.float32)
    tts = np.array([0.6, -0.6, 0.5, -0.5, 0.0], dtype=np.float32)

    out = mixer.mix(music, tts, music_gain=1.0, tts_gain=1.0)

    # Pre-clip: [2.1, -2.1, 1.3, -1.3, 2.0] -> clip to [-1, 1]
    assert out.max() <= 1.0
    assert out.min() >= -1.0
    expected = np.array([1.0, -1.0, 1.0, -1.0, 1.0], dtype=np.float32)
    np.testing.assert_allclose(out, expected, atol=1e-6)


# ---------- Ducker ----------

def test_ducker_idle_returns_one():
    clock = FakeClock(0.0)
    duck = Ducker(duck_to=0.25, ramp_s=0.2, hangover_s=0.5, clock=clock)
    assert duck.update(speech_active=False) == pytest.approx(1.0)
    clock.advance(1.0)
    assert duck.update(speech_active=False) == pytest.approx(1.0)


def test_ducker_ramps_down_when_speech_active():
    clock = FakeClock(0.0)
    duck = Ducker(duck_to=0.25, ramp_s=0.2, hangover_s=0.5, clock=clock)

    # Speech turns on at t=0.
    g0 = duck.update(speech_active=True)
    assert g0 == pytest.approx(1.0, abs=1e-3)  # hasn't started ramping yet

    # Halfway through the ramp, gain should be roughly halfway between 1.0 and 0.25.
    clock.advance(0.1)
    g_half = duck.update(speech_active=True)
    assert 0.4 < g_half < 0.95  # somewhere mid-ramp
    assert g_half < g0

    # After ramp_s elapses, gain should be at duck_to.
    clock.advance(0.1)
    g_end = duck.update(speech_active=True)
    assert g_end == pytest.approx(0.25, abs=1e-3)

    # And it stays there as long as speech is active.
    clock.advance(1.0)
    assert duck.update(speech_active=True) == pytest.approx(0.25, abs=1e-3)


def test_ducker_ramps_back_after_speech_ends_with_hangover():
    clock = FakeClock(0.0)
    duck = Ducker(duck_to=0.25, ramp_s=0.2, hangover_s=0.5, clock=clock)

    # Drive ducker to fully-ducked state. (t = 0.0 -> 0.2)
    duck.update(speech_active=True)
    clock.advance(0.2)
    assert duck.update(speech_active=True) == pytest.approx(0.25, abs=1e-3)

    # Speech ends at t = 0.3.
    clock.advance(0.1)
    g_in_hangover = duck.update(speech_active=False)
    assert g_in_hangover == pytest.approx(0.25, abs=1e-3)

    # Still in hangover at t = 0.5 (0.2s into hangover).
    clock.advance(0.2)
    assert duck.update(speech_active=False) == pytest.approx(0.25, abs=1e-3)

    # Past hangover at t = 0.9 (0.1s into ramp-back, hangover ended at t = 0.8).
    clock.advance(0.4)
    g_ramping = duck.update(speech_active=False)
    assert 0.25 < g_ramping < 1.0

    # After ramp_s fully elapses past hangover-end (t = 1.0), fully restored.
    clock.advance(0.1)
    g_restored = duck.update(speech_active=False)
    assert g_restored == pytest.approx(1.0, abs=1e-3)
