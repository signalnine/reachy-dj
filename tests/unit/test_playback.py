"""Tests for the PlaybackEngine (pure-logic, no real audio I/O).

A FakeOutputStream stands in for sounddevice.OutputStream. It records
construction arguments, exposes start/stop/close as no-ops that flip flags,
and provides a `tick(n_frames)` test helper that synchronously invokes the
callback once with an outdata buffer of the requested size. All callback
invocations (their outdata snapshots) are recorded for inspection.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

from reachy_mini_dance_party_app.music.playback import PlaybackEngine


# ---------- FakeOutputStream ----------


class FakeOutputStream:
    """Stand-in for sounddevice.OutputStream usable from unit tests.

    Drives the user-supplied callback synchronously via `tick(n_frames)`
    instead of running on a PortAudio worker thread. Records every callback
    invocation's outdata snapshot for assertions.
    """

    def __init__(
        self,
        samplerate: int,
        channels: int,
        callback: Callable[..., None],
        dtype: str = "float32",
        **kwargs: Any,
    ) -> None:
        self.samplerate = samplerate
        self.channels = channels
        self.callback = callback
        self.dtype = dtype
        self.kwargs = kwargs

        self.start_called = False
        self.stop_called = False
        self.close_called = False

        # Each entry is a copy of the outdata buffer the callback wrote.
        self.callback_outputs: list[np.ndarray] = []

    def start(self) -> None:
        self.start_called = True

    def stop(self) -> None:
        self.stop_called = True

    def close(self) -> None:
        self.close_called = True

    # Test helper — not part of the real sounddevice API.
    def tick(self, n_frames: int) -> np.ndarray:
        """Invoke the callback once with an outdata buffer of n_frames."""
        outdata = np.zeros((n_frames, self.channels), dtype=np.float32)
        # sounddevice callback signature: (outdata, frames, time, status)
        self.callback(outdata, n_frames, None, None)
        self.callback_outputs.append(outdata.copy())
        return outdata


# ---------- Helpers ----------


def _write_wav(path: Path, data: np.ndarray, samplerate: int) -> None:
    """Write a float32 wav file using soundfile (a librosa dep, already installed)."""
    import soundfile as sf

    sf.write(str(path), data, samplerate, subtype="FLOAT")


@pytest.fixture
def stereo_ramp_wav(tmp_path: Path) -> tuple[Path, np.ndarray, int]:
    """A 1-channel float32 wav with values [0, 1, 2, ..., N-1] / N for easy ordering checks."""
    sr = 22050
    n = 2000
    # Use an unambiguous monotonic ramp scaled to [0, ~1) so we can verify ordering.
    data = (np.arange(n, dtype=np.float32) / float(n)).reshape(-1)
    path = tmp_path / "ramp.wav"
    _write_wav(path, data, sr)
    return path, data, sr


# ---------- Tests ----------


def test_load_then_playback_time_starts_at_zero(stereo_ramp_wav):
    path, _, sr = stereo_ramp_wav
    engine = PlaybackEngine(stream_factory=FakeOutputStream)
    engine.load(path)

    assert engine.playback_time() == pytest.approx(0.0)
    assert engine.duration_s == pytest.approx(2000 / sr, abs=1e-6)
    assert engine.is_playing is False


def test_callback_advances_playback_time(stereo_ramp_wav):
    path, _, sr = stereo_ramp_wav
    engine = PlaybackEngine(stream_factory=FakeOutputStream)
    engine.load(path)
    engine.start()

    stream: FakeOutputStream = engine._stream  # type: ignore[assignment]
    stream.tick(1024)

    assert engine.playback_time() == pytest.approx(1024 / sr, abs=1e-9)

    # Monotonic across a second tick.
    stream.tick(512)
    assert engine.playback_time() == pytest.approx((1024 + 512) / sr, abs=1e-9)


def test_callback_consumes_buffer_in_order(stereo_ramp_wav):
    path, data, _sr = stereo_ramp_wav
    engine = PlaybackEngine(stream_factory=FakeOutputStream)
    engine.load(path)
    engine.start()

    stream: FakeOutputStream = engine._stream  # type: ignore[assignment]
    out1 = stream.tick(500)
    out2 = stream.tick(500)

    # Mono input is broadcast across channels(=1); compare the first channel.
    np.testing.assert_allclose(out1[:, 0], data[0:500], atol=1e-6)
    np.testing.assert_allclose(out2[:, 0], data[500:1000], atol=1e-6)


def test_callback_outputs_silence_after_buffer_exhausted(stereo_ramp_wav):
    path, data, _sr = stereo_ramp_wav
    engine = PlaybackEngine(stream_factory=FakeOutputStream)
    engine.load(path)
    engine.start()

    stream: FakeOutputStream = engine._stream  # type: ignore[assignment]

    # Drain everything in one huge tick (more than the buffer length).
    n_total = len(data)
    out = stream.tick(n_total + 256)

    # First n_total samples match the buffer; the trailing 256 are silence.
    np.testing.assert_allclose(out[:n_total, 0], data, atol=1e-6)
    np.testing.assert_allclose(out[n_total:, 0], np.zeros(256, dtype=np.float32))

    # Subsequent ticks are pure silence.
    out2 = stream.tick(128)
    np.testing.assert_allclose(out2, np.zeros_like(out2))


def test_stop_closes_stream(stereo_ramp_wav):
    path, _, _ = stereo_ramp_wav
    engine = PlaybackEngine(stream_factory=FakeOutputStream)
    engine.load(path)
    engine.start()

    stream: FakeOutputStream = engine._stream  # type: ignore[assignment]
    assert stream.start_called is True
    assert engine.is_playing is True

    engine.stop()

    assert stream.stop_called is True
    assert stream.close_called is True
    assert engine.is_playing is False
