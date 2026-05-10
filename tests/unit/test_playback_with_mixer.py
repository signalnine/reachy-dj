"""Tests for PlaybackEngine after the mixer/ducker refactor (Task 15).

PlaybackEngine now combines a music buffer with a TTS chunk inlet, runs both
through the existing :class:`Mixer` + :class:`Ducker`, and writes the result
to its OutputStream. All audio I/O is faked via :class:`FakeOutputStream` and
the ducker uses a :class:`FakeClock` so envelope behaviour is deterministic.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

from reachy_mini_dance_party_app.music.mixer import Ducker, Mixer
from reachy_mini_dance_party_app.music.playback import PlaybackEngine


# ---------- Fakes ----------


class FakeOutputStream:
    """Stand-in for sounddevice.OutputStream usable from unit tests."""

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

        self.callback_outputs: list[np.ndarray] = []

    def start(self) -> None:
        self.start_called = True

    def stop(self) -> None:
        self.stop_called = True

    def close(self) -> None:
        self.close_called = True

    # Test helper — not part of the real sounddevice API.
    def tick(self, n_frames: int) -> np.ndarray:
        outdata = np.zeros((n_frames, self.channels), dtype=np.float32)
        self.callback(outdata, n_frames, None, None)
        self.callback_outputs.append(outdata.copy())
        return outdata


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.now = float(start)

    def __call__(self) -> float:
        return self.now

    def advance(self, dt: float) -> None:
        self.now += float(dt)


# ---------- Helpers ----------


def _write_wav(path: Path, data: np.ndarray, samplerate: int) -> None:
    import soundfile as sf

    sf.write(str(path), data, samplerate, subtype="FLOAT")


def _pcm16_bytes(samples_float: np.ndarray) -> bytes:
    """Convert a float32 array in [-1, 1] to little-endian PCM16 bytes."""
    clipped = np.clip(samples_float, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def _make_engine(
    mixer: Mixer,
    ducker: Ducker,
    *,
    sample_rate: int = 48000,
    channels: int = 1,
) -> tuple[PlaybackEngine, type[FakeOutputStream]]:
    engine = PlaybackEngine(
        mixer=mixer,
        ducker=ducker,
        sample_rate=sample_rate,
        channels=channels,
        stream_factory=FakeOutputStream,
    )
    return engine, FakeOutputStream


# ---------- Tests ----------


def test_callback_with_no_music_no_tts_outputs_silence():
    """With nothing loaded and no TTS fed, the callback writes zeros."""
    mixer = Mixer()
    ducker = Ducker(clock=FakeClock(0.0))
    engine, _ = _make_engine(mixer, ducker)
    engine.start()

    stream: FakeOutputStream = engine._stream  # type: ignore[assignment]
    out = stream.tick(256)

    np.testing.assert_allclose(out, np.zeros((256, 1), dtype=np.float32))


def test_callback_mixes_music_and_tts(tmp_path: Path):
    """Output equals music + TTS (gains 1.0) sample-wise, modulo small clip."""
    sr = 48000
    n = 512

    # Music buffer: constant 0.1.
    music_data = np.full(n, 0.1, dtype=np.float32)
    wav_path = tmp_path / "music.wav"
    _write_wav(wav_path, music_data, sr)

    mixer = Mixer()
    # No ramp so the ducker stays at 1.0 with speech_active=False.
    ducker = Ducker(ramp_s=0.0, hangover_s=0.0, clock=FakeClock(0.0))
    engine, _ = _make_engine(mixer, ducker, sample_rate=sr)
    engine.load(wav_path)

    # TTS: constant 0.2 at the playback sample rate (no resampling needed).
    tts_samples = np.full(n, 0.2, dtype=np.float32)
    engine.feed_tts_chunk(_pcm16_bytes(tts_samples), sample_rate=sr)

    engine.start()
    stream: FakeOutputStream = engine._stream  # type: ignore[assignment]
    out = stream.tick(n)

    # PCM16 round-trip introduces ~3e-5 quantization error; allow for it.
    expected = np.full(n, 0.1 + 0.2, dtype=np.float32)
    np.testing.assert_allclose(out[:, 0], expected, atol=2e-4)


def test_set_speech_active_ducks_music(tmp_path: Path):
    """After advancing past the ramp, music is scaled to ~duck_to (0.25)."""
    sr = 48000
    n = 256
    music_data = np.full(n, 0.4, dtype=np.float32)
    wav_path = tmp_path / "music.wav"
    _write_wav(wav_path, music_data, sr)

    mixer = Mixer()
    clock = FakeClock(0.0)
    ducker = Ducker(duck_to=0.25, ramp_s=0.2, hangover_s=0.5, clock=clock)
    engine, _ = _make_engine(mixer, ducker, sample_rate=sr)
    engine.load(wav_path)
    engine.start()

    # Activate speech. The ducker needs one update call at the segment-start
    # time to anchor the ramp, then time advances and a subsequent callback
    # picks up the fully-ducked gain.
    engine.set_speech_active(True)
    stream: FakeOutputStream = engine._stream  # type: ignore[assignment]
    stream.tick(n)  # anchors ducker segment_start at t=0; buffer drains here
    clock.advance(0.25)  # past ramp_s=0.2

    # Reload music so we have samples to play after the ramp elapses.
    engine._read_pos = 0  # cheap reset of the music cursor for the test

    out = stream.tick(n)

    # Music at 0.4 * 0.25 = 0.1, no TTS.
    expected = np.full(n, 0.4 * 0.25, dtype=np.float32)
    np.testing.assert_allclose(out[:, 0], expected, atol=1e-5)


def test_set_speech_active_false_restores_after_hangover(tmp_path: Path):
    """Toggling speech off and waiting past hangover + ramp restores gain to 1.0."""
    sr = 48000
    n = 256
    music_data = np.full(n, 0.4, dtype=np.float32)
    wav_path = tmp_path / "music.wav"
    _write_wav(wav_path, music_data, sr)

    mixer = Mixer()
    clock = FakeClock(0.0)
    ducker = Ducker(duck_to=0.25, ramp_s=0.2, hangover_s=0.5, clock=clock)
    engine, _ = _make_engine(mixer, ducker, sample_rate=sr)
    engine.load(wav_path)
    engine.start()

    stream: FakeOutputStream = engine._stream  # type: ignore[assignment]

    # Drive ducker fully down: anchor at t=0, then advance past ramp_s.
    engine.set_speech_active(True)
    stream.tick(n)  # anchor at t=0; also drains the music buffer
    clock.advance(0.2)
    engine._read_pos = 0  # reset cursor so the next tick still sees music
    stream.tick(n)  # gain reaches duck_to here

    # Speech ends; ducker enters HANGOVER on the next callback tick (which
    # also anchors _speech_end_t at the current clock time).
    engine.set_speech_active(False)
    engine._read_pos = 0
    stream.tick(n)  # transition DUCKING → HANGOVER, _speech_end_t set here

    # Advance past hangover + ramp.
    clock.advance(0.5 + 0.2 + 0.05)
    engine._read_pos = 0

    out = stream.tick(n)
    expected = np.full(n, 0.4, dtype=np.float32)  # gain restored to 1.0
    np.testing.assert_allclose(out[:, 0], expected, atol=1e-5)


def test_tts_resamples_24k_to_48k():
    """A 24kHz TTS chunk gets resampled to the 48kHz playback rate.

    Feed N samples of TTS at 24kHz; after resampling to 48kHz there should be
    ~2N samples available for the callback to consume across two ticks.
    """
    play_sr = 48000
    tts_sr = 24000
    n_tts = 480  # 20ms at 24kHz; expect ~960 samples at 48kHz

    mixer = Mixer()
    ducker = Ducker(ramp_s=0.0, hangover_s=0.0, clock=FakeClock(0.0))
    engine, _ = _make_engine(mixer, ducker, sample_rate=play_sr)
    engine.start()

    # A constant-0.5 TTS chunk at 24kHz; constant signals resample to constants
    # (modulo edge effects), making this easy to assert on.
    tts_24k = np.full(n_tts, 0.5, dtype=np.float32)
    engine.feed_tts_chunk(_pcm16_bytes(tts_24k), sample_rate=tts_sr)

    stream: FakeOutputStream = engine._stream  # type: ignore[assignment]

    # Pull the TTS in two halves of 480 samples; both halves should be ~0.5.
    first = stream.tick(480)
    second = stream.tick(480)

    # Trim a few samples on each end to dodge resampler edge ringing.
    np.testing.assert_allclose(first[20:-20, 0], 0.5, atol=0.05)
    np.testing.assert_allclose(second[20:-20, 0], 0.5, atol=0.05)

    # After the inlet is drained, more ticks return silence.
    third = stream.tick(480)
    np.testing.assert_allclose(third[:, 0], 0.0, atol=1e-5)
