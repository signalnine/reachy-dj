from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from scipy.io import wavfile

from reachy_mini_dance_party_app.music.analysis import analyze_beats
from reachy_mini_dance_party_app.music.beat import BeatGrid

SR = 22050
DURATION_S = 5.0
CLICK_TIMES = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]


def _fixtures_dir() -> Path:
    return Path(__file__).parent.parent / "fixtures"


def _generate_click_wav(path: Path) -> None:
    """5s of silence with short ~5ms sine bursts at every 0.5s (120 BPM)."""
    n_samples = int(SR * DURATION_S)
    audio = np.zeros(n_samples, dtype=np.float32)
    burst_len = int(SR * 0.005)  # 5ms
    burst_freq = 1000.0  # Hz
    t_burst = np.arange(burst_len) / SR
    burst = (0.8 * np.sin(2 * np.pi * burst_freq * t_burst)).astype(np.float32)
    # Short hann envelope to avoid clicks at edges
    envelope = np.hanning(burst_len).astype(np.float32)
    burst = burst * envelope
    for click_t in CLICK_TIMES:
        start = int(click_t * SR)
        end = start + burst_len
        if end <= n_samples:
            audio[start:end] += burst
    # Convert to int16 for wav
    audio_i16 = np.clip(audio, -1.0, 1.0)
    audio_i16 = (audio_i16 * 32767).astype(np.int16)
    wavfile.write(str(path), SR, audio_i16)


def _generate_silence_wav(path: Path) -> None:
    n_samples = int(SR * DURATION_S)
    audio = np.zeros(n_samples, dtype=np.int16)
    wavfile.write(str(path), SR, audio)


@pytest.fixture(scope="session", autouse=True)
def _ensure_fixture_wavs() -> None:
    fixtures = _fixtures_dir()
    fixtures.mkdir(parents=True, exist_ok=True)
    click_path = fixtures / "120bpm_click.wav"
    silence_path = fixtures / "silence.wav"
    if not click_path.exists():
        _generate_click_wav(click_path)
    if not silence_path.exists():
        _generate_silence_wav(silence_path)


def _clear_cache(wav_path: Path) -> None:
    for suffix in (".beats.npy", ".tempo.txt"):
        cache = wav_path.with_suffix("").with_suffix(suffix) if False else wav_path.parent / (wav_path.stem + suffix)
        if cache.exists():
            cache.unlink()


@pytest.fixture
def click_wav(tmp_path: Path) -> Path:
    """Copy the session-scoped click fixture to a tmp path so cache is isolated per test."""
    src = _fixtures_dir() / "120bpm_click.wav"
    dst = tmp_path / "120bpm_click.wav"
    dst.write_bytes(src.read_bytes())
    return dst


@pytest.fixture
def silence_wav(tmp_path: Path) -> Path:
    src = _fixtures_dir() / "silence.wav"
    dst = tmp_path / "silence.wav"
    dst.write_bytes(src.read_bytes())
    return dst


def test_analyze_click_track_detects_120bpm(click_wav: Path) -> None:
    grid = analyze_beats(click_wav)
    assert isinstance(grid, BeatGrid)
    assert abs(grid.tempo - 120.0) <= 5.0, f"tempo={grid.tempo} not within +/-5 of 120"
    assert 9 <= len(grid.beat_times) <= 11, f"got {len(grid.beat_times)} beats; expected 9-11"


def test_analyze_silence_returns_synthetic_120(silence_wav: Path) -> None:
    grid = analyze_beats(silence_wav)
    assert isinstance(grid, BeatGrid)
    assert grid.tempo == 120.0
    assert len(grid.beat_times) > 0


def test_second_call_reads_cache(click_wav: Path) -> None:
    import reachy_mini_dance_party_app.music.analysis as analysis_mod

    real_beat_track = analysis_mod.librosa.beat.beat_track
    with patch.object(
        analysis_mod.librosa.beat, "beat_track", wraps=real_beat_track
    ) as spy:
        grid1 = analyze_beats(click_wav)
        grid2 = analyze_beats(click_wav)
    assert spy.call_count == 1, f"beat_track called {spy.call_count} times; expected 1"
    assert grid1.tempo == grid2.tempo
    assert np.array_equal(grid1.beat_times, grid2.beat_times)


def test_cache_files_have_expected_names(click_wav: Path) -> None:
    grid = analyze_beats(click_wav)
    beats_path = click_wav.parent / (click_wav.stem + ".beats.npy")
    tempo_path = click_wav.parent / (click_wav.stem + ".tempo.txt")
    assert beats_path.exists(), f"missing {beats_path}"
    assert tempo_path.exists(), f"missing {tempo_path}"
    cached_beats = np.load(beats_path)
    cached_tempo = float(tempo_path.read_text().strip())
    assert np.array_equal(cached_beats, grid.beat_times)
    assert cached_tempo == grid.tempo
