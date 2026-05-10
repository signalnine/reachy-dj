"""Librosa beat-analysis wrapper with on-disk cache."""
from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

from .beat import BeatGrid

SAMPLE_RATE = 22050
FALLBACK_TEMPO = 120.0


def _cache_paths(wav_path: Path) -> tuple[Path, Path]:
    beats = wav_path.parent / (wav_path.stem + ".beats.npy")
    tempo = wav_path.parent / (wav_path.stem + ".tempo.txt")
    return beats, tempo


def _read_cache(wav_path: Path) -> BeatGrid | None:
    beats_path, tempo_path = _cache_paths(wav_path)
    if not (beats_path.exists() and tempo_path.exists()):
        return None
    try:
        beat_times = np.load(beats_path)
        tempo = float(tempo_path.read_text().strip())
    except (OSError, ValueError):
        return None
    return BeatGrid(tempo=tempo, beat_times=beat_times)


def _write_cache(wav_path: Path, grid: BeatGrid) -> None:
    beats_path, tempo_path = _cache_paths(wav_path)
    np.save(beats_path, np.asarray(grid.beat_times))
    tempo_path.write_text(f"{grid.tempo}\n")


def analyze_beats(wav_path: Path) -> BeatGrid:
    """Compute tempo + beat times for a wav file using librosa.

    Caches the result alongside the wav as ``<stem>.beats.npy`` and
    ``<stem>.tempo.txt``. Subsequent calls with the same path read from cache
    without re-analyzing.

    On empty detection (no beats found, e.g. silence), returns
    ``BeatGrid.synthetic(120.0, duration=<wav_duration>)`` as the fallback.
    """
    wav_path = Path(wav_path)
    cached = _read_cache(wav_path)
    if cached is not None:
        return cached

    audio, sr = librosa.load(str(wav_path), sr=SAMPLE_RATE, mono=True)
    tempo, beat_times = librosa.beat.beat_track(y=audio, sr=sr, units="time")

    # tempo can come back as an array; coerce to plain float.
    tempo_val = float(np.asarray(tempo).reshape(-1)[0])
    beat_times = np.asarray(beat_times, dtype=np.float64)

    if beat_times.size == 0:
        duration = float(len(audio)) / float(sr)
        grid = BeatGrid.synthetic(FALLBACK_TEMPO, duration=duration)
    else:
        grid = BeatGrid(tempo=tempo_val, beat_times=beat_times)

    _write_cache(wav_path, grid)
    return grid
