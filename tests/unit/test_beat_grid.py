import numpy as np
import pytest
from reachy_mini_dance_party_app.music.beat import BeatGrid

def test_next_beat_at_returns_first_future_beat():
    grid = BeatGrid(tempo=120.0, beat_times=np.array([0.0, 0.5, 1.0, 1.5, 2.0]))
    assert grid.next_beat_at(0.6, n=1) == [1.0]

def test_next_beat_at_n_returns_n_beats():
    grid = BeatGrid(tempo=120.0, beat_times=np.array([0.0, 0.5, 1.0, 1.5, 2.0]))
    assert grid.next_beat_at(0.0, n=3) == [0.5, 1.0, 1.5]

def test_next_beat_at_past_end_returns_empty():
    grid = BeatGrid(tempo=120.0, beat_times=np.array([0.0, 0.5, 1.0]))
    assert grid.next_beat_at(2.0, n=3) == []

def test_beats_per_second_from_tempo():
    grid = BeatGrid(tempo=120.0, beat_times=np.array([0.0]))
    assert grid.beats_per_second == pytest.approx(2.0)

def test_beats_in_window_returns_count_at_tempo():
    grid = BeatGrid(tempo=120.0, beat_times=np.array([0.0]))
    assert grid.beats_in_window(2.0) == 4   # 4 beats in 2.0s at 120 BPM

def test_synthetic_grid_factory_for_empty_detection():
    grid = BeatGrid.synthetic(tempo=120.0, duration=10.0)
    assert len(grid.beat_times) == 20
    assert grid.beat_times[0] == 0.0
    assert grid.beat_times[-1] == pytest.approx(9.5)
