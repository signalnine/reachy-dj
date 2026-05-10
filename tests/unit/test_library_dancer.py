"""Tests for LibraryDancer — beat-aligned library-move enqueuer thread.

These tests drive the dancer's ``_step_once()`` method directly with a fake
playback clock and a fake move queue, so no real threading or sleeping
happens. The clock is advanced deterministically between iterations.
"""
from __future__ import annotations

import random

import numpy as np
import pytest

from reachy_mini_dance_party_app.dance.library_dancer import LibraryDancer
from reachy_mini_dance_party_app.dance.picker import MoveSpec
from reachy_mini_dance_party_app.music.beat import BeatGrid


class _FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _make_grid(tempo: float = 120.0, duration: float = 30.0) -> BeatGrid:
    return BeatGrid.synthetic(tempo=tempo, duration=duration)


def _make_dancer(
    catalog: list[MoveSpec],
    clock: _FakeClock,
    grid: BeatGrid,
    enqueued: list[tuple[str, float]],
    *,
    rng: random.Random | None = None,
    recent_window: int = 4,
    fit_window_s: float = 0.15,
) -> LibraryDancer:
    dancer = LibraryDancer(
        catalog=catalog,
        get_playback_time=clock.now,
        enqueue_move=lambda name, t: enqueued.append((name, t)),
        get_grid=lambda: grid,
        recent_window=recent_window,
        fit_window_s=fit_window_s,
        rng=rng or random.Random(0),
    )
    # Replace the sleeper with a deterministic fast-forward of the fake clock,
    # so _step_once doesn't actually block.
    dancer._sleep_until = lambda target: clock.advance(  # type: ignore[method-assign]
        max(0.0, target - clock.now())
    )
    return dancer


def test_dancer_enqueues_at_least_4_moves_in_4_simulated_seconds() -> None:
    # 4 seconds at 120 BPM = 8 beats. Catalog of small (1- and 2-beat) moves
    # so the dancer can fit >=4 of them in the window.
    catalog = [
        MoveSpec(name="one_beat_a", duration_s=0.5),
        MoveSpec(name="one_beat_b", duration_s=0.5),
        MoveSpec(name="two_beat_a", duration_s=1.0),
        MoveSpec(name="two_beat_b", duration_s=1.0),
    ]
    clock = _FakeClock()
    grid = _make_grid(tempo=120.0, duration=30.0)
    enqueued: list[tuple[str, float]] = []
    dancer = _make_dancer(catalog, clock, grid, enqueued, recent_window=2)

    # Drive until either the simulated clock reaches 4s OR we've collected
    # enough moves to satisfy the assertion. The clock advances inside
    # _step_once via the _sleep_until override, so this terminates.
    while clock.now() < 4.0:
        dancer._step_once()

    assert len(enqueued) >= 4, (
        f"expected >=4 moves enqueued during 4s simulated playback, got "
        f"{len(enqueued)}: {enqueued}"
    )


def test_enqueue_times_are_beat_aligned() -> None:
    catalog = [
        MoveSpec(name="one_beat", duration_s=0.5),
        MoveSpec(name="two_beat", duration_s=1.0),
        MoveSpec(name="four_beat", duration_s=2.0),
    ]
    clock = _FakeClock()
    grid = _make_grid(tempo=120.0, duration=30.0)
    enqueued: list[tuple[str, float]] = []
    dancer = _make_dancer(catalog, clock, grid, enqueued)

    while clock.now() < 4.0:
        dancer._step_once()

    assert len(enqueued) > 0
    beat_times = grid.beat_times
    for name, t in enqueued:
        # nearest beat
        idx = int(np.argmin(np.abs(beat_times - t)))
        nearest = float(beat_times[idx])
        assert abs(t - nearest) <= 0.05, (
            f"{name} enqueued at t={t} is {abs(t - nearest):.3f}s away from "
            f"nearest beat {nearest}"
        )


def test_no_two_consecutive_moves_have_same_name() -> None:
    # recent_window=2 with 3 catalog items means after picking any move, that
    # name is excluded from the next pick — so consecutive duplicates can't
    # happen, and we still always have >=1 viable candidate (no fallback to
    # __fill__).
    catalog = [
        MoveSpec(name="a", duration_s=0.5),  # 1 beat at 120 BPM
        MoveSpec(name="b", duration_s=1.0),  # 2 beats
        MoveSpec(name="c", duration_s=2.0),  # 4 beats
    ]
    clock = _FakeClock()
    grid = _make_grid(tempo=120.0, duration=30.0)
    enqueued: list[tuple[str, float]] = []
    dancer = _make_dancer(catalog, clock, grid, enqueued, recent_window=2)

    while clock.now() < 8.0:
        dancer._step_once()

    assert len(enqueued) >= 4, f"expected >=4 enqueues over 8s, got {enqueued}"
    for prev, curr in zip(enqueued, enqueued[1:]):
        assert prev[0] != curr[0], (
            f"consecutive moves share name: {prev} -> {curr}; full list {enqueued}"
        )


def test_dancer_enqueues_fill_when_no_fit() -> None:
    # 0.7s move at 120 BPM: nearest integer beats = round(0.7*2)=1, residual
    # = |0.7 - 0.5| = 0.2 > fit_window=0.15 => fill is the only option.
    catalog = [MoveSpec(name="bad_fit", duration_s=0.7)]
    clock = _FakeClock()
    grid = _make_grid(tempo=120.0, duration=30.0)
    enqueued: list[tuple[str, float]] = []
    dancer = _make_dancer(catalog, clock, grid, enqueued, fit_window_s=0.15)

    # Run a few steps so we get at least one enqueue.
    for _ in range(5):
        dancer._step_once()
        if enqueued:
            break

    assert enqueued, "dancer never enqueued anything"
    assert enqueued[0][0] == "__fill__", (
        f"expected fill as first enqueue, got {enqueued[0]}"
    )
