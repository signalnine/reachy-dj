"""Tests for the library move picker."""
from __future__ import annotations

import random

import pytest

from reachy_mini_dance_party_app.dance.picker import MoveSpec, choose_move


def test_picker_returns_close_fit():
    # At 120 BPM, one beat = 0.5s. A 2.0s move = exactly 4 beats (residual ~0).
    catalog = [MoveSpec(name="four_beat_move", duration_s=2.0)]
    rng = random.Random(0)
    result = choose_move(
        catalog=catalog,
        tempo=120.0,
        exclude_recent=set(),
        rng=rng,
    )
    assert result.name == "four_beat_move"
    assert result.duration_s == pytest.approx(2.0)


def test_picker_returns_fill_when_no_fit():
    # At 120 BPM, one beat = 0.5s. A 0.7s move sits ~0.2s off the nearest
    # integer beat count (1 beat = 0.5s, residual = 0.2 > 0.15 default window).
    catalog = [MoveSpec(name="bad_fit", duration_s=0.7)]
    rng = random.Random(0)
    result = choose_move(
        catalog=catalog,
        tempo=120.0,
        exclude_recent=set(),
        rng=rng,
    )
    assert result.name == "__fill__"
    assert result.duration_s == pytest.approx(60.0 / 120.0)


def test_picker_excludes_recent():
    # Two equally-good fits; the perfect one is excluded.
    catalog = [
        MoveSpec(name="a", duration_s=2.0),  # 4 beats at 120 BPM, residual 0
        MoveSpec(name="b", duration_s=1.5),  # 3 beats at 120 BPM, residual 0
    ]
    rng = random.Random(0)
    result = choose_move(
        catalog=catalog,
        tempo=120.0,
        exclude_recent={"a"},
        rng=rng,
    )
    assert result.name == "b"


def test_picker_excludes_recent_falls_back_to_fill():
    # Only candidate is excluded -> fill.
    catalog = [MoveSpec(name="only_one", duration_s=2.0)]
    rng = random.Random(0)
    result = choose_move(
        catalog=catalog,
        tempo=120.0,
        exclude_recent={"only_one"},
        rng=rng,
    )
    assert result.name == "__fill__"


def test_picker_distribution_with_seeded_rng():
    # 100 candidates with similar (small, varied) residuals; seeded RNG must
    # produce a non-degenerate distribution across many draws.
    tempo = 120.0
    beat_s = 60.0 / tempo  # 0.5
    catalog = []
    for i in range(100):
        # Each move is i*4 beats long with a tiny perturbation, residual < 0.15.
        beats = 4 + (i % 3)
        # Perturb by up to ~0.1s so residuals differ but stay within window.
        perturb = ((i * 37) % 11) * 0.01  # 0.00 .. 0.10
        catalog.append(
            MoveSpec(
                name=f"m{i}",
                duration_s=beats * beat_s + perturb,
            )
        )

    rng = random.Random(42)
    picks = [
        choose_move(
            catalog=catalog,
            tempo=tempo,
            exclude_recent=set(),
            rng=rng,
        ).name
        for _ in range(100)
    ]
    distinct = set(picks)
    # No move should ever be the fill, and we must see real variety.
    assert "__fill__" not in distinct
    assert len(distinct) >= 3


def test_picker_handles_empty_catalog():
    rng = random.Random(0)
    result = choose_move(
        catalog=[],
        tempo=120.0,
        exclude_recent=set(),
        rng=rng,
    )
    assert result.name == "__fill__"
    assert result.duration_s == pytest.approx(60.0 / 120.0)
