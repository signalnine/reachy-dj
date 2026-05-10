"""Beat-fit-scored library move picker.

Picks the next dance move from a catalog whose duration aligns well with an
integer number of beats at the song's current tempo. Falls back to a synthetic
``__fill__`` placeholder (one-beat) when nothing in the catalog fits or every
viable move was recently played.
"""
from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class MoveSpec:
    """A library move identified by name and its expected wall-clock duration."""

    name: str
    duration_s: float


def _fill_move(tempo: float) -> MoveSpec:
    """One-beat hold used when no catalog move fits."""
    return MoveSpec(name="__fill__", duration_s=60.0 / tempo)


def choose_move(
    catalog: list[MoveSpec],
    tempo: float,
    exclude_recent: set[str],
    fit_window_s: float | None = None,
    rng: random.Random | None = None,
) -> MoveSpec:
    """Pick the next move from ``catalog`` for the given ``tempo``.

    For each candidate, computes the nearest integer beat count and the
    residual seconds it would over- or under-shoot that target. Filters by
    ``residual <= fit_window_s`` and ``name not in exclude_recent``, then
    does a weighted-random pick where
    ``weight = 1.0 / ((residual + 0.05) * duration_s)`` — closer fits are
    preferred *and* shorter moves are preferred, since more frequent
    transitions read as more rhythmic to the audience.

    ``fit_window_s`` defaults to a quarter-beat (``60 / tempo / 4``) with a
    150 ms floor, so slower tempos (where a single beat is longer) admit a
    proportionally wider tolerance. With a fixed 100 ms window, songs
    around 90 BPM had only the 5 s moves fitting, which collapsed the
    rotation to 2-3 moves on repeat.

    Returns ``MoveSpec(name="__fill__", duration_s=60/tempo)`` if no candidate
    survives the filter.
    """
    if rng is None:
        rng = random.Random()

    beat_s = 60.0 / tempo
    if fit_window_s is None:
        fit_window_s = max(0.15, beat_s * 0.25)
    candidates: list[tuple[MoveSpec, float]] = []
    for move in catalog:
        if move.name in exclude_recent:
            continue
        target_beats = round(move.duration_s * tempo / 60.0)
        residual = abs(move.duration_s - target_beats * beat_s)
        if residual <= fit_window_s:
            candidates.append((move, residual))

    if not candidates:
        # No catalog move fits this tempo within ``fit_window_s``. Rather
        # than no-op'ing the beat, fall back to the shortest available real
        # move (excluding recents) — the next iteration realigns to the next
        # beat after it ends, so a small drift is fine and the audience
        # sees motion instead of silence.
        usable = [m for m in catalog if m.name not in exclude_recent]
        if usable:
            return min(usable, key=lambda m: m.duration_s)
        return _fill_move(tempo)

    weights = [
        1.0 / ((residual + 0.05) * max(move.duration_s, 0.5))
        for move, residual in candidates
    ]
    (chosen,) = rng.choices([m for m, _ in candidates], weights=weights, k=1)
    return chosen
