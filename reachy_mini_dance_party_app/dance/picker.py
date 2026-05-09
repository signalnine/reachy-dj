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
    fit_window_s: float = 0.15,
    rng: random.Random | None = None,
) -> MoveSpec:
    """Pick the next move from ``catalog`` for the given ``tempo``.

    For each candidate, computes the nearest integer beat count and the
    residual seconds it would over- or under-shoot that target. Filters by
    ``residual <= fit_window_s`` and ``name not in exclude_recent``, then does
    a weighted-random pick where ``weight = 1.0 / (residual + 0.05)`` (so
    closer fits are preferred but ties stay diverse).

    Returns ``MoveSpec(name="__fill__", duration_s=60/tempo)`` if no candidate
    survives the filter.
    """
    if rng is None:
        rng = random.Random()

    beat_s = 60.0 / tempo
    candidates: list[tuple[MoveSpec, float]] = []
    for move in catalog:
        if move.name in exclude_recent:
            continue
        target_beats = round(move.duration_s * tempo / 60.0)
        residual = abs(move.duration_s - target_beats * beat_s)
        if residual <= fit_window_s:
            candidates.append((move, residual))

    if not candidates:
        return _fill_move(tempo)

    weights = [1.0 / (residual + 0.05) for _, residual in candidates]
    (chosen,) = rng.choices([m for m, _ in candidates], weights=weights, k=1)
    return chosen
