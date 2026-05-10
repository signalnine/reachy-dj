"""LibraryDancer — beat-aligned library-move enqueuer thread.

This thread decides what dance move plays next and enqueues it to a primary-
move queue, aligned to song downbeats. It does NOT own the move worker — only
enqueues via an injected ``enqueue_move`` callable. It does NOT own the
playback engine — only reads playback time via an injected ``get_playback_time``
callable.

The loop algorithm (one iteration of ``_step_once``):

1. Read the current ``BeatGrid`` via ``get_grid()``. If ``None`` (no song
   loaded yet), wait briefly and return.
2. Read the current playback time and, starting from
   ``max(playback_time, _next_target)``, find the next beat at least 50 ms in
   the future. That beat is the move's start time.
3. Pick a move from the catalog via ``dance.picker.choose_move`` (excluding
   the last few names so we don't repeat the same move back-to-back).
4. Sleep until ~50 ms before that beat (testable: the sleeper is a hookable
   ``_sleep_until`` method).
5. Call ``enqueue_move(name, target_beat_time)``.
6. Record the name in ``self._recent`` and advance ``_next_target`` to
   ``target_beat_time + duration_s`` so the next iteration searches past
   this move's end.

Tests substitute ``_sleep_until`` with a fast-forward of a fake clock,
giving deterministic behavior with no real ``time.sleep`` calls.
"""
from __future__ import annotations

import random
import threading
import time
from collections import deque
from typing import Callable, Optional

from reachy_mini_dance_party_app.dance.picker import MoveSpec, choose_move
from reachy_mini_dance_party_app.music.beat import BeatGrid


class LibraryDancer(threading.Thread):
    """Picks library moves and enqueues them at the next downbeat."""

    def __init__(
        self,
        catalog: list[MoveSpec],
        get_playback_time: Callable[[], float],
        enqueue_move: Callable[[str, float], None],
        get_grid: Callable[[], Optional[BeatGrid]],
        recent_window: int = 4,
        fit_window_s: float = 0.15,
        rng: random.Random | None = None,
    ) -> None:
        super().__init__(name="LibraryDancer", daemon=True)
        self._catalog = catalog
        self._get_playback_time = get_playback_time
        self._enqueue_move = enqueue_move
        self._get_grid = get_grid
        self.fit_window_s = fit_window_s
        self._recent: deque[str] = deque(maxlen=recent_window)
        self._rng = rng if rng is not None else random.Random()
        self._stop_event = threading.Event()

        # Earliest future time to look for a beat. Set to current playback
        # time + small lead each step; advanced past the just-enqueued move
        # so we don't stack moves on the same beat.
        self._next_target: float = 0.0

        # Lead time before the target beat at which we want to enqueue
        # (gives the move worker time to schedule the start).
        self._lead_s: float = 0.05

    # ---------- Lifecycle ----------

    def run(self) -> None:
        while not self._stop_event.is_set():
            self._step_once()

    def stop(self) -> None:
        self._stop_event.set()
        if self.is_alive():
            self.join(timeout=2.0)

    # ---------- Hookable sleeper (overridden in tests) ----------

    def _sleep_until(self, target: float) -> None:
        """Sleep until ``get_playback_time()`` reaches ``target``.

        Overridden in tests to fast-forward a fake clock instead of blocking.
        """
        delay = target - self._get_playback_time()
        if delay > 0:
            time.sleep(delay)

    # ---------- Single iteration ----------

    def _step_once(self) -> None:
        grid = self._get_grid()
        if grid is None:
            # No song loaded yet — wait a tick and return; tests will not
            # exercise this branch (they always provide a grid).
            if not self._stop_event.is_set():
                time.sleep(0.1)
            return

        playback_t = self._get_playback_time()
        # Start searching from whichever is later: now (with a small lead) or
        # the time the previously-enqueued move is expected to end.
        search_from = max(playback_t + self._lead_s, self._next_target)

        next_beats = grid.next_beat_at(search_from, n=1)
        if not next_beats:
            # Past end of grid — nothing more to enqueue. Brief idle wait so
            # we don't busy-loop in real-thread mode.
            if not self._stop_event.is_set():
                time.sleep(0.1)
            return
        target_beat = float(next_beats[0])

        move = choose_move(
            catalog=self._catalog,
            tempo=grid.tempo,
            exclude_recent=set(self._recent),
            fit_window_s=self.fit_window_s,
            rng=self._rng,
        )

        # Sleep until just before the target beat so the enqueue is timely.
        self._sleep_until(target_beat - self._lead_s)

        if self._stop_event.is_set():
            return

        self._enqueue_move(move.name, target_beat)
        self._recent.append(move.name)
        self._next_target = target_beat + move.duration_s
