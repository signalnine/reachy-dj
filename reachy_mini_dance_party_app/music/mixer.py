"""Audio mixer + ducker (pure logic, no audio I/O).

The Mixer combines two float32 streams (music and TTS) with per-stream gains
and clips the result to [-1.0, 1.0]. The Ducker is a small state machine that
tracks a music gain envelope: it ramps from 1.0 down to ``duck_to`` while
speech is active, and ramps back up after speech stops + a hangover window.

Both classes are deterministic and clock-injectable so they can be tested
without real audio hardware.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable

import numpy as np


@dataclass
class Mixer:
    """Sample-wise mixer for two float32 streams with per-stream gain + clip."""

    def mix(
        self,
        music_chunk: np.ndarray,
        tts_chunk: np.ndarray,
        music_gain: float,
        tts_gain: float,
    ) -> np.ndarray:
        """Mix two equal-length float32 buffers, scale by gains, clip to [-1, 1]."""
        if music_chunk.shape != tts_chunk.shape:
            raise ValueError(
                f"music/tts shape mismatch: {music_chunk.shape} vs {tts_chunk.shape}"
            )
        music = music_chunk.astype(np.float32, copy=False)
        tts = tts_chunk.astype(np.float32, copy=False)
        out = music * np.float32(music_gain) + tts * np.float32(tts_gain)
        np.clip(out, -1.0, 1.0, out=out)
        return out


class _DuckState(Enum):
    IDLE = auto()       # gain == 1.0, no speech recently
    DUCKING = auto()    # speech active; ramping down (or held at duck_to)
    HANGOVER = auto()   # speech ended; holding duck_to until hangover_s elapses
    RECOVERING = auto() # ramping back up from duck_to to 1.0


@dataclass
class Ducker:
    """Music-gain envelope driven by a monotonic clock.

    On ``update(speech_active=True)`` the gain ramps linearly from its current
    value toward ``duck_to`` over ``ramp_s`` seconds. When ``speech_active``
    flips back to False, the gain holds at its current value for ``hangover_s``
    seconds, then ramps linearly back to 1.0 over ``ramp_s`` seconds.
    """

    duck_to: float = 0.25
    ramp_s: float = 0.2
    hangover_s: float = 0.5
    clock: Callable[[], float] = time.monotonic

    _state: _DuckState = field(default=_DuckState.IDLE, init=False)
    _gain: float = field(default=1.0, init=False)
    # Anchor for the current envelope segment: (time, gain_at_time).
    _segment_start_t: float = field(default=0.0, init=False)
    _segment_start_gain: float = field(default=1.0, init=False)
    # When speech ended (used to time the hangover window).
    _speech_end_t: float | None = field(default=None, init=False)

    def update(self, speech_active: bool) -> float:
        """Advance the envelope to ``self.clock()`` and return current music_gain."""
        now = self.clock()

        if speech_active:
            if self._state in (_DuckState.IDLE, _DuckState.HANGOVER, _DuckState.RECOVERING):
                # Start (or restart) ducking from current gain.
                self._state = _DuckState.DUCKING
                self._segment_start_t = now
                self._segment_start_gain = self._gain
                self._speech_end_t = None
            self._gain = self._compute_ramp(now, target=self.duck_to)
            return self._gain

        # speech_active is False
        if self._state == _DuckState.DUCKING:
            # Speech just ended; hold current gain through the hangover.
            self._state = _DuckState.HANGOVER
            self._speech_end_t = now
            # _segment_start_t/_gain unchanged so DUCKING ramp stops where it is
            # (we still let the held gain be the most-recent computed value).
            return self._gain

        if self._state == _DuckState.HANGOVER:
            assert self._speech_end_t is not None
            if now - self._speech_end_t >= self.hangover_s:
                # Hangover elapsed; begin recovery from the moment hangover ended
                # (so elapsed-into-ramp accounts for time already past hangover).
                self._state = _DuckState.RECOVERING
                self._segment_start_t = self._speech_end_t + self.hangover_s
                self._segment_start_gain = self._gain
                self._gain = self._compute_ramp(now, target=1.0)
                if self._gain >= 1.0:
                    self._gain = 1.0
                    self._state = _DuckState.IDLE
            # else: still in hangover, gain unchanged.
            return self._gain

        if self._state == _DuckState.RECOVERING:
            self._gain = self._compute_ramp(now, target=1.0)
            if self._gain >= 1.0:
                self._gain = 1.0
                self._state = _DuckState.IDLE
            return self._gain

        # IDLE
        self._gain = 1.0
        return self._gain

    def _compute_ramp(self, now: float, target: float) -> float:
        """Linear interpolation from segment_start_gain toward target over ramp_s."""
        if self.ramp_s <= 0.0:
            return float(target)
        elapsed = max(0.0, now - self._segment_start_t)
        frac = min(1.0, elapsed / self.ramp_s)
        start = self._segment_start_gain
        return float(start + (target - start) * frac)
