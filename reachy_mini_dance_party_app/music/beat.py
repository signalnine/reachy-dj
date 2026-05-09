from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class BeatGrid:
    tempo: float
    beat_times: np.ndarray  # seconds, monotonically increasing

    @property
    def beats_per_second(self) -> float:
        return self.tempo / 60.0

    def beats_in_window(self, duration_s: float) -> int:
        return int(round(duration_s * self.beats_per_second))

    def next_beat_at(self, t: float, n: int = 1) -> list[float]:
        idx = int(np.searchsorted(self.beat_times, t, side="right"))
        return self.beat_times[idx : idx + n].tolist()

    @classmethod
    def synthetic(cls, tempo: float, duration: float) -> "BeatGrid":
        bps = tempo / 60.0
        beats = np.arange(0.0, duration, 1.0 / bps)
        return cls(tempo=tempo, beat_times=beats)
