# reachy_mini_dance_party_app/dj.py
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

class DJState(Enum):
    IDLE = auto()
    FETCHING = auto()
    PLAYING = auto()

@dataclass(frozen=True)
class SongInfo:
    title: str
    duration_s: float
    url: str
    path: str
    query: str

@dataclass
class DJ:
    auto_dj: bool = True
    state: DJState = DJState.IDLE
    current: Optional[SongInfo] = None
    pending_query: Optional[str] = None
    last_error: Optional[str] = None
    history: list[SongInfo] = field(default_factory=list)
    _next_auto_query: Optional[str] = None

    def request_song(self, query: str) -> None:
        self.pending_query = query
        self.state = DJState.FETCHING
        self.last_error = None

    def song_fetched(self, info: SongInfo) -> None:
        self.current = info
        self.pending_query = None
        self.state = DJState.PLAYING

    def fetch_failed(self, error: str) -> None:
        self.last_error = error
        self.pending_query = None
        self.state = DJState.IDLE

    def song_ended(self) -> None:
        if self.current is not None:
            self.history.append(self.current)
        self.current = None
        if self.auto_dj and self._next_auto_query:
            self.request_song(self._next_auto_query)
            self._next_auto_query = None
        else:
            self.state = DJState.IDLE

    def stop_party(self) -> None:
        self.current = None
        self.pending_query = None
        self._next_auto_query = None
        self.state = DJState.IDLE

    def set_pending_auto_dj_query(self, query: str) -> None:
        self._next_auto_query = query
