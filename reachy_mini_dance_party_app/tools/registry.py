"""Tool registry: AppContext, Tool dataclass, and ``all_tools(ctx)`` aggregator.

The :class:`AppContext` is a thin container for cross-cutting refs that tools
call into (DJ, dancer, mixer, camera worker, etc.). Most fields are typed as
``Any`` because the real implementations live in other modules; this keeps
tools loosely coupled and trivial to mock in unit tests.

Each tool module under :mod:`reachy_mini_dance_party_app.tools` exposes a
single ``make(ctx: AppContext) -> Tool`` factory. :func:`all_tools` calls
each factory and returns the full list, ready for registration with the
OpenAI Realtime API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


class ToolError(Exception):
    """Raised by tool handlers on validation or runtime failure."""


@dataclass
class AppContext:
    """Container for cross-cutting refs that tools call into.

    Most fields are typed ``Any`` since the real implementations
    (DJ, LibraryDancer, CameraWorker, etc.) live in other modules. This
    keeps tools loosely coupled and easy to test with mocks.
    """

    dj: Any = None                # reachy_mini_dance_party_app.dj.DJ
    dancer: Any = None            # LibraryDancer
    mixer: Any = None             # reachy_mini_dance_party_app.music.mixer.Mixer
    playback: Any = None          # PlaybackEngine
    camera_worker: Any = None     # vision.camera_worker.CameraWorker
    face_tracker: Any = None      # vision.face_tracker.FaceTracker
    move_queue: Any = None        # primary-move queue (consumed by moves.MovementManager)
    fetcher: Any = None           # YouTubeFetcher
    analyzer: Any = None          # callable wav_path -> BeatGrid (analysis.analyze_beats)
    daemon_url: str = "http://localhost:8000"
    http_client: Any = None       # httpx.Client; lazy-default in tools that need it


@dataclass(frozen=True)
class Tool:
    """A single LLM-callable tool.

    Attributes:
        name: Function name as registered with the Realtime API.
        description: Human-readable description shown to the model.
        parameters: JSON Schema (Draft 2020-12) describing args.
        handler: Callable taking the parsed kwargs dict, returning a
            JSON-serializable result. May raise :class:`ToolError`.
    """

    name: str
    description: str
    parameters: dict
    handler: Callable[[dict], Any]


def all_tools(ctx: AppContext) -> list[Tool]:
    """Return every tool in the registry, bound to ``ctx``."""
    # Local import to avoid circular imports at package-load time.
    from reachy_mini_dance_party_app.tools import (  # noqa: PLC0415
        look_at,
        look_at_audience,
        move_head,
        play_emotion,
        play_song,
        set_face_tracking,
        set_volume,
        skip_song,
        stop_dance,
        stop_party,
        take_photo,
    )

    return [
        play_song.make(ctx),
        skip_song.make(ctx),
        stop_party.make(ctx),
        set_volume.make(ctx),
        take_photo.make(ctx),
        look_at_audience.make(ctx),
        set_face_tracking.make(ctx),
        move_head.make(ctx),
        play_emotion.make(ctx),
        look_at.make(ctx),
        stop_dance.make(ctx),
    ]
