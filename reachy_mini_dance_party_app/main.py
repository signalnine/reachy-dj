"""Reachy Mini Dance Party App — entry point.

Wires every component into a runnable app:

* :class:`reachy_mini.ReachyMini` SDK client
* :class:`~reachy_mini_dance_party_app.moves.MovementManager` — sole writer to
  ``robot.set_target``
* :class:`~reachy_mini_dance_party_app.vision.camera_worker.CameraWorker`
  + :class:`~reachy_mini_dance_party_app.vision.face_tracker.FaceTracker`
* :class:`~reachy_mini_dance_party_app.music.playback.PlaybackEngine` with
  :class:`~reachy_mini_dance_party_app.music.mixer.Mixer` /
  :class:`~reachy_mini_dance_party_app.music.mixer.Ducker`
* :class:`~reachy_mini_dance_party_app.dance.library_dancer.LibraryDancer`
* :class:`~reachy_mini_dance_party_app.dj.DJ` state
* :class:`~reachy_mini_dance_party_app.tools.AppContext` + tool registry
* :class:`~reachy_mini_dance_party_app.voice.openai_realtime.OpenAIRealtimeSession`
  running on its own asyncio thread
* :class:`~reachy_mini_dance_party_app.vision.audience.AudiencePush`

Lifecycle: every ``start()`` is paired with a ``callback`` on an
``ExitStack`` so shutdown happens in reverse order automatically when
``ExitStack.close()`` runs in the ``finally`` block.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import signal
import threading
from contextlib import ExitStack
from importlib import resources
from pathlib import Path
from typing import Any, Optional

import httpx

from reachy_mini import ReachyMini

from reachy_mini_dance_party_app.dance.library_dancer import LibraryDancer
from reachy_mini_dance_party_app.dance.picker import MoveSpec
from reachy_mini_dance_party_app.dj import DJ
from reachy_mini_dance_party_app.moves import MovementManager
from reachy_mini_dance_party_app.music.analysis import analyze_beats
from reachy_mini_dance_party_app.music.mixer import Ducker, Mixer
from reachy_mini_dance_party_app.music.playback import PlaybackEngine
from reachy_mini_dance_party_app.music.youtube import YouTubeFetcher
from reachy_mini_dance_party_app.tools import AppContext, all_tools
from reachy_mini_dance_party_app.vision.audience import AudiencePush
from reachy_mini_dance_party_app.vision.camera_worker import CameraWorker
from reachy_mini_dance_party_app.vision.face_tracker import FaceTracker
from reachy_mini_dance_party_app.voice.openai_realtime import OpenAIRealtimeSession


log = logging.getLogger(__name__)


_FRAME_FETCHER_DEFAULT_FOV_DEG = 80.0  # imx708 horizontal FOV (small-angle approx)


def _load_env_files() -> None:
    """Load .env from common locations into os.environ.

    The daemon's app launcher (`reachy_mini.apps.manager`) copies the daemon's
    own environment when starting an app, which omits anything the user dropped
    in ~/.env. Load it ourselves so the same configuration works whether the
    app is launched from the dashboard, via `python -m`, or via systemd.

    Existing env vars take precedence (does not overwrite). Missing files are
    silently ignored. python-dotenv is a declared dependency.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        log.debug("python-dotenv not installed; skipping .env load")
        return
    seen: set[Path] = set()
    candidates = [
        Path.home() / ".env",
        Path.cwd() / ".env",
        Path("/home/pollen/.env"),  # fallback for dashboard-launched runs
    ]
    for p in candidates:
        try:
            resolved = p.resolve()
        except OSError:
            continue
        if resolved in seen or not resolved.is_file():
            continue
        seen.add(resolved)
        load_dotenv(resolved, override=False)
        log.info("loaded environment from %s", resolved)


def load_system_prompt() -> str:
    """Load the DJ persona system prompt shipped with the package."""
    try:
        # importlib.resources is the modern way to read package data files.
        return (
            resources.files("reachy_mini_dance_party_app.prompts")
            .joinpath("system.md")
            .read_text(encoding="utf-8")
        )
    except Exception as exc:  # noqa: BLE001 - fallback for in-tree dev runs
        fallback = Path(__file__).parent / "prompts" / "system.md"
        log.debug("falling back to filesystem path for system.md: %s", exc)
        return fallback.read_text(encoding="utf-8")


def _build_dance_catalog() -> list[MoveSpec]:
    """Inspect ``reachy_mini_dances_library`` and return a MoveSpec catalog.

    Each ``DanceMove`` exposes a ``duration`` property in seconds. The
    library is Pi-only in some configurations; if it isn't importable on
    this host we log a warning and return an empty catalog so the rest of
    the app still boots (the LibraryDancer just no-ops without a grid).
    """
    try:
        from reachy_mini_dances_library.dance_move import (  # noqa: PLC0415
            AVAILABLE_MOVES,
            DanceMove,
        )
    except Exception as exc:  # noqa: BLE001 - library is optional on laptop
        log.warning("dance library unavailable; LibraryDancer will idle: %s", exc)
        return []

    catalog: list[MoveSpec] = []
    for name in AVAILABLE_MOVES:
        try:
            move = DanceMove(name)
            duration_s = float(move.duration)
        except Exception as exc:  # noqa: BLE001 - skip a single bad entry
            log.warning("skipping dance move %r (failed to construct): %s", name, exc)
            continue
        if not math.isfinite(duration_s) or duration_s <= 0.0:
            log.warning(
                "skipping dance move %r (invalid duration %s)", name, duration_s
            )
            continue
        catalog.append(MoveSpec(name=name, duration_s=duration_s))
    log.info("dance catalog built: %d moves", len(catalog))
    return catalog


def _face_offset_provider(face_tracker: FaceTracker):
    """Bridge ``FaceTracker.get_secondary_offset`` -> 6-tuple offsets.

    The MovementManager expects ``(x, y, z, roll, pitch, yaw)``. The face
    tracker exposes only yaw and pitch (small-angle head bias); the other
    axes stay at zero so the dance pose dominates.
    """

    def _provider() -> tuple[float, float, float, float, float, float]:
        offset = face_tracker.get_secondary_offset()
        yaw = float(offset.get("yaw", 0.0))
        pitch = float(offset.get("pitch", 0.0))
        return (0.0, 0.0, 0.0, 0.0, pitch, yaw)

    return _provider


class ReachyMiniDancePartyApp:
    """Runnable entry point that assembles every thread + lifecycle."""

    def __init__(self) -> None:
        self._stack = ExitStack()
        self._stop_event = threading.Event()
        log.info("ReachyMiniDancePartyApp initialized")

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Boot every component, install signal handlers, block until stop."""
        _load_env_files()
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not set. Put OPENAI_API_KEY=sk-... in ~/.env "
                "or export it before launching the app."
            )

        try:
            self._assemble(api_key)
            self._install_signal_handlers()
            log.info("ReachyMiniDancePartyApp running. SIGINT/SIGTERM to stop.")
            self._stop_event.wait()
        finally:
            log.info("Shutting down.")
            self._stack.close()

    # ------------------------------------------------------------------
    # Assembly
    # ------------------------------------------------------------------

    def _assemble(self, api_key: str) -> None:
        # 1. SDK client - shared across move worker + camera worker.
        robot = ReachyMini()

        # 2. Move worker — sole writer to set_target.
        move_manager = MovementManager(current_robot=robot)
        move_manager.start()
        self._stack.callback(move_manager.stop)

        # 3. Camera worker (no SDK head tracker; we use FaceTracker instead).
        camera = CameraWorker(reachy_mini=robot, head_tracker=None)
        camera.start()
        self._stack.callback(camera.stop)

        # 4. Face tracker — polls camera frames, feeds secondary offsets.
        face_tracker = FaceTracker(frame_source=camera)
        face_tracker.start()
        self._stack.callback(face_tracker.stop)
        # Hook face tracker as the secondary-offset producer on the move
        # worker (overrides the default camera_worker face-offset source).
        move_manager.set_face_offset_provider(_face_offset_provider(face_tracker))

        # 5. Audio: mixer, ducker, playback engine.
        mixer = Mixer()
        ducker = Ducker()
        playback = PlaybackEngine(mixer=mixer, ducker=ducker)
        # PlaybackEngine.start() is idempotent and is also called by the
        # play_song tool when a song is loaded; we don't pre-start it here
        # because that would open the audio device with an empty buffer.
        # The stop() is still registered so any started stream gets closed.
        self._stack.callback(playback.stop)

        # 6. LibraryDancer (idle until a song's BeatGrid is set).
        dance_catalog = _build_dance_catalog()
        dancer = LibraryDancer(
            catalog=dance_catalog,
            get_playback_time=playback.playback_time,
            enqueue_move=lambda name, t: move_manager.queue_move_named(
                name, scheduled_at=t
            ),
            # No external grid source — play_song.handler calls
            # dancer.start_with_grid(grid) to install one and start the
            # thread. start_with_grid is a no-op if already running.
            get_grid=lambda: None,
        )
        self._stack.callback(dancer.stop)

        # 7. DJ state machine (auto-DJ on by default).
        dj = DJ(auto_dj=True)

        # 8. AppContext + tool registry.
        ctx = AppContext(
            dj=dj,
            dancer=dancer,
            mixer=mixer,
            playback=playback,
            camera_worker=camera,
            face_tracker=face_tracker,
            move_queue=move_manager.tool_move_queue,
            fetcher=YouTubeFetcher(),
            analyzer=analyze_beats,
            http_client=httpx.Client(timeout=10.0),
        )
        tools = all_tools(ctx)

        # 9. Realtime session in its own asyncio-driven thread.
        session = OpenAIRealtimeSession(
            api_key=api_key,
            tools=tools,
            system_prompt=load_system_prompt(),
            on_tts_chunk=playback.feed_tts_chunk,
            on_speech_state=playback.set_speech_active,
        )
        self._session = session  # exposed for tests
        session_thread = threading.Thread(
            target=lambda: asyncio.run(session.run()),
            daemon=True,
            name="RealtimeSession",
        )
        session_thread.start()
        # stop() is sync but schedules the websocket close on the session
        # loop; the daemon thread exits when run() returns.
        self._stack.callback(session.stop)

        # 10. Audience push timer. Sends summary JSON via the session's
        # system-event injection.
        audience = AudiencePush(
            get_latest_frame=camera.get_latest_frame,
            push=session.inject_system_event,
        )
        audience.start()
        self._stack.callback(audience.stop)

        # Stash for tests / inspection.
        self._move_manager = move_manager
        self._camera = camera
        self._face_tracker = face_tracker
        self._playback = playback
        self._dancer = dancer
        self._dj = dj
        self._audience = audience
        self._http_client: Optional[Any] = ctx.http_client
        # Close the http client on shutdown too.
        if self._http_client is not None:
            self._stack.callback(self._http_client.close)

    # ------------------------------------------------------------------
    # Signal handling
    # ------------------------------------------------------------------

    def _install_signal_handlers(self) -> None:
        """Install SIGINT / SIGTERM handlers that flip the stop event.

        Best-effort: signals can only be installed from the main thread.
        Calls from other threads (e.g. integration tests) silently skip
        installation — the test sets ``_stop_event`` directly to shut down.
        """
        if threading.current_thread() is not threading.main_thread():
            log.debug("signal handlers skipped (not on main thread)")
            return
        try:
            signal.signal(signal.SIGINT, lambda *_: self._stop_event.set())
            signal.signal(signal.SIGTERM, lambda *_: self._stop_event.set())
        except (ValueError, OSError) as exc:
            log.warning("could not install signal handlers: %s", exc)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    ReachyMiniDancePartyApp().run()


if __name__ == "__main__":
    main()
