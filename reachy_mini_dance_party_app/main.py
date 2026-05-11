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
from reachy_mini_dance_party_app.settings_server import (
    build_app as _build_settings_app,
    start_in_thread as _start_settings_server,
)


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


def _restore_librosa_numba_cache() -> bool:
    """Copy bundled numba cache files into librosa's site-packages tree.

    Numba caches ``@njit(cache=True)`` results in
    ``librosa/.../__pycache__/*.nbi`` and ``*.nbc``. We bundle pre-built
    caches for ``linux_aarch64`` + ``cpython 3.12`` (the Reachy Mini
    wireless config) so the very first run doesn't pay the ~8-minute
    cold compile. Cache files contain an internal source hash; if librosa
    has been upgraded since the cache was built, numba ignores the stale
    file and recompiles — so falling back is safe.

    Only runs on platforms where shipping the cache makes sense (Pi-class
    aarch64 Linux on Python 3.12). x86_64 desktops JIT in seconds, no
    need to bundle.

    Returns True if any cache files were copied this call.
    """
    import platform as _plat
    import shutil as _sh
    import sys as _sys
    machine = _plat.machine().lower()
    system = _plat.system().lower()
    pyver = f"py{_sys.version_info.major}{_sys.version_info.minor}"
    if (machine, system, pyver) != ("aarch64", "linux", "py312"):
        return False
    bundled = Path(__file__).parent / "_numba_cache_linux_aarch64_py312"
    if not bundled.exists():
        return False
    try:
        import librosa
        librosa_root = Path(librosa.__file__).parent
    except Exception:  # noqa: BLE001
        return False
    copied = 0
    skipped = 0
    for src in bundled.rglob("*"):
        if src.is_dir():
            continue
        # Map .../bundled/librosa/<subpath>/_pycache/<file> →
        # .../site-packages/librosa/<subpath>/__pycache__/<file>
        # (the bundle uses _pycache so .gitignore's __pycache__/ rule doesn't
        # exclude it from the repo and shipped wheel)
        rel = src.relative_to(bundled / "librosa")
        rel_translated = Path(*[
            "__pycache__" if p == "_pycache" else p for p in rel.parts
        ])
        dst = librosa_root / rel_translated
        if dst.exists():
            skipped += 1
            continue
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            _sh.copy2(src, dst)
            copied += 1
        except Exception as exc:  # noqa: BLE001
            log.debug("librosa cache copy failed for %s: %s", rel, exc)
    if copied:
        log.info(
            "librosa numba cache restored from bundle (%d copied, %d already present)",
            copied, skipped,
        )
    return copied > 0


def _prewarm_librosa() -> None:
    """Run a tiny librosa.beat.beat_track to trigger numba JIT off the hot path."""
    try:
        import time
        import numpy as np
        import librosa  # noqa: PLC0415
        t0 = time.monotonic()
        # 1.5s of synthetic clicks at 120 BPM so beat_track has something real.
        sr = 22050
        sig = np.zeros(int(sr * 1.5), dtype=np.float32)
        for i in range(0, sig.size, sr // 2):
            sig[i : i + 200] = 0.8
        librosa.beat.beat_track(y=sig, sr=sr, units="time")
        log.info("librosa.beat pre-warm complete in %.1fs", time.monotonic() - t0)
    except Exception as exc:  # noqa: BLE001
        log.warning("librosa pre-warm failed (will JIT lazily): %s", exc)


def _prefetch_static_ffmpeg() -> None:
    """Trigger static-ffmpeg's binary download in the background.

    The package downloads ffmpeg + ffprobe (~50MB) on first use. Doing it
    here means the first play_song call doesn't stall on the download.
    No-op if a system ffmpeg is already on PATH (yt-dlp will find that
    first) or if static-ffmpeg already has the binaries cached.
    """
    try:
        import time
        import shutil as _shutil
        if _shutil.which("ffmpeg"):
            log.info("system ffmpeg present, skipping static-ffmpeg prefetch")
            return
        from static_ffmpeg.run import (
            get_or_fetch_platform_executables_else_raise,
        )
        t0 = time.monotonic()
        ffmpeg_bin, _ = get_or_fetch_platform_executables_else_raise()
        log.info(
            "static-ffmpeg ready in %.1fs (%s)",
            time.monotonic() - t0, ffmpeg_bin,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("static-ffmpeg prefetch failed: %s", exc)


def _start_song_progress_watcher(
    dj: Any,
    playback: Any,
    inject: Any,
    stop_event: threading.Event,
) -> threading.Thread:
    """Spawn a thread that nudges the model about song progress.

    Polls ``dj.current`` and ``playback.playback_time()``. When the active
    song is within ~20 s of its end, injects a one-shot system notice so
    the DJ can plan the next track. When playback passes the song's
    duration, injects a "song ended" notice and flips DJ state via
    ``dj.song_ended()``. Without this, the model is guessing — it tends to
    cut songs short when it hears a pause in the audio.
    """
    state = {
        "warned_id": None,
        "ended_id": None,
        "fetch_query": None,        # last query the watcher saw mid-fetch
        "fetch_started_at": None,   # monotonic ts when this fetch began
        "last_fetch_nudge_at": None,
    }

    import time as _time
    FETCH_NUDGE_AFTER_S = 7.0
    FETCH_NUDGE_INTERVAL_S = 8.0

    def _loop() -> None:
        while not stop_event.is_set():
            stop_event.wait(1.0)

            # Fetch-progress nudges: while the DJ is in FETCHING state, inject
            # periodic "still pulling it down" notices so the model can fill
            # the silence with brief progress updates instead of going quiet.
            dj_state_name = getattr(getattr(dj, "state", None), "name", "") or ""
            pending = getattr(dj, "pending_query", None)
            now_mono = _time.monotonic()
            if dj_state_name == "FETCHING" and pending:
                if state["fetch_query"] != pending:
                    state["fetch_query"] = pending
                    state["fetch_started_at"] = now_mono
                    state["last_fetch_nudge_at"] = None
                elapsed = now_mono - (state["fetch_started_at"] or now_mono)
                last_nudge = state["last_fetch_nudge_at"]
                if elapsed >= FETCH_NUDGE_AFTER_S and (
                    last_nudge is None or (now_mono - last_nudge) >= FETCH_NUDGE_INTERVAL_S
                ):
                    try:
                        inject(
                            f"Still fetching '{pending}' ({int(elapsed)}s in). "
                            "Toss the audience a brief filler line so it's "
                            "not silent (e.g. 'almost there', 'pulling it down')."
                        )
                    except Exception:  # noqa: BLE001
                        pass
                    state["last_fetch_nudge_at"] = now_mono
            else:
                # Reset fetch tracking when no fetch is active.
                state["fetch_query"] = None
                state["fetch_started_at"] = None
                state["last_fetch_nudge_at"] = None

            song = getattr(dj, "current", None)
            if song is None:
                continue
            duration = float(getattr(song, "duration_s", 0.0) or 0.0)
            if duration <= 0.0:
                continue
            song_id = id(song)
            try:
                t = float(playback.playback_time())
            except Exception:  # noqa: BLE001
                continue
            remaining = duration - t
            if remaining <= 20.0 and state["warned_id"] != song_id:
                try:
                    inject(
                        f"Track ending in ~{max(0, int(round(remaining)))}s. "
                        "Pick the next song now (auto-DJ) — call play_song "
                        "silently, do not announce the transition with voice."
                    )
                except Exception:  # noqa: BLE001
                    pass
                state["warned_id"] = song_id
            if remaining <= 0.0 and state["ended_id"] != song_id:
                state["ended_id"] = song_id
                try:
                    inject(
                        f"Track \"{song.title}\" finished. If no next song "
                        "is queued, ask the audience what they want to hear."
                    )
                except Exception:  # noqa: BLE001
                    pass
                try:
                    dj.song_ended()
                except Exception:  # noqa: BLE001
                    pass

    t = threading.Thread(target=_loop, daemon=True, name="SongProgress")
    t.start()
    log.info("SongProgress watcher started")
    return t


def _start_mic_capture(
    robot: Any,
    session: Any,
    stop_event: threading.Event,
) -> threading.Thread:
    """Spawn a daemon thread that pumps SDK mic samples into the realtime session.

    The SDK's GStreamer mic returns 16kHz stereo float32 chunks via
    ``media_manager.get_audio_sample()`` (returns ``None`` while the appsink
    is empty). We downmix to mono, resample 16kHz→24kHz (the Realtime API
    minimum input rate), convert to PCM16 LE, and push via the session's
    thread-safe ``push_mic_chunk`` shim.
    """
    import numpy as np
    from scipy.signal import resample_poly

    log_state = {"started": False, "first_chunk": False, "total_bytes": 0}

    def _loop() -> None:
        log.info("MicCapture thread entered _loop")
        try:
            robot.media_manager.start_recording()
            log_state["started"] = True
            log.info("mic capture started (16kHz stereo → 24kHz mono PCM16)")
        except Exception as exc:  # noqa: BLE001
            log.exception("mic start_recording failed: %s", exc)
            return
        while not stop_event.is_set():
            try:
                sample = robot.media_manager.get_audio_sample()
            except Exception as exc:  # noqa: BLE001
                log.warning("get_audio_sample raised: %s", exc)
                stop_event.wait(0.05)
                continue
            if sample is None or sample.size == 0:
                stop_event.wait(0.01)
                continue
            # Stereo (frames, 2) → mono (frames,)
            if sample.ndim == 2 and sample.shape[1] >= 2:
                mono = sample.mean(axis=1)
            else:
                mono = sample.reshape(-1)
            # 16kHz → 24kHz polyphase resample (3:2 ratio).
            mono = resample_poly(mono, up=3, down=2).astype(np.float32, copy=False)
            mono = np.clip(mono, -1.0, 1.0)
            pcm16 = (mono * 32767.0).astype("<i2").tobytes()
            try:
                session.push_mic_chunk(pcm16)
            except Exception as exc:  # noqa: BLE001
                log.warning("push_mic_chunk failed: %s", exc)
                continue
            log_state["total_bytes"] += len(pcm16)
            if not log_state["first_chunk"]:
                log.info(
                    "first mic chunk pushed to realtime (%d bytes)", len(pcm16)
                )
                log_state["first_chunk"] = True

    log.info("spawning MicCapture thread")
    t = threading.Thread(target=_loop, daemon=True, name="MicCapture")
    t.start()
    return t


class _MediaManagerStream:
    """Drives a PlaybackEngine-style callback, routing output to the SDK sink.

    Replaces ``sounddevice.OutputStream`` in environments where the daemon
    owns the speaker (Pi). The callback is invoked in a dedicated thread at
    a fixed cadence; its mono output is broadcast to the sink's channel
    count, resampled if the sink runs at a different rate, and pushed via
    ``media_manager.push_audio_sample``. PlaybackEngine's ``_frames_played``
    counter advances in step so ``playback_time()`` stays accurate for the
    dance scheduler.
    """

    _BLOCK_FRAMES = 2048  # ~128ms @ 16kHz — gives the GStreamer queue headroom

    def __init__(
        self,
        media_manager: Any,
        sink_sr: int,
        sink_channels: int,
        engine_sr: int,
        engine_channels: int,
        callback: Any,
    ) -> None:
        import numpy as np
        self._np = np
        self._mm = media_manager
        self._sink_sr = int(sink_sr)
        self._sink_channels = int(sink_channels)
        self._engine_sr = int(engine_sr)
        self._engine_channels = int(engine_channels)
        self._callback = callback
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="MusicStreamer"
        )
        self._thread.start()
        log.info(
            "MusicStreamer started: engine=%dHz/%dch sink=%dHz/%dch block=%d",
            self._engine_sr, self._engine_channels,
            self._sink_sr, self._sink_channels, self._BLOCK_FRAMES,
        )

    def stop(self) -> None:
        self._stop_event.set()

    def close(self) -> None:
        self.stop()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=1.0)
        self._thread = None

    def _run(self) -> None:
        np = self._np
        block = self._BLOCK_FRAMES
        period = block / float(self._engine_sr)
        outdata = np.zeros((block, self._engine_channels), dtype=np.float32)
        next_deadline = None
        first_logged = False
        while not self._stop_event.is_set():
            outdata.fill(0.0)
            try:
                self._callback(outdata, block, None, None)
            except Exception as exc:  # noqa: BLE001
                log.exception("MusicStreamer callback raised: %s", exc)
                self._stop_event.wait(period)
                continue
            mono = outdata[:, 0] if self._engine_channels >= 1 else outdata.mean(axis=1)
            if self._sink_sr != self._engine_sr:
                from scipy.signal import resample_poly
                # 48k → 16k = 1:3, 44.1k → 16k = 160:441 etc.
                # Keep it general via gcd reduction so resample_poly is happy.
                from math import gcd
                g = gcd(self._sink_sr, self._engine_sr)
                up = self._sink_sr // g
                down = self._engine_sr // g
                mono = resample_poly(mono, up=up, down=down).astype(np.float32, copy=False)
            if self._sink_channels > 1:
                sink_block = np.tile(mono[:, None], (1, self._sink_channels))
            else:
                sink_block = mono
            try:
                self._mm.push_audio_sample(sink_block)
            except Exception as exc:  # noqa: BLE001
                log.warning("MusicStreamer push_audio_sample failed: %s", exc)
            if not first_logged and float(np.abs(mono).max()) > 1e-4:
                log.info("MusicStreamer first non-silent block pushed")
                first_logged = True
            # Pace ourselves so we don't outrun the sink's queue.
            import time as _time
            now = _time.monotonic()
            if next_deadline is None:
                next_deadline = now + period
            else:
                next_deadline += period
            sleep_for = next_deadline - now
            if sleep_for > 0:
                self._stop_event.wait(sleep_for)
            else:
                # Clock fell behind; reset to avoid a runaway burst.
                next_deadline = now + period


def _make_media_stream_factory(
    robot: Any,
    engine_sr: int,
    engine_channels: int,
) -> Any:
    """Build a stream_factory for PlaybackEngine that routes to the SDK sink."""
    sink_sr = robot.media_manager.get_output_audio_samplerate()
    sink_channels = robot.media_manager.get_output_channels()

    def factory(**kwargs: Any) -> _MediaManagerStream:
        # PlaybackEngine passes samplerate/channels matching its own config —
        # we honor those for the callback shape and fan out to the sink rate.
        engine_sr_kw = int(kwargs.get("samplerate", engine_sr))
        engine_ch_kw = int(kwargs.get("channels", engine_channels))
        callback = kwargs["callback"]
        return _MediaManagerStream(
            media_manager=robot.media_manager,
            sink_sr=sink_sr,
            sink_channels=sink_channels,
            engine_sr=engine_sr_kw,
            engine_channels=engine_ch_kw,
            callback=callback,
        )

    return factory


def _make_sdk_tts_push(
    robot: Any,
    src_sr: int,
    dst_sr: int,
    channels: int,
) -> Any:
    """Build a callback that pushes OpenAI Realtime PCM16 chunks into the
    SDK's audio sink.

    The Realtime API delivers little-endian PCM16 mono at ``src_sr`` (24kHz on
    GA gpt-realtime). The SDK's ``media_manager.push_audio_sample`` expects
    float32 ndarray at ``dst_sr`` with ``channels`` channels. We resample with
    ``scipy.signal.resample_poly`` (C-based, no numba JIT — librosa.resample
    JIT-compiles on first call and blocked the realtime event loop for ~40s
    on the Pi, dropping the greeting before any audio reached the speaker).
    """
    import numpy as np
    from math import gcd
    from scipy.signal import resample_poly

    if src_sr != dst_sr:
        g = gcd(src_sr, dst_sr)
        up = dst_sr // g
        down = src_sr // g
    else:
        up = down = 1

    first_logged = {"done": False}

    def push(pcm_bytes: bytes) -> None:
        if not pcm_bytes:
            return
        audio_i16 = np.frombuffer(pcm_bytes, dtype="<i2")
        if audio_i16.size == 0:
            return
        audio_f32 = (audio_i16.astype(np.float32) / 32768.0)
        if up != 1 or down != 1:
            audio_f32 = resample_poly(audio_f32, up=up, down=down).astype(
                np.float32, copy=False
            )
        if channels > 1:
            # Mono → N-channel broadcast (shape (frames, channels)).
            audio_f32 = np.tile(audio_f32[:, None], (1, channels))
        try:
            robot.media_manager.push_audio_sample(audio_f32)
        except Exception as exc:  # noqa: BLE001
            log.warning("push_audio_sample failed: %s", exc)
            return
        if not first_logged["done"]:
            log.info(
                "first TTS chunk pushed to SDK sink (%d frames @ %d Hz, %d ch)",
                len(audio_i16), dst_sr, channels,
            )
            first_logged["done"] = True

    return push


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

    # Surfaced on the Reachy Mini dashboard as a "Settings" link. The daemon
    # parses this attribute directly from main.py source via regex (see
    # ``reachy_mini.apps.sources.local_common_venv._get_custom_app_url_from_file``)
    # so the literal here matters — don't replace with an f-string or constant.
    custom_app_url: str | None = "http://0.0.0.0:8050"

    def __init__(self) -> None:
        self._stack = ExitStack()
        self._stop_event = threading.Event()
        self._dj: Any = None
        self._playback: Any = None
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

        # Pre-warm librosa.beat (numba JIT compile of the onset_strength
        # ufuncs takes 60–300s on a Pi the first time it runs). Doing it on
        # a background thread before any user request avoids that delay being
        # incurred inside play_song while the realtime session looks frozen.
        # Restore librosa numba cache (synchronous; cheap file copy) BEFORE
        # spawning the prewarm thread. If the bundle was applicable, the
        # prewarm will load from cache in ~1s instead of ~8 min.
        _restore_librosa_numba_cache()
        threading.Thread(
            target=_prewarm_librosa, daemon=True, name="LibrosaPreWarm",
        ).start()
        threading.Thread(
            target=_prefetch_static_ffmpeg, daemon=True, name="FfmpegPreFetch",
        ).start()

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
        # Re-acquire media on the daemon side BEFORE constructing ReachyMini.
        # If a prior session called /api/media/release, the daemon's status
        # is "available=false, released=true" and ``ReachyMini()`` will pick
        # the WebRTC backend, which then fails to connect to a non-existent
        # signaling server. /api/media/acquire flips it back to LOCAL so the
        # SDK uses the in-process GStreamer IPC path.
        try:
            httpx.post("http://localhost:8000/api/media/acquire", timeout=5.0)
            log.info("media acquired on daemon (LOCAL backend)")
        except Exception as exc:  # noqa: BLE001
            log.warning("could not acquire media on daemon: %s", exc)

        # 1. SDK client - shared across move worker + camera worker.
        robot = ReachyMini()

        # 2. Move worker — sole writer to set_target.
        move_manager = MovementManager(current_robot=robot)
        move_manager.start()
        self._stack.callback(move_manager.stop)

        # Energize motors so the head holds upright at idle (BreathingMove
        # in moves.py needs torque to actually visibly breathe; with motors
        # disabled the robot just sags). The /api/motors/set_mode endpoint
        # routes through the daemon's same control loop the SDK is using.
        try:
            httpx.post("http://localhost:8000/api/motors/set_mode/enabled", timeout=5.0)
            log.info("motors enabled")
        except Exception as exc:  # noqa: BLE001 - non-fatal at startup
            log.warning("could not enable motors: %s", exc)

        # Re-acquire media on the daemon side. If a prior session had called
        # /api/media/release, the daemon flips the SDK's MediaBackend to
        # WebRTC mode for new clients, and our ReachyMini() construction
        # below tries to connect to a WebRTC signaling server that isn't
        # there (ConnectionRefused). For LOCAL operation — which is what we
        # want, since we're co-hosted with the daemon and want the GStreamer
        # IPC path — the daemon needs to be holding the hardware.
        try:
            httpx.post("http://localhost:8000/api/media/acquire", timeout=5.0)
            log.info("media acquired on daemon (LOCAL backend)")
        except Exception as exc:  # noqa: BLE001
            log.warning("could not acquire media on daemon: %s", exc)

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

        # 5. Audio: mixer + ducker still constructed (pure logic, used by
        # PlaybackEngine for music ducking later), but TTS goes through the
        # SDK's media_manager.push_audio_sample() which shares the daemon's
        # GStreamer audio sink. PortAudio's exclusive ALSA grab fights the
        # daemon for /dev/snd/pcmC0D0p; the SDK path is the supported one.
        mixer = Mixer()
        ducker = Ducker()
        # PlaybackEngine drives a stream_factory that we redirect to the SDK
        # sink. play_song.handler calls playback.load() and playback.start();
        # start() opens the (fake) stream and the streamer thread begins
        # pulling chunks via the engine's callback and pushing them to the
        # daemon's GStreamer audio output.
        # Match the SDK sink rate so the streamer doesn't have to resample
        # each block (boundary ringing causes audible glitches). The wav is
        # resampled once in PlaybackEngine.load instead.
        engine_sr = robot.media_manager.get_output_audio_samplerate()
        engine_channels = 1
        media_stream_factory = _make_media_stream_factory(
            robot, engine_sr=engine_sr, engine_channels=engine_channels
        )
        playback = PlaybackEngine(
            mixer=mixer,
            ducker=ducker,
            sample_rate=engine_sr,
            channels=engine_channels,
            stream_factory=media_stream_factory,
        )

        # Open the SDK's audio output and prepare a TTS push callback that
        # converts the OpenAI PCM16/24k chunks into float32 at the daemon's
        # native sample rate before pushing.
        robot.media_manager.start_playing()
        self._stack.callback(robot.media_manager.stop_playing)
        sdk_sr = robot.media_manager.get_output_audio_samplerate()
        sdk_ch = robot.media_manager.get_output_channels()
        log.info("SDK audio sink ready: sr=%d channels=%d", sdk_sr, sdk_ch)
        tts_push = _make_sdk_tts_push(robot, src_sr=24000, dst_sr=sdk_sr, channels=sdk_ch)

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
            on_tts_chunk=tts_push,
            on_speech_state=playback.set_speech_active,
        )
        self._session = session  # exposed for tests

        def _session_target() -> None:
            try:
                asyncio.run(session.run())
            except Exception:  # noqa: BLE001 — surface silent thread crashes
                log.exception("Realtime session thread crashed")

        session_thread = threading.Thread(
            target=_session_target,
            daemon=True,
            name="RealtimeSession",
        )
        session_thread.start()
        # stop() is sync but schedules the websocket close on the session
        # loop; the daemon thread exits when run() returns.
        self._stack.callback(session.stop)

        # 9b. Mic capture worker — pumps SDK mic into the realtime session.
        # Without this, server VAD never sees user audio so no responses fire.
        mic_stop = threading.Event()
        _start_mic_capture(robot, session, mic_stop)
        def _stop_mic() -> None:
            mic_stop.set()
            try:
                robot.media_manager.stop_recording()
            except Exception:  # noqa: BLE001
                pass
        self._stack.callback(_stop_mic)

        # 9c. Song-progress watcher — tells the DJ when the current song is
        # almost over / has ended, so it doesn't cut songs short by guessing.
        progress_stop = threading.Event()
        _start_song_progress_watcher(
            dj=dj,
            playback=playback,
            inject=session.inject_system_event,
            stop_event=progress_stop,
        )
        self._stack.callback(progress_stop.set)

        # 9d. Local settings UI. Reachable at the URL declared in
        # ``custom_app_url`` on this class. The Reachy Mini dashboard shows a
        # "Settings" link to it. Lets the user paste an OpenAI API key,
        # see live status, and skip/stop without using voice.
        def _do_skip() -> None:
            playback.stop()
            dancer.clear_grid()
            dj.song_ended()

        def _do_stop() -> None:
            playback.stop()
            dancer.clear_grid()
            dj.stop_party()

        settings_app = _build_settings_app(
            get_dj=lambda: self._dj,
            get_playback=lambda: self._playback,
            on_skip=_do_skip,
            on_stop=_do_stop,
            env_path=Path.home() / ".env",
        )
        settings_server, _settings_thread = _start_settings_server(
            settings_app, host="0.0.0.0", port=8050,
        )
        def _stop_settings() -> None:
            settings_server.should_exit = True
        self._stack.callback(_stop_settings)

        # 10. Audience push timer. Sends summary JSON via the session's
        # system-event injection.
        # Suppress audience pushes while a song is playing — they distract
        # the model into commenting mid-song, which the user explicitly
        # doesn't want. Pushes resume between songs (idle / fetching).
        def _audience_should_push() -> bool:
            d = self._dj
            if d is None:
                return True
            return getattr(getattr(d, "state", None), "name", "") != "PLAYING"

        audience = AudiencePush(
            get_latest_frame=camera.get_latest_frame,
            push=session.inject_system_event,
            should_push=_audience_should_push,
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
