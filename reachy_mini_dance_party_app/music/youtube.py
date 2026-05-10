"""yt-dlp wrapper with on-disk cache and structured errors.

Public API:
    - ``FetchError``: raised when yt-dlp fails for any reason.
    - ``FetchResult``: frozen dataclass returned by ``YouTubeFetcher.fetch``.
    - ``YouTubeFetcher``: blocking fetcher with a query-hash cache.

Cache layout (under ``cache_dir``):
    <hash>.wav         -- the audio (extracted to wav by FFmpegExtractAudio)
    <hash>.info.json   -- {"title", "duration_s", "url", "query"}

A hash is computed from the query string, so identical queries hit the cache
without ever invoking yt-dlp again. yt-dlp itself names downloaded files by
the YouTube video id; after the postprocessor finishes we rename
``<id>.wav`` to ``<hash>.wav`` so cache lookups are O(1) by query.
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import yt_dlp


_log = logging.getLogger(__name__)

# yt-dlp's ytsearch1 happily returns hour-long compilations / livestreams
# when a query is generic (e.g. "classic funk" → "Classic Funk Mix 3 Hours").
# librosa.beat_track on a 500MB wav grinds the Pi for many minutes and
# blocks subsequent tool calls, so we hard-cap at 8 minutes.
_MAX_SONG_DURATION_S = 8 * 60


def _duration_filter(info: dict) -> str | None:
    """yt-dlp ``match_filter`` callback: skip overly long videos."""
    duration = info.get("duration") or 0
    if duration and duration > _MAX_SONG_DURATION_S:
        title = info.get("title", "?")
        return (
            f"skipping {title!r}: duration {duration}s exceeds cap "
            f"{_MAX_SONG_DURATION_S}s (likely a compilation/mix)"
        )
    return None


def _find_ffmpeg() -> str | None:
    """Locate ffmpeg, falling back to common paths and a static download.

    Order of preference:
      1. ``shutil.which("ffmpeg")`` — picks up a system install on a normal PATH.
      2. ``/usr/bin/ffmpeg`` / ``/usr/local/bin/ffmpeg`` — the daemon's app
         launcher trims PATH so ``which`` misses these; cover the common
         install location directly.
      3. ``static_ffmpeg`` — pure-Python package that downloads platform
         binaries (linux_arm64 / linux_x64 / win64 / osx_arm64) on first
         call. Lets the app work out-of-the-box on a fresh install without
         the user having to ``apt-get install ffmpeg``.
    """
    found = shutil.which("ffmpeg")
    if found:
        return found
    for candidate in ("/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg"):
        if Path(candidate).exists():
            return candidate
    try:
        from static_ffmpeg.run import (
            get_or_fetch_platform_executables_else_raise,
        )
        ffmpeg_bin, _ffprobe_bin = get_or_fetch_platform_executables_else_raise()
        return str(ffmpeg_bin)
    except Exception as exc:  # noqa: BLE001
        _log.warning("static-ffmpeg unavailable: %s", exc)
        return None


class FetchError(Exception):
    """Raised when yt-dlp fails to fetch / extract audio for a query."""


@dataclass(frozen=True)
class FetchResult:
    """The result of a successful (cached or fresh) YouTube fetch."""

    title: str
    duration_s: float
    url: str
    path: Path
    query: str


def _query_hash(query: str) -> str:
    return hashlib.sha256(query.encode()).hexdigest()[:16]


class YouTubeFetcher:
    """Blocking yt-dlp wrapper that caches audio + metadata by query hash.

    The cache is bounded by ``max_cached_songs``: at every fetch (cache miss
    or hit) the oldest .wav files beyond that count are deleted along with
    their sidecar .info.json / .beats.npy / .tempo.txt files. /tmp is tmpfs
    on the Pi (RAM-backed, ~2 GiB total) so unbounded growth would crowd
    out other processes.
    """

    def __init__(
        self,
        cache_dir: Path = Path("/tmp/reachy_dance_party"),
        max_cached_songs: int = 12,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_cached_songs = int(max_cached_songs)

    # ----- public --------------------------------------------------------

    def fetch(self, query: str) -> FetchResult:
        """Return a ``FetchResult`` for ``query``, downloading if needed.

        Cache hit: read sibling ``.info.json`` and return immediately.
        Cache miss: invoke yt-dlp via the Python API, postprocess to wav,
        rename ``<id>.wav`` to ``<hash>.wav``, write ``<hash>.info.json``.

        Raises:
            FetchError: if yt-dlp raises ``DownloadError`` (or any other
                exception during extraction).
        """
        # Ensure cache_dir exists even if it was deleted between __init__ and now.
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Best-effort eviction so old downloads don't fill /tmp (tmpfs).
        try:
            self._evict_old_entries()
        except Exception as exc:  # noqa: BLE001
            # Cache cleanup must never block a fetch.
            import logging
            logging.getLogger(__name__).warning("cache eviction skipped: %s", exc)

        h = _query_hash(query)
        wav_path = self.cache_dir / f"{h}.wav"
        info_path = self.cache_dir / f"{h}.info.json"

        if wav_path.exists() and info_path.exists():
            return self._load_cached(wav_path, info_path)

        opts = {
            "format": "bestaudio/best",
            "outtmpl": str(self.cache_dir / "%(id)s.%(ext)s"),
            "postprocessors": [
                {"key": "FFmpegExtractAudio", "preferredcodec": "wav"},
            ],
            "quiet": True,
            "no_warnings": True,
            "match_filter": _duration_filter,
            # ytsearch5: try the top 5 hits and let match_filter discard the
            # long ones, so a generic query that happens to match a 3-hour
            # mix as the first result still finds a real song.
            "playlistend": 5,
        }
        ffmpeg_path = _find_ffmpeg()
        if ffmpeg_path:
            # Daemon-launched apps may not have /usr/bin on PATH, so point
            # yt-dlp at ffmpeg directly. yt-dlp accepts a directory or a
            # binary path; the parent dir is what it actually uses.
            opts["ffmpeg_location"] = str(Path(ffmpeg_path).parent)

        _log.info("yt-dlp fetching query=%r", query)
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                raw = ydl.extract_info(f"ytsearch5:{query}", download=True)
        except yt_dlp.utils.DownloadError as e:
            raise FetchError(str(e)) from e
        except Exception as e:  # noqa: BLE001 — yt-dlp wraps many failure modes
            raise FetchError(str(e)) from e
        _log.info("yt-dlp fetch returned for query=%r", query)

        # Walk the search result entries (after match_filter pruning) and
        # take the first one whose .wav was actually produced on disk.
        info, produced_wav = self._first_downloaded(raw)
        if info is None or produced_wav is None:
            raise FetchError(
                f"no usable results for query: {query!r} "
                f"(all matches filtered out — too long, age-gated, or unavailable)"
            )
        _log.info(
            "yt-dlp picked %r (id=%s, %.1fs)",
            info.get("title", "?"), info.get("id"), float(info.get("duration") or 0.0),
        )

        # Rename to hash-keyed filename so future cache lookups by query work.
        if produced_wav != wav_path:
            produced_wav.replace(wav_path)

        title = info.get("title") or ""
        duration_s = float(info.get("duration") or 0.0)
        url = info.get("webpage_url") or info.get("original_url") or ""

        info_path.write_text(
            json.dumps(
                {
                    "title": title,
                    "duration_s": duration_s,
                    "url": url,
                    "query": query,
                }
            )
        )

        return FetchResult(
            title=title,
            duration_s=duration_s,
            url=url,
            path=wav_path,
            query=query,
        )

    # ----- internals -----------------------------------------------------

    def _evict_old_entries(self) -> None:
        """Drop oldest .wav files (and sidecars) beyond ``max_cached_songs``.

        Also sweeps stray intermediate files yt-dlp may leave behind (.webm,
        .m4a, .part, etc.) that aren't named like our hash-based cache keys.
        """
        wavs = sorted(
            self.cache_dir.glob("*.wav"),
            key=lambda p: p.stat().st_mtime,
        )
        excess = max(0, len(wavs) - self.max_cached_songs)
        for wav in wavs[:excess]:
            stem = wav.stem
            for ext in (".wav", ".info.json", ".beats.npy", ".tempo.txt"):
                sidecar = self.cache_dir / f"{stem}{ext}"
                sidecar.unlink(missing_ok=True)

        # Sweep yt-dlp intermediate / partial files. Our cache keys are
        # 16-char lowercase hex; anything not matching is fair game.
        for entry in self.cache_dir.iterdir():
            if not entry.is_file():
                continue
            stem = entry.stem
            ext = entry.suffix
            looks_like_cache_key = (
                len(stem) == 16 and all(c in "0123456789abcdef" for c in stem)
            )
            if looks_like_cache_key and ext in (
                ".wav", ".info.json", ".beats.npy", ".tempo.txt"
            ):
                continue
            entry.unlink(missing_ok=True)

    @staticmethod
    def _load_cached(wav_path: Path, info_path: Path) -> FetchResult:
        data = json.loads(info_path.read_text())
        return FetchResult(
            title=data.get("title", ""),
            duration_s=float(data.get("duration_s", 0.0)),
            url=data.get("url", ""),
            path=wav_path,
            query=data.get("query", ""),
        )

    @staticmethod
    def _unwrap_search_result(raw: dict | None) -> dict | None:
        """``ytsearch1:`` returns a playlist-like dict with an 'entries' list."""
        if not raw:
            return None
        if "entries" in raw:
            entries = raw.get("entries") or []
            return entries[0] if entries else None
        return raw

    def _first_downloaded(
        self, raw: dict | None
    ) -> tuple[dict | None, Path | None]:
        """Return the first search-result entry whose <id>.wav exists on disk.

        ``ytsearch5`` paired with our duration ``match_filter`` may produce
        a list of entries where some have been skipped (no postprocess ran)
        and one was actually downloaded. We pick the first one that resulted
        in a wav file we can use.
        """
        if not raw:
            return None, None
        entries = raw.get("entries") if "entries" in raw else [raw]
        if not entries:
            return None, None
        for info in entries:
            if not isinstance(info, dict):
                continue
            video_id = info.get("id")
            if not video_id:
                continue
            wav = self.cache_dir / f"{video_id}.wav"
            if wav.exists():
                return info, wav
        return None, None
