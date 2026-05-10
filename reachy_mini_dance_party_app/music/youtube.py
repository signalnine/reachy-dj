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
from dataclasses import dataclass
from pathlib import Path

import yt_dlp


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
    """Blocking yt-dlp wrapper that caches audio + metadata by query hash."""

    def __init__(self, cache_dir: Path = Path("/tmp/reachy_dance_party")) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

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
        }

        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                raw = ydl.extract_info(f"ytsearch1:{query}", download=True)
        except yt_dlp.utils.DownloadError as e:
            raise FetchError(str(e)) from e
        except Exception as e:  # noqa: BLE001 — yt-dlp wraps many failure modes
            raise FetchError(str(e)) from e

        info = self._unwrap_search_result(raw)
        if info is None:
            raise FetchError(f"no results for query: {query!r}")

        video_id = info.get("id")
        if not video_id:
            raise FetchError("yt-dlp returned info without an 'id' field")

        produced_wav = self.cache_dir / f"{video_id}.wav"
        if not produced_wav.exists():
            raise FetchError(
                f"expected wav at {produced_wav} after FFmpegExtractAudio, not found"
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
