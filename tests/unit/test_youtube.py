"""Unit tests for the yt-dlp wrapper (mocked subprocess; no network)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from reachy_mini_dance_party_app.music.youtube import (
    FetchError,
    FetchResult,
    YouTubeFetcher,
)


def _hash(query: str) -> str:
    return hashlib.sha256(query.encode()).hexdigest()[:16]


def test_cache_hit_returns_existing_without_calling_ytdlp(tmp_path, mocker):
    """If <hash>.wav + <hash>.info.json already exist, fetch returns them and never invokes yt_dlp."""
    query = "daft punk one more time"
    h = _hash(query)
    wav = tmp_path / f"{h}.wav"
    info = tmp_path / f"{h}.info.json"
    wav.write_bytes(b"fake-wav-bytes")
    info.write_text(
        json.dumps(
            {
                "title": "One More Time",
                "duration_s": 320.0,
                "url": "https://youtube.com/watch?v=cached",
                "query": query,
            }
        )
    )

    ytdl_ctor = mocker.patch("yt_dlp.YoutubeDL")

    fetcher = YouTubeFetcher(cache_dir=tmp_path)
    result = fetcher.fetch(query)

    assert ytdl_ctor.call_count == 0
    assert isinstance(result, FetchResult)
    assert result.path == wav
    assert result.title == "One More Time"
    assert result.duration_s == 320.0
    assert result.url == "https://youtube.com/watch?v=cached"
    assert result.query == query


def test_first_fetch_calls_ytdlp_with_correct_opts(tmp_path, mocker):
    """First fetch invokes YoutubeDL with format='bestaudio/best' + FFmpegExtractAudio postprocessor."""
    query = "kevin macleod monkeys spinning monkeys"
    h = _hash(query)

    fake_info = {
        "id": "abc123",
        "title": "Monkeys Spinning Monkeys",
        "duration": 122.5,
        "webpage_url": "https://youtube.com/watch?v=abc123",
    }

    # Simulate the postprocessor producing the .wav at <id>.wav in cache_dir.
    def fake_extract_info(search, download=True):
        (tmp_path / f"{fake_info['id']}.wav").write_bytes(b"x" * 1024)
        return {"entries": [fake_info]}

    fake_ytdl = mocker.MagicMock()
    fake_ytdl.__enter__ = mocker.MagicMock(return_value=fake_ytdl)
    fake_ytdl.__exit__ = mocker.MagicMock(return_value=False)
    fake_ytdl.extract_info = mocker.MagicMock(side_effect=fake_extract_info)
    ytdl_ctor = mocker.patch("yt_dlp.YoutubeDL", return_value=fake_ytdl)

    fetcher = YouTubeFetcher(cache_dir=tmp_path)
    result = fetcher.fetch(query)

    assert ytdl_ctor.call_count == 1
    opts = ytdl_ctor.call_args.args[0] if ytdl_ctor.call_args.args else ytdl_ctor.call_args.kwargs.get("params")
    assert opts is not None, "YoutubeDL must be constructed with an opts dict"
    assert opts["format"] == "bestaudio/best"

    pps = opts["postprocessors"]
    assert any(
        pp.get("key") == "FFmpegExtractAudio" and pp.get("preferredcodec") == "wav"
        for pp in pps
    ), f"FFmpegExtractAudio postprocessor with wav not found: {pps}"

    # Search prefix must be ytsearch1:
    fake_ytdl.extract_info.assert_called_once()
    search_arg = fake_ytdl.extract_info.call_args.args[0]
    assert search_arg == f"ytsearch1:{query}"

    # Final wav lives at <hash>.wav
    assert result.path == tmp_path / f"{h}.wav"
    assert result.path.exists()


def test_first_fetch_writes_info_json(tmp_path, mocker):
    """info.json sibling appears after fetch with title/duration/url/query keys."""
    query = "some artist some song"
    h = _hash(query)

    fake_info = {
        "id": "xyz789",
        "title": "Some Song",
        "duration": 215.0,
        "webpage_url": "https://youtube.com/watch?v=xyz789",
    }

    def fake_extract_info(search, download=True):
        (tmp_path / f"{fake_info['id']}.wav").write_bytes(b"y" * 2048)
        return {"entries": [fake_info]}

    fake_ytdl = mocker.MagicMock()
    fake_ytdl.__enter__ = mocker.MagicMock(return_value=fake_ytdl)
    fake_ytdl.__exit__ = mocker.MagicMock(return_value=False)
    fake_ytdl.extract_info = mocker.MagicMock(side_effect=fake_extract_info)
    mocker.patch("yt_dlp.YoutubeDL", return_value=fake_ytdl)

    fetcher = YouTubeFetcher(cache_dir=tmp_path)
    fetcher.fetch(query)

    info_path = tmp_path / f"{h}.info.json"
    assert info_path.exists()
    data = json.loads(info_path.read_text())
    assert set(["title", "duration_s", "url", "query"]).issubset(data.keys())
    assert data["title"] == "Some Song"
    assert data["duration_s"] == 215.0
    assert data["url"] == "https://youtube.com/watch?v=xyz789"
    assert data["query"] == query


def test_ytdlp_error_raises_fetch_error(tmp_path, mocker):
    """When yt_dlp.utils.DownloadError fires, FetchError wraps the message."""
    import yt_dlp

    msg = "ERROR: Video unavailable"

    fake_ytdl = mocker.MagicMock()
    fake_ytdl.__enter__ = mocker.MagicMock(return_value=fake_ytdl)
    fake_ytdl.__exit__ = mocker.MagicMock(return_value=False)
    fake_ytdl.extract_info = mocker.MagicMock(
        side_effect=yt_dlp.utils.DownloadError(msg)
    )
    mocker.patch("yt_dlp.YoutubeDL", return_value=fake_ytdl)

    fetcher = YouTubeFetcher(cache_dir=tmp_path)
    with pytest.raises(FetchError) as exc_info:
        fetcher.fetch("nonexistent video")

    assert msg in str(exc_info.value)


def test_cache_dir_created_if_missing(tmp_path, mocker):
    """If cache_dir doesn't exist, YouTubeFetcher (or fetch) creates it."""
    cache_dir = tmp_path / "does_not_exist_yet" / "nested"
    assert not cache_dir.exists()

    fake_info = {
        "id": "newid",
        "title": "T",
        "duration": 10.0,
        "webpage_url": "https://example.com/v",
    }

    def fake_extract_info(search, download=True):
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / f"{fake_info['id']}.wav").write_bytes(b"z" * 256)
        return {"entries": [fake_info]}

    fake_ytdl = mocker.MagicMock()
    fake_ytdl.__enter__ = mocker.MagicMock(return_value=fake_ytdl)
    fake_ytdl.__exit__ = mocker.MagicMock(return_value=False)
    fake_ytdl.extract_info = mocker.MagicMock(side_effect=fake_extract_info)
    mocker.patch("yt_dlp.YoutubeDL", return_value=fake_ytdl)

    fetcher = YouTubeFetcher(cache_dir=cache_dir)
    fetcher.fetch("anything")

    assert cache_dir.exists() and cache_dir.is_dir()
