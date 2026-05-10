"""Live YouTube fetch test — slow + requires network. Skipped by default.

Run with: pytest -m slow tests/integration/test_youtube_real.py
"""
from __future__ import annotations

from pathlib import Path

import pytest

from reachy_mini_dance_party_app.music.youtube import YouTubeFetcher


@pytest.mark.slow
@pytest.mark.requires_network
def test_real_youtube_fetch_short_creative_commons(tmp_path):
    """Fetch a known short Creative Commons track, verify wav appears and is >100KB."""
    query = "kevin macleod monkeys spinning monkeys"
    fetcher = YouTubeFetcher(cache_dir=tmp_path)

    result = fetcher.fetch(query)

    assert result.path.exists(), f"expected wav at {result.path}"
    size = result.path.stat().st_size
    assert size > 100_000, f"wav too small: {size} bytes"
    assert result.title
    assert result.duration_s > 0
    assert result.query == query
