import pytest
from reachy_mini_dance_party_app.dj import DJ, DJState, SongInfo

def test_initial_state_is_idle():
    assert DJ().state is DJState.IDLE

def test_play_song_transitions_idle_to_fetching():
    dj = DJ()
    dj.request_song("test query")
    assert dj.state is DJState.FETCHING
    assert dj.pending_query == "test query"

def test_song_fetched_transitions_fetching_to_playing():
    dj = DJ()
    dj.request_song("test")
    info = SongInfo(title="X", duration_s=120.0, url="u", path="/tmp/x.wav", query="test")
    dj.song_fetched(info)
    assert dj.state is DJState.PLAYING
    assert dj.current is info

def test_song_ends_returns_to_idle_when_auto_dj_off():
    dj = DJ(auto_dj=False)
    dj.request_song("a"); dj.song_fetched(_song("a"))
    dj.song_ended()
    assert dj.state is DJState.IDLE

def test_song_ends_triggers_fetch_when_auto_dj_on():
    dj = DJ(auto_dj=True)
    dj.request_song("a"); dj.song_fetched(_song("a"))
    dj.set_pending_auto_dj_query("next track")
    dj.song_ended()
    assert dj.state is DJState.FETCHING
    assert dj.pending_query == "next track"

def test_history_records_played_songs():
    dj = DJ()
    dj.request_song("a"); dj.song_fetched(_song("a")); dj.song_ended()
    dj.request_song("b"); dj.song_fetched(_song("b")); dj.song_ended()
    assert [s.query for s in dj.history] == ["a", "b"]

def test_stop_party_returns_to_idle_from_any_state():
    dj = DJ()
    dj.request_song("a"); dj.song_fetched(_song("a"))
    dj.stop_party()
    assert dj.state is DJState.IDLE
    assert dj.current is None

def test_fetch_failure_returns_to_idle_with_error():
    dj = DJ()
    dj.request_song("bad")
    dj.fetch_failed("yt-dlp 410")
    assert dj.state is DJState.IDLE
    assert dj.last_error == "yt-dlp 410"

def _song(q: str) -> SongInfo:
    return SongInfo(title=q.upper(), duration_s=60.0, url=f"u/{q}", path=f"/tmp/{q}.wav", query=q)
