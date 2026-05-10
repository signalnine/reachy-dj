"""Unit tests for the LLM tool registry and every individual tool.

Each tool gets the same three-case treatment:
  1. Schema is well-formed JSON Schema (Draft 2020-12).
  2. Bad / missing params raise ``ToolError``.
  3. Good params trigger the right call on the injected ``ctx`` mocks.

A few tools (``play_song``, ``take_photo``) also assert the return value
shape because the LLM relies on it.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from reachy_mini_dance_party_app.tools import (
    AppContext,
    ToolError,
    all_tools,
)
from reachy_mini_dance_party_app.tools import (
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


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_ctx(**overrides) -> AppContext:
    """Build an AppContext where every component is a MagicMock by default."""
    fields = dict(
        dj=MagicMock(),
        dancer=MagicMock(),
        mixer=MagicMock(),
        playback=MagicMock(),
        camera_worker=MagicMock(),
        face_tracker=MagicMock(),
        move_queue=MagicMock(),
        fetcher=MagicMock(),
        analyzer=MagicMock(),
        http_client=MagicMock(),
    )
    fields.update(overrides)
    return AppContext(**fields)


@pytest.fixture
def ctx() -> AppContext:
    return _make_ctx()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_all_tools_returns_eleven_tools(self, ctx: AppContext) -> None:
        tools = all_tools(ctx)
        assert len(tools) == 11

    def test_all_tools_have_unique_names(self, ctx: AppContext) -> None:
        names = [t.name for t in all_tools(ctx)]
        assert len(names) == len(set(names))

    def test_all_tool_schemas_validate_as_jsonschema(self, ctx: AppContext) -> None:
        for tool in all_tools(ctx):
            Draft202012Validator.check_schema(tool.parameters)


# ---------------------------------------------------------------------------
# play_song
# ---------------------------------------------------------------------------


class TestPlaySong:
    def test_schema_is_valid(self, ctx: AppContext) -> None:
        Draft202012Validator.check_schema(play_song.make(ctx).parameters)

    def test_missing_query_raises(self, ctx: AppContext) -> None:
        tool = play_song.make(ctx)
        with pytest.raises(ToolError):
            tool.handler({})

    def test_handler_calls_dj_and_returns_metadata(self, ctx: AppContext) -> None:
        ctx.fetcher.fetch.return_value = MagicMock(
            title="Around the World",
            duration_s=420.0,
            url="https://youtu.be/x",
            path="/tmp/x.wav",
        )
        ctx.analyzer.return_value = MagicMock(name="grid")
        result = play_song.make(ctx).handler({"query": "daft punk"})
        ctx.fetcher.fetch.assert_called_once_with("daft punk")
        ctx.playback.start.assert_called_once()
        ctx.dancer.start_with_grid.assert_called_once()
        ctx.dj.song_fetched.assert_called_once()
        assert result == {"title": "Around the World", "duration_s": 420.0}


# ---------------------------------------------------------------------------
# skip_song
# ---------------------------------------------------------------------------


class TestSkipSong:
    def test_schema_is_valid(self, ctx: AppContext) -> None:
        Draft202012Validator.check_schema(skip_song.make(ctx).parameters)

    def test_extra_param_rejected_by_schema(self, ctx: AppContext) -> None:
        # additionalProperties is False; the schema itself enforces this when
        # the realtime layer validates. We assert the schema reflects that.
        assert skip_song.make(ctx).parameters["additionalProperties"] is False

    def test_handler_stops_playback_and_advances_dj(self, ctx: AppContext) -> None:
        skip_song.make(ctx).handler({})
        ctx.playback.stop.assert_called_once()
        ctx.dancer.stop.assert_called_once()
        ctx.dj.song_ended.assert_called_once()


# ---------------------------------------------------------------------------
# stop_party
# ---------------------------------------------------------------------------


class TestStopParty:
    def test_schema_is_valid(self, ctx: AppContext) -> None:
        Draft202012Validator.check_schema(stop_party.make(ctx).parameters)

    def test_schema_takes_no_params(self, ctx: AppContext) -> None:
        assert stop_party.make(ctx).parameters["properties"] == {}

    def test_handler_calls_stop_party_on_dj(self, ctx: AppContext) -> None:
        stop_party.make(ctx).handler({})
        ctx.playback.stop.assert_called_once()
        ctx.dancer.stop.assert_called_once()
        ctx.dj.stop_party.assert_called_once()


# ---------------------------------------------------------------------------
# set_volume
# ---------------------------------------------------------------------------


class TestSetVolume:
    def test_schema_is_valid(self, ctx: AppContext) -> None:
        Draft202012Validator.check_schema(set_volume.make(ctx).parameters)

    @pytest.mark.parametrize("bad", [{}, {"level": -1}, {"level": 101}, {"level": "loud"}])
    def test_bad_params_raise(self, ctx: AppContext, bad: dict) -> None:
        with pytest.raises(ToolError):
            set_volume.make(ctx).handler(bad)

    def test_handler_posts_to_daemon(self, ctx: AppContext) -> None:
        ctx.http_client.post.return_value = MagicMock(raise_for_status=lambda: None)
        result = set_volume.make(ctx).handler({"level": 50})
        ctx.http_client.post.assert_called_once_with(
            "http://localhost:8000/api/volume/set", json={"level": 50}
        )
        assert result == {"level": 50}


# ---------------------------------------------------------------------------
# take_photo
# ---------------------------------------------------------------------------


class TestTakePhoto:
    def test_schema_is_valid(self, ctx: AppContext) -> None:
        Draft202012Validator.check_schema(take_photo.make(ctx).parameters)

    def test_no_frame_raises(self, ctx: AppContext) -> None:
        ctx.camera_worker.get_latest_frame.return_value = None
        with pytest.raises(ToolError):
            take_photo.make(ctx).handler({})

    def test_handler_returns_b64_jpeg_and_shape(self, ctx: AppContext) -> None:
        # 16x12 RGB frame; cv2.imencode handles it as BGR but pixel order
        # doesn't matter for the encoding test.
        frame = np.zeros((12, 16, 3), dtype=np.uint8)
        frame[6, 8] = (255, 128, 64)
        ctx.camera_worker.get_latest_frame.return_value = frame
        result = take_photo.make(ctx).handler({})
        assert result["shape"] == [12, 16]
        assert isinstance(result["image_b64"], str)
        assert len(result["image_b64"]) > 0


# ---------------------------------------------------------------------------
# look_at_audience
# ---------------------------------------------------------------------------


class TestLookAtAudience:
    def test_schema_is_valid(self, ctx: AppContext) -> None:
        Draft202012Validator.check_schema(look_at_audience.make(ctx).parameters)

    def test_no_frame_raises(self, ctx: AppContext) -> None:
        ctx.camera_worker.get_latest_frame.return_value = None
        with pytest.raises(ToolError):
            look_at_audience.make(ctx).handler({})

    def test_no_faces_returns_zeros(
        self, ctx: AppContext, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Stub MediaPipe at the import path used inside the handler.
        ctx.camera_worker.get_latest_frame.return_value = np.zeros(
            (64, 64, 3), dtype=np.uint8
        )

        fake_results = MagicMock(detections=None)
        fake_detector = MagicMock()
        fake_detector.process.return_value = fake_results

        fake_mp = MagicMock()
        fake_mp.solutions.face_detection.FaceDetection.return_value = fake_detector
        monkeypatch.setitem(__import__("sys").modules, "mediapipe", fake_mp)

        result = look_at_audience.make(ctx).handler({})
        assert result == {"n_faces": 0, "dominant_centered": False, "smiles": 0}


# ---------------------------------------------------------------------------
# set_face_tracking
# ---------------------------------------------------------------------------


class TestSetFaceTracking:
    def test_schema_is_valid(self, ctx: AppContext) -> None:
        Draft202012Validator.check_schema(set_face_tracking.make(ctx).parameters)

    @pytest.mark.parametrize("bad", [{}, {"enabled": "yes"}, {"enabled": 1}])
    def test_bad_params_raise(self, ctx: AppContext, bad: dict) -> None:
        with pytest.raises(ToolError):
            set_face_tracking.make(ctx).handler(bad)

    def test_handler_toggles_tracker(self, ctx: AppContext) -> None:
        set_face_tracking.make(ctx).handler({"enabled": False})
        ctx.face_tracker.set_enabled.assert_called_once_with(False)


# ---------------------------------------------------------------------------
# move_head
# ---------------------------------------------------------------------------


class TestMoveHead:
    def test_schema_is_valid(self, ctx: AppContext) -> None:
        Draft202012Validator.check_schema(move_head.make(ctx).parameters)

    @pytest.mark.parametrize(
        "bad",
        [
            {},
            {"pitch": 0, "yaw": 0, "roll": 0},  # missing duration
            {"pitch": 0, "yaw": 0, "roll": 0, "duration_s": 0.05},  # below min
            {"pitch": 0, "yaw": 0, "roll": 0, "duration_s": 10.0},  # above max
            {"pitch": "x", "yaw": 0, "roll": 0, "duration_s": 1.0},
        ],
    )
    def test_bad_params_raise(self, ctx: AppContext, bad: dict) -> None:
        with pytest.raises(ToolError):
            move_head.make(ctx).handler(bad)

    def test_handler_enqueues_primary_head_move(self, ctx: AppContext) -> None:
        move_head.make(ctx).handler(
            {"pitch": 0.1, "yaw": -0.2, "roll": 0.0, "duration_s": 1.5}
        )
        ctx.move_queue.put.assert_called_once()
        enqueued = ctx.move_queue.put.call_args.args[0]
        assert enqueued.pitch == pytest.approx(0.1)
        assert enqueued.yaw == pytest.approx(-0.2)
        assert enqueued.duration_s == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# play_emotion
# ---------------------------------------------------------------------------


class TestPlayEmotion:
    def test_schema_is_valid(self, ctx: AppContext) -> None:
        Draft202012Validator.check_schema(play_emotion.make(ctx).parameters)

    @pytest.mark.parametrize("bad", [{}, {"name": ""}, {"name": 5}])
    def test_bad_params_raise(self, ctx: AppContext, bad: dict) -> None:
        with pytest.raises(ToolError):
            play_emotion.make(ctx).handler(bad)

    def test_handler_enqueues_emotion(self, ctx: AppContext) -> None:
        play_emotion.make(ctx).handler({"name": "celebrate"})
        ctx.move_queue.put.assert_called_once()
        enqueued = ctx.move_queue.put.call_args.args[0]
        assert enqueued.name == "celebrate"


# ---------------------------------------------------------------------------
# look_at
# ---------------------------------------------------------------------------


class TestLookAt:
    def test_schema_is_valid(self, ctx: AppContext) -> None:
        Draft202012Validator.check_schema(look_at.make(ctx).parameters)

    @pytest.mark.parametrize(
        "bad",
        [
            {},
            {"x": 1.5, "y": 0, "duration_s": 1.0},
            {"x": 0, "y": -2, "duration_s": 1.0},
            {"x": 0, "y": 0, "duration_s": 0.05},
            {"x": "left", "y": 0, "duration_s": 1.0},
        ],
    )
    def test_bad_params_raise(self, ctx: AppContext, bad: dict) -> None:
        with pytest.raises(ToolError):
            look_at.make(ctx).handler(bad)

    def test_handler_calls_face_tracker(self, ctx: AppContext) -> None:
        look_at.make(ctx).handler({"x": 0.5, "y": -0.3, "duration_s": 0.8})
        ctx.face_tracker.set_secondary_offset.assert_called_once_with(
            x=0.5, y=-0.3, duration_s=0.8
        )


# ---------------------------------------------------------------------------
# stop_dance
# ---------------------------------------------------------------------------


class TestStopDance:
    def test_schema_is_valid(self, ctx: AppContext) -> None:
        Draft202012Validator.check_schema(stop_dance.make(ctx).parameters)

    def test_schema_takes_no_params(self, ctx: AppContext) -> None:
        assert stop_dance.make(ctx).parameters["properties"] == {}

    def test_handler_stops_dancer_only(self, ctx: AppContext) -> None:
        stop_dance.make(ctx).handler({})
        ctx.dancer.stop.assert_called_once()
        ctx.playback.stop.assert_not_called()
        ctx.dj.stop_party.assert_not_called()
