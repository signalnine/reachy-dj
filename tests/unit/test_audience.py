"""Unit tests for :mod:`reachy_mini_dance_party_app.vision.audience`.

These tests drive ``AudiencePush._poll_once`` deterministically by injecting:

* a fake ``get_latest_frame`` callable returning canned numpy frames (or None),
* a fake monotonic clock,
* a ``FakeFaceDetection`` whose ``process()`` returns scripted bbox lists,
* a ``FakeFaceLandmarker`` whose ``process()`` returns scripted smile booleans,
* a recording ``push`` callable.

We never spawn the real background thread; we only call ``_poll_once`` directly
so the tests are pure-logic and free of timing flakiness.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from reachy_mini_dance_party_app.vision.audience import (
    AudiencePush,
    AudienceSummary,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeBBox:
    """Stand-in for ``mediapipe`` relative bounding box."""

    def __init__(self, xmin: float, ymin: float, width: float, height: float) -> None:
        self.xmin = xmin
        self.ymin = ymin
        self.width = width
        self.height = height


class _FakeLocationData:
    def __init__(self, bbox: _FakeBBox) -> None:
        self.relative_bounding_box = bbox


class _FakeDetection:
    def __init__(self, bbox: _FakeBBox) -> None:
        self.location_data = _FakeLocationData(bbox)


class _FakeResults:
    def __init__(self, detections: list[_FakeDetection]) -> None:
        self.detections = detections


class FakeFaceDetection:
    """Scripted FaceDetection: hands back a series of ``[bbox, ...]`` lists.

    Each entry in ``script`` is a list of ``(xmin, ymin, width, height)`` tuples
    in normalised image coordinates (0..1). ``process()`` returns one entry per
    call, looping back to the last entry when the script runs out.
    """

    def __init__(self, script: list[list[tuple[float, float, float, float]]]) -> None:
        self._script = script
        self.calls = 0

    def process(self, frame: np.ndarray) -> _FakeResults:
        idx = min(self.calls, len(self._script) - 1)
        self.calls += 1
        bboxes = [_FakeBBox(*b) for b in self._script[idx]]
        return _FakeResults([_FakeDetection(b) for b in bboxes])

    def close(self) -> None:  # mediapipe parity
        pass


class FakeFaceLandmarker:
    """Scripted FaceLandmarker that returns a per-call smile-count integer."""

    def __init__(self, smile_script: list[int] | None = None) -> None:
        self._smile_script = smile_script or [0]
        self.calls = 0

    def detect_smiles(self, frame: np.ndarray, n_faces: int) -> int:
        idx = min(self.calls, len(self._smile_script) - 1)
        self.calls += 1
        # Cap at the number of detected faces so the test wiring stays sane.
        return min(self._smile_script[idx], n_faces)

    def close(self) -> None:
        pass


class FakeClock:
    """Manually advanced monotonic clock."""

    def __init__(self, t0: float = 0.0) -> None:
        self.t = float(t0)

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += float(dt)


def _frame() -> np.ndarray:
    """Plain RGB frame; the fakes ignore its contents."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


def _make(
    *,
    frames: list[np.ndarray | None],
    detector: FakeFaceDetection,
    landmarker: FakeFaceLandmarker | None = None,
    clock: FakeClock,
    cadence_s: float = 8.0,
) -> tuple[AudiencePush, list[str]]:
    pushed: list[str] = []
    frame_iter = iter(frames)

    def get_latest() -> np.ndarray | None:
        try:
            return next(frame_iter)
        except StopIteration:
            return frames[-1] if frames else None

    landmarker = landmarker if landmarker is not None else FakeFaceLandmarker([0])

    push = AudiencePush(
        get_latest_frame=get_latest,
        push=pushed.append,
        cadence_s=cadence_s,
        clock=clock,
        face_detector_factory=lambda: detector,
        face_landmarker_factory=lambda: landmarker,
    )
    return push, pushed


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_no_frame_returns_none() -> None:
    """When the camera hands us no frame yet, _poll_once is a no-op."""
    detector = FakeFaceDetection([[]])
    clock = FakeClock()
    push, pushed = _make(frames=[None], detector=detector, clock=clock)

    out = push._poll_once()

    assert out is None
    assert pushed == []
    assert detector.calls == 0  # no frame -> no detector work either


def test_periodic_push_at_cadence() -> None:
    """With a steady face count, a push fires only after cadence_s elapses."""
    one_face = [(0.4, 0.4, 0.2, 0.2)]
    detector = FakeFaceDetection([one_face, one_face, one_face])
    clock = FakeClock(t0=100.0)
    push, pushed = _make(
        frames=[_frame(), _frame(), _frame()],
        detector=detector,
        clock=clock,
        cadence_s=8.0,
    )

    # First poll: face count goes 0 -> 1, edge fires immediately.
    first = push._poll_once()
    assert first is not None
    assert len(pushed) == 1

    # Second poll within cadence: same count, no push.
    clock.advance(2.0)
    assert push._poll_once() is None
    assert len(pushed) == 1

    # Third poll past cadence: periodic push fires.
    clock.advance(7.0)  # total 9.0s elapsed since first push
    third = push._poll_once()
    assert third is not None
    assert len(pushed) == 2
    payload = json.loads(pushed[-1])
    assert payload["n_faces"] == 1


def test_face_count_change_triggers_immediate_push() -> None:
    """Two polls 0.5s apart with different face counts both push."""
    one_face = [(0.4, 0.4, 0.2, 0.2)]
    two_faces = [(0.1, 0.4, 0.15, 0.15), (0.7, 0.4, 0.2, 0.25)]
    detector = FakeFaceDetection([one_face, two_faces])
    clock = FakeClock(t0=10.0)
    push, pushed = _make(
        frames=[_frame(), _frame()],
        detector=detector,
        clock=clock,
        cadence_s=8.0,
    )

    push._poll_once()  # 0 -> 1: edge push
    assert len(pushed) == 1

    clock.advance(0.5)
    push._poll_once()  # 1 -> 2: another edge push, well within cadence
    assert len(pushed) == 2

    second = json.loads(pushed[-1])
    assert second["n_faces"] == 2
    # since_last reflects the +1 delta from the previous push.
    assert "+1" in second["since_last"]


def test_summary_includes_required_keys() -> None:
    """Pushed JSON must carry the four documented fields."""
    detector = FakeFaceDetection([[(0.4, 0.4, 0.2, 0.2)]])
    landmarker = FakeFaceLandmarker([1])
    clock = FakeClock()
    push, pushed = _make(
        frames=[_frame()],
        detector=detector,
        landmarker=landmarker,
        clock=clock,
    )

    push._poll_once()

    assert len(pushed) == 1
    payload = json.loads(pushed[0])
    assert set(payload.keys()) >= {"n_faces", "dominant_centered", "smiles", "since_last"}
    assert payload["n_faces"] == 1
    assert payload["dominant_centered"] is True
    assert payload["smiles"] == 1
    # First push has no prior baseline, so since_last is "no change".
    assert payload["since_last"] == "no change"


def test_landmarker_only_runs_on_push() -> None:
    """Landmarker is invoked at most once per push, never on a no-op poll."""
    one_face = [(0.4, 0.4, 0.2, 0.2)]
    detector = FakeFaceDetection([one_face, one_face, one_face, one_face])
    landmarker = FakeFaceLandmarker([0, 0, 0, 0])
    clock = FakeClock(t0=0.0)
    push, pushed = _make(
        frames=[_frame(), _frame(), _frame(), _frame()],
        detector=detector,
        landmarker=landmarker,
        clock=clock,
        cadence_s=8.0,
    )

    push._poll_once()  # edge push (0 -> 1)
    assert landmarker.calls == 1
    assert len(pushed) == 1

    clock.advance(0.5)
    push._poll_once()  # no push: same count, within cadence
    assert landmarker.calls == 1  # not invoked
    assert len(pushed) == 1

    clock.advance(0.5)
    push._poll_once()  # still no push
    assert landmarker.calls == 1
    assert len(pushed) == 1

    clock.advance(8.0)  # past cadence -> periodic push
    push._poll_once()
    assert landmarker.calls == 2
    assert len(pushed) == 2


def test_dominant_centered_false_when_face_off_axis() -> None:
    """A face hugging the edge of frame is not considered centered."""
    edge_face = [(0.0, 0.0, 0.15, 0.15)]  # top-left corner
    detector = FakeFaceDetection([edge_face])
    clock = FakeClock()
    push, pushed = _make(frames=[_frame()], detector=detector, clock=clock)

    summary = push._poll_once()

    assert summary is not None
    assert summary.dominant_centered is False
    payload = json.loads(pushed[0])
    assert payload["dominant_centered"] is False


def test_no_face_yields_summary_with_zero_count() -> None:
    """An empty detection list still produces a summary on the first poll."""
    detector = FakeFaceDetection([[]])
    clock = FakeClock()
    push, pushed = _make(frames=[_frame()], detector=detector, clock=clock)

    summary = push._poll_once()

    # First poll: face count 0 with prior None counts as no edge change, but
    # we still emit the very first summary so the LLM has a baseline.
    assert summary is not None
    assert summary.n_faces == 0
    assert summary.dominant_centered is False
    assert summary.smiles == 0
    assert summary.since_last == "no change"


def test_audience_summary_dataclass_fields() -> None:
    """Light sanity check that the dataclass exposes the documented fields."""
    s = AudienceSummary(
        n_faces=2,
        dominant_centered=True,
        smiles=1,
        since_last="+1 face",
    )
    assert s.n_faces == 2
    assert s.dominant_centered is True
    assert s.smiles == 1
    assert s.since_last == "+1 face"
