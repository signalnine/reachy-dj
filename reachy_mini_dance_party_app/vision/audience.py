"""Audience summary push for the Realtime LLM session.

This module spawns a small background thread that polls the camera worker at
~5 Hz, summarises what the robot can see of the audience, and pushes a JSON
status blob into the LLM session. The push fires on two triggers:

* **Cadence** — every ``cadence_s`` seconds (default 8s) so the LLM keeps a
  fresh mental model of the room without spamming it.
* **Edge** — immediately whenever the detected face count changes, so the LLM
  can react ("oh, hey, two of you now!") without waiting for the next tick.

To keep CPU low on the Pi, the (heavier) FaceLandmarker / FaceMesh smile
detector is **only invoked on poll cycles that result in a push**. The
lightweight FaceDetection runs every poll.

Dependency injection
--------------------
The detector and landmarker are provided by factory callables so the unit
tests can swap in scripted fakes without touching mediapipe at import time.
The default factories build:

* ``mediapipe.solutions.face_detection.FaceDetection`` (model_selection=0,
  short-range, suited for ~2m distances typical of a room),
* ``mediapipe.solutions.face_mesh.FaceMesh`` (max_num_faces=4,
  refine_landmarks=False) — we use mouth-aspect-ratio on the returned
  landmarks for smile detection. We deliberately avoid the new Tasks API
  ``FaceLandmarker`` here so we don't need to ship a ``.task`` model file
  alongside the package.

Smile heuristic (MAR)
---------------------
For each detected face we sample four FaceMesh landmarks::

    left_corner   = landmark[61]
    right_corner  = landmark[291]
    upper_lip_mid = landmark[13]
    lower_lip_mid = landmark[14]

and compute ``MAR = ||left-right|| / max(eps, ||upper-lower||)``. Smiles
correspond to a stretched mouth: ``MAR > MAR_SMILE_THRESHOLD`` (default 4.5).
The threshold is intentionally generous; a mis-classification here is purely
cosmetic — it just nudges the LLM toward more upbeat banter.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray


logger = logging.getLogger(__name__)


# --- Tuning constants ---------------------------------------------------------

POLL_HZ = 5.0
POLL_PERIOD_S = 1.0 / POLL_HZ

# A face is "centered" if its bbox center sits within ±20% of the frame center
# along both axes — i.e. inside the central 40% × 40% box.
CENTERED_TOLERANCE = 0.20

# MediaPipe FaceMesh landmark indices for the mouth corners and lip midpoints.
LM_LEFT_MOUTH_CORNER = 61
LM_RIGHT_MOUTH_CORNER = 291
LM_UPPER_LIP_MID = 13
LM_LOWER_LIP_MID = 14

# Smile threshold on the mouth-aspect-ratio (corner-to-corner / lip-gap).
# A relaxed/closed mouth gives MAR ≈ 2.5–3.5 for most people; a wide smile
# pushes well above 4.5. See module docstring for derivation.
MAR_SMILE_THRESHOLD = 4.5


# --- Data class --------------------------------------------------------------


@dataclass
class AudienceSummary:
    """Summary of what the robot can see of the audience right now."""

    n_faces: int
    dominant_centered: bool
    smiles: int
    since_last: str  # e.g. "+1 face", "-1 face", "no change"

    def to_json(self) -> str:
        return json.dumps(asdict(self))


# --- Default mediapipe factories ---------------------------------------------


def _default_face_detector_factory() -> Any:
    """Build a MediaPipe FaceDetection (short-range model)."""
    import mediapipe as mp  # local import: heavy, optional in tests

    return mp.solutions.face_detection.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.5,
    )


def _default_face_landmarker_factory() -> Any:
    """Build a MediaPipe FaceMesh (used for smile MAR computation)."""
    import mediapipe as mp  # local import: heavy, optional in tests

    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=4,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


# --- AudiencePush ------------------------------------------------------------


class AudiencePush:
    """Periodic + edge-driven audience summary push to the Realtime session.

    * Polls the camera worker at ~5 Hz.
    * Pushes a summary every ``cadence_s`` seconds.
    * Pushes immediately when the face count changes (edge trigger).
    * Calls the FaceLandmarker (smile detection) only when emitting a push,
      keeping the steady-state cost dominated by the cheap FaceDetection.
    """

    def __init__(
        self,
        get_latest_frame: Callable[[], NDArray[np.uint8] | None],
        push: Callable[[str], None],
        cadence_s: float = 8.0,
        clock: Callable[[], float] = time.monotonic,
        face_detector_factory: Callable[[], Any] | None = None,
        face_landmarker_factory: Callable[[], Any] | None = None,
        poll_period_s: float = POLL_PERIOD_S,
        should_push: Callable[[], bool] | None = None,
    ) -> None:
        self._get_latest_frame = get_latest_frame
        self._push = push
        self._cadence_s = float(cadence_s)
        self._clock = clock
        self._poll_period_s = float(poll_period_s)
        # Optional gate: caller can suppress pushes (e.g. while a song is
        # playing) so the model doesn't get distracting audience notices that
        # tempt it to talk over the music.
        self._should_push = should_push or (lambda: True)

        self._detector_factory = face_detector_factory or _default_face_detector_factory
        self._landmarker_factory = (
            face_landmarker_factory or _default_face_landmarker_factory
        )

        # Lazily constructed so unit tests that never poll don't pay the cost,
        # and so a missing mediapipe install doesn't blow up at import time.
        self._detector: Any | None = None
        self._landmarker: Any | None = None

        # Push-state bookkeeping.
        self._last_push_time: float | None = None
        self._last_pushed_n_faces: int | None = None

        # Thread plumbing.
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    # -- public API -----------------------------------------------------------

    def start(self) -> None:
        """Spawn the background polling thread; idempotent."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="AudiencePush", daemon=True
        )
        self._thread.start()
        logger.debug(
            "AudiencePush started (poll=%.3fs, cadence=%.1fs)",
            self._poll_period_s,
            self._cadence_s,
        )

    def stop(self) -> None:
        """Stop the background thread and release the mediapipe resources."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        for obj in (self._detector, self._landmarker):
            if obj is None:
                continue
            close = getattr(obj, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001
                    pass
        self._detector = None
        self._landmarker = None
        logger.debug("AudiencePush stopped")

    # -- thread loop ----------------------------------------------------------

    def _run(self) -> None:
        while not self._stop_event.is_set():
            tick_start = self._clock()
            try:
                self._poll_once()
            except Exception as exc:  # noqa: BLE001
                logger.warning("AudiencePush tick error: %s", exc)
            elapsed = self._clock() - tick_start
            sleep_for = self._poll_period_s - elapsed
            if sleep_for > 0:
                self._stop_event.wait(sleep_for)

    # -- core logic -----------------------------------------------------------

    def _poll_once(self) -> AudienceSummary | None:
        """Run one poll cycle. Returns the summary if a push was emitted.

        Steps:
        1. Grab the latest frame; bail if None.
        2. Run the lightweight FaceDetection.
        3. Decide whether to push (edge on count delta, else periodic on cadence).
        4. If pushing, run the FaceLandmarker for smile counting and emit JSON.
        """
        frame = self._get_latest_frame()
        if frame is None:
            return None

        # Lazy detector init — kept inside _poll_once so tests that never poll
        # never construct a detector (and so a missing mediapipe install only
        # explodes once we actually try to detect).
        if self._detector is None:
            self._detector = self._detector_factory()

        bboxes = self._detect_faces(frame)
        n_faces = len(bboxes)
        now = self._clock()

        # Decide: edge (count changed) or cadence (interval elapsed).
        edge_triggered = (
            self._last_pushed_n_faces is not None
            and n_faces != self._last_pushed_n_faces
        )
        cadence_triggered = (
            self._last_push_time is None
            or (now - self._last_push_time) >= self._cadence_s
        )
        if not (edge_triggered or cadence_triggered):
            return None

        # Compute "since_last" relative to the previous *pushed* count.
        if self._last_pushed_n_faces is None:
            since_last = "no change"
        else:
            delta = n_faces - self._last_pushed_n_faces
            if delta == 0:
                since_last = "no change"
            else:
                noun = "face" if abs(delta) == 1 else "faces"
                since_last = f"{'+' if delta > 0 else ''}{delta} {noun}"

        dominant_centered = self._dominant_is_centered(bboxes, frame.shape)

        # Smiles: only run the heavier landmarker when actually pushing.
        smiles = self._count_smiles(frame, n_faces)

        summary = AudienceSummary(
            n_faces=n_faces,
            dominant_centered=dominant_centered,
            smiles=smiles,
            since_last=since_last,
        )

        try:
            should = self._should_push()
        except Exception:  # noqa: BLE001
            should = True
        if should:
            try:
                self._push(summary.to_json())
            except Exception as exc:  # noqa: BLE001
                logger.warning("AudiencePush push callable raised: %s", exc)

        self._last_push_time = now
        self._last_pushed_n_faces = n_faces
        return summary

    # -- detection helpers ----------------------------------------------------

    def _detect_faces(self, frame: NDArray[np.uint8]) -> list[tuple[float, float, float, float]]:
        """Return a list of normalised (xmin, ymin, w, h) tuples."""
        if self._detector is None:
            return []

        results = self._detector.process(frame)
        detections = getattr(results, "detections", None) or []
        out: list[tuple[float, float, float, float]] = []
        for det in detections:
            bbox = det.location_data.relative_bounding_box
            xmin = float(getattr(bbox, "xmin", 0.0))
            ymin = float(getattr(bbox, "ymin", 0.0))
            width = float(getattr(bbox, "width", 0.0))
            height = float(getattr(bbox, "height", 0.0))
            if width <= 0.0 or height <= 0.0:
                continue
            out.append((xmin, ymin, width, height))
        return out

    def _dominant_is_centered(
        self,
        bboxes: list[tuple[float, float, float, float]],
        frame_shape: tuple[int, ...],
    ) -> bool:
        if not bboxes:
            return False
        # Largest bbox by normalised area.
        xmin, ymin, w, h = max(bboxes, key=lambda b: b[2] * b[3])
        cx = xmin + w / 2.0
        cy = ymin + h / 2.0
        # Centered means the bbox center sits inside the central 40% × 40%
        # box of the frame.
        return (
            abs(cx - 0.5) <= CENTERED_TOLERANCE
            and abs(cy - 0.5) <= CENTERED_TOLERANCE
        )

    def _count_smiles(self, frame: NDArray[np.uint8], n_faces: int) -> int:
        if n_faces == 0:
            return 0
        if self._landmarker is None:
            self._landmarker = self._landmarker_factory()
        if self._landmarker is None:
            return 0

        # The fakes used in tests expose a convenience ``detect_smiles``
        # method. The real mediapipe FaceMesh exposes ``process`` returning
        # ``multi_face_landmarks``. Try the test-friendly path first so the
        # unit tests stay legible.
        detect_smiles = getattr(self._landmarker, "detect_smiles", None)
        if callable(detect_smiles):
            try:
                return int(detect_smiles(frame, n_faces))
            except Exception as exc:  # noqa: BLE001
                logger.warning("custom detect_smiles raised: %s", exc)
                return 0

        try:
            results = self._landmarker.process(frame)
        except Exception as exc:  # noqa: BLE001
            logger.warning("FaceMesh.process raised: %s", exc)
            return 0

        face_landmarks_list = getattr(results, "multi_face_landmarks", None) or []
        smiles = 0
        for face_landmarks in face_landmarks_list:
            lms = getattr(face_landmarks, "landmark", None)
            if lms is None:
                continue
            try:
                left = lms[LM_LEFT_MOUTH_CORNER]
                right = lms[LM_RIGHT_MOUTH_CORNER]
                upper = lms[LM_UPPER_LIP_MID]
                lower = lms[LM_LOWER_LIP_MID]
            except (IndexError, KeyError):
                continue
            corner_dist = math.hypot(right.x - left.x, right.y - left.y)
            lip_gap = math.hypot(lower.x - upper.x, lower.y - upper.y)
            if lip_gap < 1e-6:
                continue
            mar = corner_dist / lip_gap
            if mar > MAR_SMILE_THRESHOLD:
                smiles += 1
        return smiles
