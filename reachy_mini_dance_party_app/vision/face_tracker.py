"""Face tracker that produces a secondary-move head offset for the dance-party app.

This module is intentionally self-contained: it polls the latest frame from a
``CameraWorker`` at ~15 Hz, runs MediaPipe FaceDetection on it, picks the
dominant face (largest bbox), maps the face center pixel to a small head
yaw/pitch offset (small-angle approximation), low-pass smooths the result, and
fades the offset back to zero after ~1.5 s of no-face hysteresis.

It does *not* know about the move worker. The move worker (Task 16) calls
``get_secondary_offset()`` each tick to read the current desired offset and
composes it on top of the primary dance pose.

Design notes
------------
* The face logic in the conv app's ``CameraWorker`` runs ``look_at_image``
  to convert a face position into a full 4x4 target pose. That requires the
  ``ReachyMini`` SDK's camera intrinsics. Here we deliberately avoid that
  coupling: we use a simple linear pixel -> degrees mapping appropriate for
  a small offset that gets layered on top of choreographed dance moves.
* Mapping: ±200 px from image center maps to ±15 deg yaw/pitch (imx708 FOV
  is roughly 80 deg horizontal at the resolution the daemon hands us; the
  small-angle approximation is fine for sub-bbox-sized offsets).
* Smoothing: simple exponential filter ``offset = (1-a)*prev + a*new`` with
  ``a = 0.3``.
* Hysteresis: after 1.5 s without a face, we exponentially decay the smoothed
  offset toward zero using the same filter (target = 0). After ~3 s of no
  face the offset is essentially zero.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from typing import Optional, Protocol

import numpy as np
from numpy.typing import NDArray


logger = logging.getLogger(__name__)


# --- Tuning constants ---------------------------------------------------------

# Polling frequency for the face-tracking thread.
POLL_HZ = 15.0
POLL_PERIOD = 1.0 / POLL_HZ

# Pixel offset that maps to MAX_DEG of head deflection (small-angle approx).
PIXELS_FOR_MAX_DEFLECTION = 200.0
MAX_DEFLECTION_DEG = 15.0

# Low-pass filter coefficient. Higher = more responsive, more jittery.
SMOOTHING_ALPHA = 0.3

# Time without a face before we start fading offsets back toward zero.
LOST_FACE_DELAY_S = 1.5


class _FrameSource(Protocol):
    """Anything that can hand us a BGR/RGB frame ndarray (e.g. CameraWorker)."""

    def get_latest_frame(self) -> Optional[NDArray[np.uint8]]: ...


class FaceTracker:
    """Polls a frame source, detects faces, exposes a smoothed head offset.

    Thread-safe: ``get_secondary_offset()`` may be called from any thread; the
    background thread updates the offset under a lock.
    """

    def __init__(
        self,
        frame_source: _FrameSource,
        *,
        poll_hz: float = POLL_HZ,
        smoothing_alpha: float = SMOOTHING_ALPHA,
        lost_face_delay_s: float = LOST_FACE_DELAY_S,
        pixels_for_max_deflection: float = PIXELS_FOR_MAX_DEFLECTION,
        max_deflection_deg: float = MAX_DEFLECTION_DEG,
    ) -> None:
        self._frame_source = frame_source
        self._poll_period = 1.0 / float(poll_hz)
        self._alpha = float(smoothing_alpha)
        self._lost_delay = float(lost_face_delay_s)
        self._pixels_for_max = float(pixels_for_max_deflection)
        self._max_deg = float(max_deflection_deg)

        # Smoothed yaw/pitch offsets, in radians. Yaw = head turn left/right,
        # pitch = head tilt up/down. We deliberately do not produce roll.
        self._yaw_rad: float = 0.0
        self._pitch_rad: float = 0.0
        self._has_face: bool = False
        self._last_face_seen: Optional[float] = None

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Lazily constructed inside the worker thread so a missing mediapipe
        # install doesn't blow up at import time.
        self._detector = None

    # -- public API -----------------------------------------------------------

    def start(self) -> None:
        """Spawn the background polling thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="FaceTracker", daemon=True)
        self._thread.start()
        logger.debug("FaceTracker started (poll period %.3fs)", self._poll_period)

    def stop(self) -> None:
        """Stop the background thread and release the detector."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._detector is not None:
            close = getattr(self._detector, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001
                    pass
            self._detector = None
        logger.debug("FaceTracker stopped")

    def get_secondary_offset(self) -> dict:
        """Return the current smoothed secondary-move head offset.

        Returns a dict with keys ``yaw`` and ``pitch`` in radians plus a
        ``has_face`` bool. The move worker may use this to additively bias
        the head pose on top of the active dance move.
        """
        with self._lock:
            return {
                "yaw": self._yaw_rad,
                "pitch": self._pitch_rad,
                "has_face": self._has_face,
            }

    # -- thread loop ----------------------------------------------------------

    def _run(self) -> None:
        try:
            self._detector = self._build_detector()
        except Exception as exc:  # noqa: BLE001
            logger.error("FaceTracker: failed to initialize MediaPipe: %s", exc)
            return

        while not self._stop_event.is_set():
            tick_start = time.monotonic()
            try:
                self._tick()
            except Exception as exc:  # noqa: BLE001
                logger.warning("FaceTracker tick error: %s", exc)

            elapsed = time.monotonic() - tick_start
            sleep_for = self._poll_period - elapsed
            if sleep_for > 0:
                self._stop_event.wait(sleep_for)

    def _tick(self) -> None:
        frame = self._frame_source.get_latest_frame()
        if frame is None or frame.size == 0:
            self._maybe_decay()
            return

        face_center = self._detect_dominant_face_center(frame)
        now = time.monotonic()

        if face_center is not None:
            cx_px, cy_px = face_center
            h, w = frame.shape[:2]
            # Pixel offset from image center.
            dx = cx_px - (w / 2.0)
            dy = cy_px - (h / 2.0)

            # Map pixels -> degrees, clamped to MAX_DEFLECTION_DEG.
            yaw_deg = -self._pixels_to_deg(dx)  # face right of center -> turn right (negative yaw in our convention)
            pitch_deg = -self._pixels_to_deg(dy)  # face below center -> tilt down (negative pitch)

            target_yaw = math.radians(yaw_deg)
            target_pitch = math.radians(pitch_deg)

            with self._lock:
                self._yaw_rad = (1.0 - self._alpha) * self._yaw_rad + self._alpha * target_yaw
                self._pitch_rad = (1.0 - self._alpha) * self._pitch_rad + self._alpha * target_pitch
                self._has_face = True
                self._last_face_seen = now
        else:
            self._maybe_decay(now=now)

    def _maybe_decay(self, *, now: Optional[float] = None) -> None:
        """Fade offsets back toward zero after the lost-face hysteresis."""
        now = time.monotonic() if now is None else now
        with self._lock:
            self._has_face = False
            if self._last_face_seen is None:
                # Never saw a face yet — keep offsets at zero.
                self._yaw_rad = (1.0 - self._alpha) * self._yaw_rad
                self._pitch_rad = (1.0 - self._alpha) * self._pitch_rad
                return

            time_since = now - self._last_face_seen
            if time_since < self._lost_delay:
                # Still inside grace period — hold the last offset steady.
                return

            # Past the hysteresis window: exponentially decay toward zero.
            self._yaw_rad = (1.0 - self._alpha) * self._yaw_rad
            self._pitch_rad = (1.0 - self._alpha) * self._pitch_rad

    # -- helpers --------------------------------------------------------------

    def _pixels_to_deg(self, dx_px: float) -> float:
        """Map a pixel offset from center to degrees, clamped to ``max_deg``."""
        deg = (dx_px / self._pixels_for_max) * self._max_deg
        if deg > self._max_deg:
            return self._max_deg
        if deg < -self._max_deg:
            return -self._max_deg
        return deg

    def _detect_dominant_face_center(
        self, frame: NDArray[np.uint8]
    ) -> Optional[tuple[float, float]]:
        """Run MediaPipe FaceDetection, return the largest-bbox face center in pixels."""
        if self._detector is None:
            return None

        # MediaPipe expects RGB. The reachy-mini media pipeline hands us RGB
        # already (per the conv app's usage), but defensively normalize here.
        if frame.ndim != 3 or frame.shape[2] != 3:
            return None

        results = self._detector.process(frame)
        if not getattr(results, "detections", None):
            return None

        h, w = frame.shape[:2]
        best_area = -1.0
        best_center: Optional[tuple[float, float]] = None
        for det in results.detections:
            bbox = det.location_data.relative_bounding_box
            bw = max(0.0, bbox.width) * w
            bh = max(0.0, bbox.height) * h
            area = bw * bh
            if area <= 0.0:
                continue
            cx = (bbox.xmin + bbox.width / 2.0) * w
            cy = (bbox.ymin + bbox.height / 2.0) * h
            if area > best_area:
                best_area = area
                best_center = (cx, cy)
        return best_center

    @staticmethod
    def _build_detector():
        """Construct a MediaPipe FaceDetection detector (model 0, short range)."""
        import mediapipe as mp  # noqa: PLC0415

        return mp.solutions.face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.5,
        )
