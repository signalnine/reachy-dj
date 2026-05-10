"""``look_at_audience`` — summarize who's in front of the camera right now.

Runs MediaPipe FaceDetection on the latest frame to count faces and identify a
"dominant centered" face (largest bbox near image center). The smile count is
estimated via FaceLandmarker if available; otherwise reported as 0.
"""

from __future__ import annotations

from reachy_mini_dance_party_app.tools.registry import AppContext, Tool, ToolError


SCHEMA = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
}


def _count_smiles(frame) -> int:
    """Best-effort smile count using MediaPipe FaceLandmarker.

    Returns 0 if the landmarker isn't available or detects no faces.
    """
    try:
        import mediapipe as mp  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return 0

    try:
        mesh = mp.solutions.face_mesh.FaceMesh(  # type: ignore[attr-defined]
            static_image_mode=True,
            max_num_faces=4,
            refine_landmarks=False,
            min_detection_confidence=0.5,
        )
    except Exception:  # noqa: BLE001
        return 0

    try:
        results = mesh.process(frame)
        landmarks = getattr(results, "multi_face_landmarks", None) or []
        smiles = 0
        for face in landmarks:
            # Mouth corner (61) vs upper lip top (13). If corners sit visibly
            # higher than the lip top center, treat as a smile. Coarse but cheap.
            try:
                left = face.landmark[61]
                right = face.landmark[291]
                top = face.landmark[13]
                if (left.y < top.y) and (right.y < top.y):
                    smiles += 1
            except Exception:  # noqa: BLE001
                continue
        return smiles
    finally:
        close = getattr(mesh, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001
                pass


def make(ctx: AppContext) -> Tool:
    def handler(args: dict) -> dict:
        frame = ctx.camera_worker.get_latest_frame()
        if frame is None:
            raise ToolError("no camera frame available yet")

        # Local import to avoid penalizing import time of the package.
        try:
            import mediapipe as mp  # noqa: PLC0415
        except Exception as exc:  # noqa: BLE001
            raise ToolError(f"mediapipe unavailable: {exc}") from exc

        detector = mp.solutions.face_detection.FaceDetection(  # type: ignore[attr-defined]
            model_selection=0, min_detection_confidence=0.5
        )
        try:
            results = detector.process(frame)
        finally:
            close = getattr(detector, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001
                    pass

        detections = getattr(results, "detections", None) or []
        n_faces = len(detections)

        h, w = int(frame.shape[0]), int(frame.shape[1])
        dominant_centered = False
        if detections:
            best_area = -1.0
            best_cx = 0.0
            for det in detections:
                bbox = det.location_data.relative_bounding_box
                bw = max(0.0, bbox.width) * w
                bh = max(0.0, bbox.height) * h
                area = bw * bh
                if area > best_area:
                    best_area = area
                    best_cx = (bbox.xmin + bbox.width / 2.0) * w
            # "Centered" = within central third horizontally.
            dominant_centered = bool(w / 3.0 <= best_cx <= 2.0 * w / 3.0)

        smiles = _count_smiles(frame) if detections else 0

        return {
            "n_faces": int(n_faces),
            "dominant_centered": bool(dominant_centered),
            "smiles": int(smiles),
        }

    return Tool(
        name="look_at_audience",
        description=(
            "Summarize the audience: face count, whether the largest face is "
            "centered, and a rough smile count. Use sparingly to acknowledge "
            "audience changes naturally."
        ),
        parameters=SCHEMA,
        handler=handler,
    )
