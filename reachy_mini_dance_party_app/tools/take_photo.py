"""``take_photo`` — grab the latest camera frame and return it as base64 JPEG."""

from __future__ import annotations

import base64

from reachy_mini_dance_party_app.tools.registry import AppContext, Tool, ToolError


SCHEMA = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
}


def make(ctx: AppContext) -> Tool:
    def handler(args: dict) -> dict:
        frame = ctx.camera_worker.get_latest_frame()
        if frame is None:
            raise ToolError("no camera frame available yet")

        # Local import: cv2 is a heavy import we don't want at package load.
        import cv2  # noqa: PLC0415

        ok, jpeg = cv2.imencode(".jpg", frame)
        if not ok:
            raise ToolError("cv2.imencode returned False")

        h, w = int(frame.shape[0]), int(frame.shape[1])
        return {
            "image_b64": base64.b64encode(bytes(jpeg)).decode("ascii"),
            "shape": [h, w],
        }

    return Tool(
        name="take_photo",
        description=(
            "Capture a still image from the robot's camera. Returns a base64-"
            "encoded JPEG. Use sparingly; one frame is usually enough."
        ),
        parameters=SCHEMA,
        handler=handler,
    )
