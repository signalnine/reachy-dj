"""``set_face_tracking`` — turn the secondary face-tracking offset on or off."""

from __future__ import annotations

from reachy_mini_dance_party_app.tools.registry import AppContext, Tool, ToolError


SCHEMA = {
    "type": "object",
    "properties": {
        "enabled": {
            "type": "boolean",
            "description": "True to enable head tracking on detected faces, False to disable.",
        }
    },
    "required": ["enabled"],
    "additionalProperties": False,
}


def make(ctx: AppContext) -> Tool:
    def handler(args: dict) -> dict:
        enabled = args.get("enabled")
        if not isinstance(enabled, bool):
            raise ToolError("enabled must be a boolean")
        ctx.face_tracker.set_enabled(enabled)
        return {"enabled": enabled}

    return Tool(
        name="set_face_tracking",
        description="Enable or disable secondary head-tracking of audience faces.",
        parameters=SCHEMA,
        handler=handler,
    )
