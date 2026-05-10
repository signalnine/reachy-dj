"""``look_at`` — secondary head offset toward a normalized image-space target.

Unlike :mod:`move_head` (a primary move that preempts the dance), this is a
secondary offset: it blends with the current dance pose. For V1 we forward to
the face tracker's secondary-offset slot via ``set_secondary_offset(...)``.
"""

from __future__ import annotations

from reachy_mini_dance_party_app.tools.registry import AppContext, Tool, ToolError


SCHEMA = {
    "type": "object",
    "properties": {
        "x": {
            "type": "number",
            "minimum": -1.0,
            "maximum": 1.0,
            "description": "Normalized horizontal target in [-1, 1]; -1 = left, 1 = right.",
        },
        "y": {
            "type": "number",
            "minimum": -1.0,
            "maximum": 1.0,
            "description": "Normalized vertical target in [-1, 1]; -1 = down, 1 = up.",
        },
        "duration_s": {
            "type": "number",
            "minimum": 0.1,
            "maximum": 5.0,
            "description": "How long to hold the offset before relaxing.",
        },
    },
    "required": ["x", "y", "duration_s"],
    "additionalProperties": False,
}


def make(ctx: AppContext) -> Tool:
    def handler(args: dict) -> dict:
        for k in ("x", "y", "duration_s"):
            v = args.get(k)
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                raise ToolError(f"{k} must be a number")
        x = float(args["x"])
        y = float(args["y"])
        duration_s = float(args["duration_s"])
        if not -1.0 <= x <= 1.0:
            raise ToolError("x must be in [-1, 1]")
        if not -1.0 <= y <= 1.0:
            raise ToolError("y must be in [-1, 1]")
        if not 0.1 <= duration_s <= 5.0:
            raise ToolError("duration_s must be in [0.1, 5.0]")

        ctx.face_tracker.set_secondary_offset(x=x, y=y, duration_s=duration_s)
        return {"ok": True, "x": x, "y": y, "duration_s": duration_s}

    return Tool(
        name="look_at",
        description=(
            "Glance toward a point in normalized image coordinates "
            "(x, y in [-1, 1]) for duration_s seconds. Secondary move: blends "
            "smoothly on top of the active dance."
        ),
        parameters=SCHEMA,
        handler=handler,
    )
