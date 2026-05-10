"""``move_head`` — enqueue a primary head-pose move that preempts the dance."""

from __future__ import annotations

from dataclasses import dataclass

from reachy_mini_dance_party_app.tools.registry import AppContext, Tool, ToolError


SCHEMA = {
    "type": "object",
    "properties": {
        "pitch": {"type": "number", "description": "Pitch in radians (positive = look up)."},
        "yaw": {"type": "number", "description": "Yaw in radians (positive = look left)."},
        "roll": {"type": "number", "description": "Roll in radians."},
        "duration_s": {
            "type": "number",
            "minimum": 0.1,
            "maximum": 5.0,
            "description": "Duration of the move in seconds.",
        },
    },
    "required": ["pitch", "yaw", "roll", "duration_s"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class HeadMove:
    """A primary head-pose move enqueued onto the move queue."""

    pitch: float
    yaw: float
    roll: float
    duration_s: float


def make(ctx: AppContext) -> Tool:
    def handler(args: dict) -> dict:
        for k in ("pitch", "yaw", "roll", "duration_s"):
            v = args.get(k)
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                raise ToolError(f"{k} must be a number")
        duration_s = float(args["duration_s"])
        if not 0.1 <= duration_s <= 5.0:
            raise ToolError("duration_s must be in [0.1, 5.0]")

        move = HeadMove(
            pitch=float(args["pitch"]),
            yaw=float(args["yaw"]),
            roll=float(args["roll"]),
            duration_s=duration_s,
        )
        ctx.move_queue.put(move)
        return {"ok": True, "duration_s": duration_s}

    return Tool(
        name="move_head",
        description=(
            "Move the head to an explicit pitch/yaw/roll over duration_s "
            "seconds. Primary move: preempts the current dance step."
        ),
        parameters=SCHEMA,
        handler=handler,
    )
