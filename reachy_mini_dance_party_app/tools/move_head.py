"""``move_head`` — enqueue a primary head-pose move that preempts the dance."""

from __future__ import annotations

import numpy as np

from reachy_mini.motion.move import Move
from reachy_mini.utils import create_head_pose

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


# Default antenna pose used for held head-target moves; matches the neutral
# pose used by BreathingMove in moves.py (~10 deg outward to reduce shaking).
_NEUTRAL_ANTENNAS = np.array([-0.1745, 0.1745], dtype=np.float64)


class HeadMove(Move):
    """Primary head-pose move that holds a target pitch/yaw/roll for ``duration_s``.

    Implements the SDK :class:`reachy_mini.motion.move.Move` interface so the
    object can be enqueued on the real ``MovementManager`` move queue. The
    ``pitch``/``yaw``/``roll``/``duration_s`` attributes are preserved on the
    instance for tool-test introspection.
    """

    def __init__(self, pitch: float, yaw: float, roll: float, duration_s: float) -> None:
        self.pitch = float(pitch)
        self.yaw = float(yaw)
        self.roll = float(roll)
        self.duration_s = float(duration_s)
        # Pre-compute the target homogeneous pose once; evaluate() returns it
        # for the lifetime of the move so the head holds the requested pose.
        self._target_pose = create_head_pose(
            x=0,
            y=0,
            z=0,
            roll=self.roll,
            pitch=self.pitch,
            yaw=self.yaw,
            degrees=False,
            mm=False,
        )

    @property
    def duration(self) -> float:  # type: ignore[override]
        return self.duration_s

    def evaluate(self, t: float):  # type: ignore[override]
        # Hold the target pose for the full duration. The MovementManager uses
        # blending between primary moves, so this still yields a smooth motion.
        return (self._target_pose, _NEUTRAL_ANTENNAS, 0.0)


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
