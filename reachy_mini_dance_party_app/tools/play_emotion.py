"""``play_emotion`` — enqueue a named emotion animation as a primary move.

V1 STUB STATUS
==============
``reachy_mini_dances_library`` (the only animation catalog installed in our
laptop venv) ships beat-driven dance primitives (``simple_nod``,
``side_to_side_sway``, etc.) — *not* an emotion catalog with names like
``happy`` / ``sad`` / ``celebrate``. The conv app shipped its own emotion
library; that asset wasn't lifted into this app.

Rather than failing the tool call (which would cause the LLM to retry or
report an error to the user), we accept the call and enqueue a short
neutral-pose hold via :class:`EmotionMove`. The robot does nothing visible
beyond a brief pause from the dancer's primary-move queue, but the call
returns OK so the conversation flow keeps moving.

When a real emotion catalog ships, replace ``EmotionMove.evaluate`` with a
name-driven trajectory lookup.
"""

from __future__ import annotations

import numpy as np

from reachy_mini.motion.move import Move
from reachy_mini.utils import create_head_pose

from reachy_mini_dance_party_app.tools.registry import AppContext, Tool, ToolError


# Curated set of emotion names known to the dances library. Currently this
# list is *aspirational* — see module docstring above (V1 STUB STATUS).
KNOWN_EMOTIONS = (
    "happy",
    "sad",
    "surprised",
    "angry",
    "thinking",
    "celebrate",
    "wave",
    "nod",
    "shake",
)


SCHEMA = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": (
                "Emotion / gesture name. Common values: "
                + ", ".join(KNOWN_EMOTIONS)
                + "."
            ),
        }
    },
    "required": ["name"],
    "additionalProperties": False,
}


# Default antenna pose — matches the neutral used by BreathingMove / HeadMove
# (~10 deg outward to reduce shaking).
_NEUTRAL_ANTENNAS = np.array([-0.1745, 0.1745], dtype=np.float64)


class EmotionMove(Move):
    """V1 stub: holds the neutral pose for a short duration.

    Implements the SDK :class:`reachy_mini.motion.move.Move` interface so it
    can be enqueued onto the real :class:`MovementManager` move queue. The
    requested ``name`` is preserved on the instance for tool-test
    introspection and for future debugging when a real emotion catalog is
    wired up. ``duration_s`` defaults to 1.0 — long enough that the queue
    visibly serializes the call but short enough not to interrupt the
    dancing for noticeable time.
    """

    def __init__(self, name: str, duration_s: float = 1.0) -> None:
        self.name = str(name)
        self.duration_s = float(duration_s)
        # Pre-compute neutral pose once — evaluate returns it for the lifetime
        # of the move so the head simply holds neutral.
        self._neutral_pose = create_head_pose(
            x=0, y=0, z=0, roll=0, pitch=0, yaw=0, degrees=True
        )

    @property
    def duration(self) -> float:  # type: ignore[override]
        return self.duration_s

    def evaluate(self, t: float):  # type: ignore[override]
        # Hold neutral for the full duration (V1 stub behaviour).
        return (self._neutral_pose, _NEUTRAL_ANTENNAS, 0.0)


def make(ctx: AppContext) -> Tool:
    def handler(args: dict) -> dict:
        name = args.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ToolError("name must be a non-empty string")
        ctx.move_queue.put(EmotionMove(name=name))
        return {"ok": True, "name": name}

    return Tool(
        name="play_emotion",
        description=(
            "Play a named emotion animation (e.g. 'happy', 'celebrate'). "
            "Primary move: preempts the current dance step briefly."
        ),
        parameters=SCHEMA,
        handler=handler,
    )
