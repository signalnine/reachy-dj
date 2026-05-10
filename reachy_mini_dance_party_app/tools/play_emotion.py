"""``play_emotion`` — enqueue a named emotion animation as a primary move."""

from __future__ import annotations

from dataclasses import dataclass

from reachy_mini_dance_party_app.tools.registry import AppContext, Tool, ToolError


# Curated set of emotion names known to the dances library. The actual mapping
# from name -> trajectory lives in moves.py / EmotionMove (Task 16).
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


@dataclass(frozen=True)
class EmotionMove:
    """Placeholder for the real EmotionMove that lands in moves.py (Task 16)."""

    name: str


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
