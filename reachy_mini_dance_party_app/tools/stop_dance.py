"""``stop_dance`` — stop only the dance scheduler; music keeps playing."""

from __future__ import annotations

from reachy_mini_dance_party_app.tools.registry import AppContext, Tool


SCHEMA = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
}


def make(ctx: AppContext) -> Tool:
    def handler(args: dict) -> dict:
        ctx.dancer.stop()
        return {"ok": True}

    return Tool(
        name="stop_dance",
        description=(
            "Stop the dance moves but keep the music playing. The robot will "
            "go still until a new song starts or you ask it to dance again."
        ),
        parameters=SCHEMA,
        handler=handler,
    )
