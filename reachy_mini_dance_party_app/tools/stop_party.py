"""``stop_party`` — fully stop music + dance, return DJ to idle."""

from __future__ import annotations

from reachy_mini_dance_party_app.tools.registry import AppContext, Tool


SCHEMA = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
}


def make(ctx: AppContext) -> Tool:
    def handler(args: dict) -> dict:
        ctx.playback.stop()
        ctx.dancer.stop()
        ctx.dj.stop_party()
        return {"ok": True}

    return Tool(
        name="stop_party",
        description="Stop the dance party entirely: music off, robot still, DJ idle.",
        parameters=SCHEMA,
        handler=handler,
    )
