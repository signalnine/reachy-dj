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
        # clear_grid pauses the dancer without killing its thread — calling
        # ctx.dancer.stop() here would tear down the worker permanently and
        # any future play_song would have no beat scheduling.
        ctx.dancer.clear_grid()
        ctx.dj.stop_party()
        return {"ok": True}

    return Tool(
        name="stop_party",
        description="Stop the dance party entirely: music off, robot still, DJ idle.",
        parameters=SCHEMA,
        handler=handler,
    )
