"""``skip_song`` — stop the current track and signal the DJ that it ended."""

from __future__ import annotations

from reachy_mini_dance_party_app.tools.registry import AppContext, Tool


SCHEMA = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
}


def make(ctx: AppContext) -> Tool:
    def handler(args: dict) -> dict:
        # Halt audio and beat-aligned dance scheduling, then notify DJ so it
        # can either return to IDLE or trigger the next auto-DJ pick.
        ctx.playback.stop()
        ctx.dancer.stop()
        ctx.dj.song_ended()
        return {"ok": True}

    return Tool(
        name="skip_song",
        description="Stop the currently-playing song and advance to the next one (or stop).",
        parameters=SCHEMA,
        handler=handler,
    )
