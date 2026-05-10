"""``play_song`` — search YouTube and start synchronized playback + dance."""

from __future__ import annotations

from reachy_mini_dance_party_app.dj import SongInfo
from reachy_mini_dance_party_app.tools.registry import AppContext, Tool, ToolError


SCHEMA = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": (
                "Song title and/or artist to search for on YouTube "
                "(e.g. 'Daft Punk Around the World')."
            ),
        },
    },
    "required": ["query"],
    "additionalProperties": False,
}


def make(ctx: AppContext) -> Tool:
    def handler(args: dict) -> dict:
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ToolError("query must be a non-empty string")

        ctx.dj.request_song(query)
        try:
            result = ctx.fetcher.fetch(query)
            grid = ctx.analyzer(result.path)
            ctx.playback.load(result.path)
            ctx.playback.start()
            ctx.dancer.start_with_grid(grid)
        except Exception as exc:  # noqa: BLE001
            ctx.dj.fetch_failed(str(exc))
            raise ToolError(f"failed to play {query!r}: {exc}") from exc

        info = SongInfo(
            title=result.title,
            duration_s=float(result.duration_s),
            url=result.url,
            path=str(result.path),
            query=query,
        )
        ctx.dj.song_fetched(info)
        return {"title": result.title, "duration_s": float(result.duration_s)}

    return Tool(
        name="play_song",
        description=(
            "Search YouTube and start playing a song with synchronized dance "
            "moves. Blocks for ~5-15s while the audio is fetched and analyzed."
        ),
        parameters=SCHEMA,
        handler=handler,
    )
