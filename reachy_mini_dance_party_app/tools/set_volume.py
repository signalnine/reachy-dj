"""``set_volume`` — POST to the daemon's ``/api/volume/set`` endpoint."""

from __future__ import annotations

import httpx

from reachy_mini_dance_party_app.tools.registry import AppContext, Tool, ToolError


SCHEMA = {
    "type": "object",
    "properties": {
        "level": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
            "description": "Speaker volume from 0 (mute) to 100 (max).",
        }
    },
    "required": ["level"],
    "additionalProperties": False,
}


def make(ctx: AppContext) -> Tool:
    def handler(args: dict) -> dict:
        level = args.get("level")
        if not isinstance(level, int) or isinstance(level, bool) or not 0 <= level <= 100:
            raise ToolError("level must be int in [0, 100]")

        client = ctx.http_client or httpx.Client(timeout=5.0)
        try:
            resp = client.post(
                f"{ctx.daemon_url}/api/volume/set", json={"level": level}
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise ToolError(f"failed to set volume: {exc}") from exc
        return {"level": level}

    return Tool(
        name="set_volume",
        description="Set speaker volume from 0 (mute) to 100 (max).",
        parameters=SCHEMA,
        handler=handler,
    )
