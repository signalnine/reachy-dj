"""LLM tool implementations exposed via the OpenAI Realtime function-calling API.

Each tool lives in its own module and exposes a ``make(ctx) -> Tool`` factory.
The :func:`registry.all_tools` helper collects every tool, ready to register
with the Realtime session.
"""

from reachy_mini_dance_party_app.tools.registry import (
    AppContext,
    Tool,
    ToolError,
    all_tools,
)

__all__ = ["AppContext", "Tool", "ToolError", "all_tools"]
