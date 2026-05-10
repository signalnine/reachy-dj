"""Voice subsystem.

Owns the OpenAI Realtime session: WebSocket lifecycle, audio in/out
streaming, system-prompt + tool-schema configuration, and tool-call
dispatch back to the app's :class:`AppContext`.
"""

from reachy_mini_dance_party_app.voice.openai_realtime import OpenAIRealtimeSession

__all__ = ["OpenAIRealtimeSession"]
