"""Unit tests for :mod:`reachy_mini_dance_party_app.voice.openai_realtime`.

These tests drive the realtime loop with a scripted in-memory fake WebSocket
and a fake ``asyncio.sleep`` so reconnect backoff is observable without real
delays. No network, no OpenAI SDK required.
"""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import Awaitable, Callable
from typing import Any
from unittest.mock import MagicMock

import pytest

from reachy_mini_dance_party_app.tools.registry import Tool
from reachy_mini_dance_party_app.voice.openai_realtime import OpenAIRealtimeSession


# ---------------------------------------------------------------------------
# Fake WebSocket
# ---------------------------------------------------------------------------


class FakeConnectionClosed(Exception):
    """Stand-in for ``websockets.exceptions.ConnectionClosed``."""


class FakeWebSocket:
    """Minimal scripted async websocket.

    Each entry in ``incoming`` is either a ``dict`` (forwarded to the session
    as a JSON-decoded event) or an exception class/instance (raised on the
    next ``recv()`` call). Frames sent by the session via ``send()`` are
    appended to ``sent`` for later assertions.
    """

    def __init__(self, incoming: list[Any] | None = None) -> None:
        self.sent: list[dict[str, Any]] = []
        self._queue: asyncio.Queue[Any] = asyncio.Queue()
        for item in incoming or []:
            self._queue.put_nowait(item)
        self.closed = False

    def push(self, event: Any) -> None:
        """Schedule an additional event for the session to receive."""
        self._queue.put_nowait(event)

    async def send(self, payload: str | bytes) -> None:
        if self.closed:
            raise FakeConnectionClosed("send on closed socket")
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8")
        self.sent.append(json.loads(payload))

    async def recv(self) -> str:
        if self.closed:
            raise FakeConnectionClosed("recv on closed socket")
        item = await self._queue.get()
        if isinstance(item, BaseException) or (
            isinstance(item, type) and issubclass(item, BaseException)
        ):
            self.closed = True
            raise item if isinstance(item, BaseException) else item("scripted")
        return json.dumps(item)

    async def close(self) -> None:
        self.closed = True


def _make_connect_factory(
    sockets: list[FakeWebSocket],
) -> Callable[..., Awaitable[FakeWebSocket]]:
    """Return an async connect callable that yields the next scripted socket."""
    iterator = iter(sockets)

    async def connect(*_args: Any, **_kwargs: Any) -> FakeWebSocket:
        try:
            return next(iterator)
        except StopIteration as exc:
            raise RuntimeError("no more scripted websockets") from exc

    return connect


# ---------------------------------------------------------------------------
# Test tools
# ---------------------------------------------------------------------------


def _make_tools(handler: Callable[[dict], Any] | None = None) -> list[Tool]:
    if handler is None:
        handler = MagicMock(return_value={"ok": True})
    return [
        Tool(
            name="set_volume",
            description="Set speaker volume.",
            parameters={
                "type": "object",
                "properties": {"level": {"type": "integer"}},
                "required": ["level"],
                "additionalProperties": False,
            },
            handler=handler,
        ),
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _run_session_briefly(
    session: OpenAIRealtimeSession,
    timeout: float = 1.0,
) -> None:
    """Run ``session.run()`` until it stops itself or times out."""
    task = asyncio.create_task(session.run())
    try:
        await asyncio.wait_for(task, timeout=timeout)
    except asyncio.TimeoutError:
        session.stop()
        try:
            await asyncio.wait_for(task, timeout=0.5)
        except asyncio.TimeoutError:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, BaseException):
                pass
    except asyncio.CancelledError:
        pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_session_sends_session_update_on_connect() -> None:
    """First frame on connect is ``session.update`` with prompt + tools."""
    ws = FakeWebSocket(incoming=[Exception("disconnect to end the loop")])
    tools = _make_tools()
    session = OpenAIRealtimeSession(
        api_key="DUMMY",
        tools=tools,
        system_prompt="You are the DJ.",
        on_tts_chunk=lambda _b: None,
        connect_callable=_make_connect_factory([ws]),
        connection_closed_exceptions=(Exception,),
        max_reconnect_attempts=1,
    )

    await _run_session_briefly(session, timeout=1.0)

    assert ws.sent, "session must send at least one frame"
    first = ws.sent[0]
    assert first["type"] == "session.update"
    sess = first["session"]
    assert sess["instructions"] == "You are the DJ."
    tool_names = [t["name"] for t in sess["tools"]]
    assert tool_names == ["set_volume"]
    assert sess["tools"][0]["type"] == "function"


@pytest.mark.asyncio
async def test_tool_call_routes_to_handler() -> None:
    """A ``response.function_call_arguments.done`` event runs the handler."""
    handler_calls: list[dict] = []

    def handler(args: dict) -> dict:
        handler_calls.append(args)
        return {"applied": args["level"]}

    tools = _make_tools(handler=handler)

    ws = FakeWebSocket(
        incoming=[
            {
                "type": "response.function_call_arguments.done",
                "name": "set_volume",
                "call_id": "call_abc",
                "arguments": json.dumps({"level": 50}),
            },
            FakeConnectionClosed("end of script"),
        ]
    )
    session = OpenAIRealtimeSession(
        api_key="DUMMY",
        tools=tools,
        system_prompt="DJ",
        on_tts_chunk=lambda _b: None,
        connect_callable=_make_connect_factory([ws]),
        connection_closed_exceptions=(FakeConnectionClosed,),
        max_reconnect_attempts=1,
    )

    await _run_session_briefly(session, timeout=1.0)

    assert handler_calls == [{"level": 50}]
    outputs = [
        f
        for f in ws.sent
        if f.get("type") == "conversation.item.create"
        and f.get("item", {}).get("type") == "function_call_output"
    ]
    assert len(outputs) == 1
    out = outputs[0]
    assert out["item"]["call_id"] == "call_abc"
    assert json.loads(out["item"]["output"]) == {"applied": 50}


@pytest.mark.asyncio
async def test_tts_chunk_routes_to_callback() -> None:
    """``response.audio.delta`` events flow into ``on_tts_chunk``."""
    received: list[bytes] = []
    payload = b"hello-tts-bytes"
    encoded = base64.b64encode(payload).decode("ascii")

    ws = FakeWebSocket(
        incoming=[
            {"type": "response.audio.delta", "delta": encoded},
            FakeConnectionClosed("end"),
        ]
    )
    session = OpenAIRealtimeSession(
        api_key="DUMMY",
        tools=_make_tools(),
        system_prompt="DJ",
        on_tts_chunk=received.append,
        connect_callable=_make_connect_factory([ws]),
        connection_closed_exceptions=(FakeConnectionClosed,),
        max_reconnect_attempts=1,
    )

    await _run_session_briefly(session, timeout=1.0)

    assert received == [payload]


@pytest.mark.asyncio
async def test_reconnect_backoff_on_disconnect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On unexpected disconnect, session reconnects with growing backoff."""
    sleeps: list[float] = []

    real_sleep = asyncio.sleep

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)
        # Yield control so background tasks can run, but never actually wait.
        await real_sleep(0)

    monkeypatch.setattr(
        "reachy_mini_dance_party_app.voice.openai_realtime.asyncio.sleep",
        fake_sleep,
    )

    sockets = [
        FakeWebSocket(incoming=[FakeConnectionClosed("drop 1")]),
        FakeWebSocket(incoming=[FakeConnectionClosed("drop 2")]),
        FakeWebSocket(incoming=[FakeConnectionClosed("drop 3")]),
    ]
    session = OpenAIRealtimeSession(
        api_key="DUMMY",
        tools=_make_tools(),
        system_prompt="DJ",
        on_tts_chunk=lambda _b: None,
        connect_callable=_make_connect_factory(sockets),
        connection_closed_exceptions=(FakeConnectionClosed,),
        max_reconnect_attempts=3,
        reconnect_base_delay=1.0,
        reconnect_jitter=0.0,
    )

    await _run_session_briefly(session, timeout=2.0)

    # All three sockets should have been opened.
    assert all(ws.sent for ws in sockets), "every reconnect should re-send session.update"
    # Two backoff sleeps between three connect attempts (1s, 2s).
    assert len(sleeps) >= 2
    assert sleeps[0] == pytest.approx(1.0)
    assert sleeps[1] == pytest.approx(2.0)
    assert sleeps[1] > sleeps[0]
