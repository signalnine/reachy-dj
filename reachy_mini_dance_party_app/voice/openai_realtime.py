"""OpenAI Realtime API session for the dance-party app.

Forked-and-pruned from ``reachy_mini_conversation_app/openai_realtime.py``.
The conv app sits on the OpenAI SDK's ``client.realtime.connect()`` async
context manager which is hard to mock cleanly. We replace that with a thin
``connect_callable`` injection point so unit tests can drive a scripted
``FakeWebSocket`` and the production path uses the ``websockets`` library
directly.

Owned here:
- The WebSocket connection to OpenAI Realtime.
- A reconnect loop with exponential backoff.
- Session configuration (system prompt + tool schemas).
- Inbound event dispatch:
  * ``response.audio.delta`` (and ``response.output_audio.delta``)
    → ``on_tts_chunk(bytes)``
  * ``response.function_call_arguments.done``
    → look up :class:`Tool` by name, run the handler, send back a
      ``function_call_output`` frame
  * speech start/stop hooks → optional ``on_speech_state(bool)`` callback
- :meth:`inject_system_event` / :meth:`inject_image` for out-of-band
  context injection by the DJ thread and audience-summary push timer.

NOT owned here:
- The audio mixer (Task 15 wires that in via ``on_tts_chunk``).
- The DJ state machine, dancer, camera worker, etc. — those live on the
  ``AppContext`` and are reached via :class:`Tool` handlers.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import random
import uuid
from collections.abc import Awaitable, Callable
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Default OpenAI Realtime endpoint. Model + auth land in connect headers.
_DEFAULT_REALTIME_URL = "wss://api.openai.com/v1/realtime"
_DEFAULT_MODEL = "gpt-realtime-2"


# ---------------------------------------------------------------------------
# WebSocket protocol helpers (exposed for tests)
# ---------------------------------------------------------------------------


def _import_websockets_connection_closed() -> tuple[type[BaseException], ...]:
    """Lazy import — only used when no override is supplied."""
    try:
        from websockets.exceptions import (  # noqa: PLC0415
            ConnectionClosed,
            ConnectionClosedError,
            ConnectionClosedOK,
        )
        return (ConnectionClosed, ConnectionClosedError, ConnectionClosedOK)
    except Exception:  # pragma: no cover - dependency missing in some envs
        return (ConnectionError, OSError)


async def _default_connect(url: str, headers: dict[str, str]) -> Any:
    """Open a real WebSocket via the ``websockets`` library."""
    from websockets.asyncio.client import connect as ws_connect  # noqa: PLC0415

    return await ws_connect(url, additional_headers=headers, max_size=None)


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


ToolHandler = Callable[[dict], Any]


class OpenAIRealtimeSession:
    """OpenAI Realtime API session for the dance-party app."""

    def __init__(
        self,
        api_key: str,
        tools: list[Any],
        system_prompt: str,
        on_tts_chunk: Callable[[bytes], None],
        on_speech_state: Optional[Callable[[bool], None]] = None,
        model: str = _DEFAULT_MODEL,
        url: str = _DEFAULT_REALTIME_URL,
        sample_rate: int = 24000,
        # Test seams ----------------------------------------------------
        connect_callable: Optional[Callable[..., Awaitable[Any]]] = None,
        connection_closed_exceptions: Optional[tuple[type[BaseException], ...]] = None,
        max_reconnect_attempts: int = 5,
        reconnect_base_delay: float = 1.0,
        reconnect_max_delay: float = 30.0,
        reconnect_jitter: float = 0.5,
    ) -> None:
        self._api_key = api_key
        self._tools = list(tools)
        self._tool_handlers: dict[str, ToolHandler] = {t.name: t.handler for t in tools}
        self._system_prompt = system_prompt
        self._on_tts_chunk = on_tts_chunk
        self._on_speech_state = on_speech_state
        self._model = model
        self._url = url
        self._sample_rate = sample_rate

        self._connect = connect_callable or _default_connect
        self._closed_exceptions = (
            connection_closed_exceptions or _import_websockets_connection_closed()
        )
        self._max_reconnect_attempts = max_reconnect_attempts
        self._reconnect_base_delay = reconnect_base_delay
        self._reconnect_max_delay = reconnect_max_delay
        self._reconnect_jitter = reconnect_jitter

        self._stop_requested = asyncio.Event()
        self._connection: Any = None
        self._connected = asyncio.Event()
        self._audio_delta_count = 0
        # Reference to the loop ``run()`` is executing on. Captured at the top
        # of :meth:`run` so background-thread callers of ``inject_*`` (DJ,
        # AudiencePush, etc.) can target the session loop via
        # ``run_coroutine_threadsafe`` without needing a running loop in
        # their own thread.
        self._loop: asyncio.AbstractEventLoop | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Connect, configure, dispatch events. Reconnects until ``stop()``."""
        # Capture the running loop so background-thread inject_* calls can
        # target it via run_coroutine_threadsafe (they don't have a running
        # loop of their own).
        self._loop = asyncio.get_running_loop()
        logger.info(
            "Realtime session starting: url=%s model=%s tools=%d",
            self._url, self._model, len(self._tools),
        )
        attempt = 0
        while not self._stop_requested.is_set():
            attempt += 1
            try:
                await self._run_one_session()
            except self._closed_exceptions as exc:
                logger.warning(
                    "Realtime websocket closed (attempt %d/%d): %s",
                    attempt,
                    self._max_reconnect_attempts,
                    exc,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - log and reconnect
                logger.exception("Realtime session error: %s", exc)
            finally:
                self._connection = None
                self._connected.clear()

            if self._stop_requested.is_set():
                return
            if attempt >= self._max_reconnect_attempts:
                logger.error(
                    "Giving up after %d reconnect attempts", self._max_reconnect_attempts
                )
                return

            delay = self._compute_backoff(attempt)
            logger.info("Reconnecting in %.2fs", delay)
            await asyncio.sleep(delay)

    def stop(self) -> None:
        """Request graceful shutdown of the session loop."""
        self._stop_requested.set()
        conn = self._connection
        if conn is not None:
            close = getattr(conn, "close", None)
            if close is not None:
                try:
                    result = close()
                    if asyncio.iscoroutine(result):
                        # Schedule but don't await — caller may not be in an
                        # asyncio context.
                        asyncio.ensure_future(result)
                except Exception:  # noqa: BLE001
                    pass

    def inject_system_event(self, instructions: str) -> None:
        """Inject a system-role conversation item.

        Used by the DJ thread (e.g. "Track ending in ~20s") and the
        audience-summary push timer. Safe to call from any thread that has
        an event loop running.
        """
        self._send_nowait(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "system",
                    "content": [{"type": "input_text", "text": instructions}],
                },
            }
        )

    def inject_image(self, image_b64: str, mime: str = "image/jpeg") -> None:
        """Inject an image as a user-role conversation item.

        Routes the base64 string into a ``conversation.item.create`` frame
        with an ``input_image`` content part. Whether the model actually
        consumes the image is a runtime question — see TODO at module level.
        """
        self._send_nowait(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": f"data:{mime};base64,{image_b64}",
                        }
                    ],
                },
            }
        )

    async def send_audio_chunk(self, pcm16_bytes: bytes) -> None:
        """Send a microphone audio chunk to the server (PCM16 little-endian)."""
        if self._connection is None:
            return
        encoded = base64.b64encode(pcm16_bytes).decode("ascii")
        await self._send(
            {"type": "input_audio_buffer.append", "audio": encoded}
        )

    def push_mic_chunk(self, pcm16_bytes: bytes) -> None:
        """Thread-safe: schedule a mic chunk send onto the session loop.

        Called from the mic-capture worker thread. Drops the chunk silently if
        the websocket isn't connected yet (e.g. during reconnect backoff) —
        server VAD will pick up audio once the next session is up.
        """
        if not pcm16_bytes:
            return
        encoded = base64.b64encode(pcm16_bytes).decode("ascii")
        self._send_nowait(
            {"type": "input_audio_buffer.append", "audio": encoded}
        )

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    async def _run_one_session(self) -> None:
        # NOTE: do NOT send ``OpenAI-Beta: realtime=v1`` — that header pins
        # the connection to the older beta protocol which uses the flat
        # session schema (``modalities``, ``output_audio_format``, etc).
        # The GA ``gpt-realtime`` endpoint expects the nested ``audio.input``
        # / ``audio.output`` schema, and rejects ``session.audio`` /
        # ``session.type`` if the beta header is present. The OpenAI Python
        # SDK's own ``client.realtime.connect`` doesn't set this header.
        headers = {
            "Authorization": f"Bearer {self._api_key}",
        }
        url = f"{self._url}?model={self._model}"
        logger.info("Realtime: opening websocket to %s", self._url)
        conn = await self._connect(url, headers)
        logger.info("Realtime: websocket connected, sending session.update")

        # Some connect callables return an async-context-manager. Honour both.
        if hasattr(conn, "__aenter__"):
            async with conn as opened:
                await self._drive(opened)
        else:
            try:
                await self._drive(conn)
            finally:
                close = getattr(conn, "close", None)
                if close is not None:
                    try:
                        result = close()
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception:  # noqa: BLE001
                        pass

    async def _drive(self, conn: Any) -> None:
        self._connection = conn
        self._connected.set()
        await self._send_session_update()
        # Trigger an unprompted opening turn — the system prompt instructs the
        # model to greet on connect. Without this, the session stays silent
        # until the user speaks first.
        await self._send({"type": "response.create"})
        await self._event_loop()

    async def _send_session_update(self) -> None:
        tools_payload = [
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
            for tool in self._tools
        ]
        # GA gpt-realtime schema: nested `session.audio.{input,output}` config.
        # The older flat keys (``modalities``, ``input_audio_format``,
        # ``output_audio_format``, top-level ``turn_detection``) are silently
        # ignored by the GA endpoint, which means session.updated comes back
        # but no audio output gets emitted — the model speaks only in text
        # transcripts (``response.audio_transcript.delta``) with zero
        # ``response.output_audio.delta`` events. Verified against gpt-realtime
        # 2026-05 by reading openai SDK ``RealtimeSessionCreateRequestParam``
        # in apps_venv.
        frame = {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "model": self._model,
                "output_modalities": ["audio"],
                "instructions": self._system_prompt,
                "tools": tools_payload,
                "tool_choice": "auto",
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": 24000},
                        "turn_detection": {
                            "type": "server_vad",
                            "interrupt_response": True,
                        },
                    },
                    "output": {
                        "format": {"type": "audio/pcm", "rate": 24000},
                        "voice": "marin",
                    },
                },
            },
        }
        await self._send(frame)

    async def _event_loop(self) -> None:
        conn = self._connection
        assert conn is not None
        while not self._stop_requested.is_set():
            try:
                raw = await conn.recv()
            except self._closed_exceptions:
                raise
            try:
                event = json.loads(raw)
            except (TypeError, ValueError):
                logger.warning("Dropping non-JSON realtime frame: %r", raw)
                continue
            await self._handle_event(event)

    # ------------------------------------------------------------------
    # Event dispatch
    # ------------------------------------------------------------------

    async def _handle_event(self, event: dict) -> None:
        kind = event.get("type", "")
        # Track audio.delta event count without spamming a log line per chunk.
        if kind.endswith(".audio.delta") or kind.endswith(".output_audio.delta"):
            self._audio_delta_count += 1
            if self._audio_delta_count == 1:
                logger.info("Realtime: first audio chunk arrived (%s)", kind)
            elif self._audio_delta_count % 50 == 0:
                logger.info("Realtime: %d audio chunks received", self._audio_delta_count)
        else:
            logger.info("Realtime event: %s", kind)
        if kind in ("session.created", "session.updated"):
            sess = event.get("session", {})
            logger.info(
                "Session config (server view): output_modalities=%s audio.output=%s model=%s",
                sess.get("output_modalities"),
                sess.get("audio", {}).get("output"),
                sess.get("model"),
            )

        # Both naming conventions seen in the wild.
        if kind in ("response.audio.delta", "response.output_audio.delta"):
            self._dispatch_tts_chunk(event)
            return

        if kind in (
            "input_audio_buffer.speech_started",
            "input_audio_buffer.speech_stopped",
        ):
            if self._on_speech_state is not None:
                started = kind == "input_audio_buffer.speech_started"
                try:
                    self._on_speech_state(started)
                except Exception:  # noqa: BLE001
                    logger.exception("on_speech_state callback failed")
            return

        if kind == "response.function_call_arguments.done":
            await self._dispatch_tool_call(event)
            return

        if kind == "error":
            err = event.get("error") or {}
            logger.error(
                "Realtime API error [%s]: %s",
                err.get("code") or err.get("type"),
                err.get("message"),
            )
            return

    def _dispatch_tts_chunk(self, event: dict) -> None:
        delta = event.get("delta")
        if not isinstance(delta, str):
            return
        try:
            chunk = base64.b64decode(delta)
        except Exception:  # noqa: BLE001
            logger.warning("Failed to decode TTS chunk")
            return
        try:
            self._on_tts_chunk(chunk)
        except Exception:  # noqa: BLE001
            logger.exception("on_tts_chunk callback failed")

    async def _dispatch_tool_call(self, event: dict) -> None:
        name = event.get("name")
        call_id = event.get("call_id") or str(uuid.uuid4())
        args_raw = event.get("arguments") or "{}"
        if not isinstance(name, str):
            logger.error("Tool call missing 'name': %r", event)
            return

        handler = self._tool_handlers.get(name)
        if handler is None:
            logger.error("No handler registered for tool %r", name)
            await self._send_tool_output(
                call_id, {"error": f"unknown tool: {name}"}
            )
            return

        try:
            args = json.loads(args_raw) if isinstance(args_raw, str) else dict(args_raw)
        except json.JSONDecodeError as exc:
            logger.error("Tool %r got invalid JSON args: %s", name, exc)
            await self._send_tool_output(call_id, {"error": f"invalid args: {exc}"})
            return

        logger.info("Tool call: %s(%s) call_id=%s", name, args, call_id)
        try:
            # Tool handlers can be heavy (yt-dlp fetch, librosa beat-track) and
            # would otherwise block the asyncio event loop — meaning no mic
            # chunks get flushed, no incoming TTS frames are read, and the
            # session looks frozen until the handler returns. Offload sync
            # handlers to a thread so the loop stays responsive; await
            # coroutines directly.
            if asyncio.iscoroutinefunction(handler):
                result = await handler(args)
            else:
                result = await asyncio.to_thread(handler, args)
        except Exception as exc:  # noqa: BLE001 - tool errors are routine
            logger.exception("Tool %r raised", name)
            result = {"error": str(exc)}

        await self._send_tool_output(call_id, result)

    async def _send_tool_output(self, call_id: str, output: Any) -> None:
        try:
            output_str = json.dumps(output, default=str)
        except (TypeError, ValueError):
            output_str = json.dumps({"error": "tool result not JSON-serializable"})
        await self._send(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": output_str,
                },
            }
        )

    # ------------------------------------------------------------------
    # Wire helpers
    # ------------------------------------------------------------------

    async def _send(self, frame: dict) -> None:
        conn = self._connection
        if conn is None:
            return
        await conn.send(json.dumps(frame))

    def _send_nowait(self, frame: dict) -> None:
        """Schedule a send without awaiting; used by inject_* methods.

        Safe to call from any thread (DJ, AudiencePush, etc.) — uses the
        loop captured at :meth:`run` start to schedule the coroutine, so the
        caller does not need a running loop of its own.
        """
        conn = self._connection
        if conn is None:
            return
        loop = self._loop
        if loop is None or loop.is_closed():
            logger.warning(
                "inject_* called before session loop running; dropping %r",
                frame.get("type"),
            )
            return
        asyncio.run_coroutine_threadsafe(self._send(frame), loop)

    def _compute_backoff(self, attempt: int) -> float:
        base = self._reconnect_base_delay * (2 ** (attempt - 1))
        delay = min(base, self._reconnect_max_delay)
        if self._reconnect_jitter:
            delay += random.uniform(0, self._reconnect_jitter)
        return delay


# ---------------------------------------------------------------------------
# TODO
# ---------------------------------------------------------------------------
# - The OpenAI Realtime API's exact ``session.update`` schema continues to
#   evolve; the keys above are the consensus shape as of 2026-05 but may
#   need adjustment when verified against the live endpoint in Task 17.
# - Image content parts via ``input_image`` are accepted by the conv app's
#   path but the live model's actual consumption of them is unverified for
#   ``gpt-realtime``. Task 16 should probe this with a 1-line test.
# - ``inject_system_event`` / ``inject_image`` use ``run_coroutine_threadsafe``
#   targeting the loop captured at the top of :meth:`run`. Background threads
#   (DJ, AudiencePush) do not need a running loop of their own — they post
#   onto the session loop directly.
