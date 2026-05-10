"""Streaming audio playback with a monotonic position clock.

PlaybackEngine owns a sounddevice.OutputStream (or any compatible factory
injected for tests). A wav file loaded via ``load()`` is held entirely in
memory; the stream callback copies successive chunks of that buffer into the
output array and advances a frame counter. ``playback_time()`` returns
``frames_played / sample_rate`` and is monotonic across callback invocations,
which makes it suitable as the wall-clock source for the beat-aligned dance
scheduler (Task 11).

The stream factory is injectable so unit tests can drive the callback
synchronously via a ``FakeOutputStream`` instead of needing real audio
hardware. Defaults to ``sounddevice.OutputStream``.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

try:
    import sounddevice as _sd
    _DEFAULT_STREAM_FACTORY: Callable[..., Any] = _sd.OutputStream
except (OSError, ImportError):  # PortAudio not installed / not loadable.
    _DEFAULT_STREAM_FACTORY = None  # type: ignore[assignment]


@dataclass
class PlaybackEngine:
    """Streaming audio playback with a monotonic position clock.

    Owns a sounddevice.OutputStream (or compatible). The loaded buffer is fed
    to the stream callback in chunks; ``playback_time()`` returns
    ``frames_played / sample_rate``, suitable for beat-aligned scheduling.
    """

    stream_factory: Callable[..., Any] = field(default=None)  # type: ignore[assignment]

    # Internal state — not part of the public API.
    _buffer: np.ndarray = field(default_factory=lambda: np.zeros((0, 1), dtype=np.float32), init=False)
    _samplerate: int = field(default=0, init=False)
    _channels: int = field(default=1, init=False)
    _read_pos: int = field(default=0, init=False)  # next frame index to copy from _buffer
    _frames_played: int = field(default=0, init=False)
    _stream: Optional[Any] = field(default=None, init=False)
    _is_playing: bool = field(default=False, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def __post_init__(self) -> None:
        if self.stream_factory is None:
            if _DEFAULT_STREAM_FACTORY is None:  # pragma: no cover - host-dependent
                raise RuntimeError(
                    "sounddevice is not available on this host; pass an explicit "
                    "stream_factory to PlaybackEngine for testing."
                )
            self.stream_factory = _DEFAULT_STREAM_FACTORY

    # ---------- Public API ----------

    def load(self, wav_path: Path) -> None:
        """Load a wav file fully into memory as float32, shape (frames, channels)."""
        import soundfile as sf  # local import; soundfile is a librosa dep

        data, samplerate = sf.read(str(wav_path), dtype="float32", always_2d=True)
        with self._lock:
            self._buffer = np.ascontiguousarray(data, dtype=np.float32)
            self._samplerate = int(samplerate)
            self._channels = int(self._buffer.shape[1])
            self._read_pos = 0
            self._frames_played = 0

    def start(self) -> None:
        """Open the output stream and begin playback."""
        if self._samplerate == 0:
            raise RuntimeError("PlaybackEngine.start() called before load()")
        if self._stream is not None:
            return  # idempotent
        self._stream = self.stream_factory(
            samplerate=self._samplerate,
            channels=self._channels,
            callback=self._callback,
            dtype="float32",
        )
        self._stream.start()
        self._is_playing = True

    def stop(self) -> None:
        """Halt playback and close the underlying stream."""
        stream = self._stream
        if stream is None:
            self._is_playing = False
            return
        try:
            stream.stop()
        finally:
            try:
                stream.close()
            finally:
                self._stream = None
                self._is_playing = False

    def playback_time(self) -> float:
        """Return frames-played / sample-rate (seconds since playback began).

        Monotonic across callback invocations and unaffected by buffer
        exhaustion (silence is also "played" in stream-time terms).
        """
        sr = self._samplerate
        if sr == 0:
            return 0.0
        with self._lock:
            return self._frames_played / float(sr)

    @property
    def is_playing(self) -> bool:
        return self._is_playing

    @property
    def duration_s(self) -> float:
        """Total length of the loaded buffer in seconds (0.0 if nothing loaded)."""
        sr = self._samplerate
        if sr == 0:
            return 0.0
        return self._buffer.shape[0] / float(sr)

    # ---------- Callback (called on the audio thread for real streams) ----------

    def _callback(
        self,
        outdata: np.ndarray,
        frames: int,
        time_info: Any,
        status: Any,
    ) -> None:
        """Fill outdata with the next ``frames`` samples from the buffer.

        Pads with zeros once the buffer is exhausted. Always advances the
        frames-played counter by the requested frame count so playback_time()
        keeps ticking even past end-of-buffer (callers can detect end via
        ``playback_time() >= duration_s``).
        """
        with self._lock:
            buf = self._buffer
            pos = self._read_pos
            remaining = buf.shape[0] - pos
            if remaining > 0:
                n = min(remaining, frames)
                chunk = buf[pos : pos + n]
                # Buffer is (frames, channels); broadcast or copy as appropriate.
                if chunk.shape[1] == outdata.shape[1]:
                    outdata[:n] = chunk
                elif chunk.shape[1] == 1:
                    outdata[:n] = chunk  # broadcasts mono to all channels
                else:
                    # Fallback: fill min channels, zero the rest.
                    c = min(chunk.shape[1], outdata.shape[1])
                    outdata[:n, :c] = chunk[:, :c]
                    if c < outdata.shape[1]:
                        outdata[:n, c:] = 0.0
                if n < frames:
                    outdata[n:] = 0.0
                self._read_pos = pos + n
            else:
                outdata[:] = 0.0

            self._frames_played += frames
