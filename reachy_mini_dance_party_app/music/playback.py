"""Streaming audio playback with a monotonic position clock + TTS mixing.

PlaybackEngine owns a sounddevice.OutputStream (or any compatible factory
injected for tests). It is a hybrid music + TTS player:

* A wav file loaded via :meth:`load` is held entirely in memory and consumed
  by the stream callback as the "music" channel.
* Inbound TTS audio chunks (PCM16 little-endian, mono, typically 24kHz from
  the OpenAI Realtime API) are appended via :meth:`feed_tts_chunk`. They land
  in a thread-safe inlet, get resampled to the playback sample rate, and are
  consumed by the same callback as the "TTS" channel.

The two channels are combined per-callback through the existing
:class:`~reachy_mini_dance_party_app.music.mixer.Mixer` and
:class:`~reachy_mini_dance_party_app.music.mixer.Ducker`. Speech-active state
is set externally by the Realtime session via :meth:`set_speech_active` —
the ducker does not detect activity from the TTS bytes themselves; it only
applies the music-gain envelope dictated by ``speech_active``.

``playback_time()`` continues to return ``frames_played / sample_rate`` and is
monotonic across callback invocations, suitable as the wall-clock for the
beat-aligned dance scheduler.

The stream factory is injectable so unit tests can drive the callback
synchronously via a ``FakeOutputStream`` instead of needing real audio
hardware. Defaults to ``sounddevice.OutputStream``.
"""
from __future__ import annotations

import threading
from collections import deque
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from .mixer import Ducker, Mixer

try:
    import sounddevice as _sd
    _DEFAULT_STREAM_FACTORY: Callable[..., Any] = _sd.OutputStream
except (OSError, ImportError):  # PortAudio not installed / not loadable.
    _DEFAULT_STREAM_FACTORY = None  # type: ignore[assignment]


class PlaybackEngine:
    """Streaming audio playback with mixer-based TTS overlay + ducking.

    Construction:
        engine = PlaybackEngine(mixer=Mixer(), ducker=Ducker())
        engine.load(wav_path)
        engine.feed_tts_chunk(pcm16_bytes, sample_rate=24000)
        engine.set_speech_active(True)
        engine.start()
        ...
        engine.stop()
    """

    def __init__(
        self,
        mixer: Mixer,
        ducker: Ducker,
        sample_rate: int = 48000,
        channels: int = 1,
        stream_factory: Optional[Callable[..., Any]] = None,
    ) -> None:
        self._mixer = mixer
        self._ducker = ducker
        self._sample_rate = int(sample_rate)
        self._channels = int(channels)

        if stream_factory is None:
            if _DEFAULT_STREAM_FACTORY is None:  # pragma: no cover - host-dependent
                raise RuntimeError(
                    "sounddevice is not available on this host; pass an explicit "
                    "stream_factory to PlaybackEngine for testing."
                )
            stream_factory = _DEFAULT_STREAM_FACTORY
        self.stream_factory: Callable[..., Any] = stream_factory

        # Music buffer state.
        self._buffer: np.ndarray = np.zeros((0, 1), dtype=np.float32)
        self._music_sr: int = 0
        self._music_channels: int = 1
        self._read_pos: int = 0
        self._frames_played: int = 0

        # TTS inlet — a deque of float32 1D arrays, all already resampled to
        # ``self._sample_rate`` and stored mono. The callback drains samples
        # from the head; ``feed_tts_chunk`` appends to the tail.
        self._tts_inlet: deque[np.ndarray] = deque()

        # Speech state for ducker (set by Realtime session).
        self._speech_active: bool = False

        # Stream + lock.
        self._stream: Optional[Any] = None
        self._is_playing: bool = False
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, wav_path: Path) -> None:
        """Load a wav file fully into memory as float32, shape (frames, channels).

        The file's sample rate is recorded but the playback stream itself runs
        at ``self._sample_rate`` (set at construction). Music samples are
        *not* automatically resampled here — Task 15's scope assumes the
        downloaded music is already at the playback rate or close enough; if
        rate mismatch becomes an issue, resample at load time before storing.
        """
        import soundfile as sf  # local import; soundfile is a librosa dep

        data, samplerate = sf.read(str(wav_path), dtype="float32", always_2d=True)
        with self._lock:
            self._buffer = np.ascontiguousarray(data, dtype=np.float32)
            self._music_sr = int(samplerate)
            self._music_channels = int(self._buffer.shape[1])
            self._read_pos = 0
            self._frames_played = 0

    def feed_tts_chunk(self, pcm_bytes: bytes, sample_rate: int = 24000) -> None:
        """Append TTS audio to the inlet queue.

        Accepts PCM16 little-endian mono bytes (the OpenAI Realtime API
        delivers in this format). Converts to float32 in [-1, 1] and resamples
        to the playback sample rate before queueing.
        """
        if not pcm_bytes:
            return
        # PCM16 LE → int16 → float32 in [-1, 1].
        audio_int16 = np.frombuffer(pcm_bytes, dtype="<i2")
        if audio_int16.size == 0:
            return
        audio_f32 = audio_int16.astype(np.float32) / 32768.0

        if sample_rate != self._sample_rate:
            import librosa  # local import; heavy module

            audio_f32 = librosa.resample(
                y=audio_f32,
                orig_sr=int(sample_rate),
                target_sr=self._sample_rate,
            ).astype(np.float32, copy=False)

        with self._lock:
            self._tts_inlet.append(audio_f32)

    def set_speech_active(self, active: bool) -> None:
        """Hook for the Realtime session: ducker activates when True."""
        # No lock needed for a bool write; the callback reads it atomically.
        self._speech_active = bool(active)

    def start(self) -> None:
        """Open the output stream and begin playback."""
        if self._stream is not None:
            return  # idempotent
        self._stream = self.stream_factory(
            samplerate=self._sample_rate,
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
        sr = self._sample_rate
        if sr == 0:
            return 0.0
        with self._lock:
            return self._frames_played / float(sr)

    @property
    def is_playing(self) -> bool:
        return self._is_playing

    @property
    def duration_s(self) -> float:
        """Total length of the loaded music buffer in seconds.

        Uses the music file's native sample rate (``self._music_sr``), not the
        playback rate, since the buffer is stored at its source rate.
        """
        sr = self._music_sr
        if sr == 0:
            return 0.0
        return self._buffer.shape[0] / float(sr)

    # ------------------------------------------------------------------
    # Callback (called on the audio thread for real streams)
    # ------------------------------------------------------------------

    def _callback(
        self,
        outdata: np.ndarray,
        frames: int,
        time_info: Any,
        status: Any,
    ) -> None:
        """Mix music + TTS into outdata for the next ``frames`` samples.

        Steps:
          1. Pull ``frames`` mono samples from the music buffer (zeros if
             exhausted).
          2. Pull ``frames`` mono samples from the TTS inlet (zeros if empty).
          3. Compute ``music_gain = ducker.update(speech_active=...)``.
          4. ``mixed = mixer.mix(music_chunk, tts_chunk, music_gain, 1.0)``.
          5. Write ``mixed`` to all output channels.
        """
        with self._lock:
            music_chunk = self._pull_music_locked(frames)
            tts_chunk = self._pull_tts_locked(frames)
            self._frames_played += frames
            speech_active = self._speech_active

        music_gain = self._ducker.update(speech_active=speech_active)
        mixed = self._mixer.mix(
            music_chunk,
            tts_chunk,
            music_gain=music_gain,
            tts_gain=1.0,
        )

        # Broadcast mono mixed → all output channels.
        if outdata.shape[1] == 1:
            outdata[:, 0] = mixed
        else:
            outdata[:] = mixed[:, np.newaxis]

    # ------------------------------------------------------------------
    # Internal helpers (must hold _lock)
    # ------------------------------------------------------------------

    def _pull_music_locked(self, frames: int) -> np.ndarray:
        """Pull ``frames`` mono samples from the music buffer; pad with zeros.

        Multi-channel music files are downmixed to mono by averaging channels.
        """
        out = np.zeros(frames, dtype=np.float32)
        buf = self._buffer
        pos = self._read_pos
        remaining = buf.shape[0] - pos
        if remaining <= 0:
            return out
        n = min(remaining, frames)
        chunk = buf[pos : pos + n]
        if chunk.shape[1] == 1:
            out[:n] = chunk[:, 0]
        else:
            out[:n] = chunk.mean(axis=1)
        self._read_pos = pos + n
        return out

    def _pull_tts_locked(self, frames: int) -> np.ndarray:
        """Drain ``frames`` mono samples from the TTS inlet; pad with zeros."""
        out = np.zeros(frames, dtype=np.float32)
        written = 0
        while written < frames and self._tts_inlet:
            head = self._tts_inlet[0]
            need = frames - written
            if head.size <= need:
                out[written : written + head.size] = head
                written += head.size
                self._tts_inlet.popleft()
            else:
                out[written:] = head[:need]
                # Replace head with the remainder so the next callback can
                # continue draining from the partial chunk.
                self._tts_inlet[0] = head[need:]
                written = frames
        return out
