# Reachy Mini Dance-Party App — Design

**Date:** 2026-05-09
**Author:** signalnine (with Claude)
**Target:** Reachy Mini wireless edition, daemon v1.7.1, on Pi CM4

## Goal

A voice-controlled dance-party app for Reachy Mini. Start a chat session, ask for a song, fetch it from YouTube, play it through the robot's speaker, dance to it using the existing motor library, track the audience's faces, and let an LLM "DJ" auto-pick follow-up tracks based on history.

## Decisions

| # | Decision | Rationale |
|---|---|---|
| 1 | Fork `reachy_mini_conversation_app` v0.6.1 into a new package `reachy_mini_dance_party_app` | Conversation app already has the move scheduler, voice loop, face tracking, and dance-library integration we need. Fastest path to MVP. |
| 2 | All-in-one MVP (voice + YouTube + dance + face tracking + audience comments) | User wants the full vision in one demo. |
| 3 | OpenAI Realtime (gpt-realtime) as voice backend | Already used by the conversation app on this device, `OPENAI_API_KEY` configured. |
| 4 | Music pipeline: yt-dlp → /tmp wav cache → librosa beat analysis → sounddevice playback | Standard local pipeline; ~5–15s startup latency masked by LLM ("let me grab that"). |
| 5 | Dance gen: pre-recorded library moves from `reachy_mini_dances_library`, started on downbeats | Looks polished out of the gate; no choreography to write. |
| 6 | Engagement: head tracks dominant face + LLM gets periodic audience summaries | User wants tracking and conversational comments, not adapt-to-room or audience-driven song picks (V2). |
| 7 | Auto-DJ mode: LLM picks the next song based on what just played; user can interrupt | "Throw a little dance party" implies multi-song. |

## High-level architecture

New package, sibling to the conversation app:

```
reachy_mini_dance_party_app/
├── main.py                # entry point, ReachyMiniDancePartyApp class
├── moves.py               # primary/secondary move scheduler (lifted from conv app)
├── dj.py                  # auto-DJ state machine (queue, history, song lifecycle)
├── music/
│   ├── youtube.py         # yt-dlp wrapper, search + audio extract
│   ├── playback.py        # sounddevice playback w/ position callback
│   └── beat.py            # librosa beat analysis, beat-aligned scheduler
├── dance/
│   └── library_dancer.py  # picks library moves, schedules on downbeats
├── vision/
│   ├── camera_worker.py   # lifted
│   ├── face_tracker.py    # face-tracking secondary move (lifted)
│   └── audience.py        # periodic audience summary push to LLM
├── voice/
│   └── openai_realtime.py # lifted, modified system prompt + tools
├── tools/                 # LLM-callable
│   ├── play_song.py       │ skip_song.py     │ stop_party.py
│   ├── set_volume.py      │ take_photo.py    │ look_at_audience.py
│   ├── move_head.py       │ play_emotion.py  │ look_at.py
│   ├── set_face_tracking.py
│   └── stop_dance.py
└── prompts/               # system prompt + persona text
```

**Single control point preserved:** only the `moves.py` worker writes to `ReachyMini.set_target`.

**Entry point:** registered as `reachy_mini_apps:reachy_mini_dance_party_app = reachy_mini_dance_party_app.main:ReachyMiniDancePartyApp` so the daemon's app manager loads it; starts via dashboard or `POST /api/apps/start-app/reachy_mini_dance_party_app`.

## Music pipeline (`music/`)

- **`youtube.py`** — `yt-dlp` as Python module. `ytsearch1:<query>`, audio-only (`bestaudio[ext=m4a]/bestaudio`), post-process to wav. Cache `/tmp/reachy_dance_party/<sha256(query)>.{wav,info.json}`.
- **`beat.py`** — `librosa.beat.beat_track(sr=22050, units='time')` once on download completion. Returns `tempo, beat_times`. Cached as `<hash>.beats.npy`. ~1–2s analysis on CM4 for 3-min track.
- **`playback.py`** — `sounddevice.OutputStream` callback, `playback_time()` accessor (frames/sr, monotonic). Volume routed through daemon `/api/volume/set` to avoid fighting ALSA mixer state.
- **`BeatGrid.next_beat_at(t, n=1)`** — returns next *n* beat times after `t`, used by the dance scheduler.

**Failure modes**:
- yt-dlp fails → structured error → LLM apologizes, asks for another song.
- Empty beat array → fall back to 120 BPM synthetic grid, log warning.
- Audio device missing → app refuses to start.

## Dance scheduler (`dance/library_dancer.py`)

Thread that owns "what plays next." Loop:

1. `BeatGrid.next_beat_at(playback.time(), n=4)` — get next four beats.
2. Pick a candidate library move whose duration is close to N beats at current BPM (within ~150ms of an integer beat count).
3. If no fit, queue a 1-beat procedural fill (small head bob / antenna twitch).
4. Wait until ~50ms before downbeat, enqueue on the primary-move queue.
5. Track recently-played moves (ring buffer + weighted random) to avoid monotony.

**Tempo handling:** moves play at recorded speed; we adapt by *choosing*, not by time-stretching. For tempos <70 or >160 BPM, widen the fit window or use procedural fills.

**Integration:** zero changes to `moves.py` semantics. Library dances are **primary** (sequential, exclusive). Face tracking + a new lightweight beat-bob secondary (±2° pitch on each beat) blend on top via `compose_world_offset`.

**Stop conditions:** DJ → "stop" → drain queue, ramp head to neutral over 1.5s.

## Voice loop & tools (`voice/`, `tools/`, `prompts/`)

**Realtime backend** — `openai_realtime.py` lifted, persona swapped, tool registry replaced. Reads `OPENAI_API_KEY` from environment.

**Audio routing** — single `sounddevice.OutputStream` we own; both music and Realtime TTS mix in software:

```
music_buffer ─ ×music_gain ─┐
                            ├─→ mix → callback → speaker
LLM_tts_chunks ─ ×tts_gain ─┘
```

**Auto-ducking:** when TTS is active or VAD detects user speech, `music_gain` ramps 1.0 → 0.25 over 200ms; restored 500ms after speech ends.

**Tools (LLM-callable function-calls):**

| Tool | Type | Behavior |
|---|---|---|
| `play_song(query)` | control | Search/download/analyze/start. Blocks ~5–15s; returns title + duration. |
| `skip_song()` | control | Stop current, ramp dance, signal DJ for next. |
| `stop_party()` | control | Full stop → IDLE. |
| `set_volume(level: 0..100)` | control | Routes to `/api/volume/set`. |
| `take_photo()` | vision | Capture latest frame, JPEG-encode, send as image content part. |
| `look_at_audience()` | vision | Returns `{n_faces, dominant_centered, smiles}`. |
| `set_face_tracking(enabled)` | vision | Toggles the secondary face-tracking offset. Default ON. |
| `move_head(pitch, yaw, roll, dur)` | motion, **primary** | Manual gaze, preempts dance. Auto-resume after 5s budget. |
| `play_emotion(name)` | motion, **primary** | One-shot emotion clip from library. |
| `look_at(direction)` | motion, **secondary** | Quick glance overlay (<1.5s), blends. |
| `stop_dance()` | motion, control | Halts LibraryDancer; music keeps playing. |

**Auto-DJ:** ~20s before track end, inject `{"type": "session.update", "instructions": "Track ending in ~20s. Pick something complementary."}` into the Realtime session. LLM speaks ("up next…") and calls `play_song()` for handoff. User overrides any time.

**System prompt sketch:** warm, brief between songs; aware the room is small (1–4 people typically); avoids repeats from session history; willing to take requests; handles tool failures gracefully. Lives in `prompts/system.md`.

## Vision pipeline (`vision/`)

**`camera_worker.py`** — one thread, libcamera, ~15 fps, latest-frame-wins shared buffer (lock + most-recent frame, no queue).

**Two-tier face detection:**

| Model | Cost | Used for |
|---|---|---|
| MediaPipe FaceDetection (BlazeFace) | ~3ms/frame on CM4 | Continuous: count, bboxes, dominant pick |
| MediaPipe FaceLandmarker | ~25ms/frame | On-demand only: smile/gaze at audience-summary cadence |

**`face_tracker.py`** — secondary move at 15 Hz:
1. Pull dominant face (largest bbox) from camera worker.
2. Pixel center → desired head yaw/pitch (small-angle approx; ±200 px ≈ ±15° at imx708 FOV).
3. Set secondary offset; `moves.py` blends via `compose_world_offset`.
4. Low-pass smoothing (α=0.3) to kill jitter.
5. Lost-face hysteresis: 1.5s no-face → smooth return to zero.

**`audience.py`** — every 8s + immediate push on face-count delta. Builds:
```json
{"n_faces": 2, "dominant_centered": true, "smiles": 1, "since_last": "+1 face"}
```
Pushed to Realtime session as system-role `conversation.item.create`. System prompt instructs LLM to acknowledge sparingly between songs, not every push.

**`take_photo()`** — capture latest, JPEG-encode (~50KB at 640×480), inject as image content part. Realtime gpt-realtime accepts image parts; model "sees" it for next response.

**No frames stored to disk.** Simple privacy posture.

## Concurrency model

| # | Thread | Rate | Owns | Reads from |
|---|---|---|---|---|
| 1 | Main asyncio loop | event-driven | Realtime session, tool dispatch | dispatches → 5, 6, 7 |
| 2 | Move worker (`moves.py`) | 100 Hz | **sole writer** to `set_target` | move queue + secondary offsets |
| 3 | Camera worker | 15 fps | libcamera, latest-frame buffer | — |
| 4 | Face tracker | 15 Hz | secondary head-tracking offset | 3 |
| 5 | Library dancer | beat-driven | enqueues primary dance moves | beat-grid (from 7) → 2 |
| 6 | Audio mixer (sd callback) | 48 kHz block | speaker output, mix of music+TTS | music buffer (7), TTS chunks (1) |
| 7 | Music pipeline | per-song | yt-dlp, librosa, buffer fill | feeds 5, 6 |
| 8 | Audience push timer | 0.125 Hz + edge | summary JSON | 3 → 1 |

**Locks:** one for secondary-offsets dict, one for camera latest-frame buffer. Everything else uses queues, atomics, or per-thread state.

## State machine

```
              ┌──────► IDLE ◄─────────────┐
              │         │                 │
        stop_party    play_song           │ song ends + auto-DJ off
              │         ▼                 │
              │      FETCHING ──fail──────┤
              │         │                 │
              │         ▼                 │
              └─── PLAYING (dance + chat) ┘
                        │
              skip_song │ song ends + auto-DJ on
                        ▼
                   FETCHING (next track)
```

## Error handling

| Failure | Response |
|---|---|
| yt-dlp fails (geo-block, age gate, deleted) | Structured error → LLM apologizes + asks for another |
| Beat detection empty | Synthetic 120 BPM grid, log warning |
| Audio device missing at start | App refuses to start, daemon shows error in dashboard |
| Motor `nb_error` rises during run | Log, don't stop the party — observable via daemon, transient blips shouldn't kill music |
| OpenAI Realtime disconnects | Auto-reconnect with backoff; music keeps playing through the gap |
| Camera disappears | Disable face tracking + audience push; music + chat continue |

## Out of scope (V1)

- Adapt-to-room energy curves
- Song picks based on audience demographics
- Gradio UI / dashboard panel
- Persistence / history across app restarts
- Wake-word activation
- Multi-user voice routing
- Song seek / skip-to-chorus
- Song-section detection (verse vs chorus)
- Genre-tagged dance pools
- Time-stretching dance moves to match tempo

## Open questions for implementation

- Which `gpt-realtime` model variant exactly — pull from the conversation app's config or pin our own?
- Does the Realtime API on the version we have support image content parts in `conversation.item.create`? Verify before building `take_photo`. Fallback: caption locally with a small VLM and send text only.
- Pi CM4 thermal headroom under all 8 loops + LLM streaming — measure once running, may need to drop camera to 10 fps or push librosa to a separate process.
- yt-dlp on the Pi — check it can install from PyPI into `apps_venv` cleanly during app install (dashboard installer flow).
