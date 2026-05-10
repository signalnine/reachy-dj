# reachy_mini_dance_party_app

Voice-controlled YouTube DJ + dance app for the Reachy Mini wireless edition. Talk to the robot, ask it to play music, and it will fetch a track from YouTube, beat-align dance moves from the official move library, and DJ the room while tracking faces.

## Status

V0.1 / pre-alpha. Built fresh on top of the conversation app's primary/secondary move scheduler, camera worker, face tracker, and OpenAI Realtime voice loop. Pure-logic components have unit-test coverage (113 tests). The full assembled app boots cleanly on real hardware (`tests/integration/test_app_startup.py`) and the OpenAI Realtime session connects without protocol errors.

What is *not* yet hardware-verified end to end:

- Actual song playback through the speaker (sounddevice on the Pi audio sink)
- Beat-aligned dance moves while music is playing
- Tool dispatch from a real spoken request (`"play me Daft Punk"`)
- Face tracking biasing the head while a dance is active
- Auto-DJ handoff between songs

Full design and architecture rationale lives in `docs/plans/2026-05-09-dance-party-app-design.md`. The 18-task implementation plan is in `docs/plans/2026-05-09-dance-party-app.md`.

## Requirements

- Reachy Mini wireless edition with daemon `>= 1.7.1`
- Raspberry Pi CM4 with the Pollen `apps_venv` (`/venvs/apps_venv`) — this is the same venv the conversation app installs into and already has the heavy deps (mediapipe, librosa, opencv, gstreamer bindings, ...)
- An OpenAI API key on a tier that has access to the `gpt-realtime` model
- Network: speaker output via the Reachy Mini's GStreamer audio sink, camera via its IPC source

## Install

From a laptop with SSH key auth to the Pi:

```bash
git clone <this repo>
cd reachy
bash scripts/deploy.sh
```

The deploy script rsyncs the working tree to `/home/pollen/dance_party_app/` and runs `pip install -e` into `/venvs/apps_venv`. Override the target with `HOST=...` and `DEST=...`.

Put your OpenAI key on the Pi at `~/.env` (mode 600):

```bash
ssh pollen@192.168.1.128 'umask 077; echo "OPENAI_API_KEY=sk-..." > ~/.env'
```

The dance-party app is not (yet) registered as an officially-installed Reachy Mini app, so the dashboard's app store will not list it. Start it manually instead.

If the conversation app or any other app is currently running, stop it first so this app can grab the robot-app-lock, the camera, and the audio device:

```bash
ssh pollen@192.168.1.128 'curl -s -X POST http://localhost:8000/api/apps/stop-current-app'
```

## Usage

Start the app with the env loaded:

```bash
ssh pollen@192.168.1.128 'set -a; source ~/.env; set +a; \
  /venvs/apps_venv/bin/python -m reachy_mini_dance_party_app.main'
```

You should see `ReachyMiniDancePartyApp running` in the log. Then talk to the robot ("play me some Daft Punk", "play something else", "skip", "stop"). The DJ persona is in `reachy_mini_dance_party_app/prompts/system.md`.

Stop with Ctrl-C / SIGTERM. The shutdown hook resets the head to neutral and releases the audio device.

## Voice tools

The LLM has access to these tools (full schemas in `reachy_mini_dance_party_app/tools/`):

- `play_song` — search YouTube, fetch audio, analyze beats, start synchronized music + dance
- `skip_song` — stop the current track and advance (or stop if no queue)
- `stop_dance` — keep the music, freeze the dance moves
- `stop_party` — stop everything: music, dance, DJ idle
- `set_volume` — speaker volume 0-100
- `set_face_tracking` — toggle secondary head bias toward the largest face
- `look_at` — glance at a normalized image-coordinate point
- `look_at_audience` — orient toward the largest detected face
- `move_head` — explicit pitch/yaw/roll over a duration
- `play_emotion` — play a named emotion animation from the dance library
- `take_photo` — capture a still and return base64 (for the LLM to "see")

In addition, an audience-summary system event is auto-pushed to the model on a timer with face count + dominant face position.

## Troubleshooting

**Daemon error / motor bus down / `nb_error` climbing.** Restart the daemon:

```bash
ssh pollen@192.168.1.128 'curl -s -X POST http://localhost:8000/api/daemon/restart'
```

If that doesn't clear it, power-cycle and check the motor cable.

**`yt-dlp` extraction failing** (YouTube changes break it). Update inside `apps_venv`:

```bash
ssh pollen@192.168.1.128 '/venvs/apps_venv/bin/pip install --upgrade yt-dlp'
```

**OpenAI Realtime rate limit / 429.** The session reconnect loop will back off, but if you're getting hit hard, drop the session (Ctrl-C the app) and wait. Make sure you're on a tier with `gpt-realtime` quota.

**Camera unavailable / "media held by another process".** Stop the conversation app to release the IPC camera and audio sink:

```bash
ssh pollen@192.168.1.128 'curl -s -X POST http://localhost:8000/api/apps/stop-current-app'
```

**Motors stuck in some pose after a crash.** Set them compliant:

```bash
ssh pollen@192.168.1.128 'curl -s -X POST http://localhost:8000/api/motors/set_mode/disabled'
```

## Architecture

See `docs/plans/2026-05-09-dance-party-app-design.md` for the full design. One sentence per subsystem:

- **Music** (`music/`) — `youtube.py` (yt-dlp wrapper with cache), `analysis.py` (librosa beat-track), `beat.py` (BeatGrid for beat-time queries), `playback.py` (sounddevice engine with monotonic position clock), `mixer.py` (music + TTS mix with ducking).
- **Dance** (`dance/`) — `picker.py` (beat-fit scoring over the move library), `library_dancer.py` (worker thread that schedules moves on bar boundaries via the BeatGrid).
- **Vision** (`vision/`) — `camera_worker.py` lifts the conv app's Picamera/GStreamer worker, `face_tracker.py` runs mediapipe and produces secondary head-bias offsets, `audience.py` periodically pushes an audience-summary system event to the LLM.
- **Voice** (`voice/openai_realtime.py`) — WebSocket session to the Realtime API with reconnect/backoff, tool-call dispatch, TTS chunk callback, and thread-safe `inject_system_event` / `inject_image` for out-of-band context.
- **Tools** (`tools/`) — `Tool` registry (name, description, JSON-schema parameters, handler) consumed by both the session.update payload and the dispatch table. Each tool's handler operates on a shared `AppContext` (DJ, dancer, mixer, playback, camera, face tracker, move queue, fetcher, analyzer, http client).
- **Moves** (`moves.py`) — primary/secondary scheduler lifted from the conv app; sole writer to `robot.set_target`.
- **Prompts** (`prompts/system.md`) — the DJ persona system prompt loaded via `importlib.resources`.
- **Top-level** (`main.py`) — `ReachyMiniDancePartyApp` assembles every component on an `ExitStack` so shutdown unwinds in reverse order, installs SIGINT/SIGTERM handlers, and blocks on a stop-event.

## License

TBD — same as the Reachy Mini SDK upstream until specified otherwise.
