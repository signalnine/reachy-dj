# Reachy Mini Dance-Party App Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use conclave:executing-plans to implement this plan task-by-task.

**Goal:** Build `reachy_mini_dance_party_app` — a voice-controlled YouTube-driven dance app for Reachy Mini, forked from `reachy_mini_conversation_app` v0.6.1.

**Architecture:** Fork-and-prune. Independent package. Reuses the conv app's primary/secondary move scheduler (`moves.py`), camera worker, face tracker, and OpenAI Realtime voice loop. New code is the music pipeline (yt-dlp + librosa + sounddevice), beat-aligned dance scheduler over `reachy_mini_dances_library`, audio mixer with ducking, and an auto-DJ state machine. Deployed as a Reachy Mini app via the daemon's app manager. Full design at `docs/plans/2026-05-09-dance-party-app-design.md`.

**Tech Stack:** Python 3.12, pytest, yt-dlp, librosa, sounddevice, mediapipe, opencv-python, OpenAI Realtime API, `reachy_mini` SDK, `reachy_mini_dances_library`. Target host: Raspberry Pi CM4, Debian trixie, robot at `pollen@192.168.1.128`.

**Test strategy:**
- **Pure-logic units** (BeatGrid, DJ state machine, library picker, audio mixer, ducker) — full TDD with pytest, runs on laptop.
- **IO wrappers with deterministic fixtures** (yt-dlp wrapper with mocked subprocess, librosa wrapper on a 5-sec fixture wav) — unit tests on laptop.
- **Hardware-coupled** (camera worker, face tracker, motor-bound code, sounddevice playback) — integration tests on Pi via SSH; manual verification for visual/auditory correctness.
- **LLM loop** (Realtime session) — mocked at the WebSocket layer for unit tests; manual smoke for the real loop.
- Tests live in `tests/unit/` (laptop) and `tests/integration/` (Pi); CI not in scope for V1.

**Dev loop:** Develop locally, sync to Pi via `rsync -av --delete <repo>/ pollen@192.168.1.128:/home/pollen/dance_party_app/`, then `pip install -e /home/pollen/dance_party_app` into `apps_venv`, then restart via `POST /api/apps/start-app/reachy_mini_dance_party_app`.

---

## Wave 0 — Skeleton

### Task 1: Project skeleton

**Files:**
- Create: `pyproject.toml`
- Create: `reachy_mini_dance_party_app/__init__.py`
- Create: `reachy_mini_dance_party_app/main.py`
- Create: `tests/unit/__init__.py`
- Create: `tests/integration/__init__.py`
- Create: `tests/conftest.py`
- Create: `.gitignore`
- Create: `README.md`

**Dependencies:** none

**Step 1: Write `pyproject.toml`**

```toml
[project]
name = "reachy_mini_dance_party_app"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
  "reachy-mini>=1.7.1",
  "reachy_mini_dances_library",
  "openai>=2.30.0",
  "yt-dlp>=2025.10.0",
  "librosa>=0.11.0",
  "sounddevice>=0.5.0",
  "mediapipe==0.10.14",
  "opencv-python>=4.13.0",
  "numpy",
  "av",
  "httpx",
  "python-dotenv",
]

[project.optional-dependencies]
dev = ["pytest>=8", "pytest-asyncio", "pytest-mock", "ruff"]

[project.entry-points."reachy_mini_apps"]
reachy_mini_dance_party_app = "reachy_mini_dance_party_app.main:ReachyMiniDancePartyApp"

[project.scripts]
reachy-mini-dance-party-app = "reachy_mini_dance_party_app.main:main"

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"
```

**Step 2: Write minimal `main.py` placeholder**

```python
"""Reachy Mini Dance Party App — entry point."""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)

class ReachyMiniDancePartyApp:
    def __init__(self) -> None:
        log.info("ReachyMiniDancePartyApp initialized (skeleton)")

    def run(self) -> None:
        raise NotImplementedError("Wire-up lands in Task 14")

def main() -> None:
    logging.basicConfig(level=logging.INFO)
    ReachyMiniDancePartyApp().run()

if __name__ == "__main__":
    main()
```

**Step 3: Write `__init__.py` and `tests/conftest.py`**

`reachy_mini_dance_party_app/__init__.py`:
```python
"""Reachy Mini Dance Party App."""
__version__ = "0.1.0"
```

`tests/conftest.py`:
```python
import pytest
from pathlib import Path

@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"
```

**Step 4: Write `.gitignore`** (covers Python, venv, audio cache, IDE):
```
__pycache__/
*.py[cod]
*.egg-info/
.venv/
.pytest_cache/
.ruff_cache/
/build/
/dist/
/tests/fixtures/*.wav
!/tests/fixtures/.gitkeep
.DS_Store
.idea/
.vscode/
.env
```

**Step 5: Verify install + import works**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -c "from reachy_mini_dance_party_app.main import ReachyMiniDancePartyApp; print(ReachyMiniDancePartyApp())"
pytest --collect-only
```
Expected: install succeeds, import prints the class repr, pytest reports "0 tests collected" (no errors).

**Step 6: Commit**

```bash
git add pyproject.toml reachy_mini_dance_party_app/ tests/ .gitignore README.md
git commit -m "chore: project skeleton with entry point + dev deps"
```

---

## Wave 1 — Pure-logic units (parallel-safe)

### Task 2: BeatGrid

**Files:**
- Create: `reachy_mini_dance_party_app/music/__init__.py`
- Create: `reachy_mini_dance_party_app/music/beat.py`
- Create: `tests/unit/test_beat_grid.py`

**Dependencies:** Task 1

**Step 1: Write the failing tests (`tests/unit/test_beat_grid.py`)**

```python
import numpy as np
import pytest
from reachy_mini_dance_party_app.music.beat import BeatGrid

def test_next_beat_at_returns_first_future_beat():
    grid = BeatGrid(tempo=120.0, beat_times=np.array([0.0, 0.5, 1.0, 1.5, 2.0]))
    assert grid.next_beat_at(0.6, n=1) == [1.0]

def test_next_beat_at_n_returns_n_beats():
    grid = BeatGrid(tempo=120.0, beat_times=np.array([0.0, 0.5, 1.0, 1.5, 2.0]))
    assert grid.next_beat_at(0.0, n=3) == [0.5, 1.0, 1.5]

def test_next_beat_at_past_end_returns_empty():
    grid = BeatGrid(tempo=120.0, beat_times=np.array([0.0, 0.5, 1.0]))
    assert grid.next_beat_at(2.0, n=3) == []

def test_beats_per_second_from_tempo():
    grid = BeatGrid(tempo=120.0, beat_times=np.array([0.0]))
    assert grid.beats_per_second == pytest.approx(2.0)

def test_beats_in_window_returns_count_at_tempo():
    grid = BeatGrid(tempo=120.0, beat_times=np.array([0.0]))
    assert grid.beats_in_window(2.0) == 4   # 4 beats in 2.0s at 120 BPM

def test_synthetic_grid_factory_for_empty_detection():
    grid = BeatGrid.synthetic(tempo=120.0, duration=10.0)
    assert len(grid.beat_times) == 20
    assert grid.beat_times[0] == 0.0
    assert grid.beat_times[-1] == pytest.approx(9.5)
```

**Step 2: Run tests, confirm fail**

```bash
pytest tests/unit/test_beat_grid.py -v
```
Expected: ImportError or 6 failed.

**Step 3: Implement `BeatGrid`**

```python
# reachy_mini_dance_party_app/music/beat.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class BeatGrid:
    tempo: float
    beat_times: np.ndarray  # seconds, monotonically increasing

    @property
    def beats_per_second(self) -> float:
        return self.tempo / 60.0

    def beats_in_window(self, duration_s: float) -> int:
        return int(round(duration_s * self.beats_per_second))

    def next_beat_at(self, t: float, n: int = 1) -> list[float]:
        idx = int(np.searchsorted(self.beat_times, t, side="right"))
        return self.beat_times[idx : idx + n].tolist()

    @classmethod
    def synthetic(cls, tempo: float, duration: float) -> "BeatGrid":
        bps = tempo / 60.0
        beats = np.arange(0.0, duration, 1.0 / bps)
        return cls(tempo=tempo, beat_times=beats)
```

**Step 4: Run tests, confirm pass**

```bash
pytest tests/unit/test_beat_grid.py -v
```
Expected: 6 passed.

**Step 5: Commit**

```bash
git add reachy_mini_dance_party_app/music/ tests/unit/test_beat_grid.py
git commit -m "feat(music): BeatGrid for beat-time queries with synthetic fallback"
```

---

### Task 3: DJ state machine

**Files:**
- Create: `reachy_mini_dance_party_app/dj.py`
- Create: `tests/unit/test_dj.py`

**Dependencies:** Task 1

**Step 1: Write the failing tests (`tests/unit/test_dj.py`)**

```python
import pytest
from reachy_mini_dance_party_app.dj import DJ, DJState, SongInfo

def test_initial_state_is_idle():
    assert DJ().state is DJState.IDLE

def test_play_song_transitions_idle_to_fetching():
    dj = DJ()
    dj.request_song("test query")
    assert dj.state is DJState.FETCHING
    assert dj.pending_query == "test query"

def test_song_fetched_transitions_fetching_to_playing():
    dj = DJ()
    dj.request_song("test")
    info = SongInfo(title="X", duration_s=120.0, url="u", path="/tmp/x.wav", query="test")
    dj.song_fetched(info)
    assert dj.state is DJState.PLAYING
    assert dj.current is info

def test_song_ends_returns_to_idle_when_auto_dj_off():
    dj = DJ(auto_dj=False)
    dj.request_song("a"); dj.song_fetched(_song("a"))
    dj.song_ended()
    assert dj.state is DJState.IDLE

def test_song_ends_triggers_fetch_when_auto_dj_on():
    dj = DJ(auto_dj=True)
    dj.request_song("a"); dj.song_fetched(_song("a"))
    dj.set_pending_auto_dj_query("next track")
    dj.song_ended()
    assert dj.state is DJState.FETCHING
    assert dj.pending_query == "next track"

def test_history_records_played_songs():
    dj = DJ()
    dj.request_song("a"); dj.song_fetched(_song("a")); dj.song_ended()
    dj.request_song("b"); dj.song_fetched(_song("b")); dj.song_ended()
    assert [s.query for s in dj.history] == ["a", "b"]

def test_stop_party_returns_to_idle_from_any_state():
    dj = DJ()
    dj.request_song("a"); dj.song_fetched(_song("a"))
    dj.stop_party()
    assert dj.state is DJState.IDLE
    assert dj.current is None

def test_fetch_failure_returns_to_idle_with_error():
    dj = DJ()
    dj.request_song("bad")
    dj.fetch_failed("yt-dlp 410")
    assert dj.state is DJState.IDLE
    assert dj.last_error == "yt-dlp 410"

def _song(q: str) -> SongInfo:
    return SongInfo(title=q.upper(), duration_s=60.0, url=f"u/{q}", path=f"/tmp/{q}.wav", query=q)
```

**Step 2: Run, confirm fail.** `pytest tests/unit/test_dj.py -v`

**Step 3: Implement `dj.py`**

```python
# reachy_mini_dance_party_app/dj.py
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

class DJState(Enum):
    IDLE = auto()
    FETCHING = auto()
    PLAYING = auto()

@dataclass(frozen=True)
class SongInfo:
    title: str
    duration_s: float
    url: str
    path: str
    query: str

@dataclass
class DJ:
    auto_dj: bool = True
    state: DJState = DJState.IDLE
    current: Optional[SongInfo] = None
    pending_query: Optional[str] = None
    last_error: Optional[str] = None
    history: list[SongInfo] = field(default_factory=list)
    _next_auto_query: Optional[str] = None

    def request_song(self, query: str) -> None:
        self.pending_query = query
        self.state = DJState.FETCHING
        self.last_error = None

    def song_fetched(self, info: SongInfo) -> None:
        self.current = info
        self.pending_query = None
        self.state = DJState.PLAYING

    def fetch_failed(self, error: str) -> None:
        self.last_error = error
        self.pending_query = None
        self.state = DJState.IDLE

    def song_ended(self) -> None:
        if self.current is not None:
            self.history.append(self.current)
        self.current = None
        if self.auto_dj and self._next_auto_query:
            self.request_song(self._next_auto_query)
            self._next_auto_query = None
        else:
            self.state = DJState.IDLE

    def stop_party(self) -> None:
        self.current = None
        self.pending_query = None
        self._next_auto_query = None
        self.state = DJState.IDLE

    def set_pending_auto_dj_query(self, query: str) -> None:
        self._next_auto_query = query
```

**Step 4: Run, confirm pass.** **Step 5: Commit.**

```bash
git add reachy_mini_dance_party_app/dj.py tests/unit/test_dj.py
git commit -m "feat(dj): state machine for song lifecycle + auto-DJ"
```

---

### Task 4: Library move picker

**Files:**
- Create: `reachy_mini_dance_party_app/dance/__init__.py`
- Create: `reachy_mini_dance_party_app/dance/picker.py`
- Create: `tests/unit/test_picker.py`

**Dependencies:** Task 2 (BeatGrid)

**Step 1: Write tests** — `picker.choose_move(catalog, tempo, exclude_recent, fit_window_s=0.15) -> MoveSpec`. Cases: prefers move whose duration is closest to integer-beat multiple; falls back to "fill" when no fit exists; excludes recently-played; weighted-random across multiple equally-good fits (verify with seeded RNG that distribution is non-degenerate over 100 samples).

**Step 2: Run, confirm fail.**

**Step 3: Implement `picker.py`** — `MoveSpec` dataclass `(name: str, duration_s: float)`, function `choose_move(catalog, tempo, exclude_recent, fit_window_s=0.15, rng=None)` that:
1. computes target beat count for each candidate as `round(d * tempo/60)`,
2. computes residual `abs(d - target_beats * 60/tempo)`,
3. filters by `residual <= fit_window_s` and `name not in exclude_recent`,
4. weighted-random pick (weight = `1/(residual+0.05)`); returns `MoveSpec(name="__fill__", duration_s=60.0/tempo)` if no candidates pass.

**Step 4: Run, confirm pass.** **Step 5: Commit.**

```bash
git commit -am "feat(dance): library move picker with beat-fit scoring"
```

---

### Task 5: Audio mixer + ducker (pure logic)

**Files:**
- Create: `reachy_mini_dance_party_app/music/mixer.py`
- Create: `tests/unit/test_mixer.py`

**Dependencies:** Task 1

**Step 1: Write tests** — `Mixer` class operating on numpy float32 buffers. Cases: mixes two equal-length streams sample-wise; applies per-stream gains; `Ducker.update(speech_active=True)` ramps `music_gain` from 1.0 → 0.25 over the configured time; ramps back on `speech_active=False` after hangover; never produces samples outside [-1, 1] after clipping is applied.

**Step 2-5:** TDD as above. Implement `Ducker` as a small state machine with monotonic-clock-driven envelope (no threading, fully testable). Audio I/O integration happens in Task 9.

```bash
git commit -am "feat(music): mixer + ducker pure-logic with envelope tests"
```

---

### Task 6: Lift `moves.py` from conv app

**Files:**
- Create: `reachy_mini_dance_party_app/moves.py` (copied from Pi)
- Create: `tests/integration/test_moves_smoke.py`

**Dependencies:** Task 1; **requires Pi access** to fetch source.

**Step 1: Pull the source from the Pi**

```bash
ssh pollen@192.168.1.128 'cat /venvs/apps_venv/lib/python3.12/site-packages/reachy_mini_conversation_app/moves.py' \
  > reachy_mini_dance_party_app/moves.py
```

**Step 2: Adjust imports** — change any `from reachy_mini_conversation_app...` imports inside `moves.py` to relative imports within our package, or to the canonical `reachy_mini` SDK path.

**Step 3: Write the smoke test (`tests/integration/test_moves_smoke.py`)**

Marks itself with `pytest.mark.requires_robot`; instantiates the move worker against a real `ReachyMini` client pointed at `localhost:8000`; enqueues a no-op (current pose) "move" with duration 1.5s; verifies the worker thread didn't raise and the daemon's `nb_error` counter didn't increment.

**Step 4: Run on the Pi**

```bash
rsync -av --delete --exclude='.venv' --exclude='__pycache__' \
  ./ pollen@192.168.1.128:/home/pollen/dance_party_app/
ssh pollen@192.168.1.128 'cd /home/pollen/dance_party_app && /venvs/apps_venv/bin/pip install -e . && /venvs/apps_venv/bin/pytest tests/integration/test_moves_smoke.py -v'
```
Expected: 1 passed, no motor errors logged.

**Step 5: Commit**

```bash
git add reachy_mini_dance_party_app/moves.py tests/integration/test_moves_smoke.py
git commit -m "feat(moves): lift primary/secondary scheduler from conv app + smoke test"
```

---

### Task 7: Lift camera worker + face tracker

**Files:**
- Create: `reachy_mini_dance_party_app/vision/__init__.py`
- Create: `reachy_mini_dance_party_app/vision/camera_worker.py`
- Create: `reachy_mini_dance_party_app/vision/face_tracker.py`
- Create: `tests/integration/test_vision_smoke.py`

**Dependencies:** Task 1; **requires Pi access** to fetch source.

**Step 1: Pull `camera_worker.py` and `vision/local_vision.py`** from the Pi (same SCP pattern as Task 6). Save the face-tracking secondary-move logic from `tools/head_tracking.py` into `vision/face_tracker.py`.

**Step 2: Adjust imports** to be self-contained inside our package.

**Step 3: Write smoke test** — starts the camera worker for 2 seconds on the Pi, asserts at least one frame was captured (shape > 0). Marks `requires_robot`.

**Step 4: Run on Pi.** **Step 5: Commit.**

```bash
git commit -m "feat(vision): lift camera worker + face tracker"
```

---

## Wave 2 — Music IO (parallel-safe within wave)

### Task 8: yt-dlp wrapper

**Files:**
- Create: `reachy_mini_dance_party_app/music/youtube.py`
- Create: `tests/unit/test_youtube.py`
- Create: `tests/integration/test_youtube_real.py`

**Dependencies:** Task 1

**Step 1: Unit tests with mocked yt-dlp** — `YouTubeFetcher.fetch(query)` should:
- Hash the query, return cached path if `<hash>.wav` exists.
- On first fetch, call `yt-dlp` Python API with `ytsearch1:<query>`, audio-only, postprocess to wav.
- Write a sibling `<hash>.info.json` with title, duration, url, query.
- On `yt-dlp` error → raise `FetchError` with stderr message.

Use `pytest-mock` to patch `yt_dlp.YoutubeDL` — assert correct format string and outtmpl, no real network.

**Step 2-3-4:** Standard TDD. Implementation uses `yt_dlp.YoutubeDL(opts).extract_info(...)`, with `format='bestaudio/best'`, `postprocessors=[{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav'}]`, `outtmpl=f'{cache_dir}/{hash}.%(ext)s'`. Cache dir is `/tmp/reachy_dance_party/` (configurable).

**Step 5: Integration test (`test_youtube_real.py`)** — marked `slow` and `requires_network`. Fetches a known short Creative-Commons-licensed test track from YouTube, verifies the wav file appears and is >100KB. Skipped by default; run with `pytest -m slow` when desired.

**Step 6: Commit**

```bash
git commit -am "feat(music): yt-dlp wrapper with cache + structured errors"
```

---

### Task 9: librosa beat-analysis wrapper

**Files:**
- Create: `reachy_mini_dance_party_app/music/analysis.py`
- Create: `tests/unit/test_analysis.py`
- Create: `tests/fixtures/120bpm_click.wav` (5-sec click track at 120 BPM, generated)
- Create: `tests/fixtures/silence.wav` (5-sec silence)

**Dependencies:** Task 1, Task 2

**Step 1: Generate fixture wav files** — small script in the test setup creates `120bpm_click.wav` (sine clicks every 0.5s) and `silence.wav` programmatically into `tests/fixtures/`. Add to `conftest.py` as a session-scoped fixture if not already on disk.

**Step 2: Write tests** — `analyze_beats(wav_path) -> BeatGrid`:
- On the click track: returns `tempo` within ±5 BPM of 120 and 9-11 detected beats.
- On silence: returns `BeatGrid.synthetic(120.0, duration=5.0)` (the empty-detection fallback).
- Caches result alongside the wav (`<hash>.beats.npy` + `<hash>.tempo.txt`); second call doesn't re-analyze.

**Step 3-5:** TDD. Implementation calls `librosa.beat.beat_track(y, sr=22050, units='time')`. **Step 6: Commit.**

```bash
git commit -am "feat(music): librosa beat analysis with cache + empty-fallback"
```

---

### Task 10: sounddevice playback

**Files:**
- Create: `reachy_mini_dance_party_app/music/playback.py`
- Create: `tests/unit/test_playback.py`
- Create: `tests/integration/test_playback_real.py`

**Dependencies:** Task 1, Task 5 (Mixer)

**Step 1: Unit tests with a fake `OutputStream`** — `PlaybackEngine` should expose `load(wav_path)`, `start()`, `stop()`, `playback_time()` (returns frames-played / sr). Verify the callback consumes the buffer in order; verify `playback_time()` is monotonic across two callback ticks.

**Step 2-4:** TDD with a stub stream that drives the callback directly.

**Step 5: Integration test on Pi** — plays `tests/fixtures/120bpm_click.wav` via the real `sounddevice.OutputStream`, asserts `playback_time()` advances by ≥4.0s during a 5-second wait. Marked `requires_robot`.

**Step 6: Commit**

```bash
git commit -am "feat(music): sounddevice playback engine with monotonic position clock"
```

---

## Wave 3 — Dance scheduler thread

### Task 11: LibraryDancer thread

**Files:**
- Create: `reachy_mini_dance_party_app/dance/library_dancer.py`
- Create: `tests/unit/test_library_dancer.py`

**Dependencies:** Task 4 (picker), Task 6 (moves), Task 10 (playback time source)

**Step 1: Tests with a fake playback clock + fake move queue** — drive the dancer for 4 simulated seconds at 120 BPM, assert it enqueued ≥4 moves at beat-aligned times (within 50ms of integer-beat boundaries), assert no two consecutive moves are the same name.

**Step 2-4:** TDD. Implement `LibraryDancer` as a `threading.Thread` with `start()` / `stop()` / `set_grid(grid)` / `set_clock(callable returning playback time)` and an injected move-queue. The thread loop is the algorithm from the design doc Section 3 (look ahead 4 beats, pick a move whose duration fits within the window, sleep until ~50ms before the target downbeat, enqueue, repeat).

**Step 5: Commit**

```bash
git commit -am "feat(dance): LibraryDancer thread, beat-aligned library moves"
```

---

## Wave 4 — Tools and voice loop

### Task 12: LLM tool implementations

**Files:**
- Create: `reachy_mini_dance_party_app/tools/__init__.py`
- Create: `reachy_mini_dance_party_app/tools/play_song.py`
- Create: `reachy_mini_dance_party_app/tools/skip_song.py`
- Create: `reachy_mini_dance_party_app/tools/stop_party.py`
- Create: `reachy_mini_dance_party_app/tools/set_volume.py`
- Create: `reachy_mini_dance_party_app/tools/take_photo.py`
- Create: `reachy_mini_dance_party_app/tools/look_at_audience.py`
- Create: `reachy_mini_dance_party_app/tools/set_face_tracking.py`
- Create: `reachy_mini_dance_party_app/tools/move_head.py`
- Create: `reachy_mini_dance_party_app/tools/play_emotion.py`
- Create: `reachy_mini_dance_party_app/tools/look_at.py`
- Create: `reachy_mini_dance_party_app/tools/stop_dance.py`
- Create: `reachy_mini_dance_party_app/tools/registry.py`
- Create: `tests/unit/test_tools.py`

**Dependencies:** Tasks 3, 6, 7, 8, 9, 10, 11

**Step 1: Write `registry.py` with tool-schema dataclass**

```python
from dataclasses import dataclass
from typing import Callable, Any
@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict     # JSON-schema dict
    handler: Callable[[dict], Any]
def all_tools(ctx) -> list[Tool]: ...
```

`ctx` is an `AppContext` carrying references to DJ, dancer, mixer, camera worker, etc. Each tool module exposes `make(ctx) -> Tool`.

**Step 2: Tests** — for each tool, verify (a) JSON schema is well-formed against `jsonschema.Draft202012Validator`; (b) handler called with bad params raises `ToolError`; (c) handler called with good params triggers the right state change on the injected `ctx` mocks.

**Step 3: Implement each tool module** — small (~30 LOC each). `play_song.handler` calls `ctx.dj.request_song(query)` and dispatches an async fetch; `set_volume.handler` calls `httpx.post(...)` against the dashboard; `take_photo.handler` grabs a frame from `ctx.camera_worker` and base64-encodes; etc.

**Step 4: Run all tests, confirm pass.** **Step 5: Commit.**

```bash
git commit -am "feat(tools): LLM tool registry + 12 tool implementations + schema tests"
```

---

### Task 13: OpenAI Realtime voice loop

**Files:**
- Create: `reachy_mini_dance_party_app/voice/__init__.py`
- Create: `reachy_mini_dance_party_app/voice/openai_realtime.py`
- Create: `reachy_mini_dance_party_app/prompts/__init__.py`
- Create: `reachy_mini_dance_party_app/prompts/system.md`
- Create: `tests/unit/test_realtime_loop.py`

**Dependencies:** Task 12; **requires Pi access** to lift the conv app's working file.

**Step 1: Pull `openai_realtime.py` from the conv app and prune** — keep the WebSocket session lifecycle, audio in/out streaming, tool-call dispatch; remove the conv app's persona scaffolding and tool wiring. Replace tool list with `all_tools(ctx)` from Task 12.

**Step 2: Write the system prompt (`prompts/system.md`)** — 200-400 words, DJ persona; instructions to use `take_photo`/`look_at_audience` sparingly; auto-DJ behavior (when prompted with a "track ending" event, pick something complementary and call `play_song`); how to acknowledge audience changes naturally; how to handle tool errors.

**Step 3: Tests with mocked WebSocket** — verify session boots with our system prompt + tool schemas; verify tool-call frames trigger the right tool handler; verify reconnect-with-backoff on connection drop.

**Step 4: Run unit tests.** Manual integration verification deferred to Task 17.

**Step 5: Commit.**

```bash
git commit -am "feat(voice): OpenAI Realtime loop with DJ persona + tool dispatch"
```

---

## Wave 5 — Audience push + audio mixer wire-up

### Task 14: Audience summary push

**Files:**
- Create: `reachy_mini_dance_party_app/vision/audience.py`
- Create: `tests/unit/test_audience.py`

**Dependencies:** Task 7 (camera worker), Task 13 (Realtime session inject)

**Step 1: Tests** — `AudiencePush` runs on a fake clock + fake camera frame source. Cases:
- Pushes a summary every 8s (configurable).
- Edge-pushes immediately on face-count delta.
- Summary contains `n_faces, dominant_centered, smiles, since_last`.
- Push handler is called with the right shape.

**Step 2-4:** TDD. Implement as a thread; injects via a `push_callable` (the realtime session passes its `conversation.item.create` wrapper).

**Step 5: Commit.**

```bash
git commit -am "feat(vision): audience summary push timer with edge + cadence triggers"
```

---

### Task 15: Wire mixer into playback path

**Files:**
- Modify: `reachy_mini_dance_party_app/music/playback.py`
- Modify: `reachy_mini_dance_party_app/voice/openai_realtime.py`
- Create: `tests/unit/test_playback_with_mixer.py`

**Dependencies:** Task 5, Task 10, Task 13

**Step 1: Tests** — feed the mixer simultaneous music + TTS chunks via injected fake streams; verify ducker activates on TTS chunk arrival, ducks music to 0.25, restores after 500ms hangover. Run for 2 seconds of simulated audio, assert mixed output sample sums match expected envelope.

**Step 2-4:** Refactor `PlaybackEngine` to take a `Mixer` and a TTS-chunk inlet (`feed_tts_chunk(bytes)`). Update Realtime loop to push TTS into the mixer rather than to its own stream.

**Step 5: Commit.**

```bash
git commit -am "feat(audio): unified mixer + ducking for music and TTS"
```

---

## Wave 6 — Wire-up and deployment

### Task 16: `main.py` orchestration

**Files:**
- Modify: `reachy_mini_dance_party_app/main.py`
- Create: `tests/integration/test_app_startup.py`

**Dependencies:** All prior tasks

**Step 1: Implement `ReachyMiniDancePartyApp.run()`** — assemble:
1. `ReachyMini()` SDK client → `MoveWorker` (start)
2. `CameraWorker` (start)
3. `FaceTracker` (start; secondary offset hooked into MoveWorker)
4. `PlaybackEngine` (start, idle)
5. `LibraryDancer` (constructed, idle)
6. `DJ` (state)
7. `OpenAIRealtimeSession` (start, with tools and audience push wired)
8. `AudiencePush` (start)
9. Block until shutdown signal; on shutdown, stop in reverse order.

**Step 2: Integration test** — starts the app on the Pi, waits 5s, asserts all threads are alive and `daemon_status` reports `state: running`. Stops cleanly. Marked `requires_robot`.

**Step 3: Run on Pi.**

**Step 4: Commit.**

```bash
git commit -am "feat(main): assemble all threads + lifecycle"
```

---

### Task 17: Deploy + end-to-end smoke

**Files:**
- Create: `scripts/deploy.sh`
- Modify: `README.md` (deployment + run instructions)

**Dependencies:** Task 16

**Step 1: Write `scripts/deploy.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail
HOST=${HOST:-pollen@192.168.1.128}
DEST=${DEST:-/home/pollen/dance_party_app}
rsync -av --delete --exclude='.venv' --exclude='__pycache__' --exclude='.git' \
  ./ "$HOST:$DEST/"
ssh "$HOST" "/venvs/apps_venv/bin/pip install -e $DEST"
ssh "$HOST" "curl -s -X POST http://localhost:8000/api/apps/stop-current-app || true"
ssh "$HOST" "curl -s -X POST http://localhost:8000/api/apps/start-app/reachy_mini_dance_party_app"
```

**Step 2: Run `bash scripts/deploy.sh`**, expect: rsync succeeds, pip install succeeds, app starts.

**Step 3: Verify via dashboard API**

```bash
curl -s http://192.168.1.128:8000/api/apps/current-app-status | python3 -m json.tool
```
Expected: `name: reachy_mini_dance_party_app, state: running`.

**Step 4: Manual smoke test** — speak into the robot's mic: "play me Daft Punk Around the World." Watch / listen for:
- LLM responds in voice ("let me grab that…")
- Music starts within ~15s
- Robot dances on the beat
- Head tracks your face
- After ~1 min, ask "play something else" and verify smooth handoff
- Say "stop" → music stops, robot returns to idle

**Step 5: Commit + tag**

```bash
git add scripts/deploy.sh README.md
git commit -m "chore: deploy script + smoke-test instructions"
git tag v0.1.0
```

---

## Wave 7 — Documentation

### Task 18: README + troubleshooting

**Files:** Modify: `README.md`

**Dependencies:** Task 17

Sections to write:
- What this is, what it does
- Requirements (Reachy Mini wireless + daemon ≥1.7.1, OpenAI key)
- Install (clone, deploy with `scripts/deploy.sh`)
- Usage (start via dashboard or curl, voice commands)
- Troubleshooting (motor errors → check `nb_error`, audio device missing, yt-dlp updates, OpenAI rate limits)
- Architecture pointer to `docs/plans/2026-05-09-dance-party-app-design.md`
- License

**Commit:** `git commit -am "docs: README with install + usage + troubleshooting"`

---

## Plan validation

Multi-agent consensus is unavailable in this environment (no API keys). Plan goes to execution as-is. Risks I'd want a second opinion on if consensus were available:

- **Audio mixer ownership** — Realtime API may want its own audio stream for full-duplex VAD. If so, Task 15 needs to flip: keep Realtime's stream, route music *into* it instead of the other way around.
- **Pi thermal headroom** — 8 threads + librosa + LLM streaming on a CM4 may throttle. Monitor `vcgencmd get_throttled` after first end-to-end run; if non-zero, drop camera fps or push librosa to a subprocess.
- **yt-dlp on the Pi** — Debian trixie + Python 3.12 + yt-dlp is normally fine, but the dashboard's app installer may sandbox in a way that blocks PyPI installs of yt-dlp. Test early in Task 8.
- **Realtime image-part support** — Task 12's `take_photo` assumes the Realtime API accepts image content parts. Verify with a 1-line probe before relying on it; fallback is local captioning + text injection.

These are noted as "open questions" in the design doc.

---

## Execution

Plan complete and saved to `docs/plans/2026-05-09-dance-party-app.md`. 18 tasks across 7 waves; pure-logic tasks within a wave can be parallelized.

Two execution options:

**1. Subagent-Driven (this session)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Parallel Session (separate)** — Open a new session with `executing-plans`, batch execution with checkpoints.

Pick when ready.
