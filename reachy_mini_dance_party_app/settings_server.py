"""Local FastAPI settings UI for the dance party app.

The Reachy Mini dashboard renders ``custom_app_url`` (set on the app's main
class) as a "Settings" link. The daemon parses it from main.py source, so the
literal URL string lives there; this module just serves the FastAPI on the
matching host/port in a background thread.

Endpoints:
    GET  /                 -> HTML page (form + live status)
    GET  /api/status       -> JSON: key set?, current song, DJ state
    POST /api/openai_key   -> body {"key": "sk-..."} writes ~/.env
    POST /api/restart      -> tells the daemon to restart the app
    POST /api/skip         -> skip_song
    POST /api/stop         -> stop_party
"""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel


log = logging.getLogger(__name__)


_HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Reachy Mini Dance Party — Settings</title>
<style>
  :root {
    --bg-1: #1a0033;
    --bg-2: #4a0e7a;
    --accent: #ff3aa6;
    --accent-2: #00e8ff;
    --text: #fff7ff;
    --muted: #c2a8e8;
    --card: rgba(255, 255, 255, 0.08);
    --ok: #b6f4c2;
    --warn: #ffd166;
  }
  html, body { margin: 0; padding: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: linear-gradient(135deg, var(--bg-1), var(--bg-2));
    color: var(--text); min-height: 100vh; }
  .wrap { max-width: 720px; margin: 0 auto; padding: 2rem 1.25rem; }
  h1 { font-size: 2rem; margin: 0 0 0.5rem;
    background: linear-gradient(90deg, var(--accent), var(--accent-2));
    -webkit-background-clip: text; background-clip: text; color: transparent; }
  h2 { font-size: 1.05rem; color: var(--accent); text-transform: uppercase;
    letter-spacing: 0.05em; margin: 0 0 0.9rem; }
  section { background: var(--card); border-radius: 12px; padding: 1.3rem;
    margin-bottom: 1rem; border: 1px solid rgba(255,255,255,0.12); }
  label { display: block; font-size: 0.85rem; color: var(--muted);
    margin-bottom: 0.4rem; }
  input[type=password], input[type=text] {
    width: 100%; padding: 0.6rem 0.8rem; border-radius: 7px;
    border: 1px solid rgba(255,255,255,0.18);
    background: rgba(0,0,0,0.35); color: var(--text);
    font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; font-size: 0.9rem; }
  button {
    appearance: none; padding: 0.55rem 1.1rem; border-radius: 7px;
    font-weight: 600; cursor: pointer; border: 1px solid transparent;
    margin-right: 0.4rem; margin-top: 0.6rem;
  }
  button.primary {
    background: linear-gradient(90deg, var(--accent), var(--accent-2));
    color: #1a0033;
  }
  button.ghost {
    background: transparent; color: var(--text);
    border-color: rgba(255,255,255,0.25);
  }
  .row { display: grid; grid-template-columns: 1fr auto; gap: 0.4rem 1rem; align-items: baseline; }
  .row .k { color: var(--muted); font-size: 0.85rem; }
  .row .v { font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; }
  .ok { color: var(--ok); } .warn { color: var(--warn); }
  .toast { position: fixed; bottom: 1rem; right: 1rem;
    background: rgba(50,200,80,0.25); color: var(--ok);
    border: 1px solid rgba(50,200,80,0.45); padding: 0.6rem 1rem;
    border-radius: 7px; opacity: 0; transition: opacity 0.25s; }
  .toast.show { opacity: 1; }
  .toast.err { background: rgba(255,90,90,0.25); color: #ffc0c0;
    border-color: rgba(255,90,90,0.45); }
  small { color: var(--muted); }
</style>
</head>
<body>
<div class="wrap">
  <h1>Dance Party Settings</h1>
  <p style="color: var(--muted); margin-top: 0;">
    Running on your Reachy Mini. Changes apply immediately (or after a quick app
    restart for the API key).
  </p>

  <section>
    <h2>Status</h2>
    <div class="row" id="status">
      <div class="k">OpenAI API key</div><div class="v" id="s-key">…</div>
      <div class="k">DJ state</div><div class="v" id="s-state">…</div>
      <div class="k">Now playing</div><div class="v" id="s-song">…</div>
      <div class="k">Position</div><div class="v" id="s-pos">…</div>
    </div>
  </section>

  <section>
    <h2>OpenAI API key</h2>
    <label for="key">Paste your <code>sk-...</code> key. Written to <code>~/.env</code> on the robot.</label>
    <input id="key" type="password" autocomplete="off" placeholder="sk-..." />
    <small>The app needs to restart to pick up a new key.</small>
    <div>
      <button class="primary" onclick="saveKey()">Save key</button>
      <button class="ghost" onclick="restartApp()">Restart app</button>
    </div>
  </section>

  <section>
    <h2>Controls</h2>
    <button class="ghost" onclick="skip()">Skip song</button>
    <button class="ghost" onclick="stop()">Stop party</button>
  </section>
</div>
<div id="toast" class="toast"></div>

<script>
  const $ = (id) => document.getElementById(id);
  function toast(msg, err) {
    const t = $("toast");
    t.textContent = msg;
    t.classList.toggle("err", !!err);
    t.classList.add("show");
    setTimeout(() => t.classList.remove("show"), 2500);
  }
  function setKeyStatus(isSet) {
    const el = $("s-key");
    el.textContent = isSet ? "set" : "not set";
    el.className = "v " + (isSet ? "ok" : "warn");
  }
  async function refresh() {
    try {
      const r = await fetch("/api/status");
      const d = await r.json();
      setKeyStatus(!!d.openai_key_set);
      $("s-state").textContent = d.dj_state || "—";
      $("s-song").textContent = d.song_title || "—";
      $("s-pos").textContent = d.song_position_s != null
        ? Math.round(d.song_position_s) + " / " + Math.round(d.song_duration_s || 0) + "s"
        : "—";
    } catch (_) {}
  }
  async function saveKey() {
    const key = $("key").value.trim();
    if (!key) { toast("Paste a key first", true); return; }
    const r = await fetch("/api/openai_key", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({key}),
    });
    if (r.ok) {
      $("key").value = "";
      toast("Saved. Restart app to pick up.");
      refresh();
    } else { toast("Save failed", true); }
  }
  async function restartApp() {
    const r = await fetch("/api/restart", {method: "POST"});
    if (r.ok) toast("Restart requested");
    else toast("Restart failed", true);
  }
  async function skip() {
    const r = await fetch("/api/skip", {method: "POST"});
    toast(r.ok ? "Skipped" : "Skip failed", !r.ok);
    setTimeout(refresh, 400);
  }
  async function stop() {
    const r = await fetch("/api/stop", {method: "POST"});
    toast(r.ok ? "Stopped" : "Stop failed", !r.ok);
    setTimeout(refresh, 400);
  }
  refresh();
  setInterval(refresh, 2000);
</script>
</body>
</html>
"""


class _OpenAIKeyPayload(BaseModel):
    key: str


def build_app(
    *,
    get_dj: Any,
    get_playback: Any,
    on_skip: Any,
    on_stop: Any,
    env_path: Path,
) -> FastAPI:
    """Build the FastAPI settings app with the provided integration callables.

    Args:
        get_dj: Callable returning the current ``DJ`` instance (may be ``None``
            while assembly is still in progress).
        get_playback: Callable returning the current ``PlaybackEngine``.
        on_skip: Callable invoked when the user clicks "Skip song".
        on_stop: Callable invoked when the user clicks "Stop party".
        env_path: Path to the .env file the API key is persisted to (typically
            ``~/.env``).
    """
    app = FastAPI(title="Reachy Mini Dance Party — Settings")

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse(_HTML_PAGE)

    @app.get("/api/status")
    def status() -> dict[str, Any]:
        dj = get_dj()
        playback = get_playback()
        song = getattr(dj, "current", None) if dj is not None else None
        state = (
            dj.state.name.lower()
            if dj is not None and dj.state is not None
            else "idle"
        )
        position = (
            float(playback.playback_time())
            if playback is not None
            else None
        )
        return {
            "openai_key_set": bool(os.environ.get("OPENAI_API_KEY")),
            "dj_state": state,
            "song_title": getattr(song, "title", None),
            "song_duration_s": getattr(song, "duration_s", None),
            "song_position_s": position,
        }

    @app.post("/api/openai_key")
    def set_key(payload: _OpenAIKeyPayload) -> dict[str, Any]:
        key = payload.key.strip()
        if not key.startswith("sk-"):
            raise HTTPException(status_code=400, detail="key must start with sk-")
        # Preserve other lines in ~/.env (e.g. HF_TOKEN); rewrite only the
        # OPENAI_API_KEY line. Don't log the key itself.
        try:
            existing = env_path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            existing = []
        kept = [ln for ln in existing if not ln.startswith("OPENAI_API_KEY=")]
        kept.append(f"OPENAI_API_KEY={key}")
        env_path.write_text("\n".join(kept) + "\n", encoding="utf-8")
        try:
            env_path.chmod(0o600)
        except Exception:  # noqa: BLE001
            pass
        os.environ["OPENAI_API_KEY"] = key
        log.info("OpenAI API key updated via settings UI (written to %s)", env_path)
        return {"ok": True}

    @app.post("/api/restart")
    def restart_app() -> dict[str, Any]:
        try:
            httpx.post(
                "http://localhost:8000/api/apps/stop-current-app", timeout=5.0
            )
            httpx.post(
                "http://localhost:8000/api/apps/start-app/reachy_mini_dance_party_app",
                timeout=5.0,
            )
            return {"ok": True}
        except Exception as exc:  # noqa: BLE001
            log.warning("restart-app request failed: %s", exc)
            return {"ok": False, "error": str(exc)}

    @app.post("/api/skip")
    def skip() -> dict[str, Any]:
        try:
            on_skip()
            return {"ok": True}
        except Exception as exc:  # noqa: BLE001
            log.warning("skip via settings UI failed: %s", exc)
            return {"ok": False, "error": str(exc)}

    @app.post("/api/stop")
    def stop() -> dict[str, Any]:
        try:
            on_stop()
            return {"ok": True}
        except Exception as exc:  # noqa: BLE001
            log.warning("stop via settings UI failed: %s", exc)
            return {"ok": False, "error": str(exc)}

    return app


def start_in_thread(
    app: FastAPI,
    *,
    host: str,
    port: int,
) -> tuple[uvicorn.Server, threading.Thread]:
    """Run the FastAPI on a background thread, return (server, thread).

    ``server.should_exit = True`` requests a graceful shutdown.
    """
    config = uvicorn.Config(app=app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run, daemon=True, name="SettingsServer")
    t.start()
    log.info("SettingsServer running at http://%s:%d", host, port)
    return server, t
