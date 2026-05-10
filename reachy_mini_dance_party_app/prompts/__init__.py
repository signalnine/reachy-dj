"""System prompts.

Currently exposes :func:`load_system_prompt` which reads ``system.md``
shipped alongside this package.
"""

from __future__ import annotations

from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent
SYSTEM_PROMPT_PATH = _PROMPTS_DIR / "system.md"


def load_system_prompt() -> str:
    """Return the DJ system prompt as a UTF-8 string."""
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()


__all__ = ["SYSTEM_PROMPT_PATH", "load_system_prompt"]
