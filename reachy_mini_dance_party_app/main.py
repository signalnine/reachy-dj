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
