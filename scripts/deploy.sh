#!/usr/bin/env bash
# Deploy reachy_mini_dance_party_app to a Reachy Mini Pi.
#
# Usage:
#   bash scripts/deploy.sh
#
# Override with env vars:
#   HOST=pollen@otherhost DEST=/some/path bash scripts/deploy.sh
#
# What it does:
#   1. rsync the working tree to $HOST:$DEST (excluding venv, caches, git, claude state)
#   2. pip install -e $DEST into the daemon's apps_venv on the Pi
#
# Notes:
#   - This does NOT auto-start the app. The app is pip-installed into apps_venv
#     but is not (yet) registered as an officially-installed app in the daemon's
#     app store, so the dashboard's app launcher may not see it.
#     Start it manually instead:
#       ssh $HOST 'set -a; source ~/.env; set +a; \
#         /venvs/apps_venv/bin/python -m reachy_mini_dance_party_app.main'
#   - Make sure the conversation app (or any other app holding robot-app-lock,
#     camera, or the audio device) is stopped first:
#       ssh $HOST 'curl -s http://localhost:8000/api/apps/stop-current-app'

set -euo pipefail

HOST=${HOST:-pollen@192.168.1.128}
DEST=${DEST:-/home/pollen/dance_party_app}

echo ">>> Syncing working tree to $HOST:$DEST"
rsync -av --delete \
  --exclude='.venv' \
  --exclude='__pycache__' \
  --exclude='.git' \
  --exclude='.claude' \
  --exclude='.pytest_cache' \
  --exclude='*.egg-info' \
  ./ "$HOST:$DEST/"

echo ">>> pip install -e $DEST (into apps_venv)"
ssh "$HOST" "/venvs/apps_venv/bin/pip install -e $DEST"

echo ">>> Done. Start manually with:"
echo "    ssh $HOST 'set -a; source ~/.env; set +a; /venvs/apps_venv/bin/python -m reachy_mini_dance_party_app.main'"
