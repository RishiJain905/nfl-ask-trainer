#!/usr/bin/env bash
# Run the dashboard feed for one or more runs (inside WSL). Arguments: run names, then options.
#   bash scripts/dashboard_feed.sh smoke-5 gsmoke-1 -- --wandb
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=/mnt/d/nfl-ml-data/ask/finetune
runs=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do runs+=("$1"); shift; done
[ "${1:-}" = "--" ] && shift
for r in "${runs[@]}"; do
  ~/venvs/nfl-ask-trainer/bin/python scripts/dashboard_feed.py --run "$ROOT/$r" "$@" 2>/dev/null
done
