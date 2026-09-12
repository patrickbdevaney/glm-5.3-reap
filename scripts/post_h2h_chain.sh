#!/usr/bin/env bash
# Sequenced GPU work after the head-to-head: gate the N_ROUTED change FIRST (it decides whether
# the dsv4 arm is believable at all), then pilot the untruncation at the ceiling. Both stages
# serve ~100 GB models, so they must never overlap -- that is the whole reason they are one script
# rather than two units racing for :8080.
set -u
cd "$HOME/glm-5.3-reap"
LOG=logs/post_h2h_chain.log
echo "[$(date -Is)] chain start" >> "$LOG"
bash scripts/gate_n_routed.sh >> "$LOG" 2>&1
GRC=$?
echo "[$(date -Is)] gate_n_routed rc=$GRC" >> "$LOG"
if [ "$GRC" -ne 0 ]; then
  echo "[$(date -Is)] STOPPING: the N_ROUTED gate failed, so the dsv4 arm is void and there is" >> "$LOG"
  echo "[$(date -Is)] nothing to untruncate until it is re-measured on a gated binary." >> "$LOG"
  exit 1
fi
bash scripts/untruncate.sh >> "$LOG" 2>&1
echo "[$(date -Is)] untruncate pilot rc=$? -- chain done" >> "$LOG"
