#!/usr/bin/env bash
# flush then probe, sequentially, detached. No pgrep waits - the wave is already provably done.
set -uo pipefail
cd /home/patrickd/glm-5.3-reap
./scripts/flush_after_ship2.sh
echo "[$(date -Is)] chain: flush returned $?; starting probe" >> logs/flush.log
./scripts/probe_offload.sh
