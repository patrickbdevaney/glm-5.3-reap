#!/usr/bin/env bash
# Wait for the seqlen ladder to finish, then run the s03 equivalence gate.
# Chained rather than run concurrently: two streaming passes over the same shards wedged this
# box three times, and the equivalence gate refuses to start while the ladder holds the GPU.
set -u
cd /home/patrickd/glm-5.3-reap
LOG=logs/after_ladder.log
say(){ echo "[$(date -Is)] after-ladder: $*" | tee -a "$LOG"; }

say "waiting for ladder orchestrator PID 18057"
while kill -0 18057 2>/dev/null; do sleep 60; done
say "ladder finished"
grep 'seqlen-orch' logs/pass2_finish.log | tail -8 >> "$LOG"

sync; sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null; sleep 5
say "MemAvailable before equivalence gate: $(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)MB"

say "running s03 equivalence gate"
./scripts/gate_s03_equivalence.sh >> "$LOG" 2>&1
say "equivalence gate rc=$?"
tail -6 "$LOG"
