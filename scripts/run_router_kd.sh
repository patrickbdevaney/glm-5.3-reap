#!/usr/bin/env bash
# Detached: wait for NIAH to release the GPU, then router KD.
# Smoke-tests on 2 layers first -- the driver has never run against the real model, and an
# unattended failure 30 layers deep would waste the whole GPU window.
set -u
cd "$HOME/glm-5.3-reap"
LOG=logs/router_kd.log
echo "[$(date -Is)] waiting for NIAH to release the GPU" | tee -a "$LOG"
while pgrep -f "tools/niah_nvfp[4]" >/dev/null 2>&1 || pgrep -x glm5-server >/dev/null 2>&1; do
  sleep 60
done
echo "[$(date -Is)] GPU free; free RAM: $(free -g | awk '/^Mem:/{print $7}') GiB" | tee -a "$LOG"

echo "[$(date -Is)] SMOKE: 2 layers, 6 steps" | tee -a "$LOG"
.venv/bin/python -c "
import sys; sys.path.insert(0,'scripts')
import router_kd as R
R.run(smoke_layers=2, steps_per_layer=6, batch=1, max_len=512)
" >> "$LOG" 2>&1
rc=$?
if [ $rc -ne 0 ]; then
  echo "[$(date -Is)] SMOKE FAILED rc=$rc -- not starting the full run" | tee -a "$LOG"
  tail -30 "$LOG"
  exit 1
fi
echo "[$(date -Is)] smoke OK; starting full run" | tee -a "$LOG"
.venv/bin/python -c "
import sys; sys.path.insert(0,'scripts')
import router_kd as R
R.run(steps_per_layer=150, batch=2, max_len=1024)
" >> "$LOG" 2>&1
echo "[$(date -Is)] DONE rc=$?" | tee -a "$LOG"
