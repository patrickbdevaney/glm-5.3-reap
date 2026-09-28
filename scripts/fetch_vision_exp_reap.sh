#!/usr/bin/env bash
# Detached: fetch primitive-ai/DeepSeek-V4-Flash-Vision-Exp-REAP-145B (83.3 GB) for the
# three-way HumanEval. NOTE this artifact is OVER-PRUNED for Thor: 128/256 = 50%, where the
# fit-driven ratio is 160/256 = 37.5% (105 GB budget / 0.660 GB per expert). It therefore
# bounds the model from BELOW -- a weak result here does not condemn Vision-Exp, it condemns
# a 50% prune of it.
set -u
DEST="$HOME/models/DeepSeek-V4-Flash-Vision-Exp-REAP-145B"
LOG="$HOME/glm-5.3-reap/logs/fetch_vision_exp.log"
mkdir -p "$DEST"
avail=$(df --output=avail -BG / | tail -1 | tr -dc '0-9')
echo "[$(date -Is)] free ${avail} GB, need ~84 GB" | tee -a "$LOG"
if [ "$avail" -lt 95 ]; then
  echo "[$(date -Is)] ABORT: under 95 GB free, refusing to fill the disk" | tee -a "$LOG"
  exit 1
fi
cd "$HOME/glm-5.3-reap"
.venv/bin/python - >> "$LOG" 2>&1 <<'PY'
from huggingface_hub import snapshot_download
import os
p = snapshot_download("primitive-ai/DeepSeek-V4-Flash-Vision-Exp-REAP-145B",
                      local_dir=os.path.expanduser(
                          "~/models/DeepSeek-V4-Flash-Vision-Exp-REAP-145B"),
                      max_workers=8)
print("downloaded to", p)
PY
echo "[$(date -Is)] DONE rc=$? free now $(df -h / | awk 'NR==2{print $4}')" | tee -a "$LOG"
