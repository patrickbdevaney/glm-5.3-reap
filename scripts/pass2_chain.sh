#!/usr/bin/env bash
# Pass-2 chain: multimodal top-up -> preflight -> 12-chunk saliency sweep.
#
# Stops after saliency ON PURPOSE. P5 (measured healing gain) and P6 (split-half overlap gate)
# sit between the sweep and any materialisation, and s04b_surgery DELETES source shards as it
# writes - so it must never be chained behind an unreviewed sweep.
set -u
cd /home/patrickd/glm-5.3-reap
LOG=logs/pass2_chain.log
say() { echo "[pass2-chain $(date -Is)] $*" >> $LOG; }

say "waiting for multimodal top-up"
while pgrep -f "mm_topu[p]\.sh" > /dev/null; do sleep 60; done

MM=$(.venv/bin/python - <<'PY' 2>/dev/null
import torch, glob
n=0
for f in sorted(glob.glob("corpus/shards/multimodal/mm_*.pt")):
    try: n+=len(torch.load(f, weights_only=False))
    except Exception: pass
print(n)
PY
)
say "multimodal samples on disk: ${MM:-unknown} (pass 1 calibrated vision on 39)"

say "preflight"
.venv/bin/python scripts/preflight.py >> $LOG 2>&1
say "preflight rc=$?"

say "launching s03_saliency (12 chunks x 0.5M tokens)"
setsid nohup .venv/bin/python scripts/run_stage.py s03_saliency stages.s03_saliency \
    > logs/p2_saliency.log 2>&1 < /dev/null &
disown
say "s03 launched pid=$!"
