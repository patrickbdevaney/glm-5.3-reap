#!/usr/bin/env bash
# Re-quantise with attention included, then measure. Queued behind everything else on the GPU.
#
# Publishes NOTHING. The v2 NVFP4 repo is live; a v3 goes out only after the top-1 number is seen
# and approved. If attention at 4 bits costs more than ~0.005 top-1 this is reverted and the cost
# was two hours.
set -u
cd /home/patrickd/glm-5.3-reap
LOG=logs/pass2_finish.log
say(){ echo "" >> $LOG; echo "===== [$(date -Is)] $* =====" >> $LOG; }
run(){ say "$1"; shift; "$@" >> $LOG 2>&1; rc=$?; say "rc=$rc"; return $rc; }

say "ATTN-NVFP4: waiting for the GPU"
while pgrep -f "drafter_captur[e]\.py" >/dev/null || pgrep -f "drafter_baselin[e]\.py" >/dev/null \
   || pgrep -f "heal_ablatio[n]\.py" >/dev/null || pgrep -f "run_stage.py s09_eva[l]" >/dev/null; do
  sleep 60
done

# Disk: the existing 98.2 GiB NVFP4 stays put as the comparison arm, so the new build needs its
# own ~90 GiB. Check before starting rather than dying 30 minutes in with a truncated shard.
FREE=$(df --output=avail -BG /home/patrickd | tail -1 | tr -dc '0-9')
if [ "$FREE" -lt 100 ]; then say "only ${FREE}G free, need ~100G for the v3 build - stopping"; exit 1; fi

.venv/bin/python - >> $LOG 2>&1 <<'PY'
import sys; sys.path.insert(0,'scripts')
from common import kv_set
kv_set("nvfp4_name", "glm-5.3-flash-reap50-nvfp4-pass2-attnq")
PY
run "ATTN-NVFP4 quantise (attention included)" \
    .venv/bin/python scripts/run_stage.py s07_quantize stages.s07_quantize \
    || { say "quantise failed"; exit 1; }

.venv/bin/python - >> $LOG 2>&1 <<'PY'
import sys; sys.path.insert(0,'scripts')
from common import kv_set
kv_set("eval_student", "/home/patrickd/glm-5.3-reap/output/glm-5.3-flash-reap50-nvfp4-pass2-attnq")
PY
run "ATTN-NVFP4 evaluate" .venv/bin/python scripts/run_stage.py s09_eval stages.s09_eval
say "ATTN-NVFP4 COMPLETE - compare against eval_glm-5.3-flash-reap50-nvfp4-pass2.json before publishing"
