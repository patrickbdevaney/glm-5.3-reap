#!/usr/bin/env bash
# Re-quantise with attention included, then measure. Serialised via scripts/gpulock.sh.
# Publishes NOTHING: v3 goes out only after the top-1 number is compared to the v2 arm.
set -u
cd /home/patrickd/glm-5.3-reap
LOG=logs/pass2_finish.log
say(){ echo "" >> $LOG; echo "===== [$(date -Is)] $* =====" >> $LOG; }

OUTDIR=output/glm-5.3-flash-reap50-nvfp4-pass2-attnq
# The first attempt was killed mid-write to protect a concurrent stage, so this directory holds a
# truncated build. s07 refuses to populate a non-empty dir, and a half-quantised checkpoint is
# worse than none - remove it rather than resume into it.
[ -d "$OUTDIR" ] && { say "removing partial build from the interrupted attempt"; rm -rf "$OUTDIR"; }

.venv/bin/python - >> $LOG 2>&1 <<'PY'
import sys; sys.path.insert(0,'scripts')
from common import kv_set
kv_set("nvfp4_name", "glm-5.3-flash-reap50-nvfp4-pass2-attnq")
PY

FREE=$(df --output=avail -BG /home/patrickd | tail -1 | tr -dc '0-9')
if [ "$FREE" -lt 95 ]; then say "only ${FREE}G free, need ~95G - stopping"; exit 1; fi

./scripts/gpulock.sh attnq-quantise .venv/bin/python scripts/run_stage.py s07_quantize stages.s07_quantize \
    || { say "attnq quantise failed"; exit 1; }

.venv/bin/python - >> $LOG 2>&1 <<'PY'
import sys; sys.path.insert(0,'scripts')
from common import kv_set
kv_set("eval_student", "/home/patrickd/glm-5.3-reap/output/glm-5.3-flash-reap50-nvfp4-pass2-attnq")
PY
./scripts/gpulock.sh attnq-eval .venv/bin/python scripts/run_stage.py s09_eval stages.s09_eval
say "ATTN-NVFP4 COMPLETE - compare with eval_glm-5.3-flash-reap50-nvfp4-pass2.json before publishing"
