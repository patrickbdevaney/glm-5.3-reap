#!/usr/bin/env bash
# Wait out the healing revert, then re-measure the corrected checkpoint and rebuild NVFP4 from it.
#
# The revert is pure disk I/O and holds no GPU, so it is NOT under gpulock; everything after it is.
set -u
cd /home/patrickd/glm-5.3-reap
PY=./.venv/bin/python
LOG=logs/revert_chain.log
say(){ echo "[$(date -Is)] $*" >> $LOG; }

while kill -0 "$1" 2>/dev/null; do sleep 30; done
say "revert process $1 exited"
if ! grep -q "^PASS:" logs/heal_revert.log; then
  say "FATAL: heal_revert did not report PASS - stopping, nothing downstream is trustworthy"
  exit 1
fi
say "revert verified against the pre-heal probe"

# Keep the per-expert measurement. It is the evidence for the revert; overwriting it with the
# corrected run would delete the very number the model card now cites.
cp -n artifacts/eval/eval_glm-5.3-flash-reap50-fp8-pass2.json \
      artifacts/eval/eval_pass2_fp8_perexpert_healed.json 2>/dev/null
cp -n artifacts/eval/student_glm-5.3-flash-reap50-fp8-pass2.pt \
      artifacts/eval/student_pass2_fp8_perexpert_healed.pt 2>/dev/null

say "eval: corrected FP8 pass 2 (expect top-1 ~0.84238, matching the ablation)"
$PY - <<'EOF' >> $LOG 2>&1
import sys; sys.path.insert(0,'scripts')
from common import kv_set
kv_set("eval_student", "/home/patrickd/glm-5.3-reap/output/glm-5.3-flash-reap50-fp8-pass2")
EOF
./scripts/gpulock.sh eval-fp8-corrected $PY scripts/run_stage.py s09_eval stages.s09_eval >> $LOG 2>&1
say "eval rc=$?"
$PY - <<'EOF' >> $LOG 2>&1
import json
d = json.load(open('artifacts/eval/eval_glm-5.3-flash-reap50-fp8-pass2.json'))
print(f"corrected FP8 pass2: top1={d['top1_agreement']:.5f} dNLL={d['dNLL_mean']:.5f} "
      f"KL={d['topk_KL']:.5f}")
print("ablation predicted    top1=0.84238 dNLL=0.17601 KL=0.65248")
print("MATCH" if abs(d['top1_agreement']-0.84238) < 2e-3 else
      "MISMATCH - the revert did not reproduce the ablation, investigate before publishing")
EOF
say "chain done"
