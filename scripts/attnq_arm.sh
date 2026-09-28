#!/usr/bin/env bash
# The attention-NVFP4 arm: same corrected FP8 base, same mask, same healing, ONE variable moved -
# q/k/v/o and the MLA projections go from BF16 to NVFP4.
#
# Run as its own artifact and its own evaluation rather than folded into the shipped v2, because
# that is the lesson this project has now paid for twice: a build that changes two things at once
# cannot attribute either.
set -u
cd /home/patrickd/glm-5.3-reap
PY=./.venv/bin/python
LOG=logs/attnq.log
NAME=glm-5.3-flash-reap50-nvfp4-pass2-attnq
say(){ echo "[$(date -Is)] attnq: $*" >> "$LOG"; }

say "quantising attention projections (expect 98.2 -> ~90.2 GiB)"
./scripts/gpulock.sh s07-attnq $PY scripts/run_stage.py s07_quantize stages.s07_quantize >> "$LOG" 2>&1
rc=$?; say "quantise rc=$rc"
[ $rc -eq 0 ] || exit 1

$PY - >> "$LOG" 2>&1 <<'EOF'
import sys; sys.path.insert(0,'scripts')
from common import kv_set
kv_set("eval_student", "/home/patrickd/glm-5.3-reap/output/glm-5.3-flash-reap50-nvfp4-pass2-attnq")
EOF
say "eval"
./scripts/gpulock.sh eval-attnq $PY scripts/run_stage.py s09_eval stages.s09_eval >> "$LOG" 2>&1
say "eval rc=$?"

$PY - >> "$LOG" 2>&1 <<'EOF'
import json
a = json.load(open('artifacts/eval/eval_glm-5.3-flash-reap50-nvfp4-pass2-attnq.json'))
b = json.load(open('artifacts/eval/eval_glm-5.3-flash-reap50-nvfp4-pass2.json'))
print(f"\n{'metric':<20}{'attn BF16':>12}{'attn NVFP4':>12}{'delta':>12}")
for k in ("top1_agreement","dNLL_mean","topk_KL","student_nll"):
    print(f"{k:<20}{b[k]:>12.5f}{a[k]:>12.5f}{a[k]-b[k]:>+12.5f}")
print("\nper domain (top-1):")
for d in sorted(a["by_domain"], key=lambda x:-a["by_domain"][x]["tokens"]):
    if not a["by_domain"][d].get("sufficient"): continue
    print(f"  {d:<9}{b['by_domain'][d]['top1_agreement']:>9.4f}"
          f"{a['by_domain'][d]['top1_agreement']:>12.4f}"
          f"{a['by_domain'][d]['top1_agreement']-b['by_domain'][d]['top1_agreement']:>+12.4f}")
EOF
say "arm complete"
