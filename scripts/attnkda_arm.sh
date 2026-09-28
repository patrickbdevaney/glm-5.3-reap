#!/usr/bin/env bash
# The "kda" arm: quantise ONLY the KDA q/k/v projections.
#
# The recurrence hypothesis was REFUTED by the "post" arm: quantising only what sits outside the
# KDA recurrence cost 82% of the damage for 43% of the bytes. So KDA q/k/v - the tensors the
# hypothesis said were fragile - are in fact the most quantisation-tolerant part of attention,
# at 0.265 milli-top1 per GiB against o_proj+MLA's 1.621. This arm measures that half directly
# rather than inferring it by subtraction, which assumes an additivity nothing has established.
#
set -u
cd /home/patrickd/glm-5.3-reap
PY=./.venv/bin/python
LOG=logs/attnkda.log
NAME=glm-5.3-flash-reap50-nvfp4-pass2-attnkda
say(){ echo "[$(date -Is)] attnkda: $*" >> "$LOG"; }

say "quantising KDA q/k/v only"
./scripts/gpulock.sh s07-attnkda $PY scripts/run_stage.py s07_quantize stages.s07_quantize >> "$LOG" 2>&1
rc=$?; say "quantise rc=$rc"; [ $rc -eq 0 ] || exit 1

$PY - >> "$LOG" 2>&1 <<'EOF'
import sys; sys.path.insert(0,'scripts')
from common import kv_set
kv_set("eval_student",
       "/home/patrickd/glm-5.3-reap/output/glm-5.3-flash-reap50-nvfp4-pass2-attnkda")
EOF
say "eval"
./scripts/gpulock.sh eval-attnkda $PY scripts/run_stage.py s09_eval stages.s09_eval >> "$LOG" 2>&1
say "eval rc=$?"

$PY - >> "$LOG" 2>&1 <<'EOF'
import json, math, sys, torch, pathlib
sys.path.insert(0, 'scripts')
import stages.s09_eval as EV
E = pathlib.Path('artifacts/eval')
L = lambda n: json.load(open(E/f'eval_glm-5.3-flash-reap50-nvfp4-pass2{n}.json'))
base, post, allq, kda = L(''), L('-attnpost'), L('-attnq'), L('-attnkda')
gib = json.load(open('artifacts/s07_quantize.json'))['gib']
print(f"\n{'metric':<18}{'BF16':>11}{'kda':>11}{'post':>11}{'all':>11}")
for k in ("top1_agreement","dNLL_mean","topk_KL"):
    print(f"{k:<18}{base[k]:>11.5f}{kda[k]:>11.5f}{post[k]:>11.5f}{allq[k]:>11.5f}")
print(f"{'size GiB':<18}{98.2:>11.1f}{gib:>11.1f}{94.8:>11.1f}{90.2:>11.1f}")
print(f"{'top-1 delta':<18}{'-':>11}{kda['top1_agreement']-base['top1_agreement']:>+11.5f}"
      f"{post['top1_agreement']-base['top1_agreement']:>+11.5f}"
      f"{allq['top1_agreement']-base['top1_agreement']:>+11.5f}")

T = torch.load(E/'teacher.pt', weights_only=False)
A = torch.load(E/'student_glm-5.3-flash-reap50-nvfp4-pass2.pt', weights_only=False)
B = torch.load(E/'student_glm-5.3-flash-reap50-nvfp4-pass2-attnkda.pt', weights_only=False)
n = min(T['nll'].numel(), A['nll'].numel(), B['nll'].numel())
keep = torch.ones(n, dtype=torch.bool)
gold, skip, _, _ = EV.gold_tokens(); g = gold[:n]
for sid in skip: keep &= g != sid
t = T['argmax'][:n][keep]
a = (A['argmax'][:n][keep]==t); b_ = (B['argmax'][:n][keep]==t)
b = int((a & ~b_).sum()); c = int((b_ & ~a).sum()); N = int(keep.sum())
chi2 = (abs(b-c)-1)**2/(b+c)
print(f"\npaired vs attn-BF16: chi2 {chi2:.1f}  z {math.sqrt(chi2):.1f}  "
      f"delta {(c-b)/N:+.5f}  se {math.sqrt(b+c)/N:.5f}")

saved = 98.2 - gib
cost = base['top1_agreement'] - kda['top1_agreement']
print(f"\nkda saves {saved:.2f} GiB at -{cost:.5f} top-1 = {cost/saved*1000:.3f} milli-top1/GiB")
print(f"  post was 1.621, all was 0.841 milli-top1/GiB")
pred = 0.00673 - 0.00551
print(f"predicted from subtraction (assuming additivity): -{pred:.5f}; measured -{cost:.5f}"
      f"  -> additivity {'HOLDS' if abs(cost-pred) < 0.0008 else 'DOES NOT hold'}")
EOF
say "arm complete"
