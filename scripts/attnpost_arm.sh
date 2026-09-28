#!/usr/bin/env bash
# The "post" attention arm: quantise only what sits OUTSIDE the KDA recurrence.
#
# o_proj (all 46 layers) is applied to the attention OUTPUT, and the 11 DSA layers plus the MTP
# block are ordinary softmax attention with no recurrent state. KDA q/k/v stay BF16, because they
# feed the delta-rule state where 4-bit error propagates along the SEQUENCE - the same argument
# that already protects the gates, b_proj and the conv1ds.
#
# The "all" arm measured -0.00673 top-1 (z = 14.5) for 8.00 GiB. If the damage is the recurrence,
# this arm costs far less than its byte share; if it is just bulk quantisation error, it costs
# roughly the proportional -0.0024. Either answer is worth having.
set -u
cd /home/patrickd/glm-5.3-reap
PY=./.venv/bin/python
LOG=logs/attnpost.log
NAME=glm-5.3-flash-reap50-nvfp4-pass2-attnpost
say(){ echo "[$(date -Is)] attnpost: $*" >> "$LOG"; }

say "quantising o_proj + MLA projections only"
./scripts/gpulock.sh s07-attnpost $PY scripts/run_stage.py s07_quantize stages.s07_quantize >> "$LOG" 2>&1
rc=$?; say "quantise rc=$rc"; [ $rc -eq 0 ] || exit 1

$PY - >> "$LOG" 2>&1 <<'EOF'
import sys; sys.path.insert(0,'scripts')
from common import kv_set
kv_set("eval_student",
       "/home/patrickd/glm-5.3-reap/output/glm-5.3-flash-reap50-nvfp4-pass2-attnpost")
EOF
say "eval"
./scripts/gpulock.sh eval-attnpost $PY scripts/run_stage.py s09_eval stages.s09_eval >> "$LOG" 2>&1
say "eval rc=$?"

$PY - >> "$LOG" 2>&1 <<'EOF'
import json, math, sys, torch, pathlib
sys.path.insert(0, 'scripts')
import stages.s09_eval as EV
E = pathlib.Path('artifacts/eval')
base = json.load(open(E/'eval_glm-5.3-flash-reap50-nvfp4-pass2.json'))
allq = json.load(open(E/'eval_glm-5.3-flash-reap50-nvfp4-pass2-attnq.json'))
post = json.load(open(E/'eval_glm-5.3-flash-reap50-nvfp4-pass2-attnpost.json'))
gib  = json.load(open('artifacts/s07_quantize.json'))['gib']
print(f"\n{'metric':<18}{'attn BF16':>12}{'post':>12}{'all':>12}")
for k in ("top1_agreement","dNLL_mean","topk_KL"):
    print(f"{k:<18}{base[k]:>12.5f}{post[k]:>12.5f}{allq[k]:>12.5f}")
print(f"{'size GiB':<18}{98.2:>12.1f}{gib:>12.1f}{90.2:>12.1f}")
print(f"{'top-1 delta':<18}{'-':>12}{post['top1_agreement']-base['top1_agreement']:>+12.5f}"
      f"{allq['top1_agreement']-base['top1_agreement']:>+12.5f}")

# Paired McNemar against the BF16-attention build.
T = torch.load(E/'teacher.pt', weights_only=False)
A = torch.load(E/'student_glm-5.3-flash-reap50-nvfp4-pass2.pt', weights_only=False)
B = torch.load(E/'student_glm-5.3-flash-reap50-nvfp4-pass2-attnpost.pt', weights_only=False)
n = min(T['nll'].numel(), A['nll'].numel(), B['nll'].numel())
keep = torch.ones(n, dtype=torch.bool)
gold, skip, _, _ = EV.gold_tokens(); g = gold[:n]
for sid in skip: keep &= g != sid
t = T['argmax'][:n][keep]
a = (A['argmax'][:n][keep] == t); b_ = (B['argmax'][:n][keep] == t)
b = int((a & ~b_).sum()); c = int((b_ & ~a).sum()); N = int(keep.sum())
chi2 = (abs(b-c)-1)**2/(b+c); print(f"\npaired McNemar vs attn-BF16: chi2 {chi2:.1f}  z {math.sqrt(chi2):.1f}"
      f"  delta {(c-b)/N:+.5f}  se {math.sqrt(b+c)/N:.5f}")
saved = 98.2 - gib
print(f"\nGiB saved {saved:.2f} of the 8.00 that 'all' saves ({saved/8.0:.0%})")
print(f"top-1 cost {base['top1_agreement']-post['top1_agreement']:.5f} of the 0.00673 that "
      f"'all' costs ({(base['top1_agreement']-post['top1_agreement'])/0.00673:.0%})")
print("-> recurrence hypothesis SUPPORTED" if
      (base['top1_agreement']-post['top1_agreement'])/0.00673 < saved/8.0 * 0.6
      else "-> looks like bulk quantisation error, not the recurrence")
EOF
say "arm complete"
