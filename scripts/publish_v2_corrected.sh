#!/usr/bin/env bash
# Correct both v2 cards and push both v2 repositories, once the corrected NVFP4 exists.
#
# Serial by design. Both uploads share one uplink, and interleaving them would leave BOTH repos
# half-corrected for the whole window instead of one of them briefly - and a repo whose weights
# and card disagree is worse than one that is simply behind.
set -u
cd /home/patrickd/glm-5.3-reap
PY=./.venv/bin/python
LOG=logs/publish_v2.log
say(){ echo "[$(date -Is)] publish: $*" >> "$LOG"; }

while kill -0 "$1" 2>/dev/null; do sleep 60; done
say "rebuild chain exited"

FP8=glm-5.3-flash-reap50-fp8-pass2
NV=glm-5.3-flash-reap50-nvfp4-pass2

# Both artifacts must exist and both must have a measured eval, or the cards cannot be honest.
for d in "$FP8" "$NV"; do
  [ -f "output/$d/model.safetensors.index.json" ] || { say "FATAL: output/$d incomplete"; exit 1; }
  [ -f "artifacts/eval/eval_$d.json" ] || { say "FATAL: no eval for $d"; exit 1; }
done

$PY - >> "$LOG" 2>&1 <<'EOF'
import json, sys
for n in ("glm-5.3-flash-reap50-fp8-pass2", "glm-5.3-flash-reap50-nvfp4-pass2"):
    d = json.load(open(f"artifacts/eval/eval_{n}.json"))
    print(f"{n}: top1={d['top1_agreement']:.5f} dNLL={d['dNLL_mean']:+.5f} KL={d['topk_KL']:.5f}")
    # The corrected FP8 must reproduce the ablation; NVFP4 must not be far below its FP8 parent.
    if n.endswith("fp8-pass2") and abs(d['top1_agreement'] - 0.84238) > 2e-3:
        print("FP8 does not match the ablation - refusing to publish"); sys.exit(1)
EOF
[ $? -eq 0 ] || { say "eval gate failed"; exit 1; }
say "eval gate passed"

for d in "$FP8" "$NV"; do
  $PY scripts/card_correct_healing.py --name "$d" >> "$LOG" 2>&1 || { say "card fix failed for $d"; exit 1; }
  $PY scripts/card_addendum.py       --name "$d" >> "$LOG" 2>&1 || { say "addendum failed for $d"; exit 1; }
done
say "cards corrected"

push(){  # $1 = local dir, $2 = repo
  say "uploading $1 -> $2 ($(du -sh "output/$1" | cut -f1))"
  for a in $(seq 1 40); do
    if .venv/bin/hf upload "$2" "output/$1" . --repo-type model >> "$LOG" 2>&1; then
      say "$2 COMPLETE after $a attempt(s)"; return 0
    fi
    say "$2 attempt $a failed; resuming in 60s"; sleep 60
  done
  say "$2 GAVE UP"; return 1
}

push "$FP8" patrickbdevaney/GLM-5.3-Flash-REAP50-FP8-v2
push "$NV"  patrickbdevaney/GLM-5.3-Flash-REAP50-NVFP4-v2
say "done"
