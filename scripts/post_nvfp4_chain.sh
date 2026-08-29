#!/usr/bin/env bash
# When the NVFP4 evaluation lands: fold it into that card, publish the card, then run the
# per-expert-vs-scalar healing ablation. Sequenced, not parallel - the ablation needs the GPU and
# contention with a scoring pass is what got a stage killed by memguard earlier in this project.
set -u
cd /home/patrickd/glm-5.3-reap
LOG=logs/pass2_finish.log
say(){ echo "" >> $LOG; echo "===== [$(date -Is)] $* =====" >> $LOG; }
run(){ say "$1"; shift; "$@" >> $LOG 2>&1; rc=$?; say "rc=$rc"; return $rc; }

say "POST-NVFP4: waiting for the NVFP4 evaluation"
while pgrep -f "run_stage.py s09_eva[l]" > /dev/null; do sleep 60; done

NVFP=$(.venv/bin/python -c "import sys;sys.path.insert(0,'scripts');from common import kv_get;print(kv_get('nvfp4_name'))")
if [ ! -s "artifacts/eval/eval_${NVFP}.json" ]; then
  say "no NVFP4 evaluation artifact - not touching the card, not running the ablation"
  exit 1
fi
run "card addendum (NVFP4)" .venv/bin/python scripts/card_addendum.py --name "$NVFP"
run "publish NVFP4 card"    .venv/bin/hf upload patrickbdevaney/GLM-5.3-Flash-REAP50-NVFP4-v2 \
        "output/$NVFP/README.md" README.md --repo-type model

# --- ablation: per-expert healing vs the per-layer scalar, on the SHIPPED checkpoint ----------
run "ABLATION per-expert vs scalar healing" .venv/bin/python scripts/heal_ablation.py
say "POST-NVFP4 COMPLETE"
