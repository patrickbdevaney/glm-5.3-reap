#!/usr/bin/env bash
# Pass 2, sweep -> published artifacts, autonomously.
#
# ORDERING IS LOAD-BEARING, not stylistic:
#   * s04_sweep must precede heal_refit, or the gains are measured against pass 1's keep-set.
#     s05_heal now refuses mismatched stamps, so getting this wrong fails loudly rather than
#     shipping a wrong correction - but it should simply not happen.
#   * s09_eval on the TEACHER must precede s04b_surgery, because surgery unlinks the source
#     shards as it writes and the source IS the teacher. After surgery there is no teacher to
#     score against, and pass 2 would have no evaluation at all.
set -u
cd /home/patrickd/glm-5.3-reap
LOG=logs/pass2_finish.log
say(){ echo "" >> $LOG; echo "===== [$(date -Is)] $* =====" >> $LOG; }
run(){ say "$1"; shift; "$@" >> $LOG 2>&1; rc=$?; say "rc=$rc"; return $rc; }
stage(){ run "STAGE $1" .venv/bin/python scripts/run_stage.py "$1" "stages.$1"; }

say "waiting for the saliency sweep"
while pgrep -f "run_stage.py s03_saliency" > /dev/null; do sleep 300; done
if [ ! -s state/s03_chunks.json ]; then say "no chunk ledger - aborting"; exit 1; fi
DONE=$(.venv/bin/python -c "import json;d=json.load(open('state/s03_chunks.json'));print(len(d['done']),d['chunks'])")
say "sweep finished: chunks $DONE"

say "waiting for the gates (P5/P6/P7)"
while pgrep -f "pass2_gate[s]\.sh" > /dev/null; do sleep 120; done

# --- pass-2 keep-set, then gains measured against IT --------------------------------------
stage s04_sweep      || { say "s04_sweep failed - stopping"; exit 1; }

# Materialise the pass-2 keep-set BEFORE measuring gains against it.
#
# reap_retained_experts.json is written by s04b_SURGERY, not s04_sweep - so without this step
# heal_refit measures against pass 1's mask, stamps it, and s05_heal then refuses the mismatch
# and halts the run. And heal_refit cannot simply be deferred until after surgery: it reads
# e_score_correction_bias from the SOURCE, which surgery deletes. The only window where both the
# pass-2 mask and the teacher exist is right here.
run "materialise the pass-2 keep-set (pure function of the saliency)" .venv/bin/python - <<'PY'
import sys, json; sys.path.insert(0,'scripts')
import stages.s04b_surgery as SG
from common import ARTIFACTS, kv_get
ratio = float(kv_get("target_sparsity", 0.5) or 0.5)
retained = SG.compute_retained(ratio)
n_keep = len(next(iter(retained.values())))
mtp = SG._mtp_keep_set(n_keep, SG._original_expert_count())
if mtp:
    retained[f"model.language_model.layers.{SG.MTP_LAYER}.mlp"] = mtp
(ARTIFACTS / "reap_retained_experts.json").write_text(json.dumps(retained))
print(f"keep-set: {len(retained)} layers, {n_keep} experts, MTP={bool(mtp)}")
PY

run "heal_refit vs the pass-2 keep-set" .venv/bin/python scripts/heal_refit.py
run "split_half (full budget)"          .venv/bin/python scripts/split_half.py
run "criterion shootout (full budget)"  .venv/bin/python scripts/criterion_shootout.py

# --- P9.5: capture the teacher and baseline pass 1, BEFORE surgery destroys the source -----
say "P9.5 teacher capture + pass-1 baseline (MUST precede surgery)"
.venv/bin/python - >> $LOG 2>&1 <<'PY'
import sys; sys.path.insert(0,'scripts')
from common import kv_set
kv_set("eval_student", "/home/patrickd/glm-5.3-reap/output/glm-5.3-flash-reap50-fp8")
PY
stage s09_eval || { say "P9.5 eval failed - NOT proceeding to surgery, the teacher is still needed"; exit 1; }
if [ ! -f artifacts/eval/teacher.pt ]; then
  say "teacher capture missing - refusing to run surgery (it would delete the teacher)"; exit 1
fi
say "teacher captured; surgery is now safe"

# --- materialise pass 2 -------------------------------------------------------------------
stage s04b_surgery || { say "surgery failed"; exit 1; }
stage s05_heal     || { say "heal failed"; exit 1; }
stage s06_emit     || { say "emit failed"; exit 1; }

# --- evaluate pass 2 against the cached teacher -------------------------------------------
.venv/bin/python - >> $LOG 2>&1 <<'PY'
import sys; sys.path.insert(0,'scripts')
from common import kv_set, kv_get
kv_set("eval_student", "/home/patrickd/glm-5.3-reap/output/" + str(kv_get("emit_name")))
PY
run "P13 evaluate pass 2" .venv/bin/python scripts/run_stage.py s09_eval stages.s09_eval

# --- quantise and publish -----------------------------------------------------------------
stage s07_quantize || { say "quantize failed"; exit 1; }
run "upload FP8 v2"   .venv/bin/hf upload patrickbdevaney/GLM-5.3-Flash-REAP50-FP8-v2 \
        "output/$(.venv/bin/python -c "import sys;sys.path.insert(0,'scripts');from common import kv_get;print(kv_get('emit_name'))")" . --repo-type model
run "upload NVFP4 v2" .venv/bin/hf upload patrickbdevaney/GLM-5.3-Flash-REAP50-NVFP4-v2 \
        "output/$(.venv/bin/python -c "import sys;sys.path.insert(0,'scripts');from common import kv_get;print(kv_get('nvfp4_name'))")" . --repo-type model
say "PASS 2 COMPLETE"
