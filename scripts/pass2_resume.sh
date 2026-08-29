#!/usr/bin/env bash
# Resume pass 2 after the healing-ledger fix, from a healed pass-2 checkpoint.
#
# WHY THIS EXISTS
# ---------------
# `pass2_finish.sh` ran through s06_emit before it was noticed that s05_heal had loaded a PASS-1
# idempotence ledger. Shard filenames repeat exactly across runs, so every pass-2 shard was
# already named in it: the stage skipped all 62, applied the correction to zero tensors, and
# reported success. s06_emit then wrote a card describing per-expert healing on a checkpoint that
# had none. The run was stopped at P13, the ledger scoped to a {target, keep_set_sha} fingerprint,
# and s05_heal re-run against the emitted tree.
#
# This picks up from there. It deliberately re-verifies the healing landed rather than assuming
# it, because the failure it is recovering from was precisely a stage that reported success while
# doing nothing.
set -u
cd /home/patrickd/glm-5.3-reap
LOG=logs/pass2_finish.log
say(){ echo "" >> $LOG; echo "===== [$(date -Is)] $* =====" >> $LOG; }
run(){ say "$1"; shift; "$@" >> $LOG 2>&1; rc=$?; say "rc=$rc"; return $rc; }

say "RESUME: waiting for the re-run of s05_heal"
while pgrep -f "run_stage.py s05_hea[l]" > /dev/null; do sleep 30; done

run "verify healing actually landed" .venv/bin/python scripts/verify_heal.py \
    || { say "healing verification FAILED - stopping before anything is published"; exit 1; }

# --- evaluate pass 2 (FP8) ----------------------------------------------------------------
.venv/bin/python - >> $LOG 2>&1 <<'PY'
import sys; sys.path.insert(0,'scripts')
from common import kv_set, kv_get
kv_set("eval_student", "/home/patrickd/glm-5.3-reap/output/" + str(kv_get("emit_name")))
PY
run "P13 evaluate pass 2 (FP8)" .venv/bin/python scripts/run_stage.py s09_eval stages.s09_eval

# --- quantise -----------------------------------------------------------------------------
run "STAGE s07_quantize" .venv/bin/python scripts/run_stage.py s07_quantize stages.s07_quantize \
    || { say "quantize failed"; exit 1; }

# Evaluate the NVFP4 artifact too. It, not the 161 GiB FP8, is what is size-matched against
# aggressively-quantised GGUFs of the unpruned model.
.venv/bin/python - >> $LOG 2>&1 <<'PY'
import sys; sys.path.insert(0,'scripts')
from common import kv_set, kv_get
kv_set("eval_student", "/home/patrickd/glm-5.3-reap/output/" + str(kv_get("nvfp4_name")))
PY
run "P13b evaluate the NVFP4 artifact" .venv/bin/python scripts/run_stage.py s09_eval stages.s09_eval

# --- fold the measured evaluation into both cards BEFORE publishing -----------------------
EMIT=$(.venv/bin/python -c "import sys;sys.path.insert(0,'scripts');from common import kv_get;print(kv_get('emit_name'))")
NVFP=$(.venv/bin/python -c "import sys;sys.path.insert(0,'scripts');from common import kv_get;print(kv_get('nvfp4_name'))")
run "card addendum (FP8)"   .venv/bin/python scripts/card_addendum.py --name "$EMIT"
run "card addendum (NVFP4)" .venv/bin/python scripts/card_addendum.py --name "$NVFP"

# --- publish ------------------------------------------------------------------------------
run "upload FP8 v2"   .venv/bin/hf upload patrickbdevaney/GLM-5.3-Flash-REAP50-FP8-v2 \
        "output/$EMIT" . --repo-type model
run "upload NVFP4 v2" .venv/bin/hf upload patrickbdevaney/GLM-5.3-Flash-REAP50-NVFP4-v2 \
        "output/$NVFP" . --repo-type model
say "PASS 2 COMPLETE"
