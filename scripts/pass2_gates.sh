#!/usr/bin/env bash
# Run the pass-2 gates as soon as their data exists, and then STOP.
#
# P5 fires after chunk 1 rather than chunk 10 on purpose: if the healing correction is wrong,
# that is worth knowing ~16 hours early. P6 and P7 need the whole sweep.
#
# This deliberately does NOT chain into materialisation. s04b_surgery DELETES source shards as it
# writes, so it must never run behind gates a human has not read.
set -u
cd /home/patrickd/glm-5.3-reap
LOG=logs/pass2_gates.log
say() { echo "" >> $LOG; echo "===== [$(date -Is)] $* =====" >> $LOG; }

# ---- P5: healing re-fit, after the first chunk ------------------------------------------------
say "waiting for chunk 1 (router cache + accumulators)"
while :; do
  [ -f artifacts/router_cache/chunk_000.pt ] && [ -s state/s03_chunks.json ] && break
  pgrep -f "run_stage.py s03_saliency" > /dev/null || { say "s03 gone before chunk 1"; exit 1; }
  sleep 120
done
say "P5 heal_refit (nice'd; the sweep owns the box)"
nice -n 15 .venv/bin/python scripts/heal_refit.py >> $LOG 2>&1

# ---- P6 / P7: after the sweep finishes ---------------------------------------------------------
say "waiting for the sweep to finish"
while pgrep -f "run_stage.py s03_saliency" > /dev/null; do sleep 300; done

say "P5 heal_refit (full budget)"
.venv/bin/python scripts/heal_refit.py >> $LOG 2>&1
say "P6 split_half gate"
.venv/bin/python scripts/split_half.py >> $LOG 2>&1
say "P7 criterion shootout"
.venv/bin/python scripts/criterion_shootout.py >> $LOG 2>&1
say "GATES COMPLETE - nothing materialised, by design. Read these before running s04b_surgery."
