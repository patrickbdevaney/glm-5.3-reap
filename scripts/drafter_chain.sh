#!/usr/bin/env bash
# Drafter baseline, queued behind the NVFP4 eval and the healing ablation.
#
# Strictly sequential. Every GPU stage on this box has one at a time; the one occasion two
# overlapped, memguard killed the loser 14 layers into a 90-minute pass.
set -u
cd /home/patrickd/glm-5.3-reap
LOG=logs/pass2_finish.log
say(){ echo "" >> $LOG; echo "===== [$(date -Is)] $* =====" >> $LOG; }
run(){ say "$1"; shift; "$@" >> $LOG 2>&1; rc=$?; say "rc=$rc"; return $rc; }

say "DRAFTER: waiting for the NVFP4 eval and the healing ablation to finish"
while pgrep -f "run_stage.py s09_eva[l]" > /dev/null \
   || pgrep -f "heal_ablatio[n]\.py" > /dev/null \
   || pgrep -f "post_nvfp4_chai[n]\.sh" > /dev/null; do sleep 60; done

run "DRAFTER capture: dense taps + target greedy over held-out text" \
    .venv/bin/python scripts/drafter_capture.py --seqs 48 --max-len 1024 \
    || { say "capture failed - no baseline"; exit 1; }

run "DRAFTER baseline: stock DFlash 2 acceptance vs the REAP target" \
    .venv/bin/python scripts/drafter_baseline.py --label stock

say "DRAFTER BASELINE COMPLETE"
