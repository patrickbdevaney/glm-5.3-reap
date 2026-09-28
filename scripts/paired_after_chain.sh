#!/usr/bin/env bash
# Runs the paired McNemar test once revert_chain.sh has produced the corrected capture.
set -u
cd /home/patrickd/glm-5.3-reap
LOG=logs/revert_chain.log
while kill -0 "$1" 2>/dev/null; do sleep 60; done
echo "[$(date -Is)] revert_chain exited; paired test" >> $LOG
if [ -f artifacts/eval/student_pass2_fp8_perexpert_healed.pt ]; then
  ./.venv/bin/python scripts/heal_paired_test.py >> $LOG 2>&1
  echo "[$(date -Is)] paired test rc=$?" >> $LOG
else
  echo "[$(date -Is)] no preserved per-expert capture - skipping paired test" >> $LOG
fi
