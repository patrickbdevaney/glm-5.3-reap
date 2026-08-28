#!/usr/bin/env bash
# Restore the multimodal calibration bucket after the source re-stage finishes.
#
# The shards were collected (kv said 1802) and then deleted by a later disk reclaim, leaving
# 128 samples. Vision is a first-class capability and R3 - vision-serving experts being pruned
# because calibration carried no image tokens - is the named escalation risk, so calibrating
# pass 2 on 7% of the intended vision weight is not acceptable.
#
# Waits for the download rather than competing with it for the 105 MB/s link.
set -u
cd /home/patrickd/glm-5.3-reap
LOG=logs/mm_topup.log
echo "[mm-topup $(date -Is)] waiting for s01_source to finish" >> $LOG
while pgrep -f "run_stage.py s01_sourc[e]" > /dev/null; do sleep 60; done
echo "[mm-topup $(date -Is)] source stage done; collecting multimodal" >> $LOG
.venv/bin/python - >> $LOG 2>&1 <<'PY'
import sys
sys.path.insert(0, "scripts")
import corpus_spec as SPEC
from stages.s02_corpus import collect_multimodal
quota = round(SPEC.TOTAL_SAMPLES * SPEC.MIXTURE["multimodal"])
print(f"target multimodal quota: {quota}", flush=True)
n = collect_multimodal(quota)
print(f"multimodal on disk now: {n}", flush=True)
PY
echo "[mm-topup $(date -Is)] done rc=$?" >> $LOG
