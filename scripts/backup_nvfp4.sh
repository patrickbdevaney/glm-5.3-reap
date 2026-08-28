#!/usr/bin/env bash
# Push the healed FP8 checkpoint to HF. This is the MASTER artifact - every other
# quantisation (NVFP4, GGUF, INT4, AWQ) derives from it - and right now it exists in exactly
# one place, on one consumer NVMe, representing ~20 hours of work.
#
# Runs detached and resumable: huggingface-cli upload skips files already present with a
# matching hash, so an interrupted run costs only the file in flight.
set -u
cd /home/patrickd/glm-5.3-reap
SRC=output/glm-5.3-flash-reap50-nvfp4
REPO=patrickbdevaney/GLM-5.3-Flash-REAP50-NVFP4
LOG=logs/backup_nvfp4.log
say(){ echo "[backup-nvfp4 $(date -Is)] $*" >> "$LOG"; }
say "starting upload of $SRC -> $REPO ($(du -sh $SRC | cut -f1))"
for attempt in $(seq 1 40); do
  if .venv/bin/hf upload "$REPO" "$SRC" . --repo-type model >> "$LOG" 2>&1; then
    say "UPLOAD COMPLETE after $attempt attempt(s)"
    exit 0
  fi
  say "attempt $attempt failed; sleeping 60s then resuming"
  sleep 60
done
say "gave up after 40 attempts"
exit 1
