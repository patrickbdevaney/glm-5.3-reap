#!/usr/bin/env bash
# Healing ablation, re-queued. The first attempt never ran: post_nvfp4_chain.sh correctly refused
# because artifacts/eval/eval_<nvfp4>.json was absent - s09_eval only gained the per-student write
# AFTER that eval process had already imported the module, so it wrote s09_eval.json alone. The
# guard did its job; the artifact has since been recovered from s09_eval.json and both cards are
# published. This just runs the ablation, once the GPU is free.
set -u
cd /home/patrickd/glm-5.3-reap
LOG=logs/pass2_finish.log
say(){ echo "" >> $LOG; echo "===== [$(date -Is)] $* =====" >> $LOG; }
run(){ say "$1"; shift; "$@" >> $LOG 2>&1; rc=$?; say "rc=$rc"; return $rc; }

say "ABLATION: waiting for the drafter chain to release the GPU"
while pgrep -f "drafter_captur[e]\.py" > /dev/null \
   || pgrep -f "drafter_baselin[e]\.py" > /dev/null \
   || pgrep -f "run_stage.py s09_eva[l]" > /dev/null; do sleep 60; done

run "ABLATION per-expert vs scalar healing" .venv/bin/python scripts/heal_ablation.py
say "ABLATION COMPLETE"
