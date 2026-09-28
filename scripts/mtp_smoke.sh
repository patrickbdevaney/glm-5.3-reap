#!/usr/bin/env bash
# End-to-end check of the MTP speculative path: does blk.45, lifted into a standalone draft,
# actually draft tokens the target accepts?
#
# The head has not been fine-tuned yet - this is GLM-5.3's own MTP module as it shipped, run
# against a REAP-pruned body it was never trained on. Acceptance here is the honest baseline the
# fine-tune has to beat, and the mechanism is what is really on trial: a nonzero acceptance rate
# means the hidden-state plumbing, the KV rollback and the enorm/hnorm wiring are all correct.
set -uo pipefail
cd /home/patrickd/glm-5.3-reap
LOG=logs/mtp_smoke.log
BIN=$HOME/glm5-llama.cpp/build-cuda/bin
M=gguf/GLM-5.3-Flash-REAP50-IQ3_M.gguf
D=gguf/GLM-5.3-Flash-REAP50-IQ3_M-MTP-draft.gguf
say() { echo "[$(date -Is)] mtp: $*" >> "$LOG"; }

say "pausing extraction chain"
./scripts/overnight_ctl.sh stop >> "$LOG" 2>&1
sleep 5

PROMPT="Write a Python function that returns the nth Fibonacci number."

# Baseline first, on the same model at the same ngl with the extraction paused. The 2.56 t/s
# figure recorded earlier was measured while the trace extraction was competing for the GPU, so
# it is not a fair comparison for a speculative run that has the machine to itself.
say "=== baseline (no draft) ==="
timeout 2400 "$BIN/llama-completion" -m "$M" -ngl 24 -c 2048 -n 32 -t 8 \
    -p "$PROMPT" --seed 1 >> "$LOG" 2>&1
say "=== baseline rc=$? ==="

for DR in 1 2 3; do
  say "=== draft-max $DR ==="
  timeout 2400 "$BIN/llama-speculative-simple" \
      -m "$M" -md "$D" -ngl 24 -ngld 24 \
      --draft-max $DR --draft-min 1 -c 2048 -n 32 -t 8 \
      -p "$PROMPT" \
      >> "$LOG" 2>&1
  say "=== draft-max $DR rc=$? ==="
done

say "resuming extraction chain"
./scripts/overnight_ctl.sh resume >> "$LOG" 2>&1
say "done"
