#!/usr/bin/env bash
# Is the draft model's presence on the GPU what is costing the verification batch?
#
# Two measurements point that way. At -ngl 24 a 138-token batch costs 52 ms/token against 335 ms
# for a single decode, so this target does amortise batching - by 6.4x. But raising the target to
# -ngl 36 collapses that to nothing (277 ms/token batched vs 264 ms single), which is what memory
# pressure looks like on unified memory: the offloaded weights are non-evictable and the batched
# path needs the larger compute buffers. The speculative runs put a second model on the same GPU
# with -ngld 99, and their verification batches behave like the -ngl 36 case rather than the
# -ngl 24 one.
#
# The draft is 2.28 GiB and decodes in 5.3 ms/token; even entirely on the CPU it would be cheap
# next to a 335 ms target step. If moving it off the GPU restores the target's batching, the
# speculative loss is a placement problem, not an architectural ceiling.
set -uo pipefail
cd /home/patrickd/glm-5.3-reap
LOG=logs/mtp_draftcpu.log
BIN=$HOME/glm5-llama.cpp/build-cuda/bin
M=gguf/GLM-5.3-Flash-REAP50-IQ3_M.gguf
D=gguf/GLM-5.3-Flash-REAP50-IQ3_M-MTP-draft.gguf
say() { echo "[$(date -Is)] draftcpu: $*" >> "$LOG"; }

say "pausing extraction chain"
./scripts/overnight_ctl.sh stop >> "$LOG" 2>&1
sleep 5

PROMPT="Write a Python function that returns the nth Fibonacci number."

for NGLD in 0 99; do
  for DR in 2 3; do
    say "=== ngld $NGLD depth $DR ==="
    timeout 2400 "$BIN/llama-speculative-simple" -m "$M" -md "$D" -ngl 24 -ngld $NGLD \
        --draft-max $DR --draft-min 1 -c 2048 -n 48 -t 8 -p "$PROMPT" >> "$LOG" 2>&1
    say "=== ngld $NGLD depth $DR rc=$? ==="
  done
done

say "resuming extraction chain"
./scripts/overnight_ctl.sh resume >> "$LOG" 2>&1
say "done"
