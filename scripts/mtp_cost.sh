#!/usr/bin/env bash
# Where does the speculative iteration actually spend its time?
#
# The mechanism works - 73.7% acceptance at depth 1, verified - but end to end it is slower than
# not speculating at all. Three things could be eating it: the extra forward pass the recurrent
# rollback needs on a rejection, the 145 MiB state checkpoint taken every iteration, or the draft
# model itself. The draft is one MoE block, but it carries the full 154880-row embedding and
# output matrices, so "one layer" is not the same as "cheap". This measures the draft on its own
# so the three can be told apart.
set -uo pipefail
cd /home/patrickd/glm-5.3-reap
LOG=logs/mtp_cost.log
BIN=$HOME/glm5-llama.cpp/build-cuda/bin
M=gguf/GLM-5.3-Flash-REAP50-IQ3_M.gguf
D=gguf/GLM-5.3-Flash-REAP50-IQ3_M-MTP-draft.gguf
say() { echo "[$(date -Is)] cost: $*" >> "$LOG"; }

say "pausing extraction chain"
./scripts/overnight_ctl.sh stop >> "$LOG" 2>&1
sleep 5

PROMPT="Write a Python function that returns the nth Fibonacci number."

# The draft alone. It is run as an ordinary autoregressive model here, which is not what it does
# in the speculative loop (it gets a hidden state, not its own history), but the per-token cost of
# the graph is the same and that is what is being measured.
say "=== draft standalone ==="
timeout 1200 "$BIN/llama-completion" -m "$D" -ngl 99 -c 2048 -n 32 -t 8 \
    -p "$PROMPT" --seed 1 >> "$LOG" 2>&1
say "=== draft standalone rc=$? ==="

# Depth 1 again on the rebuilt binary, to confirm the deferred replay did not change the result.
say "=== depth 1 (rebuilt) ==="
timeout 2400 "$BIN/llama-speculative-simple" \
    -m "$M" -md "$D" -ngl 24 -ngld 99 \
    --draft-max 1 --draft-min 1 -c 2048 -n 32 -t 8 \
    -p "$PROMPT" >> "$LOG" 2>&1
say "=== depth 1 rc=$? ==="

say "resuming extraction chain"
./scripts/overnight_ctl.sh resume >> "$LOG" 2>&1
say "done"
