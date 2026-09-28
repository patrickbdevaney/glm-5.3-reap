#!/usr/bin/env bash
# The batch-size cost curve, which is the hard ceiling on any speculative speedup.
#
# Verifying a draft of depth K means decoding a batch of K+1 tokens. Speculation can only pay if
# that batch costs less than K+1 single-token decodes - the acceptance rate decides how much of the
# saving is realised, but the saving itself is set here, by the model and the hardware, before any
# draft head exists.
#
# Two numbers are already known and they disagree about what to expect: a 138-token batch costs
# 52 ms/token against 335 ms for one at a time (6.4x of headroom), but a 2-token verification batch
# in the real speculative loop looked like ~292 ms/token (barely 1.15x). If that is right the
# amortisation only arrives at batch sizes far larger than a draft ever produces, which would cap
# GGUF speculation on this model no matter how good the head gets. This measures the small end of
# the curve directly instead of inferring it by subtraction from a mixed run.
set -uo pipefail
cd /home/patrickd/glm-5.3-reap
LOG=logs/mtp_batchcurve.log
BIN=$HOME/glm5-llama.cpp/build-cuda/bin
M=gguf/GLM-5.3-Flash-REAP50-IQ3_M.gguf
say() { echo "[$(date -Is)] curve: $*" >> "$LOG"; }

say "pausing extraction chain"
./scripts/overnight_ctl.sh stop >> "$LOG" 2>&1
sleep 5

say "=== batch curve, ngl 24 ==="
timeout 3600 "$BIN/llama-bench" -m "$M" -ngl 24 -p 2,3,4,5,8,16,32 -n 1 -r 2 -t 8 >> "$LOG" 2>&1
say "=== rc=$? ==="

say "resuming extraction chain"
./scripts/overnight_ctl.sh resume >> "$LOG" 2>&1
say "done"
