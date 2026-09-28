#!/usr/bin/env bash
# Does lowering the op-offload threshold remove the batch-32 cliff?
#
# The batch cost curve at -ngl 24 is flat from 1 to 16 tokens (334 -> 233 ms/token) and then drops
# 5.4x at 32 (43 ms/token). ggml_backend_cuda_device_offload_op is
# `get_op_batch_size(op) >= dev_ctx->op_offload_min_batch_size`, and that threshold defaults to 32:
# below it the 22 layers that -ngl 24 leaves in system memory are computed by 8 CPU threads, above
# it their matmuls are handed to the GPU.
#
# That default is tuned for a discrete GPU, where offloading an op means copying its weights across
# PCIe and only pays for a big batch. Thor's memory is unified - there is no bus to cross - so the
# threshold may be costing far more than it saves. It matters for speculative decoding because a
# draft of depth K produces a batch of K+1, which is always on the wrong side of 32.
#
# GGML_OP_OFFLOAD_MIN_BATCH is read at backend registration, so this needs no rebuild.
set -uo pipefail
cd /home/patrickd/glm-5.3-reap
LOG=logs/mtp_offload.log
BIN=$HOME/glm5-llama.cpp/build-cuda/bin
M=gguf/GLM-5.3-Flash-REAP50-IQ3_M.gguf
say() { echo "[$(date -Is)] offload: $*" >> "$LOG"; }

say "pausing extraction chain"
./scripts/overnight_ctl.sh stop >> "$LOG" 2>&1
sleep 5

for MB in 1 8; do
  say "=== GGML_OP_OFFLOAD_MIN_BATCH=$MB, ngl 24 ==="
  GGML_OP_OFFLOAD_MIN_BATCH=$MB timeout 3600 "$BIN/llama-bench" \
      -m "$M" -ngl 24 -p 2,3,4,8,16,32 -n 1 -r 2 -t 8 >> "$LOG" 2>&1
  say "=== rc=$? ==="
done

say "resuming extraction chain"
./scripts/overnight_ctl.sh resume >> "$LOG" 2>&1
say "done"
