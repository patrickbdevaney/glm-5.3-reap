#!/usr/bin/env bash
# CUDA build of the fork. Runs at -j4, not -j$(nproc): the GGUF gate is CPU-bound and running
# concurrently, and starving it just moves the bottleneck.
set -u
cd ~/glm5-llama.cpp
LOG=/home/patrickd/glm-5.3-reap/logs/build_cuda.log
echo "[$(date -Is)] configuring CUDA build (sm_110, Thor)" > "$LOG"
cmake -B build-cuda -DGGML_CUDA=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES=110 >> "$LOG" 2>&1 || { echo "configure failed" >> "$LOG"; exit 1; }
echo "[$(date -Is)] building" >> "$LOG"
cmake --build build-cuda --target llama-completion llama-quantize llama-perplexity llama-imatrix \
      test-mhc-sinkhorn test-glm5-next-logits -j4 >> "$LOG" 2>&1
echo "[$(date -Is)] build rc=$?" >> "$LOG"
