#!/usr/bin/env bash
# The decisive gate: does the REAL model produce coherent text through the new port?
#
# The tiny fixture proves the graph matches transformers on random weights. It cannot prove the
# 165B model is right - a subtly wrong mHC or KDA path still matches a random-weight reference to
# a few e-3 while producing fluent nonsense at scale. Coherent, on-topic output from real weights
# is the check that actually discriminates.
#
# CPU only: ggml_mhc_sinkhorn has no CUDA kernel yet. The model is 163 GiB against 101 GiB of
# available RAM, so it streams from NVMe - but only 8 of 144 experts are read per token, so the
# steady-state working set is far smaller than the file.
set -u
cd /home/patrickd/glm-5.3-reap
LOG=logs/gguf_smoke.log
M=gguf/GLM-5.3-Flash-REAP50-Q8_0.gguf
echo "[$(date -Is)] smoke test on $M" >> "$LOG"
# llama-completion, not llama-cli: this build's cli is a chat frontend and drops into an
# interactive prompt even with -no-cnv.
#
# --no-repack is required, not a tuning knob: the ARM CPU backend repacks weights into an
# optimised layout, which is a full 166 GB copy into 122 GB of RAM. With it off the file stays
# memory-mapped and only the ~8-of-144 experts each token touches are resident.
for P in "The capital of France is Paris. The capital of Japan is" \
         "def fibonacci(n):" \
         "Q: What is 17 multiplied by 23?\nA:"; do
  echo "" >> "$LOG"; echo "=== PROMPT: $P" >> "$LOG"
  ~/glm5-llama.cpp/build-cpu/bin/llama-completion \
      -m "$M" -t "$(nproc)" -c 512 -n 48 --temp 0 --no-warmup --no-repack \
      -p "$P" >> "$LOG" 2>&1
done
echo "[$(date -Is)] smoke rc=$?" >> "$LOG"
