#!/usr/bin/env bash
# Quantise one level from the Q8_0 intermediate, verify it loads and generates, then hand off.
# Push/delete is a separate step so a bad quant never reaches the Hub.
set -u
cd /home/patrickd/glm-5.3-reap
Q="${1:?usage: gguf_quant.sh Q4_K_M}"
SRC=gguf/GLM-5.3-Flash-REAP50-Q8_0.gguf
OUT="gguf/GLM-5.3-Flash-REAP50-${Q}.gguf"
LOG="logs/gguf_quant_${Q}.log"
say(){ echo "[$(date -Is)] $*" >> "$LOG"; }

say "quantising -> $Q"
df -h /home/patrickd | tail -1 >> "$LOG"
# --allow-requantize: the source is Q8_0, which measured +0.78% on top of the 4-bit error
# versus quantising from F32 - see scripts/gguf_intermediate_cost.py.
~/glm5-llama.cpp/build-cpu/bin/llama-quantize --allow-requantize \
    "$SRC" "$OUT" "$Q" "$(nproc)" >> "$LOG" 2>&1
rc=$?; say "quantize rc=$rc"
[ $rc -eq 0 ] || exit 1
ls -la "$OUT" >> "$LOG"

say "verifying: load + generate"
~/glm5-llama.cpp/build-cpu/bin/llama-completion \
    -m "$OUT" -t "$(nproc)" -c 512 -n 32 --temp 0 --no-warmup --no-repack \
    -p "Q: What is 17 multiplied by 23?
A:" >> "$LOG" 2>&1
say "verify rc=$?"
df -h /home/patrickd | tail -1 >> "$LOG"
