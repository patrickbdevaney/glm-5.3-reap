#!/usr/bin/env bash
# FP8 release checkpoint -> Q8_0 GGUF intermediate.
#
# Q8_0 rather than BF16 because BF16 needs 308 GiB and only 262 GiB is free, which would force
# deleting the local FP8 master. The usual "never requantize" warning does not bite here: the
# source is FP8 E4M3 with 3 mantissa bits (~6% relative precision per value), while Q8_0 is int8
# against a per-32-block fp16 scale (~0.4% of block max). Q8_0 represents the FP8 values
# essentially exactly, so FP8 -> Q8_0 -> Q4_K_M has ONE real lossy step, not two.
#
# MTP is kept: llama.cpp cannot execute it, but the weights ship so the files are ready if it
# ever gains support. Costs ~2 GiB at Q4_K_M.
set -u
cd /home/patrickd/glm-5.3-reap
SRC=output/glm-5.3-flash-reap50-fp8-pass2
OUT=gguf/GLM-5.3-Flash-REAP50-Q8_0.gguf
LOG=logs/gguf_convert.log
echo "[$(date -Is)] converting $SRC -> $OUT (expect ~165 GiB)" >> "$LOG"
df -h /home/patrickd | tail -1 >> "$LOG"
GLM5_KEEP_MTP=1 ./.venv/bin/python ~/glm5-llama.cpp/convert_hf_to_gguf.py "$SRC" \
    --outfile "$OUT" --outtype q8_0 >> "$LOG" 2>&1
rc=$?
echo "[$(date -Is)] convert rc=$rc" >> "$LOG"
ls -la "$OUT" >> "$LOG" 2>&1
df -h /home/patrickd | tail -1 >> "$LOG"
