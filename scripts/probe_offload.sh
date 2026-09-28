#!/usr/bin/env bash
# Does GLM-5.3 fit on Thor's GPU well enough to be a teacher? One measurement, then we know
# whether the NVFP4 CUDA server gates the draft-head work or not.
#
# Thor is unified memory: -ngl offloads into the SAME 122 GB the weights are already mmap'd in,
# so a model that mmaps fine at -ngl 0 can still OOM the machine when offloaded (measured before
# at 93 GiB). So this walks -ngl UP from 0 and stops at the first failure instead of jumping to
# 99 and taking the box down. IQ3_M is used because it is the smallest quant and therefore the
# one with the most headroom - if it cannot offload, none of them can.
set -uo pipefail
cd /home/patrickd/glm-5.3-reap
LOG=logs/probe_offload.log
BIN=$HOME/glm5-llama.cpp/build-cuda/bin
M=gguf/GLM-5.3-Flash-REAP50-IQ3_M.gguf
say() { echo "[$(date -Is)] probe: $*" >> "$LOG"; }

say "wave verified complete at 00:18; proceeding"

if [ ! -f "$M" ]; then
  say "fetching IQ3_M (smallest quant = most offload headroom)"
  ./.venv/bin/hf download patrickbdevaney/GLM-5.3-Flash-REAP50-GGUF \
      GLM-5.3-Flash-REAP50-IQ3_M.gguf --local-dir gguf >> "$LOG" 2>&1 \
      || { say "FATAL: download failed"; exit 1; }
fi
say "model $(du -h "$M" | cut -f1); mem available $(free -g | awk 'NR==2{print $7}') GiB"

for NGL in 0 8 16 24 32 45; do
  AVAIL=$(free -g | awk 'NR==2{print $7}')
  say "--- ngl=$NGL (available ${AVAIL} GiB) ---"
  timeout 3600 $BIN/llama-bench -m "$M" -ngl $NGL -p 512 -n 32 -r 1 \
      >> "$LOG" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    say "ngl=$NGL FAILED (rc=$rc) - this is the ceiling; stopping before it takes the box down"
    break
  fi
  say "ngl=$NGL ok"
done
say "done - see the pp512 (prefill) and tg32 (decode) rows above"
say "prefill tok/s is what decides teacher extraction time; decode is what decides serving"
