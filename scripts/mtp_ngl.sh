#!/usr/bin/env bash
# Does speculative decoding have any headroom on this target, and at what offload level?
#
# At -ngl 24 a 2-token verification batch costs 585 ms against 328 ms for a single token: batching
# buys almost nothing, so verifying K+1 tokens costs nearly K+1 decodes and no acceptance rate can
# pay for that. Two things could cause it - the 22 layers still on the CPU, whose cost is compute
# bound and therefore linear in batch size, or the MoE itself, where 2 tokens route to nearly twice
# as many experts and so read nearly twice the weights however fast the device is.
#
# The prompt-eval / decode ratio separates them. It is the batching efficiency of the target and it
# is the hard ceiling on any speculative speedup: if a batch of 17 only costs 1.36x less per token
# than one at a time, then no draft head, however good, can do better than 1.36x.
#
# -ngl 36 rather than 99 on purpose: offloaded layers are non-evictable on unified memory and
# taking the whole 67 GiB device-side alongside everything else risks taking the machine down with
# the trace extraction on it.
set -uo pipefail
cd /home/patrickd/glm-5.3-reap
LOG=logs/mtp_ngl.log
BIN=$HOME/glm5-llama.cpp/build-cuda/bin
M=gguf/GLM-5.3-Flash-REAP50-IQ3_M.gguf
D=gguf/GLM-5.3-Flash-REAP50-IQ3_M-MTP-draft.gguf
say() { echo "[$(date -Is)] ngl: $*" >> "$LOG"; }

say "pausing extraction chain"
./scripts/overnight_ctl.sh stop >> "$LOG" 2>&1
sleep 5

# A long-ish prompt so prompt eval is a real batched measurement rather than a rounding error.
P=$(head -c 1400 artifacts/decode_prompts.txt | tr '\n' ' ')

for NGL in 24 36; do
  say "=== ngl $NGL baseline ==="
  timeout 1800 "$BIN/llama-completion" -m "$M" -ngl $NGL -c 2048 -n 24 -t 8 \
      -p "$P" --seed 1 >> "$LOG" 2>&1
  say "=== ngl $NGL baseline rc=$? ==="

  say "=== ngl $NGL depth 2 ==="
  timeout 1800 "$BIN/llama-speculative-simple" -m "$M" -md "$D" -ngl $NGL -ngld 99 \
      --draft-max 2 --draft-min 1 -c 2048 -n 24 -t 8 -p "$P" >> "$LOG" 2>&1
  say "=== ngl $NGL depth 2 rc=$? ==="
done

say "resuming extraction chain"
./scripts/overnight_ctl.sh resume >> "$LOG" 2>&1
say "done"
