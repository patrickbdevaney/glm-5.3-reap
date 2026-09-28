#!/usr/bin/env bash
# Decode-speed verification for the draft head.
#
# Acceptance rate alone does not tell you the speedup. A head can accept 70% of its drafts and
# still lose, if drafting costs more than the tokens it saves. The only number that settles it
# is end-to-end tokens/sec on the SAME quant, with and without the draft, at the same ngl.
#
#   ./scripts/verify_decode.sh                      # baseline only
#   ./scripts/verify_decode.sh <draft.gguf> [quant.gguf]
set -uo pipefail
cd /home/patrickd/glm-5.3-reap
BIN=$HOME/glm5-llama.cpp/build-cuda/bin
DRAFT=${1:-}
M=${2:-gguf/GLM-5.3-Flash-REAP50-IQ3_M.gguf}
NGL=${NGL:-24}          # serving config, not the extraction config: nothing else contends here
LOG=logs/verify_decode.log
say() { echo "[$(date -Is)] decode: $*" >> "$LOG"; }

[ -f "$M" ] || { say "missing $M"; exit 1; }
say "=== baseline: $(basename "$M") ngl=$NGL ==="
"$BIN/llama-bench" -m "$M" -ngl $NGL -p 512 -n 128 -r 2 >> "$LOG" 2>&1
say "baseline recorded (tg row is the decode number that matters)"

if [ -z "$DRAFT" ]; then
  say "no draft given; baseline only. Re-run with a draft gguf to get the speedup."
  exit 0
fi
[ -f "$DRAFT" ] || { say "missing draft $DRAFT"; exit 1; }

# Acceptance is prompt-dependent, so measure it on the same capability mix the head was
# trained for rather than on one cherry-picked prompt.
#
# llama-speculative-simple, NOT llama-speculative: the older tool carries its own decode loop
# and never calls common_speculative_draft(), so it cannot drive an MTP head at all. It would
# run happily and report a speedup for a completely different mechanism.
say "=== speculative: draft=$(basename "$DRAFT") ==="
while read -r PROMPT; do
  [ -z "$PROMPT" ] && continue
  for DR in 1 2 3; do
    say "--- depth $DR :: ${PROMPT:0:52} ---"
    timeout 1800 "$BIN/llama-speculative-simple" -m "$M" -md "$DRAFT" -ngl $NGL -ngld $NGL \
        --draft-max $DR --draft-min 1 -c 4096 -n 128 -t 8 --top-k 1 \
        -p "$PROMPT" >> "$LOG" 2>&1
  done
done < artifacts/decode_prompts.txt

# llama-speculative-simple prints n_drafted / n_accepted and the end-to-end t/s. Acceptance is
# accepted/drafted; the number that decides whether to ship is the t/s against the baseline
# above, because a head can accept most of its drafts and still lose if drafting costs more
# than the tokens it saves.
#
# Depths 1-3 only. GLM-5.3 has ONE nextn layer, so depth > 1 reuses the same block on its own
# output; error compounds and the reference deployments draft one token. Depth is measured
# rather than assumed, but do not expect it to keep paying past 2-3.
say "=== done - grep the log for 'accept' and 'tokens per second' ==="
