#!/usr/bin/env bash
# First forward pass through the vision tower in this project's history.
#
# REAP pruned only the language-model FFN experts - the ViT was untouched - but the language
# model is what consumes the visual tokens, and the teacher-forced eval could never measure it:
# after excluding image-placeholder positions only 532 vision tokens remained, with a NEGATIVE
# dNLL (student apparently beating its teacher), which is the signature of the placeholder bug
# that inverted every headline earlier in this project. So vision is not merely unmeasured, the
# one number we have is probably wrong.
#
# The image has verifiable content on purpose: a black "42", a red square, a blue circle. The
# answer is pass/fail, not a judgement call.
set -u
cd /home/patrickd/glm-5.3-reap
LOG=logs/vision_smoke.log
BIN=~/glm5-llama.cpp/build-cuda/bin
: > "$LOG"
say(){ echo "[$(date -Is)] $*" >> "$LOG"; }

# Bracket trick: a plain pattern also matches the caller's own shell when its command line
# happens to contain the string, which is exactly how this guard blocked itself twice.
if pgrep -x "llama-completion|llama-perplexity|llama-imatrix|llama-quantize|llama-mtmd-cli" > /dev/null; then
  say "ABORT: another llama process is running"; exit 1
fi
AVAIL=$(free -g | awk '/^Mem:/{print $7}')
[ "$AVAIL" -lt 40 ] && { say "ABORT: only ${AVAIL}G available"; exit 1; }
say "available ${AVAIL}G"

M=gguf/GLM-5.3-Flash-REAP50-Q4_K_M.gguf
[ -f "$M" ] || M=gguf/GLM-5.3-Flash-REAP50-Q8_0.gguf
say "model $M"

for P in "What number is written in this image?" \
         "Describe the shapes and their colors in this image."; do
  say "PROMPT: $P"
  $BIN/llama-mtmd-cli -m "$M" --mmproj gguf/mmproj-GLM-5.3-Flash-REAP50-F16.gguf \
      --image artifacts/test_image.png -p "$P" \
      -ngl 0 --no-repack -c 4096 -n 96 --temp 0 >> "$LOG" 2>&1
  say "  rc=$?"
done
say "done"
