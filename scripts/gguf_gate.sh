#!/usr/bin/env bash
# Short gate for one GGUF. A tripwire, not a benchmark.
#
# TWO model loads, not four: each load of a 93 GiB file costs ~8 minutes, and the first version
# of this gate spent half an hour loading and gave four chances to collide with another process.
#
# -ngl 0 ON PURPOSE. Thor is unified memory, so offloaded layers become NON-EVICTABLE device
# buffers - -ngl 99 on a 93 GiB model against 122 GB total OOMed the machine. With -ngl 0 and
# --no-repack the file stays mmap'd, pages stay evictable, and the run cannot take the box down.
# Slower, and worth it: this is a correctness tripwire, not a throughput measurement.
set -u
cd /home/patrickd/glm-5.3-reap
M="${1:?usage: gguf_gate.sh <model.gguf>}"
LOG="logs/gate_$(basename "$M" .gguf).log"
BIN=~/glm5-llama.cpp/build-cuda/bin
NGL="${GLM5_NGL:-0}"
: > "$LOG"
say(){ echo "[$(date -Is)] $*" >> "$LOG"; }

# Never run two of these at once, and never start without headroom. Two processes each holding
# a 93 GiB model is an OOM, and it reports as a content failure rather than a resource one.
if pgrep -x "llama-completion|llama-perplexity|llama-imatrix|llama-quantize|llama-mtmd-cli" > /dev/null; then
  say "ABORT: another llama process is running; refusing to compete for memory"; exit 1
fi
AVAIL=$(free -g | awk '/^Mem:/{print $7}')
if [ "$AVAIL" -lt 40 ]; then
  say "ABORT: only ${AVAIL}G available; refusing to start"; exit 1
fi
say "gate for $M | available ${AVAIL}G | ngl=$NGL (mmap, evictable)"

fail=0

say "1/2 generation"
OUT=$($BIN/llama-completion -m "$M" -c 512 -n 160 --temp 0 -ngl "$NGL" \
      --no-warmup --no-repack -p "Q: What is 17 multiplied by 23?
A:" 2>&1 | tr '\r' '\n')
if echo "$OUT" | grep -qE "error loading model|failed to load model|ggml_abort|CUDA error"; then
  say "  ERROR  model failed to load (resource problem, not a bad quant)"; fail=1
elif echo "$OUT" | grep -q "391"; then
  say "  PASS  arithmetic correct"
else
  say "  FAIL  no 391 - got: $(echo "$OUT" | tail -4 | tr '\n' ' ' | cut -c1-200)"; fail=1
fi

# -b/-ub 512: without them llama_params_fit probes CUDA and reserves a compute buffer sized for
# the default 2048 batch, which aborts in sched_reserve even at -ngl 0. llama-completion survives
# because it batches smaller. Bounding the batch is the fix, not lowering -ngl further.
say "2/2 perplexity on held-out text"
PLOG=$($BIN/llama-perplexity -m "$M" -f artifacts/ppl_gate.txt -c 512 -b 512 -ub 512 --chunks 8 \
       -ngl "$NGL" --no-warmup --no-repack 2>&1 | tr '\r' '\n')
V=$(echo "$PLOG" | grep -oE "Final estimate: PPL = [0-9.]+" | grep -oE "[0-9.]+$" | tail -1)
if [ -z "$V" ]; then
  say "  FAIL  no perplexity: $(echo "$PLOG" | tail -3 | tr '\n' ' ' | cut -c1-200)"; fail=1
elif awk "BEGIN{exit !($V > 0 && $V < 40)}"; then
  say "  PASS  ppl $V in (0, 40)"
  echo "$V" > "artifacts/ppl_$(basename "$M" .gguf).txt"
else
  say "  FAIL  ppl $V outside (0, 40)"; fail=1
fi

say "GATE $([ $fail -eq 0 ] && echo PASS || echo FAIL)"
exit $fail
