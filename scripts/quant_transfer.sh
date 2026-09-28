#!/usr/bin/env bash
# Measure whether the IQ3_M-trained draft head will transfer to the other published quants.
# Downloads a second quant (network only - no cloud spend, nothing deleted), briefly pauses
# extraction to reuse the GPU, then hands the chain straight back.
set -uo pipefail
cd /home/patrickd/glm-5.3-reap
LOG=logs/quant_transfer.log
BIN=$HOME/glm5-llama.cpp/build-cuda/bin
REF=gguf/GLM-5.3-Flash-REAP50-IQ3_M.gguf
ALT_NAME=${ALT_NAME:-GLM-5.3-Flash-REAP50-IQ4_XS.gguf}
ALT=gguf/$ALT_NAME
NPROMPT=${NPROMPT:-30}
SEP='<#sep#>'
say() { echo "[$(date -Is)] qt: $*" >> "$LOG"; }

# Build the probe as a PREFIX of an already-extracted chunk, so the IQ3_M side is free:
# the reference rows are just the first sum(lens[:N]) rows of a bin we already have.
say "building probe from the first $NPROMPT prompts of chunk_0000"
./.venv/bin/python - "$NPROMPT" <<'PY' >> "$LOG" 2>&1 || exit 1
import sys, numpy as np
n = int(sys.argv[1]); SEP = "<#sep#>"
txt = open("chunks/chunk_0000.txt", encoding="utf-8").read().split(SEP)[:n]
open("artifacts/qt_probe.txt", "w", encoding="utf-8").write(SEP.join(txt))
lens = np.fromfile("traces/chunk_0000.bin.lens", dtype=np.int32)[:n]
ntok = int(lens.sum())
D = 4096
# reference slice, taken from the run we already paid for
ref = np.fromfile("traces/chunk_0000.bin", dtype=np.float16, count=ntok*D)
ref.tofile("artifacts/qt_ref.bin")
np.fromfile("traces/chunk_0000.bin.ids", dtype=np.int32, count=ntok).tofile("artifacts/qt_ref.ids")
print(f"probe: {n} prompts, {ntok} tokens")
PY

FREE=$(df --output=avail -BG / | tail -1 | tr -dc '0-9')
if [ ! -f "$ALT" ]; then
  # Remaining extraction needs ~131G; refuse to squeeze it for a diagnostic.
  if [ "$FREE" -lt 200 ]; then say "only ${FREE}G free - refusing to download $ALT_NAME"; exit 1; fi
  say "downloading $ALT_NAME (${FREE}G free)"
  ./.venv/bin/hf download patrickbdevaney/GLM-5.3-Flash-REAP50-GGUF "$ALT_NAME" \
      --local-dir gguf >> "$LOG" 2>&1 || { say "download failed"; exit 1; }
fi
say "have $ALT ($(du -h "$ALT"|cut -f1))"

# Pause the chain rather than race it for memory. Chunk-granular, so this costs at most the
# chunk in flight, which is re-run on resume.
say "pausing extraction chain"
./scripts/overnight_ctl.sh stop >> "$LOG" 2>&1
sleep 5

say "extracting probe under $ALT_NAME"
rm -f artifacts/qt_alt.bin artifacts/qt_alt.bin.ids artifacts/qt_alt.bin.lens
LLAMA_EMBD_BIN=artifacts/qt_alt.bin timeout 5400 "$BIN/llama-embedding" -m "$ALT" -ngl 16 \
    -c 2048 -b 2048 -ub 512 --pooling none --embd-normalize -1 \
    --embd-separator "$SEP" -f artifacts/qt_probe.txt > /dev/null 2>> "$LOG"
rc=$?; say "probe extraction rc=$rc"

say "resuming extraction chain"
./scripts/overnight_ctl.sh resume >> "$LOG" 2>&1

if [ $rc -eq 0 ] && [ -f artifacts/qt_alt.bin.ids ]; then
  say "--- IQ3_M vs $ALT_NAME ---"
  ./.venv/bin/python scripts/quant_transfer.py \
      artifacts/qt_ref.bin artifacts/qt_ref.ids \
      artifacts/qt_alt.bin artifacts/qt_alt.bin.ids >> "$LOG" 2>&1
else
  say "probe failed; comparison skipped"
fi
say "done"
