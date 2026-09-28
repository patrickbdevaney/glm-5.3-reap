#!/usr/bin/env bash
# Capture NVFP4 teacher states for the probe corpus and measure how far they sit from the GGUF
# ones already captured.
#
# The two-head decision was made on a number: IQ3_M vs IQ4_XS final hidden states agree to 0.891
# mean cosine against a 0.999 same-quant control, so a draft head is quant-specific and the GGUF
# head cannot be reused for the NVFP4 server. That was measured between two GGUF quants. This
# measures the one that actually matters - NVFP4 against the quant whose traces we already have -
# and it does it on 30 sequences before committing the GPU to a full capture.
#
# It also validates the capture path end to end. If the streaming forward is wrong, the drift will
# not look like drift: it will look like noise, and the control below is what tells them apart.
set -uo pipefail
cd /home/patrickd/glm-5.3-reap
LOG=logs/nvfp4_probe.log
say() { echo "[$(date -Is)] nvfp4_probe: $*" >> "$LOG"; }

say "pausing extraction chain"
./scripts/overnight_ctl.sh stop >> "$LOG" 2>&1
sleep 5

say "=== capture ==="
./.venv/bin/python scripts/nvfp4_trace_capture.py \
    --out artifacts/qt_nvfp4.bin >> "$LOG" 2>&1
RC=$?
say "=== capture rc=$RC ==="

if [ $RC -eq 0 ]; then
  # Against IQ3_M, which is what the 20M-token capture is producing.
  say "=== NVFP4 vs IQ3_M ==="
  ./.venv/bin/python scripts/quant_transfer.py \
      traces/chunk_0000.bin traces/chunk_0000.bin.ids \
      artifacts/qt_nvfp4.bin artifacts/qt_nvfp4.bin.ids \
      traces/chunk_0000.bin.lens artifacts/qt_nvfp4.bin.lens >> "$LOG" 2>&1
  say "=== rc=$? ==="

  # And against IQ4_XS, so the NVFP4 distance can be read against the 0.891 already measured
  # between the two GGUF quants rather than against nothing.
  say "=== NVFP4 vs IQ4_XS ==="
  ./.venv/bin/python scripts/quant_transfer.py \
      artifacts/qt_alt.bin artifacts/qt_alt.bin.ids \
      artifacts/qt_nvfp4.bin artifacts/qt_nvfp4.bin.ids \
      artifacts/qt_alt.bin.lens artifacts/qt_nvfp4.bin.lens >> "$LOG" 2>&1
  say "=== rc=$? ==="
fi

say "resuming extraction chain"
./scripts/overnight_ctl.sh resume >> "$LOG" 2>&1
say "done"
