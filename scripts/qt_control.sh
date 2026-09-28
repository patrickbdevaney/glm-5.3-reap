#!/usr/bin/env bash
# Control for the quant-transfer measurement.
#
# The IQ3_M-vs-IQ4_XS comparison came back at cosine 0.89, which is either a real and important
# finding or a broken comparison. There is one way to tell them apart: run the SAME probe under
# the SAME quant the reference came from. Two IQ3_M runs of identical tokens must agree to ~1.0.
# If they do, the 0.89 is genuine quant drift. If they do not, the fault is in the harness -
# row layout, sequence offsets, or batching nondeterminism - and the 0.89 means nothing.
#
# This is cheap (IQ3_M is 67 GB and runs at ~78 tok/s, versus the 8 tok/s the 88 GB IQ4_XS
# managed under the same ngl), so the chain only pauses for a few minutes.
set -uo pipefail
cd /home/patrickd/glm-5.3-reap
LOG=logs/qt_control.log
BIN=$HOME/glm5-llama.cpp/build-cuda/bin
REF=gguf/GLM-5.3-Flash-REAP50-IQ3_M.gguf
SEP='<#sep#>'
say() { echo "[$(date -Is)] qtc: $*" >> "$LOG"; }

say "pausing extraction chain"
./scripts/overnight_ctl.sh stop >> "$LOG" 2>&1
sleep 5

say "re-extracting the probe under IQ3_M (the reference quant)"
rm -f artifacts/qt_ctl.bin artifacts/qt_ctl.bin.ids artifacts/qt_ctl.bin.lens
LLAMA_EMBD_BIN=artifacts/qt_ctl.bin timeout 5400 "$BIN/llama-embedding" -m "$REF" -ngl 16 \
    -c 2048 -b 2048 -ub 512 --pooling none --embd-normalize -1 \
    --embd-separator "$SEP" -f artifacts/qt_probe.txt > /dev/null 2>> "$LOG"
rc=$?; say "control extraction rc=$rc"

say "resuming extraction chain"
./scripts/overnight_ctl.sh resume >> "$LOG" 2>&1

say "--- CONTROL: IQ3_M vs IQ3_M (same quant, must be ~1.0) ---"
./.venv/bin/python scripts/quant_transfer.py \
    artifacts/qt_ref.bin artifacts/qt_ref.ids \
    artifacts/qt_ctl.bin artifacts/qt_ctl.bin.ids \
    traces/chunk_0000.bin.lens artifacts/qt_ctl.bin.lens >> "$LOG" 2>&1
say "done"
