#!/usr/bin/env bash
# Importance matrix, then the IQ quants that need it.
#
# WHY Q4_K_M AND NOT Q8_0. imatrix wants the highest-precision model available, but with -ngl 0
# the weights stream per chunk, and MoE does not help: over a 512-token chunk essentially every
# expert is touched, so all 175 GB of Q8_0 moves each time. At ~84 MB/s that is ~35 min/chunk
# against the ~100 chunks a useful imatrix needs. Q4_K_M is 92.5 GiB and FITS IN RAM (122 GB), so
# it runs at memory speed. The cost is quantisation noise in the activation statistics - real,
# but the alternative is not running at all.
#
# Calibration is HELD-OUT text, not the REAP mixture: tuning the imatrix on the same data the
# prune was fitted to would couple two things that should stay independent.
set -u
cd /home/patrickd/glm-5.3-reap
BIN=~/glm5-llama.cpp/build-cuda/bin
LOG=logs/gguf_imatrix.log
Q8=gguf/GLM-5.3-Flash-REAP50-Q8_0.gguf
SRC=gguf/GLM-5.3-Flash-REAP50-Q4_K_M.gguf
IMAT=artifacts/imatrix-GLM-5.3-Flash-REAP50.dat
say(){ echo "[$(date -Is)] imatrix: $*" >> "$LOG"; }

if pgrep -x "llama-completion|llama-perplexity|llama-imatrix|llama-quantize|llama-mtmd-cli" > /dev/null; then
  say "ABORT: another llama process is running"; exit 1
fi

if [ ! -s "$IMAT" ]; then
  # Rebuild Q4_K_M purely as the imatrix vehicle; it was deleted after shipping.
  if [ ! -f "$SRC" ]; then
    free=$(df --output=avail -BG /home/patrickd | tail -1 | tr -dc 0-9)
    [ "$free" -lt 95 ] && { say "ABORT: ${free}G free, need ~95G"; exit 1; }
    say "rebuilding Q4_K_M as the imatrix source"
    $BIN/llama-quantize --allow-requantize "$Q8" "$SRC" Q4_K_M "$(nproc)" >> "$LOG" 2>&1 \
      || { say "FATAL: quantise failed"; exit 1; }
  fi
  # llama-imatrix has NO --no-warmup (that flag lives in cli/server only) -- passing it aborts
  # the run instantly. --no-ppl skips the perplexity pass we do not need here. --output-format
  # dat keeps the artifact readable by older llama-quantize builds the community may be on.
  #
  # NO --save-frequency. It does not overwrite one checkpoint, it writes a NUMBERED file per
  # save: 34 of them at 264 MB each filled the last 8.5 GB on this disk, and the final write
  # then failed silently (llama-imatrix does not check fwrite/fclose), leaving a truncated
  # artifact and exit code 0. The resilience flag destroyed the run it was meant to protect.
  # The guard that actually helps is verifying the artifact before deleting the vehicle, below.
  say "computing imatrix (held-out calibration, ~77k tokens)"
  $BIN/llama-imatrix -m "$SRC" -f artifacts/imatrix_calib.txt -o "$IMAT" \
      -c 512 -b 512 -ub 512 -ngl 0 --no-repack --no-ppl --output-format dat \
      -t "$(nproc)" >> "$LOG" 2>&1
  say "imatrix rc=$? -> $(ls -la "$IMAT" 2>/dev/null | awk '{print $5}') bytes"
  [ -s "$IMAT" ] || { say "FATAL: no imatrix produced"; exit 1; }
  # rc=0 is NOT enough. llama-imatrix does not check its final fwrite/fclose, so a full disk
  # yields exit 0 and a file truncated at a stdio flush boundary -- which happened here, and was
  # only caught by llama-quantize long AFTER this line deleted the one cheap way to retry.
  # Parse the artifact before destroying the vehicle that produced it.
  if ! ./.venv/bin/python scripts/imatrix_verify.py "$IMAT" >> "$LOG" 2>&1; then
    say "FATAL: imatrix is corrupt or truncated - KEEPING the vehicle so this can be redone"
    exit 1
  fi
  say "imatrix verified structurally"
  rm -f "$SRC"; say "removed the Q4_K_M vehicle; free $(df -h /home/patrickd | tail -1 | awk '{print $4}')"
fi

say "imatrix ready: $IMAT"
