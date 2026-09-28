#!/usr/bin/env bash
# Second shipping wave: the imatrix-dependent and smaller quants, then vision characterisation
# and a card refresh. Strictly serial - disk holds one ~90 GiB quant at a time beside the 175 GB
# Q8_0 source, and two llama workers on a 122 GB box is an OOM.
#
# Quant ladder chosen for who can actually run it:
#   IQ4_XS ~82 GiB  better quality-per-bit than Q4_K_S at a smaller size (needs the imatrix)
#   Q3_K_M ~70 GiB  comfortable on 96 GB machines
#   IQ3_M  ~66 GiB  reaches 80 GB machines
# Q5_K_M (~113 GiB) is deliberately skipped: it barely fits 128 GB and buys little over Q4_K_M.
set -u
cd /home/patrickd/glm-5.3-reap
BIN=~/glm5-llama.cpp/build-cuda/bin
REPO=patrickbdevaney/GLM-5.3-Flash-REAP50-GGUF
Q8=gguf/GLM-5.3-Flash-REAP50-Q8_0.gguf
IMAT=artifacts/imatrix-GLM-5.3-Flash-REAP50.dat
LOG=logs/gguf_ship2.log
export HF_XET_FIXED_UPLOAD_CONCURRENCY=8
export HF_XET_CHUNK_CACHE_SIZE_BYTES=$((2 * 1024 * 1024 * 1024))
say(){ echo "[$(date -Is)] ship2: $*" >> "$LOG"; }

# Wait out the imatrix run. Two ways to get this wrong, both hit once:
# 1. Waiting on the artifact alone: --save-frequency checkpoints to the FINAL path, so file-exists alone means "partial", not
#    "done", and a truncated imatrix quantises cleanly and is just quietly worse.
# 2. Waiting on the process alone: if the producer already DIED, nothing is running and we fall
#    straight through to FATAL -- which is exactly how this wave died at 16:38.
# So: wait out any live producer, then require the file.
for _ in $(seq 1 720); do
  pgrep -x "llama-imatrix|llama-quantize" > /dev/null || break
  sleep 60
done
pgrep -x "llama-imatrix|llama-quantize" > /dev/null \
  && { say "FATAL: producer still running after 12h"; exit 1; }
[ -s "$IMAT" ] || { say "FATAL: no imatrix at $IMAT"; exit 1; }
say "imatrix present ($(stat -c%s "$IMAT") bytes)"

for Q in IQ4_XS Q3_K_M IQ3_M; do
  F="gguf/GLM-5.3-Flash-REAP50-${Q}.gguf"
  NAME="$(basename "$F")"

  if ! ./.venv/bin/python -c "
import sys
from huggingface_hub import HfApi
n='$NAME'
have = any(s.rfilename==n for s in HfApi().model_info('$REPO', files_metadata=False).siblings)
sys.exit(0 if have else 1)" 2>/dev/null; then

    if [ ! -f "$F" ]; then
      free=$(df --output=avail -BG /home/patrickd | tail -1 | tr -dc 0-9)
      [ "$free" -lt 90 ] && { say "ABORT: ${free}G free, need ~90G for $Q"; exit 1; }
      say "quantising $Q (with imatrix)"
      $BIN/llama-quantize --allow-requantize --imatrix "$IMAT" "$Q8" "$F" "$Q" "$(nproc)" \
          >> "logs/gguf_quant_${Q}.log" 2>&1 || { say "FATAL: quantise $Q failed"; exit 1; }
    fi
    say "$Q built ($(du -h "$F" | cut -f1))"

    GATED="artifacts/ppl_$(basename "$F" .gguf).txt"
    if [ -s "$GATED" ]; then say "$Q already gated (ppl $(cat "$GATED"))"
    else
      ./scripts/gguf_gate.sh "$F" || { say "FATAL: $Q failed the gate - keeping it"; exit 1; }
      say "$Q passed the gate"
    fi

    # Characterise vision HERE, against a quant that is still on disk. Doing it at the end of the
    # run would find only the 175 GB Q8_0 left, which does not fit in RAM.
    if [ ! -s artifacts/vision_characterization.json ]; then
      say "vision characterisation (ChartQA, held-out) against $Q"
      ./.venv/bin/python scripts/vision_characterize.py "$F" 24 >> logs/vision_characterize.log 2>&1 \
        && say "vision: $(./.venv/bin/python -c "import json;d=json.load(open('artifacts/vision_characterization.json'));print(f\"{d['correct']}/{d['n']} = {d['accuracy']}\")" 2>/dev/null)" \
        || say "vision characterisation failed (non-fatal, continuing)"
    fi

    ok=0
    for a in $(seq 1 40); do
      if .venv/bin/hf upload "$REPO" "$F" "$NAME" --repo-type model >> "$LOG" 2>&1; then
        ok=1; say "$Q uploaded after $a attempt(s)"; break
      fi
      say "$Q upload attempt $a failed; retry in 60s"; sleep 60
    done
    [ $ok -eq 1 ] || { say "FATAL: $Q upload gave up"; exit 1; }

    if .venv/bin/python - "$REPO" "$NAME" "$F" >> "$LOG" 2>&1 <<'EOF'
import sys, pathlib
from huggingface_hub import HfApi
repo, name, local = sys.argv[1], sys.argv[2], sys.argv[3]
sz = {s.rfilename: s.size for s in HfApi().model_info(repo, files_metadata=True).siblings}
want = pathlib.Path(local).stat().st_size
print(f"hub check {name}: local {want} remote {sz.get(name)}")
sys.exit(0 if sz.get(name) == want else 1)
EOF
    then rm -f "$F"; say "$Q verified on the Hub and removed; free $(df -h /home/patrickd | tail -1 | awk '{print $4}')"
    else say "FATAL: $Q size mismatch on the Hub - keeping local"; exit 1; fi
  else
    say "$Q already on the Hub - skipping"
  fi
done

# Upload the imatrix itself: it lets anyone reproduce these quants or make their own.
.venv/bin/hf upload "$REPO" "$IMAT" "$(basename "$IMAT")" --repo-type model >> "$LOG" 2>&1 \
  && say "imatrix uploaded"

# Vision is characterised inside the loop above, against a quant that is still on disk. There is
# deliberately no call here: by this point every quant has been uploaded and deleted, and the
# only local model left is the 175 GB Q8_0.

./.venv/bin/python scripts/gguf_card.py >> "$LOG" 2>&1
.venv/bin/hf upload "$REPO" gguf_README.md README.md --repo-type model >> "$LOG" 2>&1
say "card refreshed"
say "wave 2 complete"
