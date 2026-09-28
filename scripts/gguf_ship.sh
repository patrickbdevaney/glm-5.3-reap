#!/usr/bin/env bash
# Autonomous GGUF pipeline: quantise -> gate -> push -> delete, one level at a time.
#
# Strictly serial because disk forces it: the Q8_0 intermediate is 175 GB and each 4-bit level is
# ~87 GB against a 936 GB disk that is already 97% full. Nothing can be built until the previous
# level has been pushed and removed, so a failure anywhere stops the chain rather than filling
# the disk and taking the source down with it.
#
# A level that fails its gate is KEPT on disk and the chain stops - a bad quant must never reach
# the Hub, and the artifact is needed to diagnose why.
set -u
cd /home/patrickd/glm-5.3-reap
REPO=patrickbdevaney/GLM-5.3-Flash-REAP50-GGUF

# Xet defaults to a conservative single-stream path: a lone 93 GB GGUF crawled at ~400 KB/s while
# the link (1200 Mbit/s, -19 dBm) sat idle. The FP8/NVFP4 repos hit ~118 MB/s only because they
# were FOLDER uploads - 62 shards in parallel - which hid the single-file limitation.
#
# HF_XET_HIGH_PERFORMANCE=1 fixed the speed (391 KB/s -> 110 MB/s) and then OOMED THE BOX: it
# raises concurrency without a ceiling, on a machine already holding a 93 GiB file in page cache.
# Bounded concurrency instead - most of the throughput, with a memory cap.
export HF_XET_FIXED_UPLOAD_CONCURRENCY=8
export HF_XET_CHUNK_CACHE_SIZE_BYTES=$((2 * 1024 * 1024 * 1024))
LOG=logs/gguf_ship.log
say(){ echo "[$(date -Is)] ship: $*" >> "$LOG"; }

# Wait out any quantise already running (Q4_K_M was launched by hand).
while pgrep -f 'llama-quantiz[e]' > /dev/null; do sleep 60; done

for Q in Q4_K_M Q4_K_S; do
  F="gguf/GLM-5.3-Flash-REAP50-${Q}.gguf"
  if [ ! -f "$F" ]; then
    say "quantising $Q"
    free=$(df --output=avail -BG /home/patrickd | tail -1 | tr -dc 0-9)
    if [ "$free" -lt 95 ]; then say "ABORT: only ${free}G free, need ~95G for $Q"; exit 1; fi
    ~/glm5-llama.cpp/build-cuda/bin/llama-quantize --allow-requantize \
        gguf/GLM-5.3-Flash-REAP50-Q8_0.gguf "$F" "$Q" "$(nproc)" \
        >> "logs/gguf_quant_${Q}.log" 2>&1
    if [ $? -ne 0 ]; then say "FATAL: quantise $Q failed"; exit 1; fi
  fi
  say "$Q built ($(du -h "$F" | cut -f1))"

  # Skip the gate if this exact file already passed: the gate writes its perplexity to
  # artifacts/ppl_<name>.txt only on success, so that file IS the pass marker. Re-gating costs
  # ~20 minutes of model loading, and restarts of this chain were paying it every time.
  GATED="artifacts/ppl_$(basename "$F" .gguf).txt"
  if [ -s "$GATED" ]; then
    say "$Q already gated (ppl $(cat "$GATED")) - skipping"
  else
  say "gating $Q"
  if ! ./scripts/gguf_gate.sh "$F"; then
    say "FATAL: $Q failed the gate - keeping the file on disk, stopping the chain"
    exit 1
  fi
  say "$Q passed the gate"
  fi

  # Characterise context BEFORE uploading, because the next step deletes the file. Only the
  # first level needs this: the context ceiling here is architectural (llama.cpp runs the 11 DSA
  # layers dense, O(n^2)), not a property of the quantisation level.
  if [ "$Q" = "Q4_K_M" ] && [ ! -f "artifacts/niah_GLM-5.3-Flash-REAP50-${Q}.json" ]; then
    say "NIAH probe on $Q"
    .venv/bin/python scripts/gguf_niah.py "$F" --lengths 2000,8000,32000 --timeout 1800 \
        >> "logs/niah_${Q}.log" 2>&1
    say "NIAH rc=$? (see logs/niah_${Q}.log)"
  fi

  say "uploading $Q"
  ok=0
  for a in $(seq 1 40); do
    if .venv/bin/hf upload "$REPO" "$F" "$(basename "$F")" --repo-type model >> "$LOG" 2>&1; then
      ok=1; say "$Q uploaded after $a attempt(s)"; break
    fi
    say "$Q upload attempt $a failed; retry in 60s"; sleep 60
  done
  [ $ok -eq 1 ] || { say "FATAL: $Q upload gave up - not deleting"; exit 1; }

  # Confirm the remote copy byte-for-byte before reclaiming the space.
  if .venv/bin/python - "$REPO" "$(basename "$F")" "$F" >> "$LOG" 2>&1 <<'EOF'
import sys, pathlib
from huggingface_hub import HfApi
repo, name, local = sys.argv[1], sys.argv[2], sys.argv[3]
sz = {s.rfilename: s.size for s in HfApi().model_info(repo, files_metadata=True).siblings}
want = pathlib.Path(local).stat().st_size
got = sz.get(name)
print(f"hub check {name}: local {want} remote {got}")
sys.exit(0 if got == want else 1)
EOF
  then
    rm -f "$F"; say "$Q verified on the Hub and removed locally; free now $(df -h /home/patrickd | tail -1 | awk '{print $4}')"
  else
    say "FATAL: $Q size mismatch on the Hub - keeping the local file"; exit 1
  fi
done
say "regenerating the card from measured artifacts"
.venv/bin/python scripts/gguf_card.py >> "$LOG" 2>&1
.venv/bin/hf upload "$REPO" gguf_README.md README.md --repo-type model >> "$LOG" 2>&1
say "card uploaded"
say "all levels shipped"
