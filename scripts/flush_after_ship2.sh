#!/usr/bin/env bash
# Reclaim 325 GB once the GGUF wave is provably complete. Detached; costs no agent tokens.
#
# Two deletions with different proofs, because they have different recovery stories:
#
#   gguf/GLM-5.3-Flash-REAP50-Q8_0.gguf (164 GB) is NOT published anywhere. Its proof is that
#   everything derived from it is on the Hub, so it is spent. If it is ever wanted again it
#   re-converts from FP8-v2 in ~32 min, CPU-only.
#
#   output/glm-5.3-flash-reap50-fp8-pass2 (161 GB) IS the only local copy of the parent, and its
#   proof has to be that the Hub copy is byte-for-byte recoverable - per-file sha256, not a
#   filename match. flush_verify_fp8.py does that and this script deletes nothing if it fails.
set -uo pipefail
cd /home/patrickd/glm-5.3-reap
LOG=logs/flush.log
say() { echo "[$(date -Is)] flush: $*" >> "$LOG"; }

say "wave verified complete at 00:18; proceeding"
say "shipping wave is done"

# Gate 1: the Hub must actually hold everything this wave was supposed to produce.
EXPECTED="GLM-5.3-Flash-REAP50-Q4_K_M.gguf GLM-5.3-Flash-REAP50-Q4_K_S.gguf \
GLM-5.3-Flash-REAP50-IQ4_XS.gguf GLM-5.3-Flash-REAP50-Q3_K_M.gguf GLM-5.3-Flash-REAP50-IQ3_M.gguf \
mmproj-GLM-5.3-Flash-REAP50-F16.gguf imatrix-GLM-5.3-Flash-REAP50.dat \
glm5-next-llama.cpp.patch glm5-next-llama.cpp.bundle README.md"
if ! ./.venv/bin/python - "$EXPECTED" >> "$LOG" 2>&1 <<'PY'
import sys
from huggingface_hub import HfApi
have = set(HfApi().list_repo_files("patrickbdevaney/GLM-5.3-Flash-REAP50-GGUF"))
missing = [f for f in sys.argv[1].split() if f not in have]
print("missing from the Hub:", missing if missing else "none")
sys.exit(1 if missing else 0)
PY
then
  say "FATAL: the GGUF repo is missing expected files - deleting NOTHING"
  exit 1
fi
say "gate 1 passed: every expected file is on the Hub"

# Deletion 1: the spent quantisation vehicle.
if [ -f gguf/GLM-5.3-Flash-REAP50-Q8_0.gguf ]; then
  rm -f gguf/GLM-5.3-Flash-REAP50-Q8_0.gguf
  say "removed Q8_0 vehicle; free $(df -h / | awk 'NR==2{print $4}')"
fi

# Gate 2 + deletion 2: the parent, only after per-file sha256 proof.
say "verifying fp8-pass2 against the Hub (sha256 per file; this takes a few minutes)"
if ./.venv/bin/python scripts/flush_verify_fp8.py >> "$LOG" 2>&1; then
  rm -rf output/glm-5.3-flash-reap50-fp8-pass2
  say "removed fp8-pass2 (recoverable from patrickbdevaney/GLM-5.3-Flash-REAP50-FP8-v2)"
else
  say "fp8-pass2 did NOT verify - keeping it. Q8_0 was still removed."
fi

say "done. free $(df -h / | awk 'NR==2{print $4}')"
say "kept: nvfp4-pass2 (serving + drafter teacher), corpus, artifacts"
