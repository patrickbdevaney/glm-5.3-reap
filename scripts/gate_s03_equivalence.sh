#!/usr/bin/env bash
# Does splitting the layer sweep across processes change the numbers?
#
# This is the question the block plan gate CANNOT answer. gate_s03_blocks.py proves every layer
# is scheduled exactly once; it cannot prove the accumulators that come out are the same. The
# risk being tested is silent: a block boundary that loses a layer's contribution, or replays
# one, produces a perfectly valid-looking saliency file with the wrong numbers in it -- and the
# mask is chosen from those numbers.
#
# Both arms run through the SAME worker path and differ ONLY in block size, so nothing but the
# boundary is varied. Accumulators are float32 sums over the same batches in the same order, and
# states round-trip through torch.save in bf16 without loss, so the result should be BIT-EXACT.
# Anything less than exact equality is a real difference and the gate says so.
set -u
cd /home/patrickd/glm-5.3-reap
PY=./.venv/bin/python

if pgrep -f 'exp_seqlen_saliency\.py' > /dev/null 2>&1; then
  echo "REFUSING: the seqlen ladder is running; two streaming passes wedged this box three times" >&2
  exit 3
fi

T=artifacts/_s03_eqtest
rm -rf $T; mkdir -p $T/A $T/B $T/state
# Small but REAL: 4 layers over the true checkpoint and the true corpus. A toy model would test
# the plumbing and not the thing that matters.
COMMON="S03_MAX_LEN=512 S03_CALIB_TOKENS=8192 S03_CHUNK_TOKENS=4096"

run_arm(){ # dir, layers_per_block, label
  echo "--- arm $3: layers_per_block=$2 ---"
  env $COMMON \
      S03_SALIENCY_DIR="$T/$1" \
      S03_STATES_DIR="$T/$1/states" \
      S03_BLOCK_LEDGER="$T/state/$1.json" \
      S03_ROUTER_CACHE_DIR="$T/$1/router" \
      S03_SNAPSHOT_DIR="$T/$1/snap" \
      S03_LAYERS_PER_BLOCK="$2" \
      PYTORCH_CUDA_ALLOC_CONF= \
      $PY scripts/run_stage.py s03_saliency stages.s03_saliency > "$T/$1.log" 2>&1
  rc=$?
  echo "arm $3 rc=$rc ($(ls $T/$1/*.pt 2>/dev/null | wc -l) layer dumps)"
  return $rc
}

# A: one block covering all layers -- the unblocked reference
run_arm A 64 "A (single block = unblocked reference)" || { echo "FAIL: arm A rc!=0"; tail -20 $T/A.log; exit 1; }
# B: the same sweep cut into blocks of 2, so every boundary is exercised
run_arm B 2  "B (blocks of 2 -- boundaries exercised)" || { echo "FAIL: arm B rc!=0"; tail -20 $T/B.log; exit 1; }

$PY - <<'PYEOF'
import sys, torch
from pathlib import Path
A = Path("artifacts/_s03_eqtest/A"); B = Path("artifacts/_s03_eqtest/B")
fa = sorted(p.name for p in A.glob("*.pt")); fb = sorted(p.name for p in B.glob("*.pt"))
fail = 0
if not fa:
    print("FAIL: arm A produced no layer dumps"); sys.exit(1)
if fa != fb:
    print(f"FAIL: different layers dumped\n  A={fa}\n  B={fb}"); sys.exit(1)
print(f"PASS: both arms dumped the same {len(fa)} layers")
worst = 0.0
for n in fa:
    da = torch.load(A/n, weights_only=False); db = torch.load(B/n, weights_only=False)
    for k, va in da.items():
        if not torch.is_tensor(va):
            continue
        vb = db.get(k)
        if vb is None:
            print(f"FAIL: {n}: key {k} missing from B"); fail = 1; continue
        if va.shape != vb.shape:
            print(f"FAIL: {n}:{k} shape {tuple(va.shape)} vs {tuple(vb.shape)}"); fail = 1; continue
        if torch.equal(va, vb):
            continue
        d = (va.float() - vb.float()).abs().max().item()
        worst = max(worst, d)
        print(f"FAIL: {n}:{k} differs, max|delta|={d:.6g}")
        fail = 1
if not fail:
    print("PASS: every accumulator tensor is BIT-EXACT between blocked and unblocked")
    print("      -> block boundaries neither lose nor replay a layer's contribution")
else:
    print(f"worst max|delta| = {worst:.6g}")
sys.exit(fail)
PYEOF
rc=$?
echo "---"; [ $rc = 0 ] && echo "GATE PASS" || echo "GATE FAIL"
exit $rc
