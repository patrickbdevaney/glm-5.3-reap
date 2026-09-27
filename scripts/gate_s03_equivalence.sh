#!/usr/bin/env bash
# Does splitting the layer sweep across processes change the numbers?
#
# This is the question the block plan gate CANNOT answer. gate_s03_blocks.py proves every layer
# is scheduled exactly once; it cannot prove the accumulators that come out are the same. The
# risk being tested is silent: a block boundary that loses a layer's contribution, or replays
# one, produces a perfectly valid-looking saliency file with the wrong numbers in it -- and the
# mask is chosen from those numbers.
#
# Four arms, because two are not enough. A and B differ only in block size; C and D rerun them
# unchanged to establish how much two runs disagree for reasons that have nothing to do with
# blocking. See scripts/s03_eq_compare.py for why bit-exactness was the wrong bar and what
# replaced it.
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
# 6 layers, not 45: the unblocked reference arm must itself be runnable, and an unblocked
# 45-layer sweep is the 259 GiB case that cannot run on this box at all. 6 x 5.76 = 35 GiB.
COMMON="S03_MAX_LEN=512 S03_CALIB_TOKENS=8192 S03_CHUNK_TOKENS=4096 S03_MAX_LAYERS=6"

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
run_arm A 6 "A (one 6-layer block = unblocked reference)" || { echo "GATE FAIL: arm A rc!=0" | tee $T/VERDICT; tail -20 $T/A.log; exit 1; }
# B: the same sweep cut into blocks of 2, so every boundary is exercised
run_arm B 2 "B (three 2-layer blocks -- every boundary exercised)" || { echo "GATE FAIL: arm B rc!=0" | tee $T/VERDICT; tail -20 $T/B.log; exit 1; }

# Arms C and D rerun A and B unchanged: the NOISE FLOOR. The gate is meaningless without them.
# bf16 rounding moves borderline top-8 routing decisions, so no two runs agree bit-for-bit, and
# "the arms differ" says nothing until you know how much a rerun differs.
run_arm C 6 "C (unblocked RERUN -- noise floor)" || { echo "GATE FAIL: arm C rc!=0" | tee $T/VERDICT; exit 1; }
run_arm D 2 "D (blocked RERUN -- noise floor)"   || { echo "GATE FAIL: arm D rc!=0" | tee $T/VERDICT; exit 1; }

$PY scripts/s03_eq_compare.py
rc=$?
# Write the verdict to a file as well as exiting with it. An exit code alone has already been
# shown tonight to be an unreliable signal (a worker that never ran exited 0), so the verdict is
# recorded where it can be read back and cannot be confused with a wrapper's status.
echo "---"
if [ $rc = 0 ]; then echo "GATE PASS" | tee $T/VERDICT
else echo "GATE FAIL (rc=$rc)" | tee $T/VERDICT; fi
exit $rc
