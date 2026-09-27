#!/usr/bin/env bash
# Sequence-length ladder, run as one process PER ARM with a verified memory recovery between.
#
# WHY THIS SHAPE. MEASURED 2026-09-26/27 by scripts/probe_layer_mem.py: each layer's forward
# consumes ~5.76 GiB of host memory that is NEVER returned while the process lives --
# empty_cache() gave back 0.03 GiB against 13.54 GiB consumed. Four arms x 12 layers in one
# process needs 276 GiB on a 122 GiB box, which is what wedged this machine three times. One arm
# is 12 x 5.76 = 69 GiB, inside the ~97 GiB usable budget.
#
# Only two things recover that memory, and BOTH are required:
#     process exit (tears down the CUDA context) ..... 88.71 -> 93.66 GiB
#     then `echo 3 > drop_caches` .................... 93.66 -> 121.23 GiB  (full baseline)
# `echo 1` is not enough and in-process drop_caches is a documented no-op against a live
# allocation. On Tegra the GPU allocates through NvMap from the same physical memory and cannot
# reclaim page-cached pages the way MemAvailable implies, so a small allocation can fail with
# GiB "available".
set -u
cd /home/patrickd/glm-5.3-reap
PY=./.venv/bin/python
LOG=logs/pass2_finish.log
say(){ echo "[$(date -Is)] seqlen-orch: $*" | tee -a "$LOG"; }

avail_mb(){ awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo; }

recover(){   # process exit alone is not enough; this is the measured second half
  sync
  sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null
  sleep 3
}

# ---- gate 0: nothing else heavy may be running -------------------------------------------
if pgrep -f 'run_stage\.py' > /dev/null 2>&1; then
  say "REFUSING: a pipeline stage (run_stage.py) is running; two streaming passes wedged this box"
  exit 3
fi

# ---- gate 1: static lookahead for every arm, before anything allocates --------------------
for S in 2048 8192 16384; do
  if ! $PY scripts/mem_lookahead.py --seq "$S" --attn sparse --quiet; then
    say "REFUSING: arm S=$S predicted not to fit; see mem_lookahead.py --seq $S --attn sparse"
    exit 4
  fi
done
say "lookahead: all arms predicted to fit"

# ---- one process per arm, with a verified recovery between --------------------------------
BASE=$(avail_mb)
say "baseline MemAvailable ${BASE}MB"
rc=0
for arm in S2048 S8192 S16384 control_S2048_disjoint; do
  if [ -f "artifacts/exp_seqlen/$arm/_done" ]; then say "$arm already done, skipping"; continue; fi

  recover
  A=$(avail_mb)
  # gate 2: refuse to start an arm that cannot fit. 12 layers x 5.76 GiB = 69 GiB, +20 reserve.
  if [ "$A" -lt 89000 ]; then
    say "ABORT before $arm: only ${A}MB available, an arm needs ~69 GiB + 20 GiB reserve."
    say "  Memory did not return after the previous arm. Investigate before rerunning."
    rc=5; break
  fi
  say "starting $arm with ${A}MB available"

  ./scripts/gpulock.sh seqlen-exp $PY scripts/exp_seqlen_saliency.py 12 "$arm"
  arc=$?
  say "$arm exited rc=$arc"
  if [ "$arc" != 0 ]; then rc=$arc; break; fi
done

recover
say "final MemAvailable $(avail_mb)MB (baseline was ${BASE}MB)"

# ---- comparison, only when every arm exists ------------------------------------------------
if [ "$rc" = 0 ]; then
  n=$(ls -d artifacts/exp_seqlen/*/ 2>/dev/null | grep -cE '/(S2048|S8192|S16384|control_S2048_disjoint)/$')
  if [ "$n" = 4 ]; then
    say "all 4 arms present; running comparison + publishing"
    $PY scripts/exp_seqlen_saliency.py 12 && $PY scripts/publish_exp_results.py
    rc=$?
  else
    say "only $n/4 arms present; not comparing"
  fi
fi
say "done rc=$rc"
exit $rc
