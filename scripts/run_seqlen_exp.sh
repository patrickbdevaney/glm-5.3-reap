#!/usr/bin/env bash
# Detached runner for the sequence-length saliency test (CLAUDE.md §1: never a child of a
# Claude Code Bash call).
#
# ADMISSION CONTROL. MEASURED 2026-09-26, twice: this experiment and the pipeline's s03_saliency
# both stream the same 306 GB of shards, and running them together drove MemAvailable to ~200 MB
# and wedged the box. memguard then killed s03 -- the healthy 47-minute-old job -- because the
# experiment was not on its licence. The gpulock does NOT cover this: glm53-reap.service runs
# pipeline.py, which never takes it. So the check is explicit here.
#
# Refusing is correct rather than queueing: the pipeline stage is ~12 h and this experiment is
# ~40 min, so a queue would silently park it for half a day and look like a hang.
set -u
cd /home/patrickd/glm-5.3-reap

if pgrep -f 'run_stage\.py' > /dev/null 2>&1; then
  echo "[$(date -Is)] REFUSING: a pipeline stage (run_stage.py) is running." >&2
  echo "  Two streaming passes over the same shards wedged this box twice on 2026-09-26." >&2
  echo "  Stop glm53-reap.service first, or wait for the stage to finish." >&2
  exit 3
fi

# MANDATORY LOOKAHEAD. Every wedge on 2026-09-26 was a ceiling discovered by hitting it, and
# every recovery mechanism failed in turn: tier 1 cannot reclaim a live allocation, tier 2's
# SIGKILL does not land on a process blocked in the GPU driver (it killed the right PID twice
# while MemAvailable kept falling), and MemoryMax does not bind Tegra unified allocations. When
# detection, killing and cgroup limits all fail, not starting is the only control left.
for S in 2048 8192 16384; do
  if ! ./.venv/bin/python scripts/mem_lookahead.py --seq "$S" --attn sparse --quiet; then
    echo "[$(date -Is)] REFUSING: arm S=$S is predicted not to fit. Run" >&2
    echo "  ./.venv/bin/python scripts/mem_lookahead.py --seq $S --attn sparse" >&2
    echo "  for the binding term." >&2
    exit 4
  fi
done

# MemoryMax is not tuning -- it keeps a bug in THIS script from taking the box down. memguard's
# floor is deliberately 250 MB (a higher floor killed healthy runs twice, see memguard.sh), so
# the cgroup ceiling is the thing that bounds a runaway, not the guard.
exec systemd-run --user --scope --quiet \
  -p MemoryMax=72G -p MemorySwapMax=0 \
  ./scripts/gpulock.sh seqlen-exp ./.venv/bin/python scripts/exp_seqlen_saliency.py 12
