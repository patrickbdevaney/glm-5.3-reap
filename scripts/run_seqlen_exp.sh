#!/usr/bin/env bash
# Detached runner for the sequence-length saliency test (CLAUDE.md §1: never a child of a
# Claude Code Bash call). Holds the GPU lock so it cannot race the pass-3 chain.
#
# MemoryMax is not tuning -- it is the thing that keeps a bug in THIS script from taking the box
# down. On 2026-09-26 this experiment drove MemAvailable to 530 MB and wedged the machine with no
# oom-kill line, and memguard could not help: its last-resort kill tier fires below 250 MB, but
# the box is already unusable at ~1 GB. A cgroup ceiling makes the kernel kill this process
# instead of letting it starve everything else. 96G leaves ~21 GiB for the desktop and the
# driver on a 117 GiB box.
set -u
cd /home/patrickd/glm-5.3-reap
exec systemd-run --user --scope --quiet \
  -p MemoryMax=96G -p MemorySwapMax=0 \
  ./scripts/gpulock.sh seqlen-exp ./.venv/bin/python scripts/exp_seqlen_saliency.py 12
