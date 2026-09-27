#!/usr/bin/env bash
# Detached runner for the sequence-length saliency test (CLAUDE.md §1: never a child of a
# Claude Code Bash call). Holds the GPU lock so it cannot race the pass-3 chain.
set -u
cd /home/patrickd/glm-5.3-reap
exec ./scripts/gpulock.sh seqlen-exp ./.venv/bin/python scripts/exp_seqlen_saliency.py 12
