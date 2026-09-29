#!/bin/bash
# Per-3s MemAvailable sampler. Output lives in logs/ because two earlier samplers were lost:
# one died as a child of a Claude Code Bash call, one had its scratchpad wiped between sessions.
out="${1:-logs/memcurve.csv}"
while true; do
  printf '%s,%s\n' "$(date +%s)" "$(awk '/MemAvailable/{printf "%.2f", $2/1048576}' /proc/meminfo)" >> "$out"
  sleep 3
done
