#!/bin/bash
# Wait out the orphaned llama-quantize, verify it actually finished, then run ship2 fresh.
#
# Why this exists: editing a shell script while bash is executing it corrupts the run. bash keeps
# a file OFFSET (fd 255) and resumes reading there, so inserting lines shifts everything after the
# current position and the tail misparses. The for-loop had already been parsed, so the old
# in-memory copy would also have deleted each quant before the new vision step could see one.
cd /home/patrickd/glm-5.3-reap
LOG=logs/gguf_ship2.log
say(){ echo "[$(date -Is)] supervise: $*" >> "$LOG"; }

while pgrep -x llama-quantize > /dev/null; do sleep 60; done

F=gguf/GLM-5.3-Flash-REAP50-IQ4_XS.gguf
if grep -q 'quantize time' logs/gguf_quant_IQ4_XS.log 2>/dev/null && [ -s "$F" ]; then
  say "IQ4_XS quantise completed ($(du -h "$F" | cut -f1)); starting ship2"
else
  say "IQ4_XS quantise did NOT complete - removing the partial file so ship2 redoes it"
  rm -f "$F"
fi
exec ./scripts/gguf_ship2.sh
