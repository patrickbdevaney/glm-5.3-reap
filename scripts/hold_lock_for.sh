#!/usr/bin/env bash
# Hold the GPU mutex on behalf of a process that was started before the mutex existed.
set -u
exec 9>/tmp/glm53-gpu.lock
flock 9
while pgrep -f "$1" > /dev/null; do sleep 15; done
