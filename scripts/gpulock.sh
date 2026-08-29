#!/usr/bin/env bash
# Serialise GPU stages with a real mutex.
#
# The previous pattern - each chain polling `pgrep` for the absence of the others - is not a lock.
# MEASURED 2026-08-29 02:03: the ablation had finished capture and not yet started scoring when
# attnq_chain's 60-second poll fired, so both passed their checks and ran concurrently. Available
# memory fell to 7 GiB with memguard dropping caches every two minutes, which is how a stage gets
# killed 14 layers into a 90-minute pass.
#
# Usage:  ./scripts/gpulock.sh <name> <command...>
# Blocks until the lock is free, then runs the command holding it. flock releases the fd on exit,
# including on kill, so a dead holder never wedges the queue.
set -u
cd /home/patrickd/glm-5.3-reap
LOCK=/tmp/glm53-gpu.lock
NAME="$1"; shift
LOG=logs/pass2_finish.log
exec 9>"$LOCK"
echo "" >> $LOG
echo "===== [$(date -Is)] GPULOCK $NAME waiting =====" >> $LOG
flock 9
echo "" >> $LOG
echo "===== [$(date -Is)] GPULOCK $NAME acquired =====" >> $LOG
"$@" >> $LOG 2>&1
rc=$?
echo "" >> $LOG
echo "===== [$(date -Is)] GPULOCK $NAME released rc=$rc =====" >> $LOG
exit $rc
