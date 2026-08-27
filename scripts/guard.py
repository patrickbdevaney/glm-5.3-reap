#!/usr/bin/env python
"""Refuse to start heavy ad-hoc work while a heavy pipeline stage is running.

Written after doing exactly that: a 13.78 GiB validation was run alongside s03's 21.3 GiB
working set and the kernel OOM-killer took out the pipeline stage (dmesg confirms
'Killed process 223145 (python)'). Losing 12 minutes of a 5-hour pass to an avoidable OOM is
cheap to prevent.

Usage:  .venv/bin/python scripts/guard.py && <heavy command>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import db

HEAVY = {"s03_saliency", "s04b_surgery", "s07_quantize", "s05_heal"}

def _alive(pid):
    if not pid:
        return False
    try:
        with open(f"/proc/{pid}/stat") as fh:
            return fh.read().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return False


# A status of 'running' can be stale if the process was killed; check the pid too.
with db() as con:
    running = [n for n, pid in con.execute(
        "SELECT name, pid FROM stages WHERE status='running'")
        if n in HEAVY and _alive(pid)]

if running:
    print(f"REFUSING: heavy stage(s) running: {running}. "
          f"Concurrent GPU/RAM work risks OOM-killing them.", file=sys.stderr)
    sys.exit(1)
print("clear: no heavy stage running")
