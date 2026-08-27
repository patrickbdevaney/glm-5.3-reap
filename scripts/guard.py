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

with db() as con:
    running = [n for n, in con.execute(
        "SELECT name FROM stages WHERE status='running'") if n in HEAVY]

if running:
    print(f"REFUSING: heavy stage(s) running: {running}. "
          f"Concurrent GPU/RAM work risks OOM-killing them.", file=sys.stderr)
    sys.exit(1)
print("clear: no heavy stage running")
