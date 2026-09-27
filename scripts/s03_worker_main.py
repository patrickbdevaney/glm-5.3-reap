"""Entrypoint for one s03_saliency (chunk, block) worker.

Deliberately NOT run_stage.py. That wrapper takes an exclusive per-stage flock, and the
orchestrator is itself `run_stage.py s03_saliency` holding that very lock -- so every worker
spawned through it failed to acquire the lock and **exited 0 without running**. The orchestrator
saw rc=0, recorded the block as done, and produced no saliency at all. Caught by
gate_s03_equivalence.sh, which is the only reason it was not discovered as an empty mask.

A zero exit code from a process that never started work is indistinguishable from success, so
the orchestrator also verifies that dumps actually advanced. This file exists so the worker
never contends for the stage lock in the first place.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "stages"))

if __name__ == "__main__":
    if os.environ.get("S03_ROLE") != "worker":
        print("refusing: S03_ROLE must be 'worker'", file=sys.stderr)
        sys.exit(2)
    import s03_saliency
    res = s03_saliency.run() or {}
    print(f"[s03 worker] {res}", flush=True)
    sys.exit(0)
