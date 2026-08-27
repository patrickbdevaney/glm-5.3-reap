#!/usr/bin/env python
"""Run one stage in its own process and record the outcome durably.

Used for stages marked background=True so they do not block the orchestrator loop. The child
owns its own status transitions, so a killed orchestrator does not lose the result.
"""
import importlib
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, db, log, now  # noqa: E402


def set_status(name, st, **kw):
    with db() as con:
        con.execute("INSERT OR IGNORE INTO stages(name,status,attempts) VALUES(?,?,0)", (name, st))
        con.execute("UPDATE stages SET status=? WHERE name=?", (st, name))
        for k in ("finished_at", "error", "result"):
            if k in kw:
                con.execute(f"UPDATE stages SET {k}=? WHERE name=?", (kw[k], name))


if __name__ == "__main__":
    name, module = sys.argv[1], sys.argv[2]

    # Exclusive per-stage lock, held by the CHILD for its whole life.
    #
    # The orchestrator's "is it already running?" check reads a pid from SQLite, which loses a
    # race: two s01_source processes were once started together and spent minutes deleting and
    # re-fetching the same shards from under each other, driving the source tree backwards from
    # 34/62 to 29/62. A pid check cannot fix that; a lock the second process fails to take can.
    import fcntl
    lock_path = ROOT / "state" / f"{name}.lock"
    lock_fh = open(lock_path, "w")
    try:
        fcntl.flock(lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        log(f"another {name} already holds {lock_path.name}; exiting without running", name,
            "WARN")
        sys.exit(0)
    lock_fh.write(str(__import__("os").getpid()))
    lock_fh.flush()

    try:
        res = importlib.import_module(module).run() or {}
        set_status(name, "done", finished_at=now(), result=str(res)[:4000], error=None)
        log(f"DONE (background) -> {str(res)[:300]}", name)
        sys.exit(0)
    except Exception as e:
        (ROOT / "logs" / f"{name}.traceback.log").write_text(traceback.format_exc())
        set_status(name, "retry", error=f"{type(e).__name__}: {e}")
        log(f"FAILED (background): {type(e).__name__}: {e}", name, "ERROR")
        sys.exit(1)
