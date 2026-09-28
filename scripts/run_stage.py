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

    # Central memory preflight -- EVERY stage, no exceptions.
    #
    # On 2026-09-28 s03_saliency was OOM-killed three times in a row, ~10 min into the same block,
    # with no traceback and no error line: a SIGKILL leaves nothing behind. Each attempt had
    # logged "114.8 GiB available" from /proc/meminfo while the cgroup enforced MemoryMax=72G.
    # A refusal here costs seconds and says why; a kill costs the run and says nothing.
    sys.path.insert(0, str(ROOT / "scripts"))
    from memceiling import describe, oom_kills, peak_gib, preflight
    oom_before = oom_kills()
    try:
        preflight(name)
        log(f"memory preflight OK: {describe()}", name)
    except MemoryError as e:
        set_status(name, "retry", error=f"MemoryError: {e}")
        log(f"REFUSED before start: {e}", name, "ERROR")
        sys.exit(1)

    try:
        res = importlib.import_module(module).run() or {}
        pk = peak_gib()
        if pk is not None:
            # Feeds the budget table with a MEASURED number so the next run stops guessing.
            log(f"peak cgroup memory {pk:.1f} GiB (budget said "
                f"{__import__('memceiling').STAGE_PEAK_GIB.get(name, float('nan')):.1f})", name)
        set_status(name, "done", finished_at=now(), result=str(res)[:4000], error=None)
        log(f"DONE (background) -> {str(res)[:300]}", name)
        sys.exit(0)
    except Exception as e:
        (ROOT / "logs" / f"{name}.traceback.log").write_text(traceback.format_exc())
        # Attribute the failure honestly: a child killed for memory is not a code bug, and
        # retrying it unchanged just reproduces the crash loop.
        if oom_kills() > oom_before:
            log(f"OOM-KILLED by the cgroup during {name} (peak {peak_gib() or 0:.1f} GiB, "
                f"{describe()}). This is a memory ceiling, not a code fault -- lower the stage "
                f"budget or raise the unit's MemoryMax.", name, "ERROR")
            set_status(name, "retry", error=f"cgroup OOM kill during {name}")
            sys.exit(1)
        set_status(name, "retry", error=f"{type(e).__name__}: {e}")
        log(f"FAILED (background): {type(e).__name__}: {e}", name, "ERROR")
        sys.exit(1)
