#!/usr/bin/env python
"""End-to-end orchestrator for the GLM-5.3-Flash REAP + heal pipeline.

Design constraints (operator, 2026-08-27):
  * must survive the SSH tunnel collapsing  -> runs under systemd --user, linger enabled
  * must run to the finished weights without hand-back -> autonomous retry + dependency graph
  * must preserve work at every step -> SQLite state + local artifacts + HF publishing

Resume semantics: every stage is idempotent and re-entrant. Re-running the pipeline skips
'done' stages, resumes 'running' ones (their own internal checkpointing takes over), and
retries 'failed' ones up to max_attempts.

A stage whose module does not exist yet is parked as 'awaiting_impl' and polled, rather than
failing the run. That lets the long early stages (source staging, corpus build) start now
while later stage code is still being written.
"""
from __future__ import annotations

import importlib
import os
import subprocess
import signal
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, db, log, now, kv_set, free_gib  # noqa: E402


@dataclass
class Stage:
    name: str
    module: str
    deps: list[str] = field(default_factory=list)
    soft_deps: list[str] = field(default_factory=list)   # wait for terminal, tolerate failure
    max_attempts: int = 3
    retry_backoff: int = 300          # seconds, doubled each attempt
    needs_gib: float = 0.0
    critical: bool = True            # False -> pipeline continues past a hard failure
    background: bool = False         # run in its own process so it cannot block the loop


STAGES: list[Stage] = [
    Stage("s00_smoke",    "stages.s00_smoke",     [],                          max_attempts=2, needs_gib=5),
    Stage("s02_corpus",   "stages.s02_corpus",    [],                          max_attempts=12,
          needs_gib=60, background=True),
    Stage("s01_source",   "stages.s01_source",    ["s00_smoke"],               max_attempts=5, needs_gib=20),
    Stage("s01b_load",    "stages.s01b_loadcheck",["s01_source"],              max_attempts=2, needs_gib=20),
    Stage("s03_saliency", "stages.s03_saliency",  ["s01b_load", "s02_corpus"], max_attempts=3, needs_gib=40),
    Stage("s04_sweep",    "stages.s04_sweep",     ["s03_saliency"],            max_attempts=3, needs_gib=10),
    # Healing improves the artifact but must never block it. critical=False + soft dep.
    Stage("s05_heal",     "stages.s05_heal",      ["s04_sweep"],               max_attempts=2,
          needs_gib=10, critical=False),
    Stage("s06_emit",     "stages.s06_emit",      ["s04_sweep"],               max_attempts=3,
          needs_gib=20, soft_deps=["s05_heal"]),
    Stage("s07_quantize", "stages.s07_quantize",  ["s06_emit"],                max_attempts=3, needs_gib=100),
    Stage("s08_document", "stages.s08_document",  ["s07_quantize"],            max_attempts=2, needs_gib=1),
]
BY_NAME = {s.name: s for s in STAGES}

TERMINAL_OK = {"done", "skipped"}
_stop = False


def _sig(signum, _frame):
    global _stop
    _stop = True
    log(f"received signal {signum}; will stop after the current stage", "pipeline", "WARN")


signal.signal(signal.SIGTERM, _sig)
signal.signal(signal.SIGINT, _sig)


def status(name: str) -> str:
    with db() as con:
        row = con.execute("SELECT status FROM stages WHERE name=?", (name,)).fetchone()
    return row[0] if row else "pending"


def set_status(name: str, st: str, **kw) -> None:
    with db() as con:
        con.execute(
            "INSERT INTO stages(name,status,attempts,started_at,finished_at,error,result) "
            "VALUES(?,?,0,?,?,?,?) ON CONFLICT(name) DO UPDATE SET status=excluded.status",
            (name, st, kw.get("started_at"), kw.get("finished_at"),
             kw.get("error"), kw.get("result")))
        for k in ("started_at", "finished_at", "error", "result"):
            if k in kw:
                con.execute(f"UPDATE stages SET {k}=? WHERE name=?", (kw[k], name))


def bump_attempts(name: str) -> int:
    with db() as con:
        con.execute("INSERT OR IGNORE INTO stages(name,status,attempts) VALUES(?,?,0)",
                    (name, "pending"))
        con.execute("UPDATE stages SET attempts=attempts+1 WHERE name=?", (name,))
        return con.execute("SELECT attempts FROM stages WHERE name=?", (name,)).fetchone()[0]


def attempts(name: str) -> int:
    with db() as con:
        row = con.execute("SELECT attempts FROM stages WHERE name=?", (name,)).fetchone()
    return row[0] if row else 0


def ready(s: Stage) -> bool:
    if not all(status(d) in TERMINAL_OK for d in s.deps):
        return False
    # soft deps only need to be *finished*, however they finished
    return all(status(d) in TERMINAL_OK | {"failed", "blocked"} for d in s.soft_deps)


def blocked_forever(s: Stage) -> bool:
    """A dep that has permanently failed blocks this stage permanently."""
    return any(status(d) == "failed" for d in s.deps)


def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def stage_pid(name: str) -> int | None:
    with db() as con:
        row = con.execute("SELECT pid FROM stages WHERE name=?", (name,)).fetchone()
    return row[0] if row and row[0] else None


def launch_background(s: Stage) -> bool:
    """Spawn the stage in its own process. Returns True if it is now running."""
    pid = stage_pid(s.name)
    if _pid_alive(pid):
        return True
    n = bump_attempts(s.name)
    lf = ROOT / "logs" / f"{s.name}.log"
    log(f"START background (attempt {n}/{s.max_attempts}) -> {lf.name}", s.name)
    with lf.open("ab") as fh:
        p = subprocess.Popen(
            [str(ROOT / ".venv" / "bin" / "python"), str(ROOT / "scripts" / "run_stage.py"),
             s.name, s.module],
            stdout=fh, stderr=subprocess.STDOUT, start_new_session=True,
            cwd=str(ROOT), env={**os.environ, "PYTHONUNBUFFERED": "1"})
    with db() as con:
        con.execute("UPDATE stages SET status='running', started_at=?, pid=? WHERE name=?",
                    (now(), p.pid, s.name))
    return True


def run_stage(s: Stage) -> bool:
    if s.needs_gib:
        have = free_gib()
        if have < s.needs_gib:
            log(f"insufficient disk for {s.name}: need {s.needs_gib:.0f} GiB, have {have:.0f}",
                s.name, "ERROR")
            set_status(s.name, "failed", error=f"disk: need {s.needs_gib} have {have:.0f}")
            return False

    n = bump_attempts(s.name)
    set_status(s.name, "running", started_at=now())
    log(f"START (attempt {n}/{s.max_attempts})", s.name)
    t0 = time.time()
    try:
        mod = importlib.import_module(s.module)
        importlib.reload(mod)
        result = mod.run() or {}
        dt = time.time() - t0
        set_status(s.name, "done", finished_at=now(), result=str(result)[:4000], error=None)
        log(f"DONE in {dt/60:.1f} min -> {str(result)[:300]}", s.name)
        return True
    except ModuleNotFoundError as e:
        if s.module.split(".")[-1] in str(e):
            set_status(s.name, "awaiting_impl", error=str(e))
            log(f"module not implemented yet; parking and will poll", s.name, "WARN")
            return False
        raise
    except Exception as e:
        tb = traceback.format_exc()
        (ROOT / "logs" / f"{s.name}.traceback.log").write_text(tb)
        log(f"FAILED attempt {n}: {type(e).__name__}: {e}", s.name, "ERROR")
        if n >= s.max_attempts:
            set_status(s.name, "failed", finished_at=now(), error=f"{type(e).__name__}: {e}")
            lvl = "ERROR" if s.critical else "WARN"
            log(f"exhausted {s.max_attempts} attempts; marking failed"
                + ("" if s.critical else " (non-critical; pipeline continues)"), s.name, lvl)
        else:
            set_status(s.name, "retry", error=f"{type(e).__name__}: {e}")
            back = s.retry_backoff * (2 ** (n - 1))
            log(f"backing off {back}s before retry", s.name, "WARN")
            for _ in range(back):
                if _stop:
                    break
                time.sleep(1)
        return False


def summary() -> str:
    with db() as con:
        rows = con.execute(
            "SELECT name,status,attempts FROM stages ORDER BY name").fetchall()
    return " | ".join(f"{n}:{st}" for n, st, _ in rows) or "(nothing yet)"


def main() -> int:
    log(f"pipeline start; free disk {free_gib():.0f} GiB", "pipeline")
    kv_set("pipeline_pid", os.getpid())
    idle_polls = 0
    while not _stop:
        progressed = False
        for s in STAGES:
            if _stop:
                break
            st = status(s.name)
            if st in TERMINAL_OK:
                continue
            if st == "failed" and attempts(s.name) >= s.max_attempts:
                continue
            if blocked_forever(s):
                if st != "blocked":
                    set_status(s.name, "blocked", error="upstream dependency failed")
                    log("blocked: an upstream dependency failed permanently", s.name, "ERROR")
                continue
            if not ready(s):
                continue
            if s.background:
                if st == "running" and _pid_alive(stage_pid(s.name)):
                    continue                      # still going; do not block on it
                if st == "running":               # process died without recording an outcome
                    set_status(s.name, "retry", error="background process vanished")
                    log("background process vanished; will relaunch", s.name, "WARN")
                if attempts(s.name) < s.max_attempts:
                    launch_background(s)
                    progressed = True
                continue
            if run_stage(s):
                progressed = True

        pend = [s for s in STAGES
                if status(s.name) not in TERMINAL_OK
                and not (status(s.name) == "failed" and attempts(s.name) >= s.max_attempts)
                and status(s.name) != "blocked"]
        if not pend:
            log(f"all stages terminal. {summary()}", "pipeline")
            return 0
        if not progressed:
            idle_polls += 1
            if idle_polls % 20 == 1:
                log(f"waiting on: {[s.name for s in pend]}  |  {summary()}", "pipeline")
            for _ in range(60):
                if _stop:
                    break
                time.sleep(1)
        else:
            idle_polls = 0
    log(f"pipeline stopping. {summary()}", "pipeline", "WARN")
    return 130


if __name__ == "__main__":
    sys.exit(main())
