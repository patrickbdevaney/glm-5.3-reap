"""Shared plumbing: durable state, structured logging, metrics, HF artifact publishing.

Everything important lands in SQLite (state/state.db), never only in a log file. The pipeline
is expected to be killed and resumed repeatedly across several days, so every helper here is
written to be safe to call twice.
"""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "state" / "state.db"
LOGS = ROOT / "logs"
ARTIFACTS = ROOT / "artifacts"
VENV_PY = ROOT / ".venv" / "bin" / "python"

MODEL_ID = "zai-org/GLM-5.3-Flash"
HF_USER = "patrickbdevaney"
HF_PREFIX = f"{HF_USER}/GLM-5.3-Flash-REAP"

for d in (DB.parent, LOGS, ARTIFACTS):
    d.mkdir(parents=True, exist_ok=True)

SCHEMA = """
CREATE TABLE IF NOT EXISTS stages (
    name TEXT PRIMARY KEY, status TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
    started_at TEXT, finished_at TEXT, error TEXT, result TEXT, pid INTEGER);
CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, stage TEXT NOT NULL,
    key TEXT NOT NULL, value REAL, tag TEXT);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, stage TEXT,
    level TEXT NOT NULL, msg TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS metrics_stage ON metrics(stage, key);
"""


@contextmanager
def db():
    con = sqlite3.connect(DB, timeout=60)
    con.execute("PRAGMA journal_mode=WAL")
    try:
        con.executescript(SCHEMA)
        yield con
        con.commit()
    finally:
        con.close()


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def log(msg: str, stage: str | None = None, level: str = "INFO") -> None:
    line = f"[{now()}] {level:5s} {stage or '-':22s} {msg}"
    print(line, flush=True)
    try:
        with db() as con:
            con.execute("INSERT INTO events(ts,stage,level,msg) VALUES(?,?,?,?)",
                        (now(), stage, level, msg))
    except Exception:
        pass  # never let telemetry take down the pipeline


def metric(stage: str, key: str, value: float, tag: str | None = None) -> None:
    with db() as con:
        con.execute("INSERT INTO metrics(ts,stage,key,value,tag) VALUES(?,?,?,?,?)",
                    (now(), stage, key, float(value), tag))


def kv_get(k: str, default=None):
    with db() as con:
        row = con.execute("SELECT v FROM kv WHERE k=?", (k,)).fetchone()
    return json.loads(row[0]) if row else default


def kv_set(k: str, v) -> None:
    with db() as con:
        con.execute("INSERT INTO kv(k,v) VALUES(?,?) ON CONFLICT(k) DO UPDATE SET v=excluded.v",
                    (k, json.dumps(v)))


def hf_token() -> str | None:
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if tok:
        return tok.strip()
    p = Path.home() / ".cache/huggingface/token"
    return p.read_text().strip() if p.exists() else None


MAX_AUTO_PUBLISH_GIB = float(os.environ.get("MAX_AUTO_PUBLISH_GIB", "8"))


def publish(local: Path, repo_suffix: str, path_in_repo: str = ".",
            private: bool = True, stage: str | None = None,
            allow_large: bool = False) -> str | None:
    """Push an intermediate artifact to HF. Never fatal: a failed upload must not lose work
    that is already safely on local disk.

    Large weight directories are SKIPPED by default. The link measures ~105 MB/s, so the
    157 GiB pruned checkpoint would take ~7 hours and the 91 GiB NVFP4 one ~4 - blocking the
    pipeline on bandwidth for longer than the compute it is publishing. Small artifacts
    (reports, configs, cards, saliency) publish automatically; weights are opt-in via
    allow_large, so uploading them is a decision rather than an accident.
    """
    tok = hf_token()
    if not tok:
        log("no HF token; skipping publish", stage, "WARN")
        return None
    repo = f"{HF_PREFIX}-{repo_suffix}" if repo_suffix else HF_PREFIX
    if local.is_dir() and not allow_large:
        gib = sum(p.stat().st_size for p in local.rglob("*") if p.is_file()) / 2**30
        if gib > MAX_AUTO_PUBLISH_GIB:
            log(f"NOT auto-publishing {local.name} ({gib:.0f} GiB > "
                f"{MAX_AUTO_PUBLISH_GIB:.0f} GiB): at ~105 MB/s that is ~{gib*1024/105/3600:.1f} h. "
                f"Push explicitly when wanted:  huggingface-cli upload {repo} {local}",
                stage, "WARN")
            return None
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=tok)
        api.create_repo(repo, private=private, exist_ok=True, repo_type="model")
        if local.is_dir():
            api.upload_folder(folder_path=str(local), repo_id=repo, path_in_repo=path_in_repo)
        else:
            api.upload_file(path_or_fileobj=str(local), repo_id=repo,
                            path_in_repo=path_in_repo if path_in_repo != "." else local.name)
        log(f"published {local.name} -> {repo}/{path_in_repo}", stage)
        return repo
    except Exception as e:
        log(f"publish failed ({type(e).__name__}: {e}); artifact remains local at {local}",
            stage, "WARN")
        return None


def free_gib(path: Path = ROOT) -> float:
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 2**30


def require_space(gib: float, stage: str) -> None:
    have = free_gib()
    if have < gib:
        raise RuntimeError(f"insufficient disk: need {gib:.0f} GiB, have {have:.0f} GiB")
    log(f"disk ok: {have:.0f} GiB free (need {gib:.0f})", stage)


def run(cmd: list[str], stage: str, log_name: str | None = None, env: dict | None = None,
        timeout: int | None = None) -> subprocess.CompletedProcess:
    e = {**os.environ, **(env or {})}
    lf = LOGS / (log_name or f"{stage}.log")
    log(f"$ {' '.join(cmd[:6])}{' ...' if len(cmd) > 6 else ''}  (-> {lf.name})", stage)
    with lf.open("ab") as fh:
        fh.write(f"\n=== {now()} {' '.join(cmd)}\n".encode())
        fh.flush()
        return subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=e,
                              timeout=timeout, check=False)
