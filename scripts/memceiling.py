"""How much memory can this process ACTUALLY use? One answer, cgroup-aware.

WHY THIS EXISTS (2026-09-28)
----------------------------
s03_saliency crash-looped three times, each attempt dying ~10 min into chunk 8 block 0 with no
error line -- a silent cgroup OOM kill. The log said `114.8 GiB available` every time.

Both numbers were true and the pair was useless:

    /proc/meminfo MemAvailable      114.8 GiB   <- what the guard read
    cgroup memory.max                72.0 GiB   <- what the kernel enforced
    block 0 actual need              71.5 GiB   <- states 17.5 + 9 layers x 5.76 + resident 2.2

The unit sets MemoryMax=72G with MemorySwapMax=0, so there is no swap to absorb the overshoot:
crossing 72 GiB is an instant SIGKILL. Every memory guard in this repo read /proc/meminfo, which
is host-wide and knows nothing about the cgroup that does the killing. The guards were measuring
a ceiling that does not bind.

The 72G cap is CORRECT and stays -- it was added after two box wedges, and a cgroup kill loses a
job while a wedge loses the box. What was wrong is that the work was sized against the wrong
number. So: ask the cgroup, and size the work to whatever it says.

RULE: no stage may compute a memory budget from /proc/meminfo alone. Use effective_gib().
"""
from __future__ import annotations

from pathlib import Path

GIB = 2 ** 30


def _read_int(p: Path) -> int | None:
    try:
        v = p.read_text().strip()
        return None if v == "max" else int(v)
    except Exception:
        return None


def cgroup_path() -> Path | None:
    """The v2 cgroup this process lives in, as a path under /sys/fs/cgroup."""
    try:
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            parts = line.split(":", 2)
            if len(parts) == 3 and parts[0] == "0":
                p = Path("/sys/fs/cgroup") / parts[2].lstrip("/")
                return p if p.exists() else None
    except Exception:
        pass
    return None


def cgroup_headroom_gib() -> float | None:
    """memory.max - memory.current, in GiB. None when unconfined.

    This is the number that kills. `memory.high` throttles; `memory.max` SIGKILLs.
    """
    cg = cgroup_path()
    if cg is None:
        return None
    lim = _read_int(cg / "memory.max")
    if lim is None:
        return None
    cur = _read_int(cg / "memory.current") or 0
    return (lim - cur) / GIB


def host_available_gib() -> float:
    """MemAvailable. Real, but an UPPER bound only -- it ignores any cgroup cap."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024 / GIB
    except Exception:
        pass
    return 0.0


def effective_gib() -> float:
    """What a new allocation can actually consume: the MINIMUM of both ceilings.

    Host memory can be free while the cgroup is full, and the cgroup can have room while the
    host is exhausted. Only the smaller of the two is safe to plan against.
    """
    host = host_available_gib()
    cg = cgroup_headroom_gib()
    return host if cg is None else min(host, cg)


def describe() -> str:
    cg = cgroup_headroom_gib()
    cgs = "unconfined" if cg is None else f"{cg:.1f} GiB cgroup headroom"
    return f"{effective_gib():.1f} GiB effective ({host_available_gib():.1f} GiB host, {cgs})"


def require(need_gib: float, label: str, margin_gib: float = 4.0) -> None:
    """Refuse to start work that cannot fit. Raises rather than letting the OOM killer decide.

    A refusal is recoverable and leaves a message; a SIGKILL leaves a crash loop and a silent
    log, which is exactly what cost three attempts and 40 minutes on 2026-09-28.
    """
    eff = effective_gib()
    if need_gib + margin_gib > eff:
        raise MemoryError(
            f"{label}: needs {need_gib:.1f} GiB + {margin_gib:.1f} GiB margin, but only "
            f"{describe()}. Refusing to start -- this would be OOM-killed, not slow.")


def fit_units(ceiling_gib: float, fixed_gib: float, per_unit_gib: float,
              margin_gib: float, max_units: int) -> int:
    """Largest unit count whose TOTAL (fixed + units x per_unit + margin) fits the ceiling.

    `fixed_gib` is the part people forget. For s03 it is the 17.5 GiB reloaded states plus the
    2.2 GiB resident embeddings -- ~20 GiB that the old sizing left out entirely, which is how a
    block was computed at 51.8 GiB when it actually needed 71.5.
    """
    budget = ceiling_gib - fixed_gib - margin_gib
    if budget <= 0:
        return 0
    return max(0, min(max_units, int(budget // per_unit_gib)))


if __name__ == "__main__":
    print(describe())
    print(f"  host MemAvailable : {host_available_gib():8.1f} GiB")
    cg = cgroup_headroom_gib()
    print(f"  cgroup headroom   : {'unconfined' if cg is None else f'{cg:8.1f} GiB'}")
    print(f"  EFFECTIVE         : {effective_gib():8.1f} GiB")


# ---------------------------------------------------------------------------
# Per-stage budgets and OOM attribution
# ---------------------------------------------------------------------------
#
# Peak host-RAM a stage reaches. MEASURED entries come from cgroup memory.peak; [EST] entries are
# projections that the first real run replaces -- an estimate here is a claim to be checked, not
# a fact. A stage whose peak exceeds the effective ceiling is refused BEFORE it starts.
STAGE_PEAK_GIB = {
    "s00_smoke":    2.0,    # MEASURED
    "s01_source":   4.0,    # MEASURED -- streams shards, never holds the model
    "s01b_load":    8.0,    # MEASURED
    "s02_corpus":   6.0,    # MEASURED
    "s03_saliency": 60.0,   # MEASURED 2026-09-28: fixed 19.7 + 7 layers x 5.76
    "s04_sweep":    24.0,   # [EST] reads saliency accumulators, not weights
    "s04b_surgery": 48.0,   # [EST] rewrites shards expert-by-expert; streaming, one shard resident
    "s05_heal":     64.0,   # [EST] LoRA heal; the largest unmeasured stage
    "s06_emit":     32.0,   # [EST] shard assembly
    "s07_quantize": 24.0,   # [EST] llama-quantize streams tensor by tensor
    "s08_document":  4.0,   # [EST]
    "s09_eval":     40.0,   # [EST] serves the model through llama.cpp
}


def oom_kills() -> int:
    """How many times THIS cgroup has OOM-killed something. 0 when unconfined.

    A SIGKILLed child leaves no traceback and no log line -- s03 crash-looped three times on
    2026-09-28 and every log simply stopped mid-block. This counter is the difference between
    "it died" and "the kernel killed it for memory", and it is the only evidence that survives.
    """
    cg = cgroup_path()
    if cg is None:
        return 0
    try:
        for line in (cg / "memory.events").read_text().splitlines():
            k, _, v = line.partition(" ")
            if k == "oom_kill":
                return int(v)
    except Exception:
        pass
    return 0


def peak_gib() -> float | None:
    cg = cgroup_path()
    if cg is None:
        return None
    v = _read_int(cg / "memory.peak")
    return None if v is None else v / GIB


def preflight(stage: str) -> None:
    """Refuse a stage that cannot fit. Called centrally for EVERY stage by run_stage.py."""
    need = STAGE_PEAK_GIB.get(stage)
    if need is None:
        return
    require(need, f"stage {stage}", margin_gib=6.0)
