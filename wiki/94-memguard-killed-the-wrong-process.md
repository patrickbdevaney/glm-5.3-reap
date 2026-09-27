# 94 — Two box wedges, and a guard that killed the healthy job

*2026-09-26. Both incidents MEASURED from `logs/memguard.log` and the systemd journal.*

## What happened

Two hard wedges in ~30 minutes, each ending in a manual reset. **Neither produced an
`Out of memory: Killed` line** — the journal is continuous across both windows and contains none.
MemAvailable simply sat at 200 MB–1 GB for minutes with memguard logging
`nothing reclaimable ... held by a live allocation, not cache`.

## Root cause: a service I never checked was already running the same work

`glm53-reap.service` is **enabled** and auto-starts `pipeline.py` on every boot, which runs
`s03_saliency` — a 12-hour streaming pass over all 306 GB of shards. I had disabled
`reap-pass3.service` and assumed that was the only thing that could launch saliency. It was not.
The enabled set also includes `glm53-traces.service`, `reap-watchdog.service` and
`reap-publish.service`.

So when the sequence-length ladder was launched, **two independent streaming passes over the same
306 GB were running at once.** The pipeline stage had been healthy for 47 minutes at ~30 GiB with
54 GiB available; the second reader is what exhausted the box.

`gpulock.sh` did not prevent this: `pipeline.py` never takes the lock. The mutex protects the
stages that call it and is invisible to the one service that matters most.

## The worse finding: memguard fired, and killed the wrong process

```
21:20:51  !!! MemAvailable 488MB falling 6298MB/s — killing stage s03_saliency pid 2197
```

The guard worked exactly as written. Its victim licence is a **whitelist of known stage
runners** — `run_stage.py`, the MiMo stages, the llama workers, the Hub uploader. The ladder
experiment was not on it. So the guard:

1. correctly detected a runaway,
2. searched its licence for a victim,
3. found the *healthy 47-minute-old pipeline stage*, which was the only licensed process,
4. killed it, and
5. left the actual runaway alive — after which the box wedged anyway.

> A job outside the licence is not merely unkillable. **Its presence makes the guard kill
> something else instead.** The narrow licence, written specifically to avoid killing the wrong
> thing, is what caused the wrong thing to be killed. `[EST]`

The pipeline then retried, which is why `s03_saliency` reached `attempt 5/6`.

## Fixes

**1. Victim selection rewritten** (`scripts/memguard.sh`, gated 5/5 by
`scripts/gate_memguard_victim.sh` against real processes):

- The licence now matches on script **basename**, never on `" scripts/<name>"`. The
  path-anchored form missed an absolute-path launch of the very script that caused the incident —
  the same class of miss that left the runaway unlicensed to begin with.
- Among licensed candidates it now prefers the one that **started most recently**. The newcomer is
  what turned a working situation into a failing one; preferring `run_stage.py` was exactly
  backwards, since it is usually the oldest and most expensive to lose.
- Ranking deliberately does **not** use RSS. RSS is the number proven not to track Tegra unified
  allocations, which is why the kernel OOM killer picks wrong on this box.

**2. Admission control** (`scripts/run_seqlen_exp.sh`): refuses to start while any `run_stage.py`
is running, with exit 3 and the reason. Verified against the live stage. It *refuses* rather than
queues — the stage is ~12 h and the experiment ~40 min, so a queue would park it for half a day
and look like a hang.

**3. Cgroup ceilings**, because the guard is not what bounds a runaway:

| unit | MemoryMax |
|---|---|
| `glm53-reap.service` (drop-in, persistent) | 72G, swap 0 |
| ladder experiment scope | 72G, swap 0 |

**4. Page-cache reclaim** added to the experiment per layer and per arm — it had reproduced s03's
sweep with **neither** of s03's safeguards (`_reclaim_page_cache`, `CHUNK_TOKENS`).

## What was deliberately NOT changed

**memguard's 250 MB floor stays.** It looks far too low, and it is twice-documented as correct:
a 4000 MB floor killed healthy runs, and a 900 MB floor killed a healthy `s07` at 866 MB, because
this workload's normal plateau is 2–3 GiB available. A guard that must not false-positive cannot
have a floor above the plateau — which is precisely why the **cgroup ceiling**, not the floor, is
the thing that has to bound a runaway. Raising the floor would trade two wedges for a steady
stream of killed healthy stages. `[EST]`

## Open

`pipeline.py` still does not take `gpulock`. Admission control currently lives in the *callers*,
which means every future heavy script must remember to check. That is the same shape of fragility
as the whitelist. The durable fix is for `pipeline.py` to take the lock. `[OPEN]`
