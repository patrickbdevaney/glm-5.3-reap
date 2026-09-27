# 93 — Two results from the ladder run, before any saliency number came out of it

*2026-09-26. Both MEASURED on the aborted first attempt.*

## 1. Sequence length is nearly free at a fixed token budget  `[EST]`

The arms run the **same 131,072 tokens**, only differently shaped:

| arm | shape | 12 layers, wall-time |
|---|---|---|
| S=2,048 | 64 rows × 2,048 | **20.9 min** |
| S=8,192 | 16 rows × 8,192 | **21.4 min** |

**A 4× longer sequence costs +2.4 % wall-time.** Not 4×, not 16×.

The reason is architectural, and it is the same bounded-state property that
[92-long-context-program.md](92-long-context-program.md) §2 uses to argue for saturation:

- KDA is **linear** in context — its recurrent state is fixed-size, so cost per token is flat.
- DSA is capped at `index_topk = 2048` — it attends to at most 2,048 selected keys however long
  the context is, so it never reaches the quadratic regime.

Neither mechanism grows quadratically, so calibration cost is governed by **token count**, not by
sequence length.

### What this settles

The question "do we need longer runs for a better REAP?" is answered **no** for the
length dimension. `s03_saliency` currently truncates at `MAX_LEN=2048` and thereby discards
**73.5 % of the tokens already collected** ([91-capability-coverage.md](91-capability-coverage.md)
§2). Raising `MAX_LEN` uses tokens we already have, at essentially unchanged compute.

That is a rare shape of lever: more of the corpus, longer coherent context, ~2 % more wall-time,
**no new data collection**. It does not depend on the ladder's saliency result — the ladder tells
us whether it *matters*, not whether it is affordable. `[EST]`

---

## 2. The run wedged the box, and memguard was not the missing piece  `[EST]`

The first attempt died at the S=8192 → S=16384 arm boundary. Timeline from `logs/memguard.log`:

```
21:40:20  S8192 finishes layer 12/12
21:40:21  S16384 arm starts
21:40:24  memguard tier1 drop_caches at 7882MB -> 78058MB   (reclaimed ~70 GB)
21:40:33  available 530MB
21:40:43+ "nothing reclaimable ... memory is held by a live allocation, not cache"
```

Available memory sat near 1 GB for two minutes and the machine had to be reset.

**No oom-kill line was produced** — the journal is continuous across the window and contains no
`Out of memory: Killed`. That is not an anomaly; it is the signature `s03_saliency`'s own
docstring already records:

> Tegra under-reports it in MemAvailable and will not reclaim it fast enough to satisfy a driver
> allocation, **which is how this stage died six times with no oom-kill line in dmesg**.

### Cause

`exp_seqlen_saliency.py` was written against s03's layer sweep but reproduced **neither** of its
memory safeguards. Grep count of `_reclaim_page_cache` and `CHUNK_TOKENS`: s03 has both, the
experiment had **zero** of either. Each layer faults in ~7 GB of mmap'd shard; 12 layers × 4 arms
is ~336 GB of clean file-backed pages that are never re-read.

> This is a documented failure mode in this repository, with the fix sitting in the file the
> script was modelled on, and it was omitted. Copying a loop is not copying the reasons the loop
> is shaped the way it is. `[EST]`

### Two fixes

1. `_reclaim_page_cache()` now runs per layer **and at every arm boundary** — the collapse began
   three seconds into a new arm.
2. `run_seqlen_exp.sh` runs under `systemd-run --scope -p MemoryMax=96G -p MemorySwapMax=0`.

The second matters more than the first. **memguard was running the whole time** — it is an
enabled user service, independent of the chain, and its log is the evidence above. It could not
help because the memory was a live allocation, not cache. And its last-resort kill tier fires
below **250 MB**, while the box is already unusable at ~1 GB, so the kill never armed. A cgroup
ceiling makes the kernel kill the offending process instead of letting it starve the machine.

**Open item:** memguard's kill threshold is below the level at which this box becomes
unresponsive. Either raise it, or accept that every heavy stage must carry its own `MemoryMax`.
The latter is more honest — a global threshold cannot know which process is the one to kill.
`[OPEN]`
