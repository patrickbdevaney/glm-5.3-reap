# 90 — Data handling: what is scratch, what is precious, what may be deleted

*2026-09-28, after a per-chunk scratch filename filled a 936 GB disk to 100% and killed pass 3
with `OSError: [Errno 28] No space left on device`.*

## The failure

`artifacts/s03_states` reached **132 GB** — `chunk_000.pt` … `chunk_008.pt`, ~17 GB each. The
states path was `chunk_{CI:03d}.pt`, so every chunk wrote a NEW file instead of reusing one. The
comment beside it asserted the file was *"OVERWRITTEN rather than deleted"*; the per-chunk
filename made that false. The guarantee and its violation were on adjacent lines.

It took sqlite down with it (`database or disk is full`), which is worse than stopping: a
half-written 17 GB tensor and a half-written ledger are both garbage, and the ledger is what
makes the run resumable.

> Memory has had a pre-flight since the first wedge (`mem_lookahead.py`, `memfence.py`). Disk had
> none, which is how 132 GB grew unnoticed next to a 308 GB source tree. `[EST]`

## The three classes

Every path in this repo is exactly one of these, and the class determines whether it may be
deleted.

### 1. Scratch — reused, bounded, deletable at any time

| path | bound | lifetime |
|---|---|---|
| `artifacts/s03_states/states.pt` | ~17 GiB | between the blocks of ONE chunk |
| `artifacts/_s03_eqtest/` | ~1 GiB | one gate run |

Scratch must have a **fixed name**. A path with a counter in it is not scratch, it is an
accumulator wearing scratch's clothes — that is exactly the bug above.

### 2. Derived — reproducible, expensive, delete only with a reason

| path | cost to regenerate |
|---|---|
| `artifacts/router_cache` | one saliency pass (~20 h) |
| `artifacts/exp_seqlen` | one ladder run (~1.5 h) |
| `artifacts/saliency_pass1`, `_pass2` | historical, cannot be regenerated at all — the models they came from are gone |

### 3. Precious — cannot be regenerated cheaply or at all

| path | why |
|---|---|
| `source/GLM-5.3-Flash` (308 GB) | 328 GB re-download |
| `artifacts/saliency` (0.25 GB) | **the accumulators the mask is built from.** 20 h of GPU in 251 MiB. The single highest value-per-byte object in the repo. |
| `output/`, `gguf/` | shipped artifacts |

## What was deleted on 2026-09-28, and how it was verified first

125 GB, with approval, after confirming each file was dead rather than merely large:

```
ledger: blocks per chunk {0:5, 1:5, 2:5, 3:5, 4:5, 5:5, 6:5, 7:5}  -- all 5/5
artifacts/saliency: 42 layer dumps, 251 MiB, count sum 15,633,048 on a spot check
```

- `chunk_000.pt` … `chunk_007.pt` (119 GB) — those chunks were **fully swept**; their entire
  contribution is in the accumulators. Nothing would ever read them again.
- `chunk_008.pt` (5.3 GB) — a partial write from the ENOSPC. Corrupt.
- `artifacts/_s03_eqtest` (0.5 GB) — gate scratch, removed unilaterally to regain enough free
  space to run a shell command at all. Flagged rather than quietly done.

**The verification is the point.** "Fully swept" came from the ledger, and "the work survives"
came from opening an accumulator and checking it held 15.6M counted routed slots. Deleting on the
strength of a filename would have been the same class of error as the bug that caused it.

## Rules now enforced in code

1. **`scripts/gate_disk_budget.py`** — every scratch path with the size it is *allowed* to reach.
   A path over budget is a leak, not a big file, and the gate says so.
2. **`_require_disk()` in `s03_saliency`** — refuses to start a chunk's prepare without 25 GiB
   free. Stopping is a recoverable failure; a truncated tensor file and a truncated ledger are
   not.
3. **Scratch paths take fixed names.** If a path needs a counter, it is not scratch and belongs in
   class 2 or 3 with a stated retention.
