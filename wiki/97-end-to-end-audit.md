# 97 — End-to-end audit of the pass-3 pipeline

*2026-09-27, after five defects in `s03_saliency` were found by gating rather than by running.
This audits every remaining stage for the same failure classes before pass 3 reaches them.*

The five defects shared a shape worth naming, because it is what the audit looks for:

> **Four of the five produced a plausible wrong answer rather than a crash.** An empty mask, a
> 10x per-layer data imbalance, a silently skipped worker. Only one announced itself. A stage
> that fails loudly is a stage that is working.

## Failure classes

| class | what it looks like |
|---|---|
| **A. Per-layer leak** | sweeps N layers in one process; `empty_cache()` returns 0.03 GiB against 13.54 consumed ([96](96-the-real-oom-cause.md)) |
| **B. Trusting an exit code** | a child that never ran exits 0; parent records success |
| **C. Resume double-count** | checkpoint granularity finer than resume granularity → replay into cumulative totals |
| **D. Conditional accumulate** | a load guarded by the wrong predicate → silently starts from zero and overwrites |

## Findings

| stage | A leak | B exit | C resume | D accumulate | verdict |
|---|---|---|---|---|---|
| `s03_saliency` | **was** | **was** | **was** | **was** | FIXED + gated |
| `s04_sweep` | no layer loop | no subprocess | no resume state | no accumulate | **clear** |
| `s04b_surgery` | no layer loop | no subprocess | per-shard, **idempotent** | none | **clear** |
| `s05_heal` | no layer loop | no subprocess | per-layer, independent | none | **clear** |
| `s06_emit` | no layer loop | no subprocess | none | none | **clear** |
| `s07_quantize` | no layer loop | no subprocess | 4 refs, per-shard | none | **clear** |
| `s09_eval` | **YES — 45 layers, one process** | no subprocess | none | none | **AT RISK** |

### s04b_surgery is safe for a specific reason worth recording

Its resume is per-shard, and its own comment states *"the output file existing IS the resume
record now that names are 1:1"*. Writing a shard twice produces the same shard, so replay is
**idempotent** — which is exactly what `s03`'s accumulators were not. Accumulation is the
dangerous operation; a pure function of its inputs is not. `[EST]`

This matters more than usual because `s04b_surgery` **deletes source shards as it writes
survivors**. It is the one irreversible stage, and it is the one whose resume cannot corrupt.

### s09_eval carries class A  `[EST]`

```python
for li in range(tcfg.num_hidden_layers):        # 45
    layer = _build_layer(tcfg, li, reader, torch.bfloat16)
    ...
    del layer
    reader.release(); gc.collect(); torch.cuda.empty_cache()
```

Structurally identical to `s03`'s sweep before the fix. The leak is dominated by **layer weights
that are never returned**, not by activations, so it is roughly constant per layer regardless of
batch size — a smaller eval batch does not avoid it.

`s09_eval` is what produces the numbers pass 3 is gated on (FP8 top-1 **0.84249**, dNLL 0.17563,
ballast top-1 0.58008). A stage that crashes partway through an eval and resumes is a stage whose
gate numbers are not trustworthy, which is the same reason the `s03` leak mattered.

**Not urgent**: `s09_eval` runs after surgery, heal, emit and quantize, so pass 3 does not reach
it for many hours. It needs the same block treatment before it does. `[OPEN]`

## Standing rule added

> Any stage that loops over `num_hidden_layers` in a single process is presumed to carry the
> leak until measured otherwise. `scripts/probe_layer_mem.py` measures it; `mem_lookahead.py`
> predicts it; neither is optional for a new sweep.
