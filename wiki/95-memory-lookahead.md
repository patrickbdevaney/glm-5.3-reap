# 95 — Predicting peak memory instead of discovering it

*2026-09-26. Third box wedge of the evening. This page is why there will not be a fourth.*

## Every recovery mechanism failed, in order

| mechanism | what it did |
|---|---|
| memguard tier 1 (drop_caches) | `nothing reclaimable ... held by a live allocation` — correct and useless |
| memguard tier 2 (kill) | killed the **right** PID twice, 39 s apart, and MemAvailable kept **falling**: 125MB → 97MB → 49MB |
| `MemoryMax=72G` on the cgroup | did not bind at all |

Two hard lessons, both MEASURED:

1. **SIGKILL is not a recovery mechanism for GPU work on this box.** A process blocked in the
   NVIDIA driver does not die until the driver returns. memguard picked the correct victim — the
   fix from [94](94-memguard-killed-the-wrong-process.md) worked — and it changed nothing.
2. **Cgroup limits do not bound Tegra unified allocations.** They are not charged to the process
   cgroup that requested them, which is the same property that makes RSS useless for ranking
   victims and makes `/proc/meminfo` attribute the nvmap pool to nobody.

When detection, killing and cgroup ceilings all fail, the only remaining control is **not
starting a run that cannot fit**.

## Root cause of this wedge

The ladder died entering `layer 4/12`, which is index 3 — the first
`deepseek_sparse_attention` layer. Measured allocation for the DSA indexer's score matrix,
`[B, heads, L, L]` in bf16 at 64 heads:

| L | score matrix |
|---|---|
| 2,048 | 0.50 GiB |
| 8,192 | 8.00 GiB |
| **16,384** | **32.00 GiB** |

Hidden states held were **identical across all three arms** (4.00 GiB, same 131,072 tokens), so
they were never the variable. The attention term was.

Two compounding mistakes, both mine:

- **`tcfg._attn_implementation = "eager"`** forced the naive path that materialises the full
  matrix. `s03_saliency` does not set this at all and gets the config default. Removed — matching
  s03 is both correct and the validated path.
- **The claim in [92](92-long-context-program.md) §2 that DSA "never reaches the quadratic
  regime" was wrong.** `index_topk` caps what is ATTENDED, never what is SCORED. The indexer must
  score all L keys to select the top 2,048.

## The lookahead

`scripts/mem_lookahead.py` computes peak host memory for a configuration *before* allocating
anything, reports the binding term, and exits 4 if it will not fit. It is wired as a **mandatory
preflight** in `run_seqlen_exp.sh`, which now refuses to start any arm predicted not to fit.

It is a model, so it is **validated against outcomes already observed** rather than trusted.
`scripts/gate_mem_lookahead.sh`, 5/5:

```
PASS: S=2048  eager predicted to FIT   -- it ran to completion
PASS: S=8192  eager predicted to FIT   -- it ran to completion
PASS: S=16384 eager predicted to FAIL  -- it wedged the box entering the DSA layer
PASS: S=16384 sparse predicted to FIT  -- the fix, not the naive path
PASS: peak is monotone non-decreasing in sequence length
```

A predictor that cannot reproduce the three runs we actually watched is not a predictor.

### The budget is 55 GiB, not 122

MemTotal is 122 GiB and almost none of it is usable as transient headroom:

- streaming the shards holds tens of GiB of page cache (MEASURED: one `drop_caches` released
  **88 GiB** mid-run, and it refilled within seconds). Reclaimable in principle, and *not fast
  enough to satisfy a driver allocation* — the documented Tegra failure.
- the nvmap pool is attributed to nobody and only a full shrinker pass releases it.

So the budget is set against observed outcomes, not against MemTotal.

### What it predicts for the fix

At S=16,384 on the sparse path: **31.18 GiB total**, binding term `moe_layer_weights` at
13.50 GiB — the experts, not the attention. That is the shape a healthy configuration should
have on this model.

## Standing rule

> No heavy run starts without a lookahead that covers **every** stage of its end-to-end
> execution — every arm, every layer type, at its worst sequence length. The cost of being wrong
> is a box reset, and the guard cannot save us: it has now been proven unable to reclaim, unable
> to kill, and unbounded by cgroups. `[EST]`
