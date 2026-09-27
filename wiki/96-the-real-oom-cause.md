# 96 — The real OOM cause: per-layer memory is never returned in-process

*2026-09-27, after four box resets in one evening. Everything here is MEASURED by
`scripts/probe_layer_mem.py`, not modelled. Three earlier explanations of these wedges were
wrong and are superseded by this page.*

## The measurement that settled it

One row, S=2048 — the configuration that had run to completion repeatedly, so **not** a
sequence-length problem:

```
start                            117.52 GiB
L0 linear_attn built+to(DEV)     116.10 GiB   delta  -1.30
L0 FORWARD done                  102.47 GiB   delta -13.54
L0 freed + empty_cache           102.50 GiB   delta  +0.03   <-- nothing returned
L3 deepseek_sparse FORWARD        98.16 GiB   delta  -3.88
L3 freed + empty_cache            98.17 GiB   delta  +0.01
L4 FORWARD done                   88.71 GiB   delta  -9.09
L4 freed + empty_cache            88.78 GiB   delta  +0.07
```

**5.76 GiB per layer, consumed by the forward, never returned while the process lives.**
`torch.cuda.empty_cache()` returned 0.03 GiB against 13.54 GiB consumed.

| workload | layers | memory needed | box has |
|---|---:|---:|---:|
| one ladder arm | 12 | 69 GiB | ~97 GiB usable |
| ladder, 4 arms in one process | 48 | **277 GiB** | ~97 GiB |
| `s03_saliency` | 45 | **259 GiB** | ~97 GiB |

> `s03_saliency` has been surviving this by **crashing and resuming**, not by being correct. It
> checkpoints per layer and `Restart=on-failure` brings it back, so a structural leak looked like
> flaky hardware for weeks. `attempt 5/6` was the symptom. `[EST]`

## What actually recovers the memory

Both steps are required; neither alone is sufficient:

| action | MemAvailable |
|---|---|
| in-process `empty_cache()` | +0.03 GiB |
| in-process `drop_caches` | no-op against a live allocation |
| **process exit** (CUDA context teardown) | 88.71 → 93.66 GiB |
| **then `echo 3 > drop_caches`** | 93.66 → **121.23 GiB** — full baseline |

`echo 1` is not enough; only the full shrinker pass releases the nvmap pool.

## Why every guard we built failed

Corroborated by others hitting the same hardware behaviour:

- **NvMap cannot reclaim page-cached pages the way MemAvailable implies.** The GPU allocates from
  the same physical memory, so an allocation can fail with GiB "available"
  ([vllm#35920](https://github.com/vllm-project/vllm/issues/35920),
  [goose-in-a-pond#370](https://github.com/jarida-io/goose-in-a-pond/issues/370)).
- **Memory stays "used" after workloads end.** ~50 GiB stayed used on an AGX Thor with no
  process, page-cache or slab accounting for it; the resource manager holds freed pages in its
  own pool after a CUDA context exits
  ([ollama#12283](https://github.com/ollama/ollama/issues/12283),
  [ollama#12528](https://github.com/ollama/ollama/issues/12528)).
- **SIGKILL does not land** on a process blocked in the GPU driver. memguard killed the correct
  PID twice, 39 s apart, while MemAvailable fell 125 → 97 → 49 MB.
- **cgroup `MemoryMax` does not bind** Tegra unified allocations — they are not charged to the
  requesting cgroup. The 72 GiB ceiling was inert.

Related and worth knowing even though it was not our bug: loading safetensors from a
`MAP_PRIVATE` mmap straight to an integrated GPU triggers copy-on-write into **unreclaimable
anonymous** memory, needing ~2× model size; the fix is to `clone()` off the mmap before the H2D
copy ([transformers#47257](https://github.com/huggingface/transformers/pull/47257)). Our
`_build_layer` already breaks the alias with `t.to(dtype, copy=True)`.

## A second real defect, found while researching

`glm53-reap.service` sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Upstream
explicitly disables this on Tegra boards
([unsloth-zoo#1235](https://github.com/unslothai/unsloth-zoo/pull/1235)); `is_integrated` is 1 on
this device. It did not cause the ladder failures (the experiment never set it) but it is wrong
for the pipeline. `[OPEN]`

## The architecture that follows

Layered, each layer validated rather than assumed:

1. **`mem_lookahead.py`** — static, pre-launch, refuses a configuration that cannot fit. Gated
   5/5 against outcomes already observed ([95](95-memory-lookahead.md)).
2. **One process per arm** — 12 layers × 5.76 = 69 GiB, inside the ~97 GiB usable budget. This is
   the fix; four arms in one process is 277 GiB.
3. **`drop_caches 3` between arms, then verify** ≥ 89 GiB before starting the next. If memory did
   not return, **abort rather than continue** — a run that starts below baseline is the run that
   wedges the box.
4. **`memfence.require()` before every layer allocation.** This is the only control that can
   actually stop a runaway, because nothing outside the process can: an exception unwinds Python
   and frees tensors; a wedged box does not.
5. **memguard** as last resort, now with corrected victim selection
   ([94](94-memguard-killed-the-wrong-process.md)).

## Standing rule

> On this box, **memory is not returned until a process exits**. Any sweep over more than ~16
> layers must be split across processes, with `drop_caches 3` and a verified return to baseline
> between them. Static prediction is necessary and not sufficient — the per-step fence is what
> makes it safe, because the leak depends on driver-pool state no static model can see. `[EST]`

## s03_saliency, fixed 2026-09-27

The stage is now an **orchestrator** that spawns one worker process per `(chunk, layer block)`.
The parent imports torch but never touches a device, so it never accumulates the leak; each
worker exits and its memory returns. Between workers the orchestrator runs `drop_caches 3` and
**refuses to start the next block if memory did not come back**, rather than continuing into a
wedge. Default block is 9 layers (52 GiB + 18 reserve against a 121 GiB baseline).

Hidden states are the only thing that must survive a block boundary, so they go to disk and are
**overwritten, never deleted**.

### The more important half: a silent double-count

Blocking a sweep risks something worse than a crash. The pre-existing resume path had exactly
that bug:

> Accumulators were dumped per **layer** (`SS.dump(SALIENCY)` after each one, "a kill costs at
> most one layer"), but resume skipped per **chunk**. So a crash mid-chunk reloaded cumulative
> totals that already contained that chunk's finished layers, then replayed them.

The effect is not a crash, it is a **bias**: the interrupted chunk gets double weight for the
layers it had completed and single weight for the rest. Because
[85-corpus-sources.md](85-corpus-sources.md) establishes that chunks arrive **grouped by
domain**, an interrupted chunk is domain-skewed — so the early layers of a resumed run carry a
distorted domain mixture. `[EST]`

**Pass 3 was not affected**: it never completed a chunk, so `done` was always empty, accumulators
reset to zero and the dumps were overwritten each attempt. The bug needed at least one completed
chunk plus a crash in a later one.

The ledger is now keyed on `(chunk, block)`, and `gate_s03_blocks.py` (16/16) asserts that every
layer runs exactly once at every block size and that a completed block is never rescheduled.

### Completion moved to the orchestrator

Workers return early, so the audit and `kv_set("saliency_ready", True)` had to move to the
parent — otherwise the pipeline would wait forever on a stage that had actually finished.
`_audit()` only `torch.load`s the dumps on CPU, so the parent still never initialises CUDA.

### Validated at production scale, 2026-09-27

The first real block boundary, on the full 45-layer / 10-chunk pass-3 sweep:

```
02:31:14  chunk 0 block 0 (layers 0-8) starting, 113.8 GiB available
02:53:56  chunk 0 block 0 done                         (9 layers, 21.4 min)
02:54:17  chunk 0 block 1 (layers 9-17) starting, 113.4 GiB available
```

**113.8 GiB before nine layers, 113.4 GiB after.** Unblocked, those nine layers cost ~52 GiB
that never came back. Process exit plus `drop_caches 3` returned all of it, and the orchestrator's
pre-block check confirmed the return before committing to the next worker. The mechanism holds on
the real model at real scale, not only on the 6-layer gate.

Cost: ~22 min per 9-layer block, 50 blocks (10 chunks x 5) -> **~18 h** for the full pass.

### Still open

Block-level equivalence at production scale was not separately measured — the ladder holds the GPU.
The block plan, ledger and budget arithmetic are gated; the equivalence of blocked vs unblocked
accumulators on real weights is not. `[OPEN]`

The old single-process chunk loop is left in place, marked unreachable, rather than deleted.
`[OPEN]`
