# 95 — Wall-clock optimisation

Baseline in the first `PLAN.md` was **8–14 days**. This page records every lever found, what
it is worth, and what it costs. Revised critical path: **4–7 days**.

The dominant uncertainty throughout is that **no forward-pass throughput has been measured for
`glm5_next` on this box.** Stage 0 produces that number; every estimate here should be revised
from it. Ranges are honest, not padded.

---

## L1 — MAXN + `jetson_clocks` `[EST — MEASURED, APPLIED 2026-08-27]`

The box was in **120 W mode (nvpmodel 1)** with `schedutil` governors and no clock lock.
NVIDIA's own Thor benchmarks are run at **MAXN + `jetson_clocks`**.

Applied `sudo nvpmodel -m 0 && sudo jetson_clocks`. Measured effect on the clock ceilings:

| Clock | Before (120 W, DVFS) | After (MAXN, locked) | Δ |
|---|---:|---:|---:|
| `gpu-gpc-0` max | 1386 MHz | **1575 MHz** | **+13.6%** |
| `gpu-gpc-0` current (idle) | 315 MHz | **1575 MHz** (pinned) | — |
| **EMC (memory controller)** | **2750 MHz** | **4266 MHz** | **+55%** |
| CPU (all 14) | schedutil, 972–2601 | **2601 MHz pinned**, idle states off | — |

**EMC is the important one.** MoE forward passes are memory-bandwidth-bound (expert weights are
streamed, not reused), and the 273 GB/s figure in the spec corresponds to max EMC. The box was
running its memory controller at **64% of peak**. Thermals are 39 °C idle with a 130 W envelope,
so there is ample headroom to hold this.

**Cost:** more power and heat, and DVFS ramp is disabled. Fully reversible:
`sudo nvpmodel -m 1 && sudo jetson_clocks --restore`.

> **Caveat:** these are *clock ceilings*, not measured throughput. Under sustained load DVFS
> would have ramped EMC up anyway; the guaranteed win is removing ramp latency and any
> thermal/power-cap downclocking, not a flat 1.55×. **Do not bank a 1.55× — measure it at
> stage 0.** `[EXT]`

---

## L2 — Fold the sensitivity probe into the saliency pass `[EST]` · **saves 8–24 h**

This was a **defect in the first plan**, not an optimisation. Stage 3 (sensitivity probe,
8–24 h) and stage 5 (sweep, 4–12 h) were scheduled as separate forward-pass workloads from
stage 4 (saliency). But **the probe and the sweep both consume saliency scores** — they do not
need their own saliency computation.

REAP saliency is a **running conditional mean**, so partial scores after the first chunk are
*valid*, merely less converged. Therefore:

- Run saliency chunk 1 (~5% of corpus) → **preliminary scores** → run the §6.3 go/no-go probe
  at 10–15% → **gate**.
- If the gate passes, keep accumulating. **The early chunks are not discarded**; they are the
  first terms of the same running mean.

The gate still happens early and cheaply, and the work counts toward the final scores.

---

## L3 — Evaluate the sweep by masking, not by writing checkpoints `[EXT]` · **saves 4–10 h**

Building a pruned model at ratio *r* from cached scores does **not** require materialising a
checkpoint. Mask the bottom-*r* experts at runtime and measure KL / reconstruction on the
held-out set. Each ratio then costs one held-out forward pass (~500 samples × 4K ≈ 2M tokens),
not a 160 GiB write plus reload.

Also: **do not measure 30%.** It is *provably* over the memory envelope (123.3 GiB vs a
117 GiB budget) — measuring its quality answers a question we cannot act on. Sweep **40 / 50 /
55** only, and 55 only as research data given the hard stop at 50.

---

## L4 — Fuse surgery and distillation into one streaming pass `[EXT]` · **saves ~1 day**

Original plan: stage 6 surgery (2–4 h) then stage 7 layer-local distillation (1–2 days), each
its own pass over the weights.

Under **teacher forcing**, layer-local distillation makes layers *independent*: layer *L*'s
student is trained against teacher inputs, not against the student's own layer *L−1* output. So
a single pass suffices:

```
for L in 0..45:
    stream teacher layer L  (HTTP range request, ~7.3 GB)
    prune layer L using cached saliency        → student layer L
    train student layer L to match teacher layer L on the current activation batch
    propagate TEACHER outputs forward as layer L+1's input
    write student layer L, evict teacher layer L
```

One pass, one teacher layer resident (~11 GB peak), no activation store beyond the current
batch, and surgery + repair happen together.

**A rejected idea, recorded so it is not re-litigated:** caching teacher activations during the
saliency pass to avoid a teacher forward later. It does not pay — 1,000 samples × 4K tokens ×
4096 dim × 2 B × 46 layers × 2 (in+out) ≈ **3.1 TB**, and even 200 samples is ~617 GB.
Streaming the teacher (~52 min of download, amortised across the whole pass) is far cheaper
than storing its activations. `[EST — arithmetic]`

---

## L5 — Converge-and-stop on saliency instead of fixing the token budget `[EXT]` · **saves 1–2 days**

The 200M-token corpus ([80](80-calibration.md)) is a *ceiling* derived from the REAP paper's
≥110B setting, not a measured requirement. Published pruning-calibration convergence studies
find perplexity stabilising between **32 and 128 samples**, with clear diminishing returns
beyond `[EST]` — though those are *dense* per-weight criteria (Wanda/SparseGPT-shaped), not a
per-expert mean over 288 experts, so they do **not** transfer directly.

What does transfer is the *method*: **measure convergence rather than guess it.** Saliency is a
running mean, so instrument it and stop when it stops moving:

- **Rank stability:** Kendall-τ between the bottom-50% expert set at chunk *k* and *k−1*.
  Stop when τ > 0.99 for two consecutive chunks. The bottom set is what actually gets pruned —
  it is the only ranking that matters.
- **Per-expert sample floor:** every expert must have ≥ N tokens in its active set before any
  stop is permitted. This is the guard against stopping while **rare specialists** — risk R1 —
  are still badly estimated. Routing imbalance means the rarest experts see far fewer than the
  uniform 2.78% share, so **this floor, not the rank criterion, will usually be the binding
  constraint.**

Both criteria are free (they read state we already accumulate) and are logged per chunk.

**Expected landing: 30–80M tokens rather than 200M**, i.e. ~2.5× off the saliency pass. But it
is a *measured* stop with an explicit rare-expert guard, not a budget cut — and if the floor
says 200M, we spend 200M.

---

## L6 — Use vLLM for the calibration forward passes `[EXT]` · **potentially 3–10×, unproven**

The largest *potential* lever and the least certain.

HF `transformers` + llm-compressor's calibration path runs an unoptimised eager forward,
dequantising FP8 weights to BF16 per layer. vLLM on this box already has: FP8-native MoE
kernels, the cutlass backend that is the only one that survives ≥256 experts
([60](60-quantization.md)), CUDA graphs, and continuous batching.

Relevant precedent from NVIDIA's forums: a report of Thor MoE throughput far below reference
(Qwen3-30B-A3B at 34 tok/s vs ~61 expected) turned out to be **container selection** — the
generic NGC `nvcr.io/nvidia/vllm:26.02-py3` image instead of
**`ghcr.io/nvidia-ai-iot/vllm:latest-jetson-thor`**. After switching: **81.4 tok/s, +21.5% over
reference.** `[EST]` **That Jetson-Thor image is already on this box (50.2 GB).**

**What blocks this:** REAP saliency needs per-token **router gate values** and **per-expert
output norms**. vLLM's fused MoE kernels compute the expert outputs *inside* a fused kernel and
do not expose per-expert activations. Extracting them means either a hook in the fused MoE
layer or a custom kernel variant — real work, and exactly the kind of thing that turns a
2-day saving into a 3-day detour.

**Recommendation:** do **not** put this on the critical path. Run stage 0 with the transformers
path, get a real throughput number, and only pursue the vLLM path if that number makes the
saliency pass unacceptably long. Decide with data. `[EXT]`

Related, cheap, and worth doing regardless: **generate a tuned fused-MoE config** for our shape
(`E=288,N=2048,device_name=NVIDIA_Thor.json`, and `E=144` post-prune) via
`benchmark_moe.py --tune`. vLLM logs *"Using default MoE config. Performance might be
sub-optimal!"* when the file is missing, and no config for a 288-expert shape can possibly
exist upstream. This matters for the downstream serving work even if calibration never uses vLLM.

---

## L7 — NVFP4 quantisation needs almost no calibration `[EST]` · **saves 5–20 h**

Original estimate for stage 10 was 8–24 h, implicitly assuming a GPTQ-shaped calibrated pass.
NVFP4 **weight** quantisation is a deterministic per-block transform — max-abs per 16-element
block, FP8 scale, F32 global — with **no calibration data required at all**. Only the
**activation** scales for W4A4 need calibration, and those converge on a few hundred samples.

Revised: **3–6 h**, dominated by disk I/O over the 160 GiB pruned checkpoint (487 MB/s write is
the binding constraint, not compute).

Block-scale *search* (Four Over Six / SOAR / RaZeR) would add hours and remains a
post-baseline refinement, off the critical path.

---

## L8 — Overlap everything that is not on the critical path `[EST]` · **saves ~1 day of serial time**

- **Corpus build (1–2 days) is the long pole and is pure CPU/network.** It overlaps stage 0
  (smoke test), stage 1 (staging the 328 GB source), and the early saliency chunks — provided
  the buckets are built in the order they are first consumed.
- **Staging the source (~55 min) overlaps the smoke test.**
- **Prefetch layer *L+1* while computing layer *L*** in every streaming pass. NVMe read is
  3.4 GB/s and the network is 105 MB/s; neither should ever block compute.
- **Sequence packing** in the calibration loader — variable-length financial and agentic
  samples otherwise waste 20–40% of every padded batch.

---

## Revised critical path

| # | Stage | Was | Now | Lever |
|---|---|---:|---:|---|
| 0 | `glm5_next` smoke test + throughput measurement | 2–6 h | 2–6 h | — |
| 1 | Stage FP8 source | 55 min | *overlapped* | L8 |
| 2 | Build calibration corpus | 1–2 d | *overlapped* | L8 |
| 3 | Saliency pass, gate at chunk 1, converge-and-stop | 32–72 h | **8–24 h** | L2, L5, L1 |
| 4 | Sweep 40/50/55 by masking | 4–12 h | **2–3 h** | L3 |
| 5 | Surgery + distillation, fused streaming pass | 26–52 h | **12–24 h** | L4, L1 |
| 6 | mHC + router full FT, LoRA SFT | 2–4 d | **1–2 d** | L1, damaged-layer targeting |
| 7 | Emit healed FP8 + adapters | 2–4 h | 2–4 h | — |
| 8 | NVFP4 quantise | 8–24 h | **3–6 h** | L7 |
| 9 | Document output format | 4 h | 4 h | — |
| | **TOTAL** | **8–14 days** | **4–7 days** | |

**Where the confidence sits.** L2, L3, L4, L7 and L8 are structural — they remove work that was
scheduled twice or never needed, and they hold regardless of throughput. L1 is applied and
measured at the clock level but unquantified at the workload level. **L5 is the one that could
disappear**: if the per-expert sample floor binds late, the saliency pass stays near its
original length and the total lands at the top of the 4–7 day range.

L6 is deliberately excluded from the table. It is the largest potential win and the largest
potential detour; it gets decided at stage 0 with a measured number, not now.
