# 70 — Healing: recovery training after the prune

## The enabling insight: healing is layer-local, so the teacher never needs to fit

The obvious objection to distillation-based healing is fatal-looking: the teacher is the
**unpruned 328 GB model** and Thor has **117 GiB**. Teacher and student cannot coexist. Not
even the teacher alone fits.

But REAP's objective is **layer-local reconstruction error**, and so is the repair. For
layer *L*, the target is:

```
minimise  ‖ pruned_L(x) − original_L(x) ‖   over calibration activations x
```

This needs **one layer of the teacher and one layer of the student at a time**. Both are
streamed. Peak residency is two copies of a single layer's experts — at 50% prune,
7.25 GB (teacher) + 3.62 GB (student) ≈ 11 GB in FP8 — plus the activation batch. `[EXT,
follows from the objective being layer-local]`

This is exactly the shape of published post-prune repair. **MoE-Pruner** (arXiv 2410.12013)
reports that "the gap between the pruned MoE model and the pretrained model can be largely
mitigated via expert-wise knowledge distillation with only **1000 training samples from C4,
completing in 1 hour**." `[EST]`

**Recommendation: make layer-local reconstruction distillation the primary healing stage.**
It is the cheapest, most targeted, and best-evidenced repair available, it directly optimises
the quantity REAP's saliency bounds, and it is the only healing stage that is unambiguously
feasible on this box. It also composes with the sequential-onload pass we are already
running, so it costs one extra streaming pass rather than a new pipeline.

## Stage 2 — LoRA/QLoRA recovery SFT

Base stays **frozen in FP8**; adapters are BF16. This is QLoRA in all but name, and it is why
the FP8-native decision compounds: we never materialise a 328 GB BF16 base to train against,
and the adapters are a few GB rather than a new full checkpoint.

### Target modules — the directive's widening is correct, and cheaper than expected

The directive requires the LoRA target set to include mHC and attention-adjacent projections
that read from expert outputs, not just expert/FFN weights. The reasoning (mHC was calibrated
against the *original* expert-output distribution; pruning shifts that distribution out from
under it) is sound — see [40-hybrid-fragility.md](40-hybrid-fragility.md) R2.

**mHC is 17,695,935 parameters across 135 tensors — 0.006% of the model.**

> **Do not LoRA mHC. Fully fine-tune it.** A rank-decomposed approximation of an 18M-parameter
> module is a strictly worse and more complicated answer than training the module. Full
> fine-tuning of mHC is ~35 MB of optimiser state at BF16 Adam and directly addresses the
> exact recalibration hypothesis. `[EXT]`

Module set, from the verified tensor inventory:

| Group | Tensors | Treatment |
|---|---|---|
| `hc_attn_{base,fn,scale}`, `hc_ffn_{base,fn,scale}`, `mapping_proj` | 135, 17.7 M | **full fine-tune** |
| `mlp.gate.*` routers (+`e_score_correction_bias`) | 86, 50.7 M | **full fine-tune** — routing must relearn a 144-expert simplex |
| MLA `o_proj`, KDA `o_proj` (read expert-influenced residual) | 45 | LoRA, higher rank |
| MLA `q_a/q_b/kv_a/kv_b`, KDA `q/k/v/g/f/b_proj` | — | LoRA, lower rank |
| Surviving expert `gate/up/down_proj` | 74k−ish | LoRA (or covered by stage 1 distillation) |
| Vision tower | 203 | **frozen** — untouched by pruning, nothing to repair |

**The routers deserve as much emphasis as mHC and the directive under-weights them.** After
pruning, each router still emits logits over a *renumbered, halved* expert set, and
`norm_topk_prob: true` + `routed_scaling_factor: 2.5` mean the surviving gates are
renormalised over a different support. The router is the component whose *semantics* changed
most, it is only 50.7M params, and it is free to train fully. `[EXT]`

**MoE-aware rank allocation** (DR-LoRA and successors; also arXiv 2604.26340 on module-wise
expert pruning for LoRA-MoE): `[OPEN]` whether per-expert rank allocation beats uniform rank
here. Not on the critical path — decide after seeing stage-1 per-layer reconstruction error,
which *is* a per-layer damage signal and is the natural thing to allocate rank against.

## Stage 3 — RLVR: an honest feasibility verdict

The directive asks to "establish a feasible on-Thor rollout budget." **The honest answer is
that there is not one at this model size, and I would rather say so than invent a number.**

Arithmetic, from the measured platform ([20-host-thor.md](20-host-thor.md)) and the
recomputed weight map ([60-quantization.md](60-quantization.md)):

- Bytes read per decoded token ≈ attention 6.2 GB (FP8) + 8 routed experts × 25.17 M ×
  0.5625 B × 43 layers ≈ 4.9 GB + shared 0.6 GB ≈ **11.7 GB/token**
- At 273 GB/s peak → **~23 tok/s theoretical ceiling**; prior measured work on this box lands
  at 30–50% of ceiling (DSpark-180B 7.9 tok/s, DFlash-122B 18.6 tok/s), so **~10 tok/s
  realistic** single-stream. `[EXT, calibrated against measured prior art]`
- GRPO needs group rollouts: 16 rollouts × ~1,000 tokens = ~16K tokens per prompt →
  **~27 min per prompt** single-stream. A modest 1,000-prompt run is **~460 days**.
- Batching amortises weight reads and is the only real lever; even an optimistic 15×
  aggregate speedup gives ~30 days, on a box that must also hold ~97 GiB of weights, LoRA
  optimiser state, and rollout KV — and RL is the least checkpoint-tolerant stage in the
  pipeline.

Published efficiency work (rollout pruning + difficulty scheduling for 60–70% compute
reduction; sequential adaptive rollout allocation, arXiv 2607.26253) is real but demonstrated
at **1.5B–3B on a single GPU** `[EST]` — two orders of magnitude below our target. It does
not close a 460-day gap.

> **Verdict: RLVR is out of reach for a 165B model on one Thor and should be dropped from
> this run, not scheduled optimistically and abandoned mid-way.** `[EXT]`
>
> The capability RLVR was meant to repair — long-horizon multi-step coherence — should
> instead be pursued through the **calibration and SFT mixture**, which is why the 22%
> agentic-trajectory share ([80-calibration.md](80-calibration.md)) matters more than it
> would if RLVR were on the table. Agentic trajectories in *calibration* also protect the
> experts those behaviours route through, which is the cheapest available proxy.

This is a change to directive §6.7 and needs operator acknowledgement. It does not affect the
primary deliverable.

## Recommended healing recipe

1. **Layer-local reconstruction distillation** — streamed, teacher never resident, one pass.
   Primary. Best evidence, lowest cost, directly targets REAP's own objective.
2. **Full fine-tune of routers + mHC** (68.4 M params total) — tiny, and targets the two
   components whose semantics actually changed.
3. **LoRA SFT** on attention-adjacent + surviving experts over the domain mixture.
4. ~~RLVR~~ — deferred with reasons above.

## RESOLVED: the first-moment healing gain over-corrects by 30.8% `[MEAS 2026-08-28 01:10]`

The `[OPEN]` question from the pass-2 research is now answered, and **the research was right**.

Measured on chunk 1 of the pass-2 sweep (2,345,112 cached router rows, 42 layers), by replaying
post-prune routing from the router-score cache rather than deriving the correction from pre-prune
means:

| quantity | value |
|---|---|
| shipped in pass 1 | **0.6964** |
| first-moment estimator, recomputed on pass-2 data | 0.7149 |
| **measured by router replay** | **0.9111** (per-layer 0.8460 – 0.9499) |
| **disagreement with shipped** | **30.8%** |
| gate mass, pre-prune / post-prune | **2.5000 / 2.5000** |
| real output inflation | **1.1015x** (first-moment implied 1.3925x) |

### The mechanism, confirmed exactly

`norm_topk_prob` renormalises the surviving top-8, so **the router hands back essentially all the
gate mass by itself** — measured identically at 2.5000 on both sides. A surviving expert's gate
*grows* after pruning because it divides by a smaller sum. The first-moment ratio compares
pre-prune conditional means and cannot see that, so it attributes the entire saliency gap to output
inflation and prescribes a 39% shrink where the true inflation is 10%.

### It is not an artefact of the estimator

The replay only resolves tokens with >=8 survivors inside the cached top-40, and that fraction
varies by layer (0.366 – 0.999), so the obvious worry is a biased subsample. Tested:

| subset | n | median measured gain |
|---|---|---|
| resolvable < 0.75 | 8 | 0.9177 |
| resolvable >= 0.95 | 12 | 0.9041 |
| all | 42 | 0.9111 |

Pearson r(resolvable_frac, gain) = **-0.257**. The spread between the well- and poorly-resolved
layers is ~1.5%, against a 30.8% discrepancy. The finding is not a sampling artefact.

The estimator also benefits from being a **ratio** of two same-form quantities: the
`||y||^2 ~ sum g^2||f||^2` orthogonality approximation appears in numerator and denominator alike,
so cross-term error largely cancels. The first-moment derivation has no such protection — it omits
renormalisation, which is a first-order effect.

### Consequence for the published artifact

`patrickbdevaney/GLM-5.3-Flash-REAP50-FP8` (and the NVFP4 derived from it) has **every retained
expert's output scaled by 0.6964 where 0.9111 was correct — a systematic under-scaling of
0.7643 on the entire MoE pathway in all 42 layers**, relative to attention, the shared experts and
the residual stream. It is a real defect, not a rounding issue.

It is also cheap to repair: the gain lives entirely in the F32 `weight_scale_inv` tensors, so
correcting it is a multiply by 1.3083, not a requantisation.

### Actions taken

1. `s05_heal` now **prefers the measured per-layer gains** from `artifacts/heal_refit.json` and
   falls back to the first-moment derivation only with a loud warning naming the ~30% bias.
2. Pass 2 will be healed with the measured gains.
3. Pass 1 stays as published for now, because it is the honest A/B baseline for P9.5 — what is on
   HF today. Re-healing it is a separate, cheap job and would make a clean three-way comparison
   (published / re-healed / pass 2) that isolates the value of this fix alone.

### The general lesson

This is the second time on this project that a **derivation** lost to a **measurement** in a way no
amount of care in the derivation would have caught. The first-moment correction was internally
consistent, dimensionally sound, defensible in review, and wrong by 30% — because it modelled a
router that renormalises as though it did not. The router cache cost ~0.5 GB per chunk and turned
an unanswerable question into a five-minute one.

### Convergence check: the measurement was already stable at one chunk `[MEAS 2026-08-28 05:05]`

The published FP8 was re-healed using the **chunk-1** measurement, before the sweep finished. That
is a decision acted on with 1/10 of the data, so it needs checking rather than assuming.

| | median measured gain |
|---|---|
| 1 chunk (0.5M tokens) — what shipped | 0.9111 |
| 3 chunks (1.5M tokens) | 0.9099 |
| per-layer drift | **median 0.10%, max 0.93%** |

The re-healed artifact is **0.14%** from the better estimate. No further correction is warranted,
and the decision to act early was sound.

Worth noting *why* it converges so fast, because it is not a general licence to trust one chunk:
the healing gain is a **ratio of population means over ~2.3M routed-token events per layer**, so
its standard error is tiny even at one chunk. The keep-set decision is a different animal — it
depends on the *ordering* of 288 experts, many separated by less than the noise floor, which is
exactly why P6's split-half gate exists and why the token budget matters there but not here.

Relevant to A7 in `CLOUD_COUNTERFACTUAL.md`: for this quantity, more calibration buys nothing.
