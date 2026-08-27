# 40 — The central risk: pruning a hybrid linear-attention + mHC model

The directive names this "the single largest technical risk in this project." After the
Phase 0 pass, **the risk is materially smaller than the directive assumed, and for a
specific, checkable reason.** But the reasoning that led there was sound, and one real
concern survives — a different one.

## The directive's hypothesis

> Nemotron3 (Transformer attention + Mamba SSM + MoE) underperformed baseline by ≥6.3%
> under REAP, hypothesised to be disruption of balance within the hybrid architecture.
> GLM-5.3-Flash is a hybrid — therefore expect fragility.

## What the literature actually says

### 1. There is a direct architectural precedent, and it is good

**`cerebras/Kimi-Linear-REAP-35B-A3B-Instruct`** — Cerebras's own REAP applied to
`Kimi-Linear-48B-A3B-Instruct` at **30% expert pruning**. `[EST]`

Kimi Linear (arXiv **2510.26692**) is **KDA — Kimi Delta Attention — interleaved with full
attention at a uniform 3:1 ratio**. That is *exactly* GLM-5.3-Flash's attention stack: 34 KDA
layers to 11 full-attention layers, full attention every 4th layer. GLM-5.3-Flash's config
does not merely resemble Kimi Linear; it names the mechanism, `linear_attn_config.kda_layers`.

Published results, 30% prune, 48B → 35B:

| Benchmark | base | REAP-30 | Δ |
|---|---:|---:|---:|
| HumanEval | 86.6 | 87.2 | **+0.6** |
| HumanEval+ | 82.3 | 81.1 | −1.2 |
| MBPP | 84.1 | 83.6 | −0.5 |
| MBPP+ | 66.9 | 69.3 | **+2.4** |
| LiveCodeBench | 27.6 | 30.2 | **+2.6** |
| AIME25 | 30.0 | 40.0 | **+10.0** |
| MATH-500 | 81.8 | 80.8 | −1.0 |
| GSM8k | 87.3 | 85.8 | −1.5 |
| LongBench v2 | 36.8 | 37.2 | **+0.4** |
| **FRAMES** | **55.7** | **52.3** | **−3.4** |

**A hybrid KDA + full-attention MoE tolerates 30% REAP with no sign of the hybrid-specific
collapse the directive feared.** Long-context behaviour (LongBench v2) is *unchanged*, which
is the sharpest available evidence that pruning experts does not destabilise the linear-attention
recurrence. `[EST]`

### 2. The Nemotron finding is about expert-pool size, not hybridness

Tracing the Nemotron3 claim: the source is a **biomedical-domain** study of pruned MoE
factual reliability (arXiv **2607.01444**). Its actual finding is:

> Under moderate (~50%) pruning ratios, models with **smaller expert pools** exhibit factual
> reliability degradation earlier than models with **larger expert pools**; Nemotron3 undergoes
> gradual reliability degradation at 50%.

The governing variable is **expert granularity**, which is exactly the variable the REAP paper
itself identifies (Mixtral-8, Llama-4-Scout-16 degrade; Qwen3-128, GLM-4.5-Air-128 do not).
Nemotron 3's Mamba layers are not implicated; its expert pool size is. `[EST]`

Separately, NVIDIA's own hybrid-compression work (**Nemotron-Labs-3-Puzzle-75B-A9B**,
arXiv 2607.04371; **Minitron-SSM**, arXiv 2504.11409) finds that what actually breaks hybrid
models is **pruning the Mamba/SSM state itself** — "pruning Mamba heads and Mamba head
dimension caused severe accuracy degradation, particularly with head dimension pruning."
**REAP does not touch attention or SSM state at all. It only deletes expert FFNs.** `[EST]`

> **Revised assessment.** The hybrid-fragility risk was inferred from a result that, traced to
> source, is a *granularity* result reported on a *biomedical QA* benchmark, not a hybrid-
> architecture result. The one true hybrid-specific compression failure in the literature is
> SSM-state pruning, which is out of scope for REAP. Meanwhile the closest architectural
> analogue to GLM-5.3-Flash — Kimi Linear, same KDA 3:1 stack — has a published, near-lossless
> REAP-30 checkpoint from the method's own authors. `[EXT, well-supported]`

**GLM-5.3-Flash sits at 288 experts / top-8 — the most favourable granularity in the entire
REAP literature.** On the variable that actually predicts degradation, we are better placed
than any published REAP subject.

## What the risk actually is

Three concerns survive, reordered by real severity.

### R1 — Rare-knowledge erosion, not architectural collapse *(highest)*

The Kimi-Linear-REAP card's own caveat is the tell:

> "FRAMES is a benchmark more reliant on the model's internal factual knowledge compared to
> LongBench v2, so it shows a higher accuracy drop from expert pruning."

**FRAMES −3.4 at only 30% pruning** is the largest regression in the table, and it is
precisely the failure mode the directive cares about: obscure-domain and rare-fact retention.
Reasoning (AIME +10.0) and code (LCB +2.6) survive or improve; *stored knowledge* does not.

This is the mean-vs-specialist bias of the REAP criterion showing up empirically
([30-reap.md](30-reap.md)). It is the strongest justification in the project for both the
quantile-blended saliency A/B and the broad-domain calibration mixture. `[EST]`

### R2 — mHC recalibration *(real, but cheap to fix)*

The directive's mechanism is sound and worth preserving even though the Nemotron evidence
did not support it: **mHC's connection matrices were Sinkhorn-normalised against the
*original* expert-output distribution.** Deleting half the experts shifts the statistics of
what flows into the residual streams, and mHC's doubly-stochastic constraint (`hc_mult 4`,
`hc_sinkhorn_iters 20`) was fitted to the old distribution.

mHC (arXiv **2512.24880**) constrains connection matrices to the Birkhoff polytope of doubly
stochastic matrices via Sinkhorn–Knopp, which "ensures conservation of feature means and
bounded signal propagation." A shift in expert-output means is therefore *exactly* the
perturbation mHC's invariant is defined against. `[EST for the mechanism; [OPEN] for whether
it matters empirically at 50%]`

**Mitigation is unusually cheap here: mHC is only 17.7M parameters (0.006% of the model).**
It can be *fully* fine-tuned during healing rather than LoRA-approximated. See
[70-healing.md](70-healing.md).

### R3 — MTP layer 45 *(minor, but must not be forgotten)*

Layer 45 is the multi-token-prediction block and carries its own full 288-expert MoE
(7.43B params). It must be pruned **consistently with** the layers it drafts for, or the
draft head's routing will disagree with the target model's. Speculative decoding is
downstream and out of scope, but silently corrupting the MTP block would poison that later
work. `[EXT]`

## Go/no-go gate (directive §6.3) — recalibrated

Because a hybrid-KDA REAP-30 reference now exists, the sensitivity probe has a real
comparison curve rather than only Qwen3/Kimi-K2 full-attention anchors.

- **Compare against:** Kimi-Linear-REAP-30's shape — reasoning/code flat-to-positive,
  long-context flat, factual-recall down a few points.
- **Trip the gate if:** long-context or reconstruction error degrades *disproportionately*
  relative to text KL at the same ratio. That would be the genuine linear-attention signal.
- **Do NOT trip the gate on:** a factual-recall / rare-domain drop. That is expected,
  reproduced in the reference, and is what the healing stage exists to repair.
