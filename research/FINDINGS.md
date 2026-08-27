# Phase 0 — Deep Research Findings

**Project:** GLM-5.3-Flash REAP + Heal · **Host:** Jetson AGX Thor · **Date:** 2026-08-27
**Status:** Phase 0 complete. No weights downloaded. No implementation started.

Depth lives in [`../wiki/`](../wiki/README.md); this document is the synthesis and the
decisions that follow from it. Confidence markers: `[EST]` established · `[VEN]` vendor claim
· `[EXT]` our extrapolation · `[OPEN]` no source exists.

---

## 1. The four findings that change the plan

### 1.1 The release is FP8, not BF16 — and there is no BF16 master `[EST]`

`zai-org/GLM-5.3-Flash` ships `quantization_config: {quant_method: fp8, fmt: e4m3,
weight_block_size: [128,128], activation_scheme: dynamic}`. Every routed expert is F8_E4M3
with F32 block scales. **On disk: 328.3 GB / 305.8 GiB**, not the 640 GB the directive assumed.

`unsloth/GLM-5.3-Flash` (BF16, 642 GB) is a **dequantised upcast** of that same FP8 release —
`unsloth/GLM-5.3-Flash-FP8` carries a byte-identical dtype profile to `zai-org`'s. It contains
no information the FP8 does not, in twice the space.

**Consequences.** Storage halves. The BF16 intermediate disappears from the pipeline. Pruning
becomes *lossless* (below). And the directive's primary deliverable — "healed **BF16**
weights" — should become **healed FP8 base + adapters**, with BF16 available as an optional
export rather than a pipeline stage. Operator confirmed this direction mid-session.

### 1.2 Routed experts are 96.99% of the model `[EST]`

From an exact count over all 62 shard headers (76,108 tensors, read via HTTP range requests
without downloading any weights — `research/glm53_tensors.json`):

| component | params | share |
|---|---:|---:|
| **routed experts** | **311,672,586,240** | **96.99%** |
| attention (11 MLA+DSA, 34 KDA) | 6,199,640,639 | 1.93% |
| shared expert | 1,082,196,480 | 0.34% |
| embed + lm_head | 1,268,776,960 | 0.39% |
| vision tower | 563,627,008 | 0.18% |
| routers | 50,737,248 | 0.016% |
| **mHC** | **17,695,935** | **0.006%** |

This single number organises the entire project. Expert pruning is the **only** compression
lever that matters — and, symmetrically, **every component worth protecting is too small to
be worth compressing.** The whole non-expert model is 3% of the parameters. Protecting all of
it at BF16 costs ~19 GB against a ~97 GB budget. That is why the 50% target has room.

### 1.3 The architecture is Kimi Linear's — and that de-risks the project's stated top risk `[EST]`

The directive states GLM-5.3-Flash is "**not** MLA+DSA". It is. `layer_types` is
**34 × `linear_attention` + 11 × `deepseek_sparse_attention`** in a strict **3:1** pattern
(full attention every 4th layer). The 11 full layers are MLA (`q_lora_rank 1536`,
`kv_lora_rank 512`, `qk_rope_head_dim 0`) with a DSA indexer and IndexPool compression. The
34 linear layers are **KDA — Kimi Delta Attention**; the config names them
(`linear_attn_config.kda_layers`).

That is *exactly* Kimi Linear's attention stack — and **`cerebras/Kimi-Linear-REAP-35B-A3B-Instruct`
exists**: Cerebras's own REAP at 30% on that architecture. Near-lossless, with
**LongBench v2 flat (36.8 → 37.2)** — direct evidence that deleting experts does not
destabilise the linear-attention recurrence.

Meanwhile the Nemotron3 result the directive built its top risk on traces to a
**biomedical-QA** study (arXiv 2607.01444) whose governing variable is **expert-pool size**,
not hybridness — the same variable the REAP paper itself identifies. NVIDIA's own hybrid
compression work finds the thing that actually breaks hybrids is **pruning the Mamba/SSM
state**, which REAP never touches.

> **The hybrid-fragility risk is materially lower than the directive assumed.** And on the
> variable that genuinely predicts degradation, GLM-5.3-Flash at **288 experts / top-8** is
> the most favourable subject in the entire REAP literature.

### 1.4 Pruning never leaves FP8, so the prune stage is numerically lossless `[EST]`

Experts are stored as **individual per-expert tensors** with their own block scales:

```
model.language_model.layers.{L}.mlp.experts.{E}.{gate,up,down}_proj.weight            F8_E4M3
model.language_model.layers.{L}.mlp.experts.{E}.{gate,up,down}_proj.weight_scale_inv  F32
```

Deleting an expert means deleting six tensors and renumbering. No dequantisation, no
reconstruction, **no numerical change to any surviving weight**. The pruned checkpoint is
bit-identical to the source on everything it keeps. An entire class of risk the directive
budgeted for does not exist.

---

## 2. Storage: solved, with nothing deleted

**298 GiB free vs a 305.8 GiB source.** Naively staging source + pruned output + corpus peaks
at **~496 GiB**. Holding the source at all is fatal.

**Resolution: never stage the source.** Two measurements make it work — the HF link saturates
at ~105 MB/s (parallelism buys nothing), so one full pass costs **~52 minutes**; and REAP is
layer-sequential, so nothing ever requires whole-model residency.

One catch, found while validating this: **shards are laid out in lexicographic tensor-name
order**, so layers appear as `45, 0, 1, 10, 11, …, 19, 2, 20, …`. Numeric layer order is not
shard order, and a naive download-shard-then-delete window would thrash. The fix is to stream
in **numeric layer order via range requests against the safetensors byte offsets** (already
extracted). Same total bytes, different order, and peak local footprint drops to **one layer
plus one prefetch ≈ 15 GB**.

| Stage | Peak disk |
|---|---:|
| A — saliency pass | ~85 GiB |
| B — surgery pass | ~176 GiB |
| C — healing | ~211 GiB |
| **D — quantise** | **~251 GiB** |
| final artifact | ~91 GiB |

**Worst peak ~251 GiB against 298 GiB free — fits with ~47 GiB margin, and no model,
container image, or prior-art repo needs to be deleted.** `models/DeepSeek-V4-Flash-0731-REAP`
(101 GiB) is preserved as the §7 incumbent baseline.

Optional zero-risk margin, operator's call: `docker image prune` (21 GiB unreferenced layers)
+ `thor-vllm-cache` (31 GiB regenerable compile cache) = **+52 GiB**.

---

## 3. Recomputed memory table (replaces the directive's estimates)

Weights only, NVFP4 experts at 4.5 bpw effective. Thor usable envelope ≈ **117 GiB**.

| Prune | Total params | rest @ FP8 | rest @ BF16 | Verdict |
|---:|---:|---:|---:|---|
| 30% | 227.8 B | 123.3 GiB | 129.7 GiB | **does not fit** |
| 40% | 196.7 B | 107.0 GiB | 113.4 GiB | fits, ~4–10 GiB left |
| **50%** | **165.5 B** | **90.6 GiB** | **97.1 GiB** | **target — 20–26 GiB headroom** |
| 55% | 149.9 B | 82.5 GiB | 88.9 GiB | comfortable |

The directive was optimistic by 5–12 GB; **its conclusion stands — 50% is the right target.**

**KV cache is nearly free**, and this is the hybrid paying off: only 11 of 45 layers grow a
cache, and they are MLA with `kv_lora_rank 512` → **~11 KB/token**, so 128K context costs
**~1.4 GB**. The 34 KDA layers hold ~71 MB of *fixed*, context-independent state. Long context
on Thor is weight-bound, not cache-bound.

---

## 4. What the literature does and does not establish about REAP

| Claim | Status |
|---|---|
| REAP > merging on generative tasks, especially at 50% | `[EST]` |
| ≤2% coding drop at 50% on Qwen3-Coder-480B / Kimi-K2-1T | `[EST]` |
| **Suite-wide mean coding loss at 50%: 6.9%** | `[EST]` — the honest number |
| Degradation is governed by **expert granularity** | `[EST]` |
| Any result above 50% | **`[OPEN]` — paper tests only 25% and 50%** |
| Non-uniform per-layer budgets | **`[OPEN]` in REAP; EvoESAP (arXiv 2603.06003) reports up to +19.6% on MATH-500 at 50% as a plug-in over REAP** |
| Quantile-pooled saliency to protect rare specialists | **`[OPEN]` — no prior art found; genuinely novel, must be A/B'd** |

**The criterion's structural bias is real and is the top risk.** `S_j` is a *conditional mean*
over the tokens where expert *j* fired, so it cannot see how *often* an expert matters or how
*variable* its contribution is. A rare high-magnitude specialist and a reliable mediocre
generalist are indistinguishable.

This is not theoretical. The Kimi-Linear-REAP-30 card's worst regression is **FRAMES −3.4** —
a factual-knowledge benchmark — while AIME25 gains +10.0 and LiveCodeBench +2.6. Reasoning
and code survive; **stored knowledge does not.** Cerebras say so explicitly: "FRAMES is more
reliant on the model's internal factual knowledge … so it shows a higher accuracy drop from
expert pruning."

**Obscure-domain retention is the thing this project is most likely to lose**, and it is the
justification for both the quantile-blended saliency A/B and the broad-domain corpus.

---

## 5. Calibration: the directive's corpus is too small

The REAP paper's own setting for models **≥110B** is **12,228 samples @ 16,384 tokens,
untruncated**. The directive's "Cerebras used 3,072" is not the paper's figure, and its
6,000–8,000 @ 4–8K estimate is **below** the published setting, not above it.

It should be *larger* still, because saliency is a per-expert statistic and GLM-5.3-Flash has
the largest expert pool in the literature:

| | Qwen3-Coder-480B | GLM-5.3-Flash |
|---|---:|---:|
| experts / layer | 160 | **288** |
| expected token share per expert | 5.0% | **2.8%** |

Each expert is estimated from **~1.8× fewer tokens** at equal sample count. **Under-sampling
and the criterion's mean-bias compound on exactly the same rare specialists.**

**Recommendation: 12,288 samples @ 16,384 tokens as a floor (~200M tokens).**

**One refinement to the difficulty policy.** The directive says bias every bucket toward the
harder end. Evidence supports **medium-hard, not maximal**: Hard-only calibration degrades
general perplexity **6.2–12.1%** vs 1.5–4.2% for mixed — precisely the connective tissue the
8% ballast slice exists to protect. Recommended **60% medium / 30% hard / 10% easy**. Also
worth recording: **pruning is up to 2.3× more sensitive to calibration difficulty than
quantisation is**, so curation effort belongs at the prune stage.

Sources verified against the HF API. `nvidia/Nemotron-CC-Math` is **gated (401)** —
`Nemotron-PrismMath` covers the bucket. The **finance/quant 10% bucket has no verified
source yet** and is the largest remaining gap.

---

## 6. Vision: the risk is not where the directive looked

**The vision tower has no MoE.** It is a dense 24-layer ViT, 563.6 M params, 0.18% of the
model. **REAP cannot touch it.**

The real risk is one level down. GLM-5.3-Flash is natively multimodal — image tokens project
into the shared 4096-d backbone and route through the *same* 288-expert MoE as text. If image
tokens route to experts that text-only calibration never activates, those experts have an
empty active set, score `S_j = 0`, and are deleted first.

> **Text-only calibration does not under-weight vision experts. It makes them invisible, and
> REAP deletes invisible experts with certainty, at any prune ratio.** `[EXT — follows
> directly from the saliency definition]`

The 15% multimodal share is therefore **load-bearing, not a nice-to-have**. Two concrete
requirements follow:

1. **Assert a non-zero image-token count** (`image_token_id 154854`) in the tokenised
   calibration stream before trusting any run. A collator that silently drops images
   degenerates calibration to text-only — the exact scenario that deletes vision. This is the
   highest-value single assertion in the pipeline.
2. **Measure modality-stratified routing mass during the sensitivity probe.** It comes free
   with the §3.6 routing-mass proxy, converts an `[OPEN]` into a number, and directly sets the
   required multimodal share.

**Ignore-list conflict resolved.** Upstream skips the vision tower for a *quantisation*
reason — "compressing these parameters offers little benefit" — which applies with more force
here (0.18% of mass). Skip it **on merit, not by imitation**. Prune and quantise need two
different lists; REAP needs none, since it only touches MoE layers it discovers. Keep the
`linear_attn` exclusion for an independent reason: quantising a recurrence's state-transition
parameters compounds error along the sequence instead of averaging it out.

---

## 7. Healing: one stage becomes cheap, one becomes impossible

**Layer-local distillation is the primary stage, and the teacher never needs to fit.** The
obvious objection — teacher is 328 GB, Thor is 117 GiB — dissolves because REAP's objective is
layer-local: minimise `‖pruned_L(x) − original_L(x)‖`. That needs **one teacher layer and one
student layer at a time**, both streamed; peak residency ~11 GB. MoE-Pruner (arXiv 2410.12013)
reports the pruned-vs-pretrained gap "largely mitigated via expert-wise knowledge distillation
with only 1000 training samples from C4, completing in 1 hour."

**Fully fine-tune mHC and the routers rather than LoRA them.** mHC is 17.7 M params; a
rank-decomposed approximation of an 18 M module is strictly worse and more complicated than
training it. And the directive under-weights the **routers** (50.7 M): after pruning, each
router emits logits over a *renumbered, halved* expert set, with `norm_topk_prob` renormalising
over different support. **The router is the component whose semantics changed most**, and it is
free to train fully. Combined: 68.4 M params.

**RLVR is out of reach and should be dropped, not scheduled optimistically.** `[EXT]`
~11.7 GB read per decoded token / 273 GB/s → ~23 tok/s ceiling, ~10 tok/s realistic against
measured prior art on this box. A modest 1,000-prompt GRPO run is **~460 days** single-stream,
~30 days under optimistic batching, on a box that must simultaneously hold ~97 GiB of weights.
Published rollout-efficiency work is demonstrated at **1.5B–3B**, two orders of magnitude below
target. The capability RLVR was meant to repair — long-horizon coherence — should be pursued
through the calibration and SFT mixture instead, which raises the value of the 22% agentic
share. **This changes directive §6.7 and needs operator acknowledgement.**

---

## 8. Quantisation and the Thor target

FP8 → NVFP4 is **only mildly** a double quantisation. The grids differ in opposite directions:
FP8 has a finer value grid (E4M3, 3 mantissa bits) but far coarser scale granularity (16,384
elements per scale); NVFP4 has a coarser element format (E2M1, 1 mantissa bit) but much finer
scales (16 elements). NVFP4's tight blocks partly compensate, and the dominant error term —
E2M1's mantissa — would be incurred identically from a true BF16 master. Anchor: NVFP4 within
1% of FP8 on DeepSeek-R1 `[VEN]`.

**Thor is SM110a > SM100**, so vLLM does **not** fall back to weight-only; full W4A4 is
available. The llm-compressor `< SM100` warning does not apply.

**Two traps already paid for on this exact box** (operator's prior Thor work):
- The **Marlin FP4 MoE kernel faults in-kernel at 256-expert scale**; only the **cutlass**
  backend loads large MoE models. We have **288 experts, 144 post-prune** — plan for cutlass.
- **MLA requires `TRITON_MLA`**; FLASHINFER is invalid for MLA. Our 11 full layers are MLA.

Per-component policy, following the GLM-5.2 precedent (NVFP4 on MoE + FP8 on attention,
>70% reduction with GPQA maintained `[VEN]`): **NVFP4** experts + shared expert; **FP8**
attention; **BF16 untouched** for vision tower, KDA recurrent state, mHC, routers, `lm_head`.

Export as **`compressed-tensors`** — vLLM reads it directly and it records the per-component
precision map explicitly, which is what the downstream kernel work needs.

---

## 9. Toolkit decision

**`vllm-project/llm-compressor` 0.13.0**, not `CerebrasResearch/reap`. `[EST]`

`REAPPruningModifier` (`modifiers/pruning/reap/base.py`, read directly) discovers MoE
structure **generically** via `get_moe_attrs(model, ignore)` rather than from a hardcoded
architecture list — the best available signal that `glm5_next` will work. It prunes a uniform
fraction per layer, validates that `top_k` experts remain reachable (trivially satisfied:
288→144, top-8), and warns against `moe_calibrate_all_experts` (set it **False** — REAP wants
the real routing distribution).

llm-compressor wins because it unifies prune + quantise and already solves sequential
onloading and disk offload — precisely the hard part on a 117 GiB box. `Qwen3.8-2.4T-A95B-NVFP4-REAP-25`
confirms the prune-then-quantise ordering composes in this toolkit `[VEN]`.

**Note:** `/home/patrickd/llmcompressor` is a bare venv (6.8 G), not a git checkout, and has no
`reap` module. It predates the modifier. A fresh checkout is required.

---

## 10. Risks, ordered by real severity

| # | Risk | Status |
|---|---|---|
| **R1** | **Rare-knowledge erosion** — FRAMES −3.4 at only 30% in the closest analogue; criterion bias and thin per-expert sampling compound | `[EST]` |
| **R2** | **Nothing published above 50%**; the only high-ratio artifacts (Kimi-K3-REAP-73/80) report "noticeable degradation" | `[OPEN]`, one negative data point |
| **R3** | **Vision deleted by invisible-expert** under text-only calibration | `[EXT]`, certain if it occurs |
| **R4** | **`glm5_next` tooling immaturity** (`transformers 5.16.0`; untested with llm-compressor) | `[OPEN]` |
| **R5** | **mHC recalibration** against a shifted expert-output distribution | `[OPEN]` empirically |
| **R6** | **Activation-cache blowup** — naive sequential pipeline holds ~1.6 PB at the recommended corpus size | `[EST]` arithmetic |
| **R7** | **MTP layer 45** must be pruned consistently or later spec-decode is poisoned | `[EXT]` |
| **R8** | **Marlin FP4 MoE fault ≥256 experts on Thor** | `[EST]`, reproduced here |
| **R9** | Finance/quant calibration source unresolved | `[OPEN]` |

**Recalibrated go/no-go gate (directive §6.3).** Because a hybrid-KDA REAP-30 reference now
exists, the probe has a real comparison curve. Trip the gate on **disproportionate
long-context or reconstruction damage** — the genuine linear-attention signal. **Do not trip
on a factual-recall drop**: it is expected, reproduced in the reference, and is exactly what
the healing stage exists to repair.

---

## 11. Open questions carried into Phase 2

1. Operator sign-off: RLVR removal (§7), corpus increase to 12,288 (§5), optional 52 GiB reclaim (§2).
2. `glm5_next` × llm-compressor smoke test — cheapest possible early failure (R4).
3. Modality-stratified routing mass — measurable in the probe, closes the vision `[OPEN]` (§6).
4. Quantile-blended saliency — no prior art; A/B design required (§4).
5. EvoESAP-style non-uniform allocation — real gains reported as a plug-in over REAP, but
   costs a search; decide against measured per-layer reconstruction error.
6. Finance/quant corpus source (R9).
7. NVFP4 block-scale search (Four Over Six / SOAR / RaZeR) — refinement after a baseline run,
   not critical path.
