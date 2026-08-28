# Deep research brief — maximising quality of a 50% REAP on GLM-5.3-Flash

## What this is for

We have already produced a *correct but thin* 50% REAP of `zai-org/GLM-5.3-Flash`. We are now
doing **one more pass**, and we want it to be the best that is achievable on the hardware —
not an incremental improvement on what we did.

Every recommendation must be actionable under the constraints in §2 and must change what we
actually do. Generic advice about MoE pruning is a failure. If the literature does not answer a
question, say so explicitly and label it an open risk rather than filling the gap.

## 1. The exact target

Verified from `config.json` and all 62 safetensors headers — do not re-derive, but do
challenge if a source contradicts:

| property | value |
|---|---|
| params | 321,342,220,638 total; **routed experts = 311,672,586,240 = 96.99%** |
| MoE | **288 routed experts/layer, top-8**, 1 shared expert, `moe_intermediate_size` 2048 |
| MoE layers | 42 (layers 3–44; layers 0–2 dense), plus an MTP block at layer 45 |
| routing | sigmoid scoring, `noaux_tc`, `norm_topk_prob=true`, `routed_scaling_factor=2.5`, `n_group=1` |
| attention | **34 KDA (Kimi Delta Attention) layers interleaved 3:1 with 11 MLA+DSA layers** (full attention at 3,7,…,43) |
| residual | **mHC** — Manifold-Constrained Hyper-Connections, `hc_mult=4`, 20 Sinkhorn iters (arXiv 2512.24880) |
| modality | **natively multimodal**; dense 24-layer ViT (0.18% of params, **no MoE**), image tokens route through the *same* expert pool as text |
| precision | released as **FP8 E4M3 with 128×128 block scales** — there is no BF16 master |
| context | 1M native |

**Measured routing behaviour on our own calibration run** (48.3M-token corpus, 7 domains):
all 288/288 experts fire in every layer; token counts per expert span **~250×** (min ~190,
median ~7.2k, max ~216k); per-expert saliency spans 10–17×.

## 2. Hard constraints — a recommendation that violates these is useless

- **Single Jetson AGX Thor.** 122 GiB *unified* memory (CPU and GPU share one pool), sm_110a,
  ~936 GB NVMe, ~105 MB/s link. **No cloud spend, no second machine.**
- **The model cannot be loaded.** 157–328 GiB does not fit in RAM, and `device_map="auto"`
  faults because accelerate reads unified memory as VRAM. Everything must work **one layer or
  one tensor at a time, streamed from safetensors.**
- **A KDA forward costs ~13 GiB of transient memory per 2048-token sequence**, linear in batch.
  This, not the weights, bounds calibration batch size.
- **Tegra unified memory is invisible to the OOM killer** (a touched 2 GiB `cudaMalloc` charges
  42 MiB to the cgroup), so nothing can rely on kernel memory pressure signals.
- **The output must load in stock transformers/vLLM.** `num_local_experts` is a single scalar
  applied to every layer, so **per-layer expert counts are not expressible** — we implemented
  non-uniform allocation, measured it better, and had to discard it as unloadable.
- Final artifact: FP8, convertible downstream to NVFP4 / GGUF.

## 3. What we did in pass 1 (the baseline to beat)

- Stock REAP saliency: `S_j = mean over tokens routed to j of g_j · ‖f_j‖₂`, captured before the
  gate scales the expert output.
- **256 samples × 2048 tokens = 0.52M tokens** — 1.1% of the 48.3M-token corpus we built, and
  0.26% of the REAP paper's ≥110B setting (12,228 × 16,384).
- Uniform 50% per layer. Retained saliency mass **0.643 = ×1.286 vs random**; retained routing
  mass ×0.90 (REAP keeps rare-but-strong experts over common-but-weak).
- Healing: **first-moment output-scale correction only** — per-layer gain
  `E[g‖f‖]_all / E[g‖f‖]_kept` (median 0.696) applied exactly to the F32 block scales. No
  distillation, no gradient step.
- **No evaluation of any kind.**

## 4. Questions the research must answer

Prioritise by expected quality gain per unit of Thor time.

### 4.1 Saliency criterion — is REAP's conditional mean the right objective here?
- What has superseded or beaten REAP (arXiv 2510.13999) since publication? Name specific
  criteria, with measured deltas at ~50% on high-granularity MoEs.
- REAP's mean is **frequency-invariant and variance-blind**: a rare high-magnitude specialist
  and a reliable mediocre generalist score identically. Given our measured 250× token-count
  spread, does a **quantile/max-blended** statistic (e.g. `0.6·mean + 0.4·p99`) preserve
  obscure-domain capability better? Is there prior art, or is this untested?
- Does weighting by **downstream** effect (layer output reconstruction, or logit KL) beat
  per-layer `g·‖f‖`? Is a cheap second-order/Hessian-style signal (à la SparseGPT/Wanda,
  adapted to whole experts) affordable one-layer-at-a-time?
- **Router-aware**: with `norm_topk_prob=true`, deleting an expert renormalises the gates over
  a different support. Does any criterion account for the *post-prune* routing distribution
  rather than the pre-prune one? Iterative/greedy re-scoring after each removal?

### 4.2 Calibration — how much, of what, at what length?
- What is the *measured* relationship between calibration tokens and post-prune quality for MoE
  expert pruning at ~50%? We have per-expert token counts as low as 190; where is the knee?
- Does **sequence length** matter beyond token count for a model whose architecture is a 3:1
  KDA/full-attention hybrid built for 1M context? Our pass used 2048 and truncated everything.
- Corpus composition for our specific goals: **agentic tool use, coding, world knowledge in
  science/bio/finance, and vision**. What mixtures are shown to preserve which capabilities?
  Is there evidence on *tool-call/trajectory* data specifically, as distinct from code?
- Does calibrating on **the model's own outputs** (self-generated, à la arXiv 2511.18864) beat
  human/web text for retention?

### 4.3 Beyond pure pruning
- **Expert merging vs pruning**: REAP argues pruning wins on generative tasks, but is there a
  hybrid — merge the near-duplicates, prune the rest — with evidence at 50%?
- **Within-expert / neuron-granularity** pruning (e.g. MoNE, arXiv 2510.05781) as a *second
  axis*, so 50% total compression is split across two individually-validated axes instead of
  pushing whole-expert pruning to its limit. Does this compose with REAP?
- Is there any published method that **redistributes** deleted experts' function into survivors
  (weight surgery, low-rank residual, bias correction) beyond a scalar gain?

### 4.4 Healing without gradients, or with very cheap ones
- Our first-moment scalar correction is the weakest defensible thing. What is the strongest
  repair achievable **one layer at a time, teacher streamed, on 122 GiB**?
- Specifically: layer-local reconstruction distillation (teacher-forced, so layers are
  independent) — what does it recover, at what token cost? Evidence at ≥100B scale?
- Are **router-only** or **mHC-only** updates (50.7M and 17.7M params respectively — both fully
  trainable here) enough to recover most of the loss? mHC's Sinkhorn-normalised matrices were
  fitted against the *original* expert-output distribution; is there work on repairing
  hyper-connection/residual-mixing weights after structural surgery?
- Is there evidence that **re-estimating `e_score_correction_bias`** (the aux-loss-free load
  balancing bias) after pruning materially helps?

### 4.5 Architecture-specific risks nobody has published on
- **KDA hybrid**: `cerebras/Kimi-Linear-REAP-35B-A3B` is the closest published analogue (same
  KDA 3:1 stack, 30% prune, LongBench-v2 flat, **FRAMES −3.4**). Is there anything at 50% on a
  KDA hybrid? Any evidence that linear-attention layers are more or less tolerant of MoE
  pruning than full-attention layers?
- **mHC**: any published interaction between hyper-connections and structural sparsity?
- **Native multimodality with a shared backbone**: how much of the expert pool is
  vision-serving, and does text-dominated calibration measurably degrade vision even when image
  tokens are present? What share is enough?
- **MTP block**: we exclude it (transformers does not instantiate layer 45). Is there a correct
  way to prune an MTP head consistently with its target model, for later speculative decoding?

### 4.6 Cheap evaluation that actually predicts downstream quality
- We can afford ~1–2 hours of eval, cannot run full agentic benchmarks, and have no baseline
  generations. What proxies **correlate measurably** with post-prune benchmark loss?
- Specifically: held-out perplexity, per-domain KL vs the unpruned model (we can stream the
  teacher one layer at a time), routing-mass coverage, layer reconstruction error. Which of
  these has *published* correlation with downstream loss, and which are folklore?
- Is there a small, fast, discriminative probe for **agentic/tool-use** and for
  **factual-recall** capability specifically — the two things we most expect to lose?

## 5. Deliverable

A findings document that, for each of §4.1–4.6:

1. States what is **established** (cite: arXiv ID, repo, model card, or measured result),
   what is a **vendor claim**, and what is **unknown**.
2. Gives a **specific, implementable recommendation** for our pipeline under §2's constraints,
   including its expected cost in Thor-hours.
3. Ranks recommendations by **expected quality gain per Thor-hour**, so we can build the plan
   top-down and stop when the budget runs out.

Where a technique is strong but violates §2, say so and state the precondition that would make
it usable — we lost time implementing non-uniform allocation before checking that the config
could express it, and do not want to repeat that.
