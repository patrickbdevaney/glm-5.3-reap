# 30 — REAP: method, published results, toolkit landscape

## The method

**REAP — Router-weighted Expert Activation Pruning.** Lasby, Lazarevich, Sinnadurai, Lie,
Ioannou, Thangarasa (Cerebras + U. Calgary). arXiv **2510.13999** (v3), **ICLR 2026**.
Repo `CerebrasResearch/reap`.

Saliency of expert *j*, over the calibration tokens for which *j* was actually routed:

```
S_j = (1 / |X_j|) * Σ_{x ∈ X_j}  g_j(x) · ‖ f_j(x) ‖₂
```

where `g_j(x)` is the router gate value and `f_j(x)` the expert's output. It is a
**conditional mean** — averaged only over tokens where the expert fired — derived as a bound
on layer output reconstruction error. The lowest-`S` experts in each layer are deleted and
the router's corresponding logits removed. `[EST]`

### The structural bias in the criterion

Because `S_j` is a **mean** over the expert's own active set, an expert that fires rarely but
contributes enormously when it does scores the *same* as one that fires constantly and
contributes moderately. What the criterion cannot see is **how often** the expert matters,
nor the **variance** of its contribution. Two experts with identical means — one a reliable
generalist, one a rare high-magnitude specialist — are indistinguishable. `[EST, from the formula]`

The directive proposes blending in an upper quantile, e.g. `0.6·mean + 0.4·p99`, to protect
rare specialists. **No prior art was found for quantile-pooled REAP saliency.** Treat as a
novel modification requiring an A/B test, not as a known-good technique. `[OPEN]`

## Published results — what is actually established

| Claim | Status |
|---|---|
| REAP > merging on **generative** tasks, esp. at 50% | `[EST]` — the paper's central result |
| ≤2% drop on code gen at 50% for **Qwen3-Coder-480B** and **Kimi-K2-1T** | `[EST]` |
| 50% mean accuracy loss on coding **across the whole model suite**: **6.9%** | `[EST]` — the honest number |
| Merging at 50%: >20% loss ("functional subspace collapse") | `[EST]` |
| Any result at **60%, 75%** compression | **`[OPEN]` — the paper tests only 25% and 50%** |
| Non-uniform per-layer budgets | **`[OPEN]` — the paper prunes uniformly, with no ablation** |

> **Directive correction.** The directive cites "50% on Qwen3-Coder-480B retains 97.6%
> non-agentic coding". That is a real per-model number, but the *suite-wide* mean loss at
> 50% is **6.9%** on coding. Near-losslessness at 50% is a property of the **largest,
> highest-granularity** models, not of REAP in general. GLM-5.3-Flash at 288 experts / top-8
> is squarely in the favourable regime — but the favourable regime is what makes 50% safe,
> not the method alone.

### Granularity is the governing variable

The paper's own failure analysis: **low-granularity** models (Mixtral-8x7B at 8 experts,
Llama-4-Scout at 16) degrade far worse than **high-granularity** ones (Qwen3-30B at 128,
GLM-4.5-Air at 128, ERNIE-4.5-21B at 64). The llm-compressor example reproduces this —
"very high recovery" for Qwen3-30B-A3B (128 experts), "poor scaling" for Moonlight-16B
(64 experts). `[EST]`

**GLM-5.3-Flash has 288 routed experts per layer with top-8 routing — the highest granularity
of any model in the REAP literature.** 50% pruning still leaves 144 experts and top-8 routing
(density 5.6%). This is the strongest single argument that 50% is achievable here. `[EXT]`

## Calibration setup used by the paper

- Datasets: **C4, evol-codealpaca, xlam-function-calling, SWE-smith-trajectories,
  WritingPrompts, tulu-3-sft-personas-math**
- **≤110B params:** 1,024 samples @ 2,048 tokens
- **≥110B params:** **12,228 samples @ up to 16,384 tokens (no truncation)**

> **Directive correction.** The directive says "Cerebras used 3,072 samples" and proposes
> 6,000–8,000 @ 4–8K as a working estimate. The paper's own setting for models ≥110B is
> **12,228 samples @ 16K**. At 321B, GLM-5.3-Flash is far past that threshold. The directive's
> estimate is **below** the published setting, not above it. Plan for ≥12,228 samples. `[EST]`

## Toolkit landscape

### `vllm-project/llm-compressor` — **recommended base**

Release **0.13.0** (2026-08-11). REAP lives at
`src/llmcompressor/modifiers/pruning/reap/` (`base.py` 327 lines, `utils.py`, `__init__.py`),
class **`REAPPruningModifier`**.

Inspected `base.py` directly. What it does: `[EST]`

- Discovers MoE structure generically via `get_moe_attrs(model, ignore)` → `MoeModelAttrs`
  (`num_experts`, `top_k`, `n_group`, `group_size`, `top_k_group`, `moe_layer_names`).
  **It is not a hardcoded per-architecture list** — it introspects the model. Good sign for
  `glm5_next` support.
- `sparsity` is a **uniform fraction of experts removed per layer** — confirming no
  non-uniform allocation in the shipped implementation.
- Handles grouped routing (`n_group`) by dropping a proportional count per group.
  GLM-5.3-Flash has `n_group: 1`, so this path is inert.
- **Validates that `top_k` experts remain reachable after pruning** and refuses otherwise.
  For us: top_k=8, 288→144 experts, trivially satisfied.
- Prunes structurally in `on_finalize` via `prune_moe_layer`, rewriting the expert count.
- Warns if `moe_calibrate_all_experts` is on — REAP does not need it (it wants the *real*
  routing distribution, not a forced all-expert pass). **Set `moe_calibrate_all_experts=False`.**

**Why llm-compressor over `CerebrasResearch/reap`:** it unifies prune + quantise in one
toolkit, and it already solves sequential onloading and disk offload, which is precisely the
hard part on a 117 GiB box. The Cerebras repo is the reference implementation but does not
carry the memory machinery. `[EXT]`

**Local state:** `/home/patrickd/llmcompressor` is a bare venv (6.8 G), **not a git checkout**,
and contains no `reap` module. It predates the REAP modifier. A fresh checkout is required.

### Verified community precedent for prune-then-quantise ordering

`Qwen3.8-2.4T-A95B-NVFP4-REAP-25` — 25% of least-salient experts pruned **before** NVFP4
quantisation. Confirms the ordering and that the two compose in llm-compressor. `[VEN]`

## Community prune-ratio distribution

Public REAP checkpoints cluster at 19–50% (MiniMax-M2.5-REAP-19/29/39, MiniMax-M2-REAP-30,
Qwen3-Coder-REAP-20/50, Qwen3.8-REAP-25, Kimi-Linear-REAP-30). The exceptions are the
Kimi-K3 MLX ports at **REAP-73 and REAP-80** — and those are explicitly described as having
**"noticeable degradation versus full K3"**. The absence of validated high-ratio checkpoints
is real, and the one high-ratio data point that exists is a negative result. `[EST]`

## Early criterion divergence, and the noise floor that interprets it `[MEAS 2026-08-27 23:58]`

Run on **partial** pass-2 data — 11 layers of chunk 1, ~1/10 of the final token budget — so
indicative, not final. Keep-set overlap against stock REAP at 50%, 144 experts/layer:

| criterion | overlap vs REAP | experts differing (of 144) |
|---|---|---|
| `var_aware` (mean + 0.5·std) | 0.958 | 6 |
| `quantile` (0.6·mean + 0.4·p99) | 0.937 | 9 |
| `gate_only` (mean g) | 0.922 | 11 |
| `mix_sample` (sample-proportional re-weight) | 0.915 | 12 |
| `mix_codemath` (code/math-forward) | 0.911 | 13 |
| `norm_only` (mean ‖f‖, gate removed) | 0.866 | 19 |
| `frequency` (sum, not mean) | 0.804 | **28** |

Three things this already tells us:

1. **REAP's conditional mean is doing real work.** `frequency` is the control — the
   frequency-weighted ranking REAP exists to avoid — and it picks **28 different experts per
   layer**. Had it come out at 0.99, the whole conditional-mean argument would have been
   decorative. It did not.
2. **Both factors in `g·‖f‖` carry independent signal.** Removing the gate (`norm_only`) moves 19
   experts; keeping only the gate (`gate_only`) moves 11. Neither half reproduces the product, so
   the criterion is not secretly one of its factors.
3. **The mixture is not neutral.** Re-weighting the calibration mixture moves ~12 experts per
   layer — the same order as changing the criterion outright. Given the realised token mixture
   gave `code` 5.7% against a 21% sample quota, that is a live decision, not a formality.

### The methodological point: P6 sets the ruler for P7

These overlaps cannot be read on their own, because **part of every gap is sampling noise, not
criterion disagreement**. At 1/10 of the token budget, even stock REAP disagrees with *itself*
across independent halves.

That is exactly what P6's split-half overlap measures. So the two gates must be read together:

> **split-half overlap is the noise floor; criterion divergence is only meaningful above it.**

If P6 reports 0.97 at full budget, then `mix_codemath` at 0.911 is a real difference worth
materialising and evaluating. If P6 reports 0.91, then everything in the table above is noise and
the correct conclusion is that the criterion does not matter at this ratio — spend the budget on
evaluation instead. Running P7 without P6 would invite reading noise as signal, which is how a
project talks itself into materialising three arms of the same mask.

## Interim noise floor, and what it does to the criterion table `[MEAS 2026-08-28 06:55]`

Split-half at **2 chunks per half (~1M tokens each)**, i.e. 40% of the final budget:

| | |
|---|---|
| keep-set overlap | mean **0.9478**, p10 0.9167, min 0.9028 |
| layers below the 0.95 gate | **22 / 42** |
| worst layers | 35 (0.9028), 18 (0.9097), 14/31/39 (0.9167) |

Worst layers carry 380–900 tokens for their least-served expert, which is the expected driver.

**This is the ruler for P7.** Read the criterion table against it:

| criterion | overlap vs REAP | verdict at this noise floor |
|---|---|---|
| `var_aware` | 0.958 | **above the floor — indistinguishable from REAP** |
| `quantile` | 0.937 | at/below — marginal |
| `gate_only` | 0.922 | slightly below |
| `mix_sample` / `mix_codemath` | 0.915 / 0.911 | slightly below |
| `norm_only` | 0.866 | **clearly below — genuinely different** |
| `frequency` | 0.804 | **clearly below — the control, as expected** |

So the emerging answer to "does the criterion matter?" is **mostly no**: `var_aware` is inside the
noise, `quantile` marginal, and only removing the gate or switching to frequency-weighting yields
a materially different mask. That is the cheap-close outcome — it would send the budget to
evaluation rather than criterion search.

Two caveats before this is final: the shootout ran on 1 chunk and the split-half on 2+2, so the
two are **not yet measured on matched data** — both get re-run at full budget. And if the
disagreement is noise-driven, scaling 1M -> 2.75M tokens per half should lift overlap to roughly
**0.97** and pass the gate.

**If it does not pass at full budget, that is a substantive result**, not a disappointment: it
would mean 5.5M tokens still cannot determine the mask, and A7 in `CLOUD_COUNTERFACTUAL.md` moves
from "candidate non-unlock" to a real one — more calibration would be the cheapest quality
available, and a cluster would buy it.

## The split-half gate FAILED on identity and PASSED on the objective `[MEAS 2026-08-28 17:50]`

Full budget, 2.75M tokens per half:

| | 1.0M tok/half | 2.75M tok/half |
|---|---|---|
| keep-set overlap | 0.9478 | **0.9446** |
| layers below the 0.95 gate | 22/42 | 19/42 |

**Raising the budget 2.75x moved overlap by -0.003.** If the disagreement were sampling noise it
should have risen to ~0.968. It did not move at all, so the residual ~5.5% is **structural**, and
more calibration cannot fix it. That prediction was made in advance and was wrong, which is what
makes the measurement worth having.

### The disputed experts are near-ties

| | |
|---|---|
| disputed experts per layer | ~14 of 144 |
| their median distance from the decision boundary | **2.42%** |
| retained saliency mass, mask A vs mask B, scored on the FULL accumulator | 0.70004 vs 0.69973 |
| **disagreement in what REAP optimises** | **0.133% mean, 0.562% max** |

The two halves pick different experts and retain **the same saliency mass**. The experts they
argue about sit within 2.4% of the cut line — they are interchangeable, and choosing either is
not an error.

**The gate was measuring the wrong thing.** Keep-set identity treats a tie-break as a failure.
`split_half.py` now also computes retained-mass agreement and issues the verdict on *that*,
reporting `pass_on_mass` when overlap is below the identity gate but mass agrees to within 1%.
The old wording — "more tokens are the cheapest quality available" — was actively wrong here and
would have bought calibration this same run proved useless.

### Consequences

1. **Materialising is safe.** The mask is determined to 0.133% of the objective.
2. **A7 in `CLOUD_COUNTERFACTUAL.md` is settled as a NON-unlock**, now by measurement rather than
   conjecture: more calibration tokens would not produce a better mask for this model at this
   ratio. A cluster buys teacher generation, distillation healing and evaluation — not this.
3. **It sets the resolution limit for P7.** ~10% of the keep-set is an irreducible tie band, so
   criteria differing by less than that are choosing among equivalents:

| criterion | overlap vs REAP | verdict |
|---|---|---|
| `var_aware` 0.9635 | above the tie band | indistinguishable from REAP |
| `quantile` 0.9385, `mix_sample` 0.9378 | inside the band | not meaningfully different |
| `gate_only` 0.9238, `mix_codemath` 0.9231 | at the edge | marginal |
| `norm_only` 0.8940 | outside | genuinely different |
| `frequency` 0.7553 | far outside | the control, as designed |

So the criterion question closes cheaply: **stock REAP is not measurably improvable here**, and
the budget belongs to evaluation. The one criterion that clearly differs is the frequency-weighted
control REAP exists to avoid.
