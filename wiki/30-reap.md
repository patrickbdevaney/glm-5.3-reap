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
