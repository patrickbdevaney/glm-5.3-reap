# 80 — Calibration corpus

## Size: the directive's estimate is too small

REAP's own published setting for models **≥110B** is **12,228 samples at up to 16,384 tokens,
untruncated** ([30-reap.md](30-reap.md)). The directive's "Cerebras used 3,072 samples" is
not the paper's figure, and its working estimate of 6,000–8,000 @ 4–8K is **below** the
published setting rather than above it.

GLM-5.3-Flash is 321B — nearly 3× the ≥110B threshold — with **288 experts per layer**, the
largest pool in the literature. Saliency is a per-expert conditional mean, so the statistic
that matters is **tokens observed per expert**, and that divides by the pool size:

| | Qwen3-Coder-480B (REAP paper) | GLM-5.3-Flash |
|---|---:|---:|
| experts/layer | 160 | **288** |
| top-k | 8 | 8 |
| expected share of tokens per expert | 5.0% | **2.8%** |

At equal sample count, each GLM-5.3-Flash expert is estimated from **~1.8× fewer tokens**.

> **Recommendation: 12,288 samples @ 16,384 tokens as the floor, not the ceiling** — matching
> the paper's ≥110B setting, with the understanding that our per-expert statistics are
> thinner than theirs at the same count. Budget ~200M calibration tokens. `[EXT, from the
> paper's setting + routing arithmetic]`

**Rare-specialist experts are exactly the ones estimated from the fewest tokens, and exactly
the ones the mean-based criterion already disadvantages** ([30-reap.md](30-reap.md)). Under-
sampling and criterion bias compound in the same direction. This is the quantitative core of
risk R1.

## Difficulty biasing — the directive's instinct is right, with one refinement

Established findings: `[EST]`

- Calibration data is stratified Easy / Medium / Hard; **medium-difficulty reasoning data
  with moderate length is the best single tier** for pruning large reasoning models —
  42.2% avg vs 39.6% (short) and 40.4% (long) (arXiv 2511.18864).
- **Pruning is up to 2.3× more sensitive to calibration difficulty than quantisation is**
  (arXiv 2510.10618). Difficulty curation matters far more for our prune stage than for our
  quantise stage — spend the curation effort there.
- **Mixed-difficulty is the most balanced**: within 1.5% of Hard-only on reasoning gains,
  while limiting perplexity degradation to 1.5–4.2% vs 6.2–12.1% for Hard-only.

> **Refinement to the directive.** The directive says "bias every bucket toward the harder end
> of its source material." The evidence supports biasing toward **medium-hard**, not maximally
> hard: Hard-only calibration measurably damages general perplexity (6.2–12.1%), which is the
> "connective tissue" the 8% ballast slice exists to protect. Recommended per-bucket
> distribution: **~60% medium, ~30% hard, ~10% easy**. Keep the directive's *direction*;
> stop short of its endpoint. `[EST]`

## Mixture (operator-approved shares) and verified sources

All availability checked against the HF API on 2026-08-27.

| Share | Domain | Source | Status |
|---:|---|---|---|
| 22% | Agentic coding trajectories | `thoughtworks/agentic-coding-trajectories` (15K sessions / 618K turns), `SWE-bench/SWE-smith-trajectories`, operator's own hermes-max + Claude Code logs | **OK** (lic. `other` — check terms) / MIT |
| 18% | Code, multi-language / systems / CUDA / IaC / repo-scale | `theblackcat102/evol-codealpaca-v1` (Apache-2.0), plus repo-scale long-context from local prior-art repos | **OK** |
| **15%** | **Multimodal** | `HuggingFaceM4/the_cauldron` (aggregate: ChartQA, DocVQA, AI2D, …), `allenai/pixmo-docs` (ODC-BY), `ServiceNow/BigDocs-Bench` (CC-BY-4.0), `lmms-lab/multimodal-open-r1-8k-verified` | **OK** |
| 14% | Math + algorithm/architecture synthesis | `nvidia/Nemotron-PrismMath` (CC-BY-4.0), `allenai/tulu-3-sft-personas-math` | **OK** |
| 13% | Hard science & engineering | `nvidia/sft_datablend_v1` (CC-BY-4.0), arXiv-derived STEM | **OK** |
| 10% | Finance / quant / econometrics | TBD — no verified source yet | `[OPEN]` |
| 8% | General coherence ballast | `HuggingFaceFW/fineweb-edu` (ODC-BY) | **OK** |
| — | Tool use (folded into agentic) | `Salesforce/xlam-function-calling-60k` (CC-BY-4.0), `arcee-ai/agent-data` (MIT) | **OK** |

- **`nvidia/Nemotron-CC-Math` returns HTTP 401 — gated.** Needs an access request or a
  substitute; `Nemotron-PrismMath` is available and covers the math bucket. `[EST]`
- **`lmms-lab/DocVQA` returns 307** (redirect) — reachable but resolve the canonical ID
  before pinning. `the_cauldron` already contains DocVQA, so this is likely moot.
- The **finance/quant 10% bucket has no verified source yet** and is the single largest gap
  in the corpus plan. Flagged for Phase 2.

## The ballast slice: keep it

The directive is right and the reason is worth recording. Interpretability work finds a
**domain-invariant "standing committee"** of experts carrying the majority of routing mass
across all domains, with only a thin peripheral layer showing genuine domain specialisation —
and the specialisation that exists is often **syntactic rather than topical**. Starving the
committee damages connective tissue that every target domain routes through. `[EST, as cited
in the directive]`

This is the same shared-core / thin-periphery structure the multimodal literature reports
([50-multimodal.md](50-multimodal.md)), which is mild independent corroboration.

## Hard requirements on the build

1. **Multimodal samples must be real image-text pairs through the real processor.** Text
   descriptions of images route like text and protect nothing.
2. **Assert a non-zero image-token count** (`image_token_id 154854`, `video_token_id 154855`)
   in the tokenised calibration stream before trusting any run. A collator that silently
   drops images degenerates calibration to text-only, which deletes vision experts with
   certainty ([50-multimodal.md](50-multimodal.md)). This is the highest-value single
   assertion in the pipeline.
3. **Hold out a stratified eval split per domain, including a separate image-text slice**,
   for the §3.6 proxies. Never average vision into a text-dominated mean.
4. Shard, pre-tokenise, and store the corpus once — every sweep and both healing passes reuse it.
5. `moe_calibrate_all_experts=False` — REAP needs the *real* routing distribution.
