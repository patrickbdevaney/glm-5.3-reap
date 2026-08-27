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

---

## Finance / quant / business / econometrics — bucket resolved (R9 closed)

10% of 12,288 samples ≈ **1,229 samples**. All availability and licences verified against the
HF API on 2026-08-27.

| Sub-share | Purpose | Source | Licence |
|---:|---|---|---|
| **25%** | **Long-context financial documents** | **`kensho/DocFinQA`** — 7,437 FinQA questions re-grounded in *full* documents, average context **123k words** (vs <700 in FinQA) | MIT |
| 20% | Numerical reasoning over filings | `ibm-research/finqa` — 8,281 expert-annotated QA over S&P 500 earnings, joint table+text | CC-BY-4.0 |
| 15% | Multi-turn / conversational finance | `TheFinAI/flare-convfinqa` — 3,892 conversations, 3.6 turns avg, requires coreference + context tracking | — (verify) |
| 15% | **Multimodal finance** | `sujet-ai/Sujet-Finance-QA-Vision-100k`, `TheFinAI/FinMR` — financial charts, tables, statements as images | Apache-2.0 / verify |
| 15% | **Quantitative / program synthesis** | `kensho/bizbench` — 8 quantitative reasoning tasks, financial QA **via program synthesis** | Apache-2.0 |
| 5% | Hybrid table+text instruction form | `next-tat/tat-llm-instructions` — curated FinQA + TAT-QA + TAT-DQA | CC-BY-4.0 |
| 5% | Explicit reasoning traces | `TheFinAI/Fino1_Reasoning_Path_FinQA` | CC-BY-4.0 |

### Why this composition, specifically

**`kensho/DocFinQA` is the highest-value single item in the entire corpus plan, and not for a
finance reason.** It is the only place where a target domain and *genuine long context*
coincide — 123k-word average context. Every other bucket tops out in the low thousands of
tokens. GLM-5.3-Flash's whole architectural bet is the 3:1 KDA/full-attention hybrid for
long-context efficiency, and **long-context behaviour is the one axis on which the
Kimi-Linear-REAP-30 reference is flat (LongBench v2 36.8 → 37.2)** — i.e. the axis where we
have a published expectation to hold ourselves to. Without DocFinQA the corpus would barely
exercise the layers that make this model what it is. `[EXT]`

**`kensho/bizbench` earns its place by being program synthesis.** It is finance expressed as
*code*, so it routes through both the finance periphery and the code experts. Given that the
standing-committee/thin-periphery structure means cross-domain samples protect more experts per
token than single-domain ones, dual-routing samples are disproportionately efficient calibration.

**`Sujet-Finance-QA-Vision-100k` + `FinMR` do double duty.** They protect finance experts *and*
vision experts in the same forward pass. Both are periphery, both are what pruning erodes first
(R1, R3), and financial charts/tables are a genuinely distinct visual domain from the
screenshots and diagrams in the main multimodal bucket.

**`flare-convfinqa` is the finance analogue of the agentic bucket** — multi-turn with
coreference and context tracking across turns, which is the long-horizon coherence capability
that lost its dedicated repair stage when RLVR was dropped ([70](70-healing.md)).

### Deliberately rejected

- **`Josephgflowers/Finance-Instruct-500k`** (500k, Apache-2.0) and
  **`sujet-ai/Sujet-Finance-Instruct-177k`** (177k, Apache-2.0) — large and permissive, but
  instruction-shallow and skewed easy. The difficulty evidence above is explicit that easy data
  biases saliency toward superficially-frequent shallow experts, which is the exact failure
  mode we are trying to avoid. **Size is not the objective; per-expert signal quality is.**
  Admit at most a token amount as ballast, if at all.
- **`eloukas/edgar-corpus`** (raw 10-K filings, Apache-2.0) — excellent *long-context* material
  but unstructured and unreasoned. Hold as a fallback only if DocFinQA proves too small to fill
  the 25% long-context sub-share.
- **`TheFinAI/MultiFinBen`** — **gated (401)**. Would have added multilingual finance.

### Residual gap

**Econometrics proper** (time-series inference, causal identification, panel methods) has no
strong dedicated instruction dataset. Cover it via **arXiv `econ.EM` and `q-fin` slices** folded
into the 13% hard-science bucket rather than forcing a weak dataset into the finance bucket.
Minor, and logged rather than papered over. `[OPEN]`

---

## FINAL mixture and licence policy (operator-approved 2026-08-27)

### Correction: the "code and math are robust under pruning" claim was over-read from noise

An earlier revision of this plan cut code and maths calibration on the grounds that the
Kimi-Linear-REAP-30 table showed them robust. **That reading does not survive scrutiny and is
withdrawn.** `[EST]`

- **AIME25 30.0 → 40.0** was quoted as "+10.0". AIME 2025 has **30 problems** — that is 9 → 12
  problems. A 3-problem swing on n=30 is **noise**.
- **LiveCodeBench +2.6, MBPP+ +2.4, HumanEval +0.6, MBPP −0.5** — small-to-moderate benchmarks,
  plausibly noise in either direction.
- **FRAMES −3.4** on **824 questions** ≈ 28 questions. **This one is probably real.**

The table therefore supports exactly one conclusion — **stored knowledge erodes** — and does
**not** establish that code or maths are robust. It is also measured at **30%**, while we target
**50%**; extrapolating robustness across that gap is the same unsupported move flagged as R2
everywhere else in this project.

### The criterion is frequency-invariant — share controls variance, not rank

```
S_j = (1/|X_j|) · Σ_{x ∈ X_j}  g_j(x) · ‖f_j(x)‖₂
```

The normaliser is `|X_j|`, **the expert's own active-token count**. An expert firing on 0.1% of
tokens but firing *strongly* scores the same as one firing on 10% equally strongly.

> **A domain's calibration share does not compete for saliency mass and does not systematically
> down-rank that domain's experts. It controls the *variance* of their estimates.** `[EST — from
> the criterion's definition]`

Consequences:
- Under-representation → noisy scores → experts near the prune threshold deleted **at random**.
- Zero representation → `S_j` undefined/zero → deletion **certain** (risk R3).
- **Past a sufficiency floor, extra share buys almost nothing.** World-knowledge domains need a
  floor, not proportional representation; everything above the floor should go to the target
  capability.

**The mixture is only a prior. The guarantee is the per-expert sample floor** in the L5
convergence criterion ([95-walltime.md](95-walltime.md)) — the stop condition cannot fire until
every expert has enough tokens for a low-variance estimate. If 10% science under-samples science
specialists, the floor catches it and calibration continues.

### Final mixture

Operator intent: **a strong coder/agent that remains empirically knowledgeable about the world** —
science and bio exist so it is not "a smart coder that is dumb at the real world", not to make it
a scientist. Code and maths are the product; maths is instrumental to code. Finance is a live
commercial use case (markets, crypto).

| Share | Bucket | Was | Δ |
|---:|---|---:|---:|
| **24%** | Agentic coding trajectories | 22% | +2 |
| **21%** | Code — multi-lang, systems, CUDA/kernels, IaC, repo-scale | 18% | +3 |
| **15%** | Math + algorithm/architecture synthesis | 14% | +1 |
| **15%** | Multimodal | 15% | 0 |
| **10%** | Hard science + bio/biotech + biomedical literature | 13% | −3 |
| **8%** | Finance / quant / markets | 10% | −2 |
| **7%** | General coherence ballast | 8% | −1 |

Code-adjacent (agentic + code + maths) **54% → 60%**. World knowledge 18% — above floor, not
competing. **Multimodal held at 15%**: R3 is the one risk where failure is *certain* rather than
probable, and the directive set that share deliberately.

### Licence policy: permissive-only, one corpus

**Reverses the earlier "NC for calibration, never SFT" split.** The two-corpus design in
[85-corpus-sources.md](85-corpus-sources.md) is **withdrawn**; there is one corpus.

**GLM-5.3-Flash is MIT** — the most permissive licence a frontier base can carry. A derivative
*can* be MIT; nothing upstream prevents it. That is a real, valuable, and **irreversible**
property: you cannot un-train a model.

And the NC data buys almost nothing:

| NC source | Permissive replacement | Verdict |
|---|---|---|
| `EricLu/SCP-116K` (274K, CC-BY-NC-SA) | `open-thoughts/OpenThoughts3-1.2M` (1.2M, **Apache-2.0**) | replacement **4× larger** |
| `camel-ai/{physics,chemistry,biology}` (CC-BY-NC) | `TIGER-Lab/MMLU-Pro` (MIT), `jablonkagroup/ChemBench` (MIT), `nvidia/sft_datablend_v1` (CC-BY-4.0) | covered |
| `osunlp/UGround-V1-Data` (CC-BY-NC-SA) | `xlangai/aguvis-stage2` (Apache-2.0), `TIGER-Lab/VisualWebInstruct` (Apache-2.0) | covered |

**Also dropped: `tattabio/OG` (CC-BY-SA-4.0).** ShareAlike is *worse* than NC for this purpose —
NC limits how a derivative may be used, SA can dictate what licence the derivative must carry.
Genomics coverage moves to biomedical literature instead.

**Terms to read during the corpus build** (tagged `other`, not resolvable from the API):
`bigcode/the-stack-v2-dedup`, `ncbi/pubmed` (NCBI terms), `thoughtworks/agentic-coding-trajectories`,
`GPUMODE/KernelBook`. `[OPEN]` — cheap to close before ingest, expensive after.
