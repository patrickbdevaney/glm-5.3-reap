# PLAN.md — GLM-5.3-Flash REAP + Heal

**Derived from** [`research/FINDINGS.md`](research/FINDINGS.md) · depth in [`wiki/`](wiki/README.md)
**Date:** 2026-08-27 · **Status:** awaiting operator approval before execution

Amendments to the directive already approved by the operator:
- **§6.7 RLVR — dropped.** Long-horizon coherence pursued via the 22% agentic calibration share + SFT.
- **§6.8 deliverable — healed FP8 base + adapters**, merged at quantisation time. No BF16 export stage.

---

## 1. Toolkit

**`vllm-project/llm-compressor` 0.13.0** (fresh checkout; `/home/patrickd/llmcompressor` is a
stale venv with no `reap` module).

`REAPPruningModifier` discovers MoE structure **generically** via `get_moe_attrs(model, ignore)`
rather than from a hardcoded architecture list — the best available signal that `glm5_next`
will work untouched. It validates `top_k` reachability (288→144 at top-8: trivially satisfied)
and warns against `moe_calibrate_all_experts` (set **False** — REAP needs the *real* routing
distribution). Chosen over `CerebrasResearch/reap` because it unifies prune + quantise and
already solves sequential onloading and disk offload.

**Fallback if `get_moe_attrs` does not recognise `glm5_next`:** the expert tensor layout is
regular and fully enumerated in `research/glm53_tensors.json`, so a direct saliency+surgery
implementation is ~300 lines against safetensors. This is the R4 contingency and is not a
research problem.

## 2. Parameter accounting (measured, all 62 shard headers)

| component | params | share | dtype |
|---|---:|---:|---|
| **routed experts** (288/layer × 43 MoE blocks) | **311,672,586,240** | **96.99%** | F8_E4M3 + F32 scales |
| attention — 11 MLA+DSA, 34 KDA | 6,199,640,639 | 1.93% | BF16 4.99B + F8 1.21B |
| shared expert (1/layer) | 1,082,196,480 | 0.34% | F8_E4M3 |
| embed_tokens + lm_head (untied) | 1,268,776,960 | 0.39% | BF16 |
| dense MLP (layers 0–2) + vision MLP | 755,223,552 | 0.24% | F8 0.45B + BF16 0.30B |
| **vision tower** (dense ViT, depth 24, **no MoE**) | 563,627,008 | 0.18% | BF16 |
| routers (`mlp.gate` + `e_score_correction_bias`) | 50,737,248 | 0.016% | BF16 |
| MTP `eh_proj` | 33,554,432 | 0.010% | BF16 |
| **mHC** (`hc_*`, `mapping_proj`; 135 tensors) | **17,695,935** | **0.006%** | BF16 + F32 |
| norms | 393,216 | ~0 | BF16 |
| **TOTAL** | **321,342,220,638** | | 328.3 GB / 305.8 GiB |

**KV cost:** only 11 of 45 layers grow a cache (MLA, `kv_lora_rank 512`, `qk_rope_head_dim 0`)
→ **~11 KB/token**; 128K ctx ≈ **1.4 GB**. The 34 KDA layers hold **~71 MB fixed**,
context-independent. MTP block sits at layer index **45** with its own full 288-expert MoE.

## 3. Memory table and target

NVFP4 experts @ 4.5 bpw effective; Thor usable envelope ≈ **117 GiB**.

| Prune | Total params | rest @ FP8 | rest @ BF16 | Verdict |
|---:|---:|---:|---:|---|
| 30% | 227.8 B | 123.3 GiB | 129.7 GiB | does not fit |
| 40% | 196.7 B | 107.0 GiB | 113.4 GiB | fits, ~4–10 GiB left |
| **50%** | **165.5 B** | **90.6 GiB** | **97.1 GiB** | **TARGET — 20–26 GiB headroom** |
| 55% | 149.9 B | 82.5 GiB | 88.9 GiB | research-only, escalation required |

**Target: 50%** — REAP's *validated* ceiling, not beyond it. Nothing above 50% is published;
the only high-ratio artifacts (Kimi-K3-REAP-73/80) report "noticeable degradation."

## 4. Storage and data-flow design

**Revision from FINDINGS §2.** Free space moved to **453 GiB** outside this session. That
changes the optimal design, because chunked calibration needs **many** passes over the weights
and local NVMe read (**3.4 GB/s**) is **32× faster** than the HF link (**105 MB/s**).

| Phase | Source of weights | Why |
|---|---|---|
| A — saliency | **local staged copy** (306 GiB) | 8–24 chunked passes; network would cost 8–24 × 52 min |
| B — surgery | local, **deleting shards as consumed** | one pass; ends with source gone |
| C — healing | **teacher streamed from HF** by layer-ordered range request | layer-local KD needs one teacher layer at a time; ~52 min total |
| D — quantise | local pruned checkpoint | one pass |

| Stage | Peak disk | Free after |
|---|---:|---:|
| A | 306 (src) + 30 (corpus) = **336 GiB** | ~117 GiB |
| B | src shrinking + output growing ≈ **336 GiB** | ~292 GiB |
| C | 160.6 (pruned) + 15 (teacher window) + 30 + 5 = **211 GiB** | ~242 GiB |
| D | 160.6 + 90.6 = **251 GiB** | ~202 GiB |
| final | pruned FP8 160.6 + NVFP4 90.6 (both kept for validation) | ~202 GiB |

Fits throughout with ≥117 GiB spare. **No further deletion required.**

> **R6 — the activation-cache trap.** llm-compressor's sequential pipeline holds calibration
> activations *between* layers. At 12,288 × 16,384 × 4096 × 2 B that is **~1.6 PB**. It must be
> chunked. REAP saliency is a **running conditional mean**, so chunks compose **exactly** —
> accumulate `Σ g·‖f‖` and count per expert, divide once at the end. Chunk to ~512 samples
> (≈34 GB activation buffer, comfortably inside 122 GiB RAM with UVM). This is a correctness-
> neutral change and the single most likely cause of a silent OOM days into a run.

## 5. Calibration corpus

**12,288 samples @ up to 16,384 tokens (~200M tokens)** — the REAP paper's own ≥110B setting,
and a floor rather than a ceiling: at 288 experts each expert sees ~2.8% of tokens vs 5.0% for
Qwen3-Coder-480B, i.e. **1.8× thinner per-expert statistics at equal count**.

Difficulty **60% medium / 30% hard / 10% easy** per bucket — hard-only degrades general
perplexity 6.2–12.1% vs 1.5–4.2% mixed, which would damage the ballast the mixture exists to
protect. (Pruning is ~2.3× more difficulty-sensitive than quantisation: spend curation here.)

| Share | Domain | Sources (HF availability verified 2026-08-27) |
|---:|---|---|
| 22% | Agentic coding trajectories | `thoughtworks/agentic-coding-trajectories`, `SWE-bench/SWE-smith-trajectories`, operator hermes-max / Claude Code logs |
| 18% | Code — multi-lang, systems, CUDA, IaC, repo-scale | `theblackcat102/evol-codealpaca-v1` + local prior-art repos for long-context |
| **15%** | **Multimodal** | `HuggingFaceM4/the_cauldron`, `allenai/pixmo-docs`, `ServiceNow/BigDocs-Bench`, `lmms-lab/multimodal-open-r1-8k-verified` |
| 14% | Math + algorithm synthesis | `nvidia/Nemotron-PrismMath`, `allenai/tulu-3-sft-personas-math` |
| 13% | Hard science & engineering | `nvidia/sft_datablend_v1`, arXiv-derived STEM |
| 10% | Finance / quant / econometrics | **`[OPEN]` — no verified source (R9), resolve before Phase A** |
| 8% | General ballast | `HuggingFaceFW/fineweb-edu` |
| — | Tool use (into agentic) | `Salesforce/xlam-function-calling-60k`, `arcee-ai/agent-data` |

`nvidia/Nemotron-CC-Math` is **gated (401)** — PrismMath substitutes.

**Hard build requirements**
1. Multimodal samples are **real image-text pairs through the real processor**, never text descriptions.
2. **Assert non-zero image-token count** (`154854`/`154855`) in the tokenised stream before any
   run. A collator silently dropping images degenerates calibration to text-only, which deletes
   vision experts **with certainty**. Highest-value assertion in the pipeline.
3. Held-out stratified eval split per domain **including a separate image-text slice**.
4. Pre-tokenise and shard once; every sweep and both healing passes reuse it.

## 6. Saliency configuration — the A/B

- **Arm A (control):** stock REAP, `S_j = mean_{x∈X_j} g_j(x)·‖f_j(x)‖₂`.
- **Arm B (novel):** `S_j = 0.6·mean + 0.4·p99` over the same per-expert activation set.

**No prior art exists for quantile-pooled REAP saliency** — this is a genuine modification,
motivated by the criterion's structural blindness to *rare high-magnitude* specialists, and by
the fact that this failure is **empirically reproduced** (Kimi-Linear-REAP-30: FRAMES −3.4,
its worst regression, while AIME25 +10.0 and LiveCodeBench +2.6).

**Cost is near-zero:** both arms are computed from the *same* forward pass by additionally
accumulating a per-expert streaming quantile sketch (t-digest) alongside the running mean.
**One saliency pass yields both rankings.** Decide on the §7 proxies, especially the
obscure-domain and image-text slices.

## 7. Layer allocation

**Uniform** (llm-compressor's only shipped mode; the REAP paper's setting; no ablation exists).

EvoESAP (arXiv 2603.06003) reports up to **+19.6% on MATH-500 at 50%** as a plug-in over REAP
via searched non-uniform budgets — real, but it costs an evolutionary search. **Deferred**, and
if revisited the search should be seeded by **measured per-layer reconstruction error** from
Phase A, which is a free damage signal we will already have.

## 8. Quality gating — thresholds, not just numbers

Computed **per domain**, with a **separate held-out image-text slice** — never averaged.

| Proxy | Green | Amber | Red |
|---|---|---|---|
| Output KL (pruned ‖ base), per domain | < 0.05 | 0.05–0.15 | > 0.15 |
| Per-layer relative reconstruction error | < 2% | 2–5% | > 5% |
| Routing-mass coverage on held-out | > 0.90 | 0.80–0.90 | < 0.80 |
| **Image-text KL vs text KL ratio** | < 1.3× | 1.3–2.0× | **> 2.0×** |

Thresholds are `[EXT]` — anchored to Kimi-Linear-REAP-30 behaviour, not to published
constants. **First action in Phase A is to calibrate them against the ratio-0 baseline**, so
the sweep is gated on measured spread rather than guessed absolutes.

**Go/no-go gate (directive §6.3), recalibrated.** Probe at 10–15%.
- **Trip on:** disproportionate long-context or reconstruction damage → the genuine
  linear-attention/mHC signal. Stop and report.
- **Do NOT trip on:** factual-recall / rare-domain drop. Expected, reproduced in the
  Kimi-Linear reference, and precisely what healing exists to repair.

## 9. Healing recipe

1. **Layer-local reconstruction distillation** *(primary)* — minimise
   `‖pruned_L(x) − original_L(x)‖`. Teacher streamed one layer at a time (~11 GB peak); the
   328 GB teacher never needs to fit. Precedent: MoE-Pruner (arXiv 2410.12013) — gap "largely
   mitigated … 1000 C4 samples, 1 hour."
2. **Full fine-tune of mHC (17.7 M) + routers (50.7 M)** — 68.4 M params total.
   *Not LoRA.* Rank-decomposing an 18 M module is strictly worse than training it. The routers
   matter as much as mHC and the directive under-weights them: post-prune each router emits
   logits over a **renumbered, halved** expert set with `norm_topk_prob` renormalising over
   different support — **the component whose semantics changed most**, and free to train fully.
3. **LoRA SFT** over the domain mixture, base frozen in FP8 (QLoRA-shaped). Targets: MLA/KDA
   `o_proj` (higher rank, they read expert-influenced residual), other attention projections
   (lower rank), surviving experts. **Vision tower frozen** — untouched by pruning.
4. ~~RLVR~~ — dropped (operator-approved).

## 10. Quantisation recipe

| Component | Policy |
|---|---|
| Routed + shared experts | **NVFP4 (W4A4)** |
| Attention (MLA + DSA) | **FP8** |
| KDA recurrent state (`A_log`, `dt_bias`, gates, `*_conv1d`) | **BF16, unquantised** — recurrence compounds error along the sequence |
| Vision tower | **BF16, unquantised** — 0.18% of mass, first-class capability |
| mHC, routers, `lm_head`, embeddings | **BF16** |

`ignore = ["re:.*lm_head", "re:.*visual.*", "re:.*linear_attn.*", "re:.*hc_.*", "re:.*mapping_proj.*", "re:.*mlp\\.gate\\..*"]`

Sequential onloading **on** (default); disk offload **on**; export **`compressed-tensors``**.

**Thor serving constraints baked in now** (both reproduced previously on this box):
`VLLM_FUSED_MOE_BACKEND=cutlass` — the **Marlin FP4 MoE kernel faults in-kernel at ≥256
experts**, and we have 288 (144 post-prune); and **`TRITON_MLA`** for the MLA layers,
FLASHINFER is invalid for MLA.

Block-scale search (Four Over Six 2512.02010 / SOAR 2605.12245 / RaZeR 2501.04052) is a
**post-baseline refinement, not critical path**.

## 11. Execution order and wall-clock

| # | Stage | Wall-clock `[EXT]` | Gate |
|---|---|---|---|
| 0 | **`glm5_next` × llm-compressor smoke test** (R4) — load config, run `get_moe_attrs`, verify collator emits image tokens | **2–6 h** | **Hard gate.** Cheapest possible early failure |
| 1 | Stage FP8 source (328.3 GB @ 105 MB/s) | **~55 min** | checksum vs HF |
| 2 | Build calibration corpus *(parallel with 0–1; long pole)* | **1–2 days** | image-token assertion |
| 3 | Sensitivity probe @ 10–15% + threshold calibration | **8–24 h** | **§8 go/no-go** |
| 4 | Full saliency pass, both arms, scores cached | **1–2 days** | scores persisted |
| 5 | Prune-ratio sweep 30/40/50 on cached scores | **4–12 h** | §8 proxies; pick knee |
| 6 | Surgery at chosen ratio | **2–4 h** | tensor-count + router-width audit |
| 7 | Layer-local distillation | **1–2 days** | per-layer recon error |
| 8 | mHC + router full FT, then LoRA SFT | **2–4 days** | held-out proxies |
| 9 | **Emit healed FP8 + adapters** ← **primary deliverable** | **2–4 h** | |
| 10 | NVFP4 quantise, sequential onload + disk offload | **8–24 h** | loads on Thor via cutlass |
| 11 | Document output format (§6.10) | **4 h** | |

**Total ≈ 8–14 days.** Ranges are wide because *no forward-pass throughput has been measured
for `glm5_next` on this box* — stage 0 produces the first real number and every estimate
downstream should be revised from it.

## 12. Checkpointing and resume

Multi-day single-box run; assume interruption.

- **All long-running work detached** — `setsid`/systemd, never a child of a Claude Code Bash
  call. (House rule from prior art: an SSH drop once cost 16 h.)
- **Metrics to SQLite** at `logs/metrics.db`, not scrollback: `(stage, layer, expert, arm,
  metric, value, ts)`. Every proxy, every per-layer reconstruction error, both saliency arms.
- **Per-layer resume markers.** Saliency accumulators (`Σ g·‖f‖`, counts, t-digests) flushed
  per layer per chunk — a kill loses at most one chunk-layer, and the running-mean structure
  means partial state is *valid*, just less converged.
- **Saliency scores are the expensive artifact** (~days). Persist to `artifacts/saliency/`
  immediately; every sweep ratio and both arms are then cheap re-rankings. Never recompute.
- Source checksums retained after deletion so a re-stage can be verified.
- Commit after every stage.

## 13. Risk register (ordered)

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| **R1** | Rare-knowledge erosion (FRAMES −3.4 at 30% in closest analogue) | **High** | Arm B saliency; 12,288 samples; broad mixture; domain-stratified gates |
| **R2** | Nothing published >50%; only high-ratio artifacts degrade | **High** | Hard-stop at 50%; escalate before 55% |
| **R3** | Vision deleted by invisible-expert (S_j=0) | **High**, certain if triggered | 15% real image-text; image-token assertion; modality-stratified routing mass |
| **R4** | `glm5_next` tooling immaturity | **Medium**, early | Stage-0 smoke test; direct-implementation fallback |
| **R6** | Activation-cache blowup (~1.6 PB naive) | **Medium**, certain if unhandled | Chunked accumulation; exact by construction |
| **R5** | mHC recalibration | Medium | Full FT of mHC + routers |
| **R7** | MTP layer 45 inconsistency | Low | Prune with same policy; audit expert counts |
| **R8** | Marlin FP4 MoE fault ≥256 experts | Low, known | cutlass backend + TRITON_MLA |
| **R9** | Finance/quant corpus source | Low | Resolve before Phase A |

## 14. Escalation triggers (directive §6)

Stop and report on: any destructive disk operation; exceeding 50% prune; any measurement
suggesting vision capability loss; anything requiring cloud spend; **and the §8 go/no-go gate
tripping on disproportionate long-context or reconstruction damage.**
