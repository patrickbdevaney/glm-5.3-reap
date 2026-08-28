# 90 — Open risks, directive contradictions, decision log

## A. Contradictions with the directive (directive §6 standing rule: trust the research, flag it)

| # | Directive says | Verified reality | Impact |
|---|---|---|---|
| **C1** | "BF16 checkpoint on disk: ~640 GB"; deliverable is "healed **BF16** weights" | Release is **FP8 E4M3**, 128×128 blocks, **328.3 GB**. The 642 GB `unsloth/GLM-5.3-Flash` BF16 repo is a **dequantised upcast** with zero extra information | **Large, favourable.** Halves storage and removes the BF16 intermediate. Operator confirmed FP8-native 2026-08-27 |
| **C2** | "Hybrid sparse + linear attention (**not MLA+DSA** — new base model)" | The 11 full-attention layers **are MLA + DSA** (`q_lora_rank 1536`, `kv_lora_rank 512`, DSA indexer + IndexPool), interleaved **3:1** with 34 **KDA** linear layers | **Favourable.** This is Kimi Linear's stack, which has a published REAP-30 checkpoint from Cerebras themselves |
| **C3** | Nemotron3 hybrid degraded ≥6.3% under REAP → hybrid fragility is the top risk | Traces to a **biomedical-QA** study (arXiv 2607.01444) whose variable is **expert-pool size**, not hybridness. NVIDIA's hybrid work finds **SSM-state pruning** is what breaks hybrids — REAP never touches it | **Risk downgraded.** See [40](40-hybrid-fragility.md) |
| **C4** | "Cerebras used 3,072 samples"; estimate 6,000–8,000 @ 4–8K | Paper's ≥110B setting is **12,228 @ 16,384, untruncated**. Directive estimate is **below** it | **Corpus must grow.** See [80](80-calibration.md) |
| **C5** | 50% → ~168 B → ~92 GB, ~18 GB headroom | 50% → **165.5 B** → **97.3 GB (rest FP8) / 104.3 GB (rest BF16)**; headroom 20–26 GiB | Directive optimistic by 5–12 GB; **conclusion unchanged, 50% is right** |
| **C6** | "50% on Qwen3-Coder-480B retains 97.6%" implies REAP is ~lossless at 50% | True per-model, but the paper's **suite-wide** mean coding loss at 50% is **6.9%**. Near-losslessness is a property of high-granularity models | We are the **highest-granularity** subject in the literature (288/top-8) — favourable, but for a reason, not by default |
| **C7** | "Bias every bucket toward the harder end" | Hard-only calibration degrades general perplexity **6.2–12.1%** vs 1.5–4.2% mixed | Bias to **medium-hard** (60/30/10), not maximal |
| **C8** | §6.7 RLVR healing stage | ~10 tok/s realistic → 1,000-prompt GRPO ≈ **460 days** single-stream, ~30 days optimistically batched | **Recommend dropping RLVR.** Needs operator acknowledgement. See [70](70-healing.md) |

## B. Storage — resolved without deleting anything

**The problem.** 298 GiB free vs a 305.8 GiB source. Naively staging source + pruned output +
calibration corpus peaks at **~496 GiB** — over by ~200 GiB. Holding the source at all is fatal.

**The resolution: never stage the source.** `[EST]`

Two measurements make this work:
- HF link saturates at **~105 MB/s** and parallelism does not help → one full pass over
  328.3 GB costs **~52 min**. Re-reading from HF is cheap.
- REAP is **layer-sequential**: saliency for layer *L* needs only layer *L*'s inputs and
  outputs. Nothing requires whole-model residency at any moment.

### The shard-ordering catch, and why range requests solve it

Shards are laid out in **lexicographic tensor-name order**, so layers appear as
`45, 0, 1, 10, 11, …, 19, 2, 20, …, 29, 3, 30, …, 39, 4, 40, …, 44, 5, 6, 7, 8, 9, V`.
Numeric layer order is **not** shard order — a naive "download shard N, process, delete" window
would thrash.

**Fix:** stream in *numeric layer* order using **HTTP range requests against the safetensors
byte offsets** (already extracted into `research/glm53_tensors.json`). Each layer's ~7.3 GB of
experts is fetched directly, wherever it lives. Total bytes are identical — only the order
changes. Peak local footprint is **one layer + one prefetched layer ≈ 15 GB**. `[EST — offsets
verified, transfer rate measured]`

Vision tower lives entirely in the final shard; the MTP block (layer 45) in the first two.

### Peak disk by stage (50% prune)

| Stage | Resident | Peak |
|---|---|---:|
| A — saliency pass | layer window 15 + calib 30 + activation cache 40 | **~85 GiB** |
| B — surgery pass | layer window 15 + pruned output 160.6 | **~176 GiB** |
| C — healing | pruned 160.6 + teacher window 15 + calib 30 + adapters 5 | **~211 GiB** |
| D — quantise | pruned 160.6 + NVFP4 output 90.6 | **~251 GiB** |
| final | NVFP4 output only | **~91 GiB** |

**Worst peak ≈ 251 GiB against 298 GiB free — fits with ~47 GiB margin, and no model,
container, or prior-art directory needs to be deleted.** `[EST]`

**Recommended margin (zero-risk, operator's call):** `docker image prune` (21 GiB of
unreferenced layers) and `thor-vllm-cache` (31 GiB, regenerable compile cache) → **+52 GiB**,
taking margin to ~99 GiB. Neither touches a model, a container image in use, or any prior-art
repo. **`models/DeepSeek-V4-Flash-0731-REAP` (101 GiB) must be preserved — it is the §7
incumbent eval baseline.**

**Cost of the approach:** two full streaming passes ≈ **104 min** of download, and the
saliency scores must be **cached to disk** (they are tiny) so the prune-ratio sweep and any
re-run never re-derive them.

## C. Open risks, ordered by severity

| # | Risk | Status | Mitigation |
|---|---|---|---|
| **R1** | **Rare-knowledge / obscure-domain erosion.** Kimi-Linear-REAP-30 loses **3.4 pts on FRAMES** — its worst regression — while code and math *improve*. The mean-based criterion plus thin per-expert sampling (288 experts → 2.8% of tokens each) both disadvantage rare specialists | `[EST]` — reproduced in the closest architectural analogue | Quantile-blended saliency A/B; ≥12,288 samples; broad-domain mixture; domain-stratified proxies |
| **R2** | **No published data above 50%.** The only high-ratio REAP artifacts (Kimi-K3-REAP-73/80) report "noticeable degradation" | `[OPEN]` — literature is silent, and the one data point is negative | **Do not exceed 50%.** Directive already requires escalation; treat 55% as research-only |
| **R3** | **Vision deletion by invisible-expert.** Text-only calibration gives vision-only experts `S_j = 0` — deleted with certainty at any ratio | `[EXT]`, follows from the saliency definition | 15% real image-text calibration; **assert non-zero image-token count**; modality-stratified routing-mass measurement in the probe |
| **R4** | **`glm5_next` tooling immaturity.** `transformers 5.16.0`; llm-compressor 0.13.0's `get_moe_attrs` introspects generically but has never been run on this architecture; multimodal collator may need a model-specific implementation | `[OPEN]` | Smoke-test `get_moe_attrs` + the collator on the real config **before** any long run. Cheapest possible early failure |
| **R5** | **mHC recalibration.** Sinkhorn-normalised connection matrices were fitted against the *original* expert-output distribution | `[OPEN]` empirically; `[EST]` mechanism | Full fine-tune of mHC (17.7 M params) + routers (50.7 M) during healing |
| **R6** | **Activation-cache blowup.** llm-compressor's sequential pipeline holds calibration activations between layers; 12,288 × 16,384 × 4096 × 2 B is **~1.6 PB** if held naively | `[EST]` — arithmetic | Chunk calibration (REAP saliency is a running conditional mean, so chunks compose exactly); ~512-sample chunks @ 8K ≈ 34 GB buffer; offload to host RAM via UVM |
| **R7** | **MTP layer 45 inconsistency.** Its own 288-expert MoE must be pruned consistently or later spec-decode work is poisoned | `[EXT]` | Prune layer 45 with the same policy; verify expert counts match post-surgery |
| **R8** | **Marlin FP4 MoE fault at ≥256 experts on Thor.** We have 288 (144 post-prune) | `[EST]` — reproduced on this box previously | Use the **cutlass** MoE backend; `TRITON_MLA` for the MLA layers |
| **R9** | Finance/quant calibration bucket (10%) has no verified source | `[OPEN]` | Resolve in Phase 2 |

## D. Decision log

| Date | Decision | Rationale |
|---|---|---|
| 2026-08-27 | **Work FP8-native; never download the 642 GB BF16 upcast** | Operator confirmed. Upcast carries no extra information; halves storage |
| 2026-08-27 | **Stream the source by layer-ordered range requests; never stage 328 GB** | Only way to fit in 298 GiB; download cost is ~52 min/pass |
| 2026-08-27 | **llm-compressor over `CerebrasResearch/reap`** | Unifies prune+quantise; already solves sequential onloading and disk offload |
| 2026-08-27 | **Target 50%, do not exceed without escalation** | Validated ceiling; 30% does not fit, 40% leaves no headroom |
| 2026-08-27 | **Recommend dropping RLVR from this run** | ~460 days single-stream; pending operator acknowledgement |
| 2026-08-27 | **Fully fine-tune mHC + routers rather than LoRA them** | 68.4 M params combined; they are the components whose semantics actually changed |

---

## E. R10 — disk cannot hold source + FP8 intermediate simultaneously (found during execution, 2026-08-27)

**The rev-3 storage plan is wrong**, and it is worth recording exactly how, because the error
was in an assumption that looked safe at the time.

FINDINGS §2 assumed ~453 GiB free (after the unexplained mid-session cleanup) and derived a
worst-stage peak of ~251 GiB. Two things changed:

1. Free space at the time of measurement already reflected work-in-progress, and settled at
   **~314 GiB**, not 453.
2. The pruned FP8 intermediate is **~161 GiB**, not the ~91 GiB an NVFP4 artifact would be.

Actual arithmetic:

| item | GiB |
|---|---:|
| free now | 306 |
| source still to download | −171 |
| **free once source is staged** | **~135** |
| pruned FP8 intermediate (50%) | 161 |
| **shortfall** | **≈ 26** |

The source cannot be deleted to make room, because the model is **mmap-backed by those very
shards** while it is being pruned — deleting them mid-run corrupts the model.

### Resolution: do not write the FP8 intermediate under disk pressure

The FP8 intermediate exists only as a staging convenience between prune and quantise. Under
pressure, `s03` prunes in memory, applies the first-moment correction, and **writes NVFP4
directly** — ~91 GiB, which fits inside 135 GiB with room to spare.

The FP8 artifact is still written when space allows, because it is the more useful thing to
keep (it is the healed base the deliverable is defined as). The decision is made from measured
free space at runtime, not assumed.

**Explicitly NOT done:** deleting any model checkpoint, container image, or prior-art directory
to make room. The directive requires escalation for destructive disk operations, and the
operator is asleep. `models/DeepSeek-V4-Flash-0731-REAP` (101 GiB, the §7 eval baseline),
`s5-capture` (80 GiB), the docker images (102 GiB) and `Qwen3.6-35B-A3B-NVFP4` (24 GiB) are all
untouched. The only reclaim performed was `thor-vllm-cache` (25 GiB), which the operator had
already approved and which is a regenerable compile cache. `[EST]`

---

## F. R11 — non-uniform per-layer allocation is unloadable on this architecture (2026-08-27)

The measured case for non-uniform allocation was strong. At an identical 6,048-expert budget:

| | uniform | non-uniform |
|---|---:|---:|
| worst layer, retained saliency mass | **0.491** (×0.98 vs random — no better than chance) | **0.649** |
| layers below 0.60 | 12 / 42 | **0 / 42** |
| experts per layer | 144 flat | 86 – 187 |

It was implemented, it ran, and it produced a 157 GiB checkpoint. **That checkpoint cannot be
loaded.**

`Glm5NextTextExperts.__init__` and `Glm5NextTextTopkRouter.__init__` both read a single scalar
`config.num_local_experts` and apply it to every layer:

```python
self.num_experts = config.num_local_experts
self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * intermediate, hidden))
...
self.weight = nn.Parameter(torch.zeros(self.num_experts, self.hidden_dim))
```

There is no per-layer expert count in `glm5_next`, and vLLM and the GGUF converters read the
same field. A checkpoint whose layers hold 86–187 experts can only be loaded by something
carrying a patched model definition. `[EST — read from the modeling source]`

> **The EvoESAP/DiEP line of work assumes an architecture that can express what it produces.**
> On a scalar-expert-count model the gain is real and undeliverable. This is not a reason to
> discard the code — it is a reason to keep it behind a flag and note the precondition.

**Decision: uniform allocation.** An unloadable checkpoint is worth nothing regardless of its
saliency numbers, and the directive's priority is a usable model. The non-uniform path stays in
`compute_retained(uniform=False)` for the day the config grows a per-layer field.

**Cost of finding this late:** the non-uniform output was discarded and the source re-staged.
It would have been caught before surgery by one question — *can the target config express a
per-layer expert count?* — which is now the first thing to ask of any allocation scheme.

## R14 — the MTP draft head was dropped, and that forecloses speculative decoding `[MEAS 2026-08-27 22:32]`

**The shipped REAP FP8 has no layer 45.** Source layers are 0–45 (n=46); the REAP output has
0–44 (n=45) and zero MTP tensors. `s04b_surgery` excludes it deliberately (`MTP_LAYER = 45`,
"transformers does not instantiate it") and writes `num_nextn_predict_layers: 0`.

**The research's claim that `num_nextn_predict_layers: 0` in the source, so the block is "closed",
is FALSE.** The source config says **1**, and layer 45 is a real, fully-populated block:

| | layer 45 (MTP) | layer 44 (normal) |
|---|---|---|
| tensors | 1,760 | 1,759 |
| experts | 288 routed + shared + router | same |
| attention | **MLA + DSA indexer** | KDA linear attention |
| mHC (`hc_*`) | **absent** | present |
| MTP-specific | `eh_proj`, `enorm`, `hnorm`, `shared_head.norm` | — |

Two consequences:

1. **Cost to preserve is trivial**: layer 45's experts are ~6.75 GiB FP8 at 288, ~3.4 GiB pruned
   to 144 — about +2% on the artifact.
2. **Cost of dropping it is not**: an MTP block is exactly the draft head a speculative-decoding
   inference server needs. vLLM implements MTP spec-decode even though `transformers` does not,
   so "transformers cannot instantiate it" is a statement about the *validation* harness, not
   about the weight's value. Dropping it makes a draft-head fine-tune impossible from the
   published artifact — the weights simply are not there.

**Pass-2 decision: preserve layer 45**, prune its experts to 144 (forced — `num_local_experts` is
one scalar), and ship `num_nextn_predict_layers: 1`.

Note that layer 45 has **no hyper-connections**, so it runs on plain `[B, S, H]` and is *easier*
to stream standalone than a normal layer. Getting real saliency for it means reconstructing the
MTP input, `eh_proj(cat(hnorm(h_final), enorm(embed(next_token))))`, after the layer-44 sweep.

If that forward proves fiddly, a weight-only criterion is an acceptable fallback **here
specifically**, because the intended downstream use is to *fine-tune* the draft head — which
repairs a mediocre prune. That argument does NOT extend to the main stack.

## R15 — the eval held 19 GiB of tap features and was killed at layer 14/45 `[MEAS 2026-08-28 18:20]`

memguard killed `s09_eval` with MemAvailable at 162 MB. Cause: the tap capture stored the **full**
`[B, S, H]` hidden state for each of 5 tap layers across all 122 batches and subsampled only at
the end — **~19 GiB held persistently**, on top of ~16 GiB of activations and a ~30 GiB KDA
transient for the longest image-text sample.

Only 2% of the taps is ever used. Subsampling at capture instead of at the end: **19 GiB -> 0.38
GiB**.

Two things worth keeping:

- **The guard chain worked.** memguard killed only this project's stage; the finisher saw `rc=137`
  and refused to continue to `s04b_surgery`, which would have deleted the teacher the eval still
  needs. A failure at the worst possible moment cost 30 minutes instead of the run.
- **`s03` never hit this** because it captures no taps and chunks its batches. Reusing s03's
  streaming shape for the eval carried over its layer loop but not its memory discipline — the
  new per-batch state was the part that needed the same treatment.

Measured while diagnosing: held-out image-text sequences are much longer than calibration's
(median 4,076 vs 308 tokens) because they come from the tail shards. Calibration survived a
5,052-token sample, so length alone was not the trigger — but it narrows the margin, and the KDA
transient scales linearly with sequence length.
