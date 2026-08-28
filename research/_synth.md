# GLM-5.3-Flash REAP — Pass-2 Findings and Implementation Plan

**Audience:** the engineer running pass 2 tomorrow on the Thor box.
**Status of pass 1:** complete through `s06_emit`; FP8 artifact exists and is backed up. `s07_quantize` **did not finish** (see G0-4). No evaluation of any kind was run.
**Everything marked `[MEAS]` below I re-measured today** from `/home/patrickd/glm-5.3-reap/` — artifacts, logs, the emitted checkpoints, and the installed `transformers` source. Those numbers supersede all six input reports where they disagree.

---

## 0. Ranked table

Ranked by **expected quality gain per Thor-hour**, so you can build the plan top-down and stop when the budget runs out. Rows G0-x are gates: their "quality gain" is *preventing a wasted pass*, which is why they outrank everything.

Constraint codes: **C1** one Thor / 122 GiB unified / 936 GB NVMe / ~105 MB/s net · **C2** model cannot be loaded, stream one layer at a time · **C3** ~13 GiB KDA transient per 2048-tok sequence, linear in batch · **C4** unified memory invisible to the OOM killer · **C5** expert count is a single scalar · **C6** final artifact is FP8.

| # | Recommendation | Expected quality gain | Thor-h | Constraints | Confidence |
|---|---|---|---|---|---|
| **G0-1** | **Restore the 328.3 GB source weights — they are gone from disk.** Free space first by deleting the two output trees (one is backed up on HF, one is corrupt). | Pass 2 is impossible without this | **1.5–3** | C1 | `[MEAS]` |
| **G0-2** | **Sign-check the `s05` healing scalar before reusing it.** The 0.696 shrink is applied on top of a renormalization the router already performs. | Potentially large; wrong sign here corrupts every layer output | **~1** (free inside pass 2) | C6 | `[MEAS]` mechanism / `[OPEN]` sign |
| **G0-3** | **Do not touch the MTP block.** Already correctly excluded — this is *closed*, not open. | Saves 1–3 h of re-litigation | **0** | C5 | `[MEAS]` |
| **G0-4** | **Re-run `s07_quantize`; the current NVFP4 tree is incomplete (58/62 shards, no `config.json`).** | You have no loadable student today | **0.6** | C1, C6 | `[MEAS]` |
| **1** | **Per-domain / per-modality / second-moment accumulators + top-40 router-logit cache** — free riders on the pass you are running anyway | Converts 1 pass into ~8 offline criteria and an exact mixture sweep; removes ~5 h of would-be extra passes | **~0** marginal (+37 GB disk) | C1, C2 | `[EST]` mechanism |
| **2** | **Raise the calibration budget 0.52M → 5.5M tokens, chunked at 0.5M** | Removes the dominant known defect: 501/12096 expert slots are decided on <2000 tokens; 25.5 experts/layer sit within ±5% of the cut | **14–17** | C1, C2, C3, C4 | `[EST]` noise floor / `[OPEN]` that it moves benchmarks |
| **3** | **Offline criterion shootout** (S(b,α,β) family, AIMER, coverage round-robin) from the accumulators | Picks between masks that differ by ~30% of the keep-set, at zero pass cost | **~1** | C5 | `[EST]` computable / `[EXT]` which wins here |
| **4** | **Split-half keep-set-overlap stopping rule** | Tells you whether 5.5M was enough instead of guessing | **~0** | — | `[EST]` |
| **5** | **Router-aware staged greedy re-scoring** (needs the logit cache) | Corrects the top-8-renormalization drift that one-shot ranking ignores | **1–2** | C2 | `[EXT]`; compounding over 144 removals is `[OPEN]` |
| **6** | **AIMER weight-only criterion as a free cross-check** | Calibration-free second opinion; the only criterion natively compatible with C2 | **0.5–1** | C2, C6 | `[EST]` published / `[OPEN]` under FP8 |
| **7** | **Evaluation: paired teacher-vs-student teacher-forced ΔNLL + flip rates on the 8% held-out corpus** | The actual defect of pass 1. Nothing else is interpretable without it | **6–15** | C1, C2, C3, C4 | `[EST]` design / `[OPEN]` any generative benchmark |
| **8** | **Materialise 2–3 candidate masks and pick on measured ΔNLL** | Turns the shootout from a ranking into a decision | **1.3/mask** + eval | C1, C6 | `[EST]` cost `[MEAS]` |
| **9** | **Re-open 40% pruning as an option, with corrected sizes** | Worst-layer routing mass 0.403 vs 0.300; saliency mass 0.741 vs 0.643 | **1.3** to materialise | C1, C6 | `[MEAS]` sizes / `[EXT]` quality |
| **10** | **Train the mHC FFN hyper-connection instead of the block-scale scalar** | Replaces a mis-specified scalar with an exactly-expressible per-layer correction | **2–8**, gated on a backward pass working | C1, C2, C4, C6 | `[EST]` expressible / `[OPEN]` on this box |

---

## 1. The gates

### G0-1 — The source weights are gone. This is the blocker. `[MEAS]`

```
/home/patrickd/glm-5.3-reap/source/GLM-5.3-Flash/  → 5.1 GB total
   config.json, model.safetensors.index.json, tokenizer.json, chat_template.jinja … and no shards
/home/patrickd/glm-5.3-reap/offload/               → 17 GB (layers 29–31 expert tensors only)
df /: 936G total, 205G available
```

`logs/s01_source.log`: *"source staged and verified: 62 shards, 328.3 GB"*. Those 62 shards no longer exist. Every pass-2 stage — saliency, surgery, quantize — reads them.

**Re-download is cheap; the disk arithmetic is not.** Measured rate from the same log: 5 shards (~26 GB) per 4.5 min ≈ **96 MB/s ≈ 345 GB/h**, so 328.3 GB ≈ **1.0 h at line rate, 1.5–3 h realistically** (the log shows CAS-client and self-signed-cert failures; keep the existing 40-attempt retry loop).

Disk plan, in this order:

1. `rm -rf output/glm-5.3-flash-reap50-nvfp4` — 80 GiB, **incomplete and regenerable** (G0-4).
2. Confirm then `rm -rf output/glm-5.3-flash-reap50-fp8` — 157 GiB. It is verified pushed: `logs/backup_fp8.log` ends `UPLOAD COMPLETE after 1 attempt(s)` to `patrickbdevaney/GLM-5.3-Flash-REAP50-FP8`, 63/63 files. **Verify the remote file list before deleting.**
3. Free ≈ 459 GB. Download 328.3 GB → **≈131 GB free.**
4. **131 GB is not enough to emit a second 168 GB FP8 artifact while the source is resident.** Two options, pick one now:
   - **(a) unlink-as-you-go surgery**: `s04b` reads each source shard once; delete each consumed source shard immediately after writing its pruned counterpart. Pruned expert tensors are ~52% of source, so free space increases monotonically. Peak ≈ 328 GB + one output shard. The source is re-downloadable, so this is safe.
   - **(b) emit NVFP4 directly** from the pruned stream and re-derive FP8 later. Violates the spirit of C6 (FP8 is the master); prefer (a).

**Do this first thing tomorrow, detached** (`setsid`, per the standing rule — `s03` already shows five restarts from dropped sessions).

### G0-2 — The healing scalar is probably a double correction. `[MEAS]` + `[OPEN]`

I reproduced `s05` exactly from `artifacts/saliency/*.pt`:

| quantity | median | min | max |
|---|---|---|---|
| shipped gain (`s05_heal.json`) | 0.6964 | 0.5635 | 0.8557 |
| `E[g‖f‖]_all / E[g‖f‖]_kept`, recomputed | 0.6933 | 0.5635 | 0.8557 |

So the shipped gain **is** the ratio of per-expert conditional means under **pre-prune** routing, folded into the F32 block scales of 6,048 experts.

Now the mechanism, verified in the installed model code — `.venv/lib/python3.12/site-packages/transformers/models/glm5_next/modeling_glm5_next.py`, lines 161–183:

```python
scores            = router_logits.sigmoid()
scores_for_choice = scores + self.e_score_correction_bias   # bias used for SELECTION ONLY
...
topk_weights = scores.gather(1, topk_indices)
if self.norm_topk_prob:
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True) + 1e-20
topk_weights = topk_weights * self.routed_scaling_factor    # 2.5
```

Two consequences:

- **Good news, confirmed:** the `g_j` captured by `scripts/stream_saliency.py` (`g_j = top_k_weights[token_idx, top_k_pos]`) correctly **excludes** `e_score_correction_bias` and correctly uses the renormalized, scaled gate — the coefficient actually multiplying `f_j`. Pass 1's saliency is faithful to REAP as implemented. The architecture report's S2 concern is **closed**. `routed_scaling_factor=2.5` is a per-layer constant and cannot affect any intra-layer ranking; `n_group=topk_group=1` makes group routing a no-op.
- **Bad news:** because `norm_topk_prob=True`, after pruning the router **already restores each token's gate mass to 1.0 (×2.5) over the surviving top-8**. The layer output is not scaled down by the deleted routing mass. Multiplying survivor weights by a further 0.696 is an *additional*, unjustified 30% shrink of every MoE branch, feeding an mHC residual with `hc_mult=4` and 20 Sinkhorn iterations.

The correct first-moment target is a **per-token** quantity, not a ratio of per-expert conditional means:

```
gain_L = E_t[ Σ_{k ∈ top8(all 288)}  g_k(t)  · ‖f_k(t)‖ ]
         ────────────────────────────────────────────────
         E_t[ Σ_{k ∈ top8(survivors)} g̃_k(t) · ‖f̄_k‖   ]      g̃ = renormalized over survivors
```

With the top-40 logit cache (rec. 1) and the per-expert mean norms (rec. 1), this is **minutes of CPU arithmetic, zero forward passes**. Bounds you can reason about today: retained routing-slot mass is median 0.443/layer, retained saliency mass median 0.625 — but renormalization multiplies the surviving gates back up by ~1/0.443, so the true ratio sits somewhere between 0.64 and ≥1.0 and the sign is genuinely `[OPEN]`. **Compute it; do not re-ship 0.696 by default.**

Worst layer on both measures is **layer 3** (gain 0.5635, routing mass 0.4030) — the first sparse layer, immediately after the three dense layers.

### G0-3 — MTP is already handled. Close it. `[MEAS]`

Three of the six reports flag layer 45's un-pruned 288-expert MoE as a load-breaking blocker. It is already resolved and the resolution is documented:

- `artifacts/s00_smoke.json`: *"transformers `Glm5NextForConditionalGeneration` does not instantiate the MTP block at layer index 45 (7.45B params, its own 288-expert MoE) … Decision: exclude MTP from this run, archive its original tensors verbatim."*
- Emitted checkpoint `output/glm-5.3-flash-reap50-fp8/model.safetensors.index.json`: **0 tensors matching `.layers.45.`**; layer 3 has exactly **144** expert ids.
- Emitted `config.json`: `n_routed_experts: 144`, `num_nextn_predict_layers: 0`, `num_hidden_layers: 45` (= layers 0–44, consistent with `len(mlp_layer_types) == 45`).

The 311.67B routed-parameter figure **does** include the MTP block's 7.25B (42 × 288 × 3 × 2048 × 4096 = 304.41B, + 7.25B = 311.66B) — so the "43 blocks" arithmetic is right, but the extra block is dropped rather than pruned. Cost: MTP speculative decoding is unavailable in the artifact. That is the correct trade under C5, and your own spec-decode wins on the 119B/122B stacks came from Eagle/DFlash draft heads anyway, not native MTP.

**Do not spend a Thor-hour scoring layer 45.** If a downstream spec-decode project wants it, the original tensors are archived and re-derivable.

Also note the config key is `text_config.n_routed_experts` (DeepSeek-style), **not** `num_local_experts`. C5 is unchanged — it is still one scalar, and `mlp_layer_types` expresses only dense/sparse, never a count.

### G0-4 — The NVFP4 deliverable does not exist. `[MEAS]`

```
output/glm-5.3-flash-reap50-nvfp4/  → 58 *.safetensors of 62, index total_size: 0,
                                      no config.json, no tokenizer
logs/s07_quantize.log               → last line "shard 55/62 … eta 4 min", then nothing
logs/s07_quantize.traceback.log     → AssertionError in compressed_tensors/offload/convert/
                                      from_accelerate.py:155  (assert offload.device.type == "meta")
```

The streaming path was working (measured **~36 min for the full model, 0.6 Thor-h**) and simply did not finish; the traceback is from a *different*, accelerate-based attempt that should not be reused (it is the C2 failure mode again — accelerate reading unified memory).

What the partial output does confirm, and it is good news — the precision layout is correct:

| tensor | dtype in NVFP4 artifact |
|---|---|
| `…layers.10.mlp.experts.0.down_proj.weight_packed` | `U8` (NVFP4 packed) + `F8_E4M3 [4096,128]` block scale + `F32` global scale |
| `…layers.10.mlp.gate.weight` | `BF16 [144, 4096]` |
| `…layers.10.mlp.gate.e_score_correction_bias` | `F32 [144]` |
| `…layers.0.hc_ffn_fn` | `BF16 [24, 16384]` |
| `…layers.29.self_attn.q_proj.weight` (KDA) | `BF16 [8192, 4096]` |

So router, mHC, and KDA projections survived in high precision. **`[MEAS]` hazard for any *other* converter (GGUF, a third-party NVFP4 path):** `quantization_config.modules_to_not_convert` has 1509 entries written as `model.layers.0.hc_attn_base`, but the real keys are `model.language_model.layers.0.hc_attn_base`. **The per-layer ignore entries are not substrings of the real tensor names.** Only the generic entries (`hyper_connection`, `router`, `lm_head`, `dt_bias`, `visual`, `attn_mha`, `attn_mqa`) match. Anything doing substring matching will quantize every `hc_*` and every KDA `self_attn.*` tensor. One-line check before any downstream conversion:

```bash
python - <<'PY'
import json; c=json.load(open('config.json')); ig=c['quantization_config']['modules_to_not_convert']
keys=json.load(open('model.safetensors.index.json'))['weight_map']
bad=[k for k in keys if 'hc_' in k and not any(p in k for p in ig)]
print(len(bad), bad[:3])
PY
```

Related, from the same modeling file (line 1358): `_keep_in_fp32_modules_strict = ["e_score_correction_bias", "conv1d", "dt_bias", "A_log"]`. Preserve all four.

---

## 2. The recommendations, in implementable detail

### Rec. 1 — Accumulators and the logit cache (free riders on the pass you're already running)

**What pass 1 stored** (`artifacts/saliency/*.pt`, 42 files): exactly `['layer', 'sum_saliency', 'count', 'num_experts']`. Nothing else is recoverable. That single fact is what forces every extra pass the reports propose. Fix it once.

Patch site: `scripts/stream_saliency.py`, inside `patch_experts_for_saliency()`, where `f_j` and `g_j` already exist (lines ~67–75):

```python
f_j = F.linear(cur, self.down_proj[expert_idx])   # ungated expert output, already there
g_j = top_k_weights[token_idx, top_k_pos]         # renormalized ×2.5 gate, already there
n   = f_j.to(torch.float32).norm(dim=-1)
s   = g_j.to(torch.float32) * n
```

Accumulate, **per expert, per layer, and replicated per corpus bucket and per modality** (7 text buckets + `multimodal` = 8 streams):

| accumulator | shape | unlocks |
|---|---|---|
| `count` | `[288]` | already have |
| `sum_g` = Σ g | `[288]` | routing mass without the norm |
| `sum_n` = Σ ‖f‖ | `[288]` | MAN / MSAN / EAN family |
| `sum_n2` = Σ ‖f‖² | `[288]` | second moment of the norm |
| `sum_s` = Σ g‖f‖ | `[288]` | already have (`sum_saliency`) |
| `sum_s2` = Σ (g‖f‖)² | `[288]` | **the true within-expert SE** |
| 200-bin log-histogram of `s` | `[288, 200]` | tail/quantile criteria; 9.7 MB/layer |

Marginal compute: a handful of reductions against a `[tokens, 2048]×[2048,4096]` GEMM. Genuinely ~0.

**Why per-domain matters more than any other single line of code:** with per-bucket `sum_*` and `count`, any calibration mixture becomes an *exact offline reweighting*,
`S_j(w) = Σ_d w_d·sum_{d,j} / Σ_d w_d·count_{d,j}`.
A mixture sweep that would otherwise cost 4–5 saliency passes (~50 Thor-h) becomes arithmetic. `[EST]`

**Top-40 router-logit cache.** Store, per token per MoE layer, the top-40 `router_logits` (fp16) and their indices (int16) = 160 B/layer = **6.7 KB/token**, **37 GB at 5.5M tokens**. Budget it against the ~131 GB free after G0-1.

Why 40 and not 24: at a 50% cut, ~half a token's cached candidates are pruned in expectation. With top-24, `P(fewer than 8 survive) ≈ 3%` of tokens — and those are exactly the ambiguous tokens. With top-32 it is ~1.1e-3; with **top-40, ~1e-6**. `[EST]` (binomial arithmetic).

The cache is what makes G0-2, rec. 5, and the orphan-rate diagnostic possible. Without it, all three need another 15 Thor-h pass.

### Rec. 2 — Token budget: 0.52M → 5.5M, chunked at 0.5M

**The measured case for more tokens** — all `[MEAS]`, recomputed today from the 42 saliency files:

| statistic | value |
|---|---|
| relative saliency gap between rank 144 and rank 145 | **median 0.338%**, min 0.0089%, max 2.78% |
| experts within ±5% of the boundary value | **median 25.5 per layer**, max 41 |
| tokens routed to the *least-sampled kept* expert | median 986, **min 190** |
| expert slots below the 2000-token floor | **501 / 12096 (4.1%)** (`s03_saliency.json`) |
| CV of per-expert mean saliency, across experts | 0.61 |

The decision at the cut is being made on differences of 0.3% between experts whose scores are estimated from as few as 190 samples. Note the last row honestly: the CV that matters for the standard error is the **within-expert, per-token** CV of `g‖f‖`, and **pass 1 did not store the second moment, so it is not knowable today.** That is precisely rec. 1's `sum_s2`. Anyone quoting "CV ≈ 1.45 ⇒ SE ≈ 3.6%" is quoting a proxy.

**Budget rationale you can defend:** to bring the worst-sampled expert slot to the project's own 2000-token floor requires 2000/190 ≈ **10.5× the pass-1 token count = 5.5M tokens**. That is a stated internal criterion, not an extrapolated knee.

**Cost model, from measurement, not FLOPs** `[MEAS]`:

```
s03: 512 samples (431 text + 81 image-text), 135 batches, 21.3 GiB activations resident,
     83.2 min for ~0.52M tokens, 61–62 GiB free throughout.
     Per-layer: KDA ~2.1 min, DSA ~1.05 min. Weights read ONCE (328 GB @ ~3.4 GB/s ≈ 1.6 min).
```

So ~98% of the wall clock is compute/activation, and scaling in tokens is essentially linear. **5.5M tokens ≈ 11 chunks × (80 + 2) min ≈ 15 Thor-h** (the +2 is re-reading the weights once per chunk).

**You must chunk, and this is a C3/C4 hazard, not a preference.** `s03` currently materialises *all* batch activations up front — 21.3 GiB for 0.52M tokens = **41 KB/token** (the `hc_mult=4` expansion: 4 streams × 4096 × 2 B = 32 KB, plus residue). With 61 GiB of headroom, the hard ceiling is ~1.4M tokens per chunk, and under C4 you get a hard CUDA fault, not an OOM kill. **Chunk at 0.5M tokens (≈21 GiB), sweep all 45 layers per chunk, accumulate.** The docstring in `stream_saliency.py` already describes this loop order ("Chunk by samples, sweeping all layers per chunk"); the machinery exists.

**Corpus: use what you built. Do not re-mix.** `corpus/manifest.json` `[MEAS]`: 48.28M text tokens across `agentic 2949 / code 2580 / math 1843 / science 1229 / finance 680 / ballast 860` documents plus **1802 image-text items**, with `MIXTURE = {agentic .24, code .21, math .15, multimodal .15, science .10, finance .08, ballast .07}` and `HELDOUT_FRACTION = 0.08` stratified per bucket. Three things worth stating:

- The corpus **already** satisfies REAP's own ≥110B protocol requirement of *"tool calling and agentic trajectory data"* — `SWE-smith-trajectories`, `CoderForge-Preview`, `AgentTrove`, `SWE-rebench`, `agent-data`. `[EST]` that REAP specifies this; `[MEAS]` that you have it.
- General web text is only 7% (`ballast`: fineweb-edu, tulu-3, finemath). REAP §A8's finding — *"C4 calibration results in a collapse in accuracy, with several compressed model instances failing to produce coherent outputs"* — is the thing you already avoided. `[EST]`
- **The published mixture-ratio guidance is thinner than the input reports claimed.** The specific ratios attributed to "AIMER Table 9" (the 1:1 CodeAlpaca:C4 rule, the 48/41/11 split) come from a table that **does not exist in either version of that paper** — that was verified by the calibration reviewer. There is real correlational evidence (REAM, arXiv 2604.04356: general-domain share correlates r ≥ 0.95 with multiple-choice and r ≤ −0.82 with generative benchmarks at 50% expert reduction on GLM-4.5-Air among others), and it points the same direction you already went. Your mixture is fine. **Spend the hours on token count, not on churning the mixture** — and with rec. 1's per-domain accumulators you can test alternative mixtures for free afterward anyway.

One protocol detail worth copying from REAP and free to implement: GLM-5.3-Flash is a hybrid-reasoning model, and REAP's protocol explicitly closes `</think>` on SFT samples to disable reasoning during calibration. Check what `s02_corpus` currently emits. `[EST]`

**Sequence length: keep 2048.** The argument for 8192 is weak here and the cost is real. `index_topk=2048` means DSA runs *dense* at 2048 anyway, so you would not be exercising sparse attention without also implementing DSA top-k selection on sm_110a — where your own record has FlashInfer MLA invalid and Triton paths needing hand-holding. `[EXT]` payoff, `[MEAS]` risk. Deferred, not refuted.

### Rec. 3 — Offline criterion shootout (~1 Thor-h, no forward passes)

From rec. 1's accumulators, score every candidate mask offline. Parameterize as `S_j = N_j^(b-1) · Σ_t g_j^α ‖f_j‖^β` (the arXiv 2606.15716 family; REAP is `(1,1,1)`):

| mask | formula from accumulators |
|---|---|
| REAP (baseline) | `sum_s / count` |
| gated-EAN `(0,1,1)` | `sum_s` |
| MAN `(1,0,1)` | `sum_n / count` |
| MSAN `(1,0,2)` | `sum_n2 / count` |
| routing mass `(0,1,0)` | `sum_g` |
| `(0,2,2)` | needs Σ(g‖f‖)² → `sum_s2` |
| tail-weighted | `0.6·(sum_s/count) + 0.4·p99` from the histogram |
| coverage round-robin | per-domain scores, allocate the 144 slots round-robin across the 8 streams |
| AIMER | rec. 6, weight-only |

Report pairwise keep-set overlap. Pass-1 evidence that this matters `[MEAS]`: switching only the `b` exponent moves ~30% of the keep-set.

Honest scoping of what the literature supports:
- `(0,1,1)` and `(0,2,2)` beating REAP is measured on **OLMoE-1B-7B, DeepSeek-V2-Lite, ERNIE-21B, Qwen3-30B at 25%**, with **C4-only** calibration. The 50% results exist only for ERNIE and Qwen3. None is sigmoid + `noaux_tc`, none exceeds 128 experts, none is a KDA hybrid, none uses a 7-domain agentic corpus. `[EXT]`, weak transfer.
- The round-robin "coverage" variant buys perplexity and easy zero-shot accuracy but **regresses MMLU** (0.406 → 0.369 on Qwen1.5-MoE at 50% retain) — directly against goal #2. Compute it, do not adopt it by default. `[EST]` the numbers, `[EXT]` the transfer.
- Quantile/tail-pooled saliency at 288 experts: **`[OPEN]`. No source.** It costs a histogram, so gather it; do not ship on it without an eval delta.

### Rec. 4 — Split-half stopping rule (~0 h)

Score the two halves of the 5.5M-token set independently, then measure keep-set overlap at 50%:
**> 0.95 → converged, stop. 0.90–0.95 → accept with a noted risk. < 0.90 → the mask is calibration noise; add tokens or reduce the ratio.**

This is the only convergence test grounded in *your* model rather than someone else's. The literature is explicitly non-convergent here: the one published expert-level size sweep (AIMER Fig. 1, REAP on Qwen3-30B at 50%, C4 fixed, 0.5M → 2.1M tokens) reports *"Half of the benchmarks show significant variation. Some benchmarks improve while others degrade"* — it demonstrates instability **across its whole range** and never shows convergence. `[EST]` that it says this; **`[OPEN]` that any budget converges.**

### Rec. 5 — Router-aware staged greedy re-scoring (1–2 h)

One-shot ranking scores all 288 experts under the *original* routing. After removal, tokens re-route to survivors and gates renormalize, so the marginal value of the remaining experts changes. Staged greedy: remove in R rounds (R = 8 or 16 is plenty), recomputing scores from the cached top-40 logits after each round.

Cost is **I/O-bound, not compute-bound**: per round the work is `tokens × k` touches (per-expert token lists), not `288 × tokens × k`. The binding cost is re-reading the 37 GB logit cache per round: 16 rounds × 37 GB at ~3 GB/s ≈ 0.06 h. Even full R=144 is ~0.5 h.

Support: proxy-vs-exact bottom-ranked overlap is reported *"mostly close to or larger than 0.95"* (worst case ~0.85 at 25% on Qwen3). `[EST]` for the single-step proxy; **compounding over 144 sequential removals at 288 experts is `[OPEN]`** — which is the argument for staging at R=8–16 rather than full greedy.

### Rec. 6 — AIMER as a calibration-free cross-check (0.5–1 h)

`AIMER_j = ‖w‖₁ / (√N · ‖w‖₂)` over the concatenated `(gate_proj, up_proj, down_proj)` of expert *j* — the Hoyer concentration ratio. Weight-only, one tensor at a time, **zero forward passes**: the only criterion in the literature that is natively compatible with C2.

**Two implementation traps.** (i) It must be computed on **dequantized** blocks — the ratio is invariant to a global rescale but *not* to per-128×128 FP8 block scales, so on raw E4M3 you are measuring the scale layout, not the weights. Use `dequant_fp8_block()` which is already in `stream_saliency.py`. (ii) E4M3's 3-bit mantissa distorts an ℓ1/ℓ2 ratio by an unmeasured amount — `[OPEN]`.

Reported as best average rank on 4 of 5 model families (7B–47B, 16 benchmarks), at 0.22–2.06 s versus 0.75–2.96 h for calibration-based REAP. `[EST]` as published. It optimises *task-agnostic capability balance*, i.e. goals #2–#4, not goal #1 — use it as a keep-set to compare against, never as a replacement.

### Rec. 7 — Evaluation. The real defect of pass 1. (6–15 Thor-h)

**Gate first, before writing eval code:** there is no runtime on this box that serves GLM-5.3-Flash. Every local vLLM checkout (`dflash-dev/vllm-src`, `dllm-fork-coherent`, `nvfp4-quantize`) and the Thor container registry contain `glm4_moe*`, `glm4_1v`, `glm_ocr`, `chatglm` — **no `glm5*`**. Serving would need a vLLM port of KDA linear attention, the DSA indexer, and the mHC Sinkhorn residual, plus sm_110a NVFP4 MoE kernels. That is weeks, `[OPEN]`, and it is not on tomorrow's plan.

**Therefore: teacher-forced scoring only, via the same layer-streaming harness `s03` already uses.** Never generate from the teacher — at 328 GB and ~3.0 GB/s that is **~110 s per decoded token**. `[EST]` arithmetic.

The design:

- **Lockstep paired pass**: one streamed forward per layer for the teacher and the student, on identical items, emitting paired per-item metrics. The real justification is *paired* scoring (McNemar / flip rates) and amortising the 328 GB teacher read across N candidate masks — **not** I/O savings on a single branch (`107 s + 55 s` sequential = the same 162 s).
- **Batch size 4, not 6.** Both streams run unpruned KDA/MLA, so transient ≈ 13 GiB/seq **each**. At B=6: 12 (OS) + 22 (double-buffered layer weights) + 78 = 112 of 122 GiB, before final-layer logits (~155k vocab × seq × 2 B) and retained teacher hidden states. Under C4 that stalls without warning. **B=4.** `[EST]` arithmetic, `[MEAS]` transient.
- **Primary metric: per-domain held-out ΔNLL**, reported per domain and **max-over-domains**, never as a text-dominated mean. Use the existing stratified **8% held-out split** (`HELDOUT_FRACTION = 0.08` in `corpus_spec.py`) — it is disjoint from the calibration tokens by construction, which matters: reconstruction-style objectives fit on calibration data are known to overfit it, *"resulting in rather increased language perplexity and poor performance at downstream tasks"* (arXiv 2406.15524). Add one **out-of-mix** domain the corpus never saw. `[EST]`
- **Secondary: %flips and KL** against the teacher on the same items. Treat as a **damage tripwire, not a gate**: KLD is a documented *weak* predictor of code failures specifically (per-prompt geometric-mean ratios 1.08–1.22 on LiveCodeBench; cross-model routing accuracy 42.3–49.4%, at or below chance) — and coding is goal #1. `[EST]`
- **Integrity checks, ~free from the router traces:** P5 per-token retained gate mass; **orphan rate** (fraction of tokens whose pre-prune top-1 was pruned); **dead-expert census** (any survivor receiving ~0 tokens post-prune → index-remap or gate-slicing bug). No literature supports these as quality predictors — they are bug detectors, and bugs are what cost you hours last time. `[EXT]`
- **Knowledge probe (goal #2):** PopQA (`akariasai/PopQA`, 14,267 items — not 1,399), ~600 items, ΔNLL on the gold answer span regressed against `log(Wikipedia monthly pageviews)`, which the dataset ships. Short entity answers, alias lists, **no grader model needed**. This directly tests the measured fact that survivors carry ×0.886 of average routing mass — i.e. that REAP kept rare-but-strong experts. `[EST]` that PopQA supports it, `[EXT]` the stratification.
- **Vision probe (goal #3):** ΔNLL on a held-out slice of the existing 15% multimodal bucket, plus MMStar (1,500 human-curated, vision-indispensable). Expect vision to be *less* damaged than text — published evidence is that vision tokens tolerate more aggressive expert reduction than text tokens. **Watch OCR-heavy items specifically**: the one shared expert (which you are not pruning) carries modality-agnostic knowledge critical to text-rich visual tasks. `[EST]` direction.
- **Agentic (goal #1):** BFCL v3 **non-live AST + irrelevance** only — no sandbox, no execution, scorable teacher-forced. Weight irrelevance ≥25%: over-triggering is the agent-killer and AST-match saturates. Pin `bfcl-eval` and count the categories locally rather than trusting mirrored numbers. `[EST]` that REAP uses BFCLv3; **`[OPEN]` that any teacher-forced proxy tracks true multi-turn AST match** — that correlation cannot be measured until a runtime exists.

**What you cannot get, and must say out loud:** there is no on-box unpruned baseline for any generative benchmark, because the teacher cannot generate. Report **relative** regressions (student vs teacher, paired, teacher-forced) and, where you need an absolute anchor, vendor-published GLM-5.3-Flash numbers with the harness mismatch stated. Do not claim a benchmark-regression number you did not measure.

### Rec. 8 — Materialise and decide (1.3 Thor-h per mask, `[MEAS]`)

Measured stage costs from `logs/`: `s04b` surgery **17.5 min (0.29 h)** · `s05` heal **~20 min (0.33 h)** · `s07` quantize **~36 min (0.60 h)**. Total **1.22 h per candidate mask**, serially, deleting between arms (see the disk plan in G0-1).

Surgery itself is an index-select on the leading dim of the fused expert tensors plus a router row-slice, one tensor at a time — peak memory is one tensor, fully C2-compatible. The two invariants to assert after every emit: (1) every kept expert index appears exactly once in the new router row order; (2) `e_score_correction_bias` is sliced with the *same* permutation as `gate.weight` (pass 1 sliced 84 tensors = 42 layers × 2 — correct).

Realistic plan: **top 2 masks**, plus the pass-1 mask as control. 3 × 1.22 h ≈ 3.7 h of materialisation, then one paired eval pass covering all three.

### Rec. 9 — Re-open 40%, with corrected sizes `[MEAS]`

`artifacts/s04_sweep.json` predicted 90.6 GiB for the 50% NVFP4 artifact. **The actual emitted tree is 80 GiB** (and that is 58/62 shards, so ~86 GiB complete). The sweep's size model is ~5–13% conservative. Corrected:

| ratio | saliency mass kept | worst-layer routing mass | NVFP4 size (from the 50% measurement) | headroom in a 117 GiB envelope |
|---|---|---|---|---|
| 0.50 (shipped) | 0.6432 | 0.2996 | **~86 GiB** | ~31 GiB |
| 0.40 | 0.7405 | 0.4031 | **~101–105 GiB** `[EXT]` | ~12–16 GiB |
| 0.55 | 0.5930 | 0.2684 | ~78 GiB | ~39 GiB |

40% is arithmetically in-envelope and buys a materially better worst layer (0.403 vs 0.300 routing mass). It leaves **~12–16 GiB for KV cache, activations and the runtime** on a pool shared with the OS, which is tight but is the same order as your 119B/122B NVFP4 precedents. It is **not** true that "50% is forced by the envelope"; it is a choice with ~12 GiB of margin at stake. Decide it on rec. 7's measured ΔNLL, not on the sweep's estimate. `[MEAS]` sizes, `[EXT]` the quality delta.

### Rec. 10 — mHC FFN healing (2–8 h, optional, gated)

If G0-2 shows the first-moment correction is nonzero and layer-dependent, the *right* place to put it is not the FP8 block scales. Verified from the checkpoint:

```
model.language_model.layers.{L}.hc_ffn_fn     BF16  [24, 16384]   # 24 = 4 + 4 + 16
model.language_model.layers.{L}.hc_ffn_base   F32   [24]
model.language_model.layers.{L}.hc_ffn_scale  F32   [3]
model.language_model.layers.{L}.hc_attn_*                          # separate — attention untouched
```

Four properties, all checked: the FFN-side hyper-connection is **per-sublayer**, so it cannot perturb the attention path; the post-mixing gate is `2σ(·) ∈ (0,2)`, and your gains span 0.5635–0.8557, entirely representable; these tensors are **BF16/F32, outside the FP8 block-scale scheme**, so an update merges into the artifact *exactly* (C6); 42 layers × 393,243 params = **16.5M trainable**.

**Preconditions before budgeting this** (all currently unproven on sm_110a, `[OPEN]`): a backward pass through a streamed layer with KDA and a 20-iteration Sinkhorn; and **you must undo the existing 0.696 block-scale gain first**, or you train a correction on top of the correction it replaces. Chunk at ≤256k tokens (~10 GiB/buffer) given the measured 41 KB/token and 61 GiB headroom.

This is the last item on the list for a reason. Do it only if the budget survives everything above.

---

## 3. Established / vendor / extrapolated / unknown

**ESTABLISHED (`[EST]`), and directly applicable**
- REAP saliency `S_j = mean_{t→j} g_j‖f_j‖`, captured before the gate scales the output (arXiv 2510.13999). Your implementation matches it, including the renormalized ×2.5 gate.
- REAP at 50% on large SMoEs is near-lossless for **single-turn coding** and collapses for **multi-turn tool use** and **multiple-choice knowledge**. Kimi-K2-Instruct (384 experts) @50%: Eval+ 0.828→0.819, SWE-Bench 0.554→0.576 (*up*), but **BFCLv3 multi-turn 0.355→0.164** and **MC 0.780→0.643**. Qwen3-Coder-480B (160 experts) @50%: multi-turn 0.380→0.371, MC 0.750→0.692. **HumanEval will tell you everything is fine. It is the wrong instrument.**
- General-web-only calibration is catastrophic for REAP (*"collapse in accuracy … 0% accuracy"*, REAP §A8). Your 7% ballast share is already safe.
- Perplexity actively rewards bad masks: a randomly-pruned model scored **better** held-out code perplexity than base (4.82 vs 5.38) while losing **57.9 points** of pass@1.
- Kimi-Linear-REAP-35B (KDA linear attention, 256→180 experts, 30%) is the only published KDA-architecture data point: FRAMES 55.7→52.3, LongBench v2 36.8→37.2, HumanEval+ 82.3→81.1, MBPP+ 66.9→69.3, LCBv6 27.6→30.2, AIME25 30.0→40.0. Knowledge down, code flat-to-up. **One model, one ratio (30%, not 50%).**

**VENDOR CLAIMS (`[VEN]`) — verify before depending on**
- bitsandbytes "Jetson Thor Blackwell" support: the changelog says only that; it does **not** say sm_110a, aarch64-JetPack, or CUDA 13. The published aarch64 wheels are built on `sbsa` runners and are documented as incompatible with the L4T/JetPack runtime (precedent: issue #1930, sm_87 omitted, Orin fails at first kernel launch). A source build with `-gencode arch=compute_110a` is required. **4–8 h, may fail.** Only relevant if you attempt rec. 10.
- The vLLM recipe for GLM-5.3-Flash specifies FlashInfer ≥0.6.17 NoPE sparse MLA on Hopper/GB200 with TP=4. Your own stack notes record FlashInfer MLA as invalid on sm_110a.

**EXTRAPOLATION (`[EXT]`) — direction plausible, magnitude not transferable**
- Every criterion comparison in the S(b,α,β) literature: ≤128 experts, softmax gating, C4-only calibration, mostly 25% ratio.
- REAM's calibration-mixture correlations (general↑ ⇒ MC↑/GEN↓): measured at 25–50% on Qwen3/GLM-4.5-Air, 128–512 experts.
- Sizes for the 40% artifact, scaled from the 50% measurement.
- Any transfer of MAESTRO/MoE-Pruner recovery percentages: those are 20–30B dense-attention models with no MLA, no KDA, no mHC, no vision.
- Escalation factors from small-model REAP results (the 1.9%/6.9% coding figures are the **small-model subset**; the same paper reports 0.16%/1.2% mean decrease at 25%/50% on Qwen3-Coder-480B, and 321B is in the large-scale regime).

**GENUINELY UNKNOWN (`[OPEN]`) — open risks, not gaps to paper over**
1. **KDA hybrids at 50%.** The single published KDA data point is 30% on a 48B model. Error compounding through 34 recurrent linear-attention layers with half the expert pool removed has never been measured. **No source exists.**
2. **mHC under sparsity.** No published work on pruning a model with manifold-constrained hyper-connections. The two competing stories in the literature — Birkhoff-polytope doubly-stochastic mixing (which would protect a scalar gain from depth amplification) versus measured "stream collapse" toward a single dominant residual pathway — cannot both be load-bearing, and neither has been checked on this model. Free diagnostic, no action attached to the answer.
3. **The sign of the healing correction** (G0-2). Cheap to resolve, currently unresolved.
4. **Quantile/tail-pooled saliency at 288 experts.** No source. Gather the histogram; do not ship on it.
5. **Whether more calibration tokens improve downstream quality** (as opposed to reducing estimator variance). The only published sweep shows instability across its entire tested range and stops at 2.1M.
6. **Whether any teacher-forced proxy tracks true multi-turn agentic performance.** Unmeasurable until a runtime exists.
7. **Compounding of greedy re-scoring over 144 sequential removals.**
8. **Whether a backward pass is possible at all on sm_110a through KDA + 20-iteration Sinkhorn.**

---

## 4. DO NOT DO

Each entry states the precondition that would make it usable.

| Technique | Why not | Precondition |
|---|---|---|
| **Non-uniform per-layer expert allocation** | `n_routed_experts` is one scalar (C5). Already implemented, already measured better (worst layer 0.491→0.649), already discarded. | A config/runtime that accepts a per-layer count. `mlp_layer_types` expresses dense/sparse only. |
| **Pruning or re-scoring the layer-45 MTP block** | Already dropped; `transformers` does not instantiate it; config says `num_nextn_predict_layers: 0`. Three reports flagged this as blocking. It is closed. | A vLLM port with MTP spec-decode, *and* a same-pass saliency for layer 45. |
| **Router KD healing** | The loss is token-level KL on the student's final vocab logits; router-only *updates* still need a full forward **and backward** through a 165B student. Does not fit 122 GiB (C1/C2). Also: the parameter count often quoted (198M) is a bytes/params confusion — it is 49.5M params = 198 MB F32. | An NVFP4 student (~86 GiB) plus a gradient-checkpointed differentiable NVFP4 MoE on sm_110a. 40–100 h of engineering, high failure risk. |
| **Self-generated calibration data** | Requires batched decode from the *unpruned* model. The KDA recurrent state is O(1) in sequence length but **not** in batch: 34 layers × H·d_k·d_v is 73 GB at 16 heads/bf16 and 146–292 GB at fp32, before KV and weights. Batch 4096 does not exist (C1/C3/C4). Real cost 60–160 h for an effect with zero evidence in whole-expert MoE pruning. | A measured KDA state size proving batch ≥2048 fits, plus a streaming decode with per-layer state management. |
| **TBYP-style difficulty/correctness filtering** | Needs n=16 sampled responses per problem from the dense teacher — 4–16× the generation budget that is already impossible. | Same as above. |
| **Any teacher generation, for any purpose** | 328 GB ÷ 3.0 GB/s ≈ 110 s per decoded token. | Nothing on one Thor. |
| **Expert merging (REAM / EEP / merge-into-survivor)** | REAP's own paper: *"expert merging necessitates re-quantization for block quantization formats."* Every merged expert needs a full FP8 128×128 requant, not a block-scale tweak (C6). REAM is otherwise C5-compatible and evaluated at exactly 50% — this is a cost objection, not a validity one. | Budget 2 materialisation passes + 10–20 h of implementation, and a code-heavy calibration mix (merging trades GEN for MC). |
| **Intra-expert / channel pruning to split the sparsity budget** | Measured on your own artifacts: block-granular selectivity ×1.005 vs expert-granular ×1.286. Also `moe_intermediate_size` sets the **shared** expert width too, so any uniform intra-expert slice must cut the shared FFN identically. | An activation-aware intra-expert criterion showing >1.0 selectivity. Weight-norm-based is dead. |
| **Re-fitting `e_score_correction_bias` toward load balance** | Under sigmoid + `noaux_tc`, slicing `b` **preserves the original router's preference order restricted to survivors** — that is the faithful restriction. Re-fitting deliberately pushes tokens off their highest-affinity surviving expert. There is no throughput benefit on one unified-memory box, which was the bias's only purpose. Expected effect neutral-to-negative. | An eval harness showing a measured win. Not before rec. 7 exists. |
| **Attention LoRA on the MLA projections** | The 11 MLA layers' `q_a/q_b/kv_a_proj_with_mqa/o_proj` are **FP8 with `weight_scale_inv`**; a merged delta is partially discarded on requant. The 34 KDA layers' `q/k/v/o_proj` are **BF16** and merge exactly. | Restrict the target list to BF16-resident modules. Note KDA has no `W_K`/`W_V` in the MAESTRO sense and MLA has neither — the recipe's module list does not name real tensors here. |
| **16k-token calibration sequences** | At 2048, `index_topk=2048` means DSA already runs dense — you gain nothing without implementing sparse DSA top-k on sm_110a. And at a fixed token budget, 16k means 8× fewer documents (coverage collapse); at fixed document count it is 8× the compute. 20–40 h dev + 4–8× compute for an `[OPEN]` payoff. | A working chunkwise KDA + sparse-DSA long-context forward, plus a token budget raised 8×. |
| **Trusting `s04_sweep.json`'s NVFP4 size estimates** | Predicted 90.6 GiB at 50%; the actual tree is 80 GiB (58/62 shards). | Re-derive from the completed G0-4 artifact. |
| **Citing "AIMER Table 9" mixture ratios, the 1:1 CodeAlpaca:C4 rule, or a 48/41/11 split** | The table does not exist in either version of the paper. | — |
| **Using `device_map="auto"` / accelerate anywhere** | This is the C2 failure and it is what killed `s07` (`assert offload.device.type == "meta"`). | Explicit `max_memory` / manual streaming only. |
| **Running anything as a child of a Claude Code Bash call** | Standing rule; `s03` already restarted five times from dropped sessions. | `setsid` / systemd, always. |

---

## 5. Proposed pass-2 pipeline

Cumulative Thor-hours. **CP = critical path.**

| # | Stage | Detail | h | Cum. | CP? |
|---|---|---|---|---|---|
| 0 | **Verify the HF backup, then free disk** | list the 63 remote files; `rm -rf` both output trees | 0.2 | 0.2 | **CP** |
| 1 | **Re-download the source** | 62 shards, 328.3 GB, detached, retry loop | 1.5–3 | 3.2 | **CP** |
| 2 | **Patch `stream_saliency.py`** | 6 accumulators + log-histogram + top-40 fp16 logit cache, all replicated per bucket and per modality | 0.5 (dev) | 3.7 | **CP** |
| 3 | **Chunk `s03`** | 0.5M tokens/chunk (~21 GiB activations), sweep 45 layers per chunk, accumulate | 0.5 (dev) | 4.2 | **CP** |
| 4 | **Saliency pass, 5.5M tokens** | 11 chunks × ~82 min; detached; watch the 61 GiB headroom | 14–17 | 21 | **CP** |
| 5 | **G0-2: correct healing gain** | per-token pre/post ratio from the logit cache + mean norms; compare to 0.696 | 0.5 | 21.5 | **CP** |
| 6 | **Split-half overlap** | keep-set overlap at 50%; gate on >0.95 | 0.2 | 21.7 | **CP** |
| 7 | **Offline criterion shootout** | 8 masks + pairwise overlap; pick 2 | 1 | 22.7 | optional* |
| 8 | **AIMER cross-check** | streamed, block-scale-dequantized | 0.5–1 | 23.7 | optional |
| 9 | **Staged greedy re-scoring** | R=8–16 on the chosen criterion | 1–2 | 25.7 | optional |
| 10 | **Materialise mask A** | surgery 0.29 + heal 0.33 + quantize 0.60 | 1.3 | 27 | **CP** |
| 11 | **Materialise mask B** (+ control) | same, serially, deleting between arms | 1.3–2.6 | 29.6 | optional |
| 12 | **Build the paired eval harness** | lockstep teacher/student streamed forward, B=4, teacher-forced only | 4–8 (dev) | 37.6 | **CP** |
| 13 | **Eval pass** | held-out per-domain ΔNLL + flips/KL + PopQA + MMStar/multimodal slice + BFCL non-live AST | 6–15 | 52.6 | **CP** |
| 14 | **Decide** | ratio (50% vs 40%), criterion, healing gain; emit final FP8 + NVFP4; push to HF | 2 | 54.6 | **CP** |
| 15 | *(optional)* **mHC FFN healing** | only if 10's preconditions hold and budget remains | 2–8 | 62.6 | optional |

\* Stage 7 is marked optional only in the sense that you could ship the REAP mask; it costs 1 h and is the highest-leverage optional item on the list.

**Critical path total: ~38–55 Thor-h to a decided, evaluated artifact.** Optional items add ~5–14 h.

**Stop-points, if the budget runs out:**
- After stage 6 (~22 h): you have a 10× better-sampled saliency and a corrected healing gain. Re-emit with the same criterion and you have strictly improved pass 1, unevaluated.
- After stage 10 (~27 h): plus one candidate mask materialised.
- After stage 13 (~53 h): the first evaluated REAP artifact this project has produced. **This is the point of pass 2.** If you have to choose between more calibration tokens and any evaluation at all, choose the evaluation — pass 1's defect was never the criterion.

**Two things to fix before you start that cost nothing:** the disk plan in G0-1 (there is no room for the naive sequence), and the confirmation that stage 4 is detached. Everything else is recoverable.