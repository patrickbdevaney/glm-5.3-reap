# 96 — Implementation notes: what stage 0 found, and how the pipeline is built

Written 2026-08-27 after stage 0 passed. Everything here is `[EST]` — measured or reproduced
on this box, not inferred.

## Environment resolution (R4, closed)

| Component | Version | Note |
|---|---|---|
| torch | **2.13.0+cu130** | official cu130 aarch64 wheel; `arch_list` includes **sm_110**, reports `NVIDIA Thor` |
| torchvision | 0.28.0+cu130 | required by `Glm5NextVideoProcessor`; **pulled torch 2.10→2.13**, re-validated |
| transformers | **5.16.1** | `glm5_next` exists **only** at ≥5.16.1 |
| llmcompressor | **0.13.1.dev51** (git main) | pypi 0.13.0 pins `transformers<=5.14.1` |
| compressed-tensors | 0.18.0 | |

**The version conflict was real and would have blocked the project.** `glm5_next` landed in
transformers **5.16.1** (verified against the GitHub tree at tags v5.14.1/v5.15.0/v5.16.0/v5.16.1
— absent in all but the last). llmcompressor 0.13.0 on pypi requires `transformers<=5.14.1`.
Those are disjoint.

Resolution: install llm-compressor from **git main**, whose `setup.py` asks for
`transformers>=5.15.0` in a dev build (`==5.15.0` only for release builds). Upstream is
actively raising this bound — a commit dated 2026-06-22 reads *"Bump tested transformers upper
bound to 5.12.1 (unblocks newer model types…)"* — so the pin is caution, not a known
incompatibility.

## Three glm5_next × llm-compressor gaps

### 1. `AutoModelForCausalLM` rejects `Glm5NextConfig`
It is a conditional-generation (multimodal) model. Use
`Glm5NextForConditionalGeneration` or `AutoModelForImageTextToText`.

### 2. `glm5_next` is not registered, but auto-derivation mostly works
`glm5_next` is absent from `modeling/moe/conversion_mappings.py::ARCH_TO_IMPORT_PATHS`
(registered GLM variants: `glm4_moe`, `glm4_moe_lite`, `glm_moe_dsa`).

That turns out to be *mostly* fine: `LinearExperts2D.get_linear_experts_cls` can derive a class
from any experts module carrying the `@use_experts_implementation` decorator, which
`Glm5NextTextExperts` does. Derivation succeeds with `has_gate=True, is_transposed=False`.

Two consequences remain:
- **Load inefficiency.** Our checkpoint stores experts as **2D per-expert tensors**; transformers
  fuses them to 3D on load; llm-compressor then linearizes back to 2D. llm-compressor warns
  about exactly this: *"may be inefficient if the model checkpoint is already linearized
  (2D → 3D → 2D). Consider registering a load converter."* **Registering one is a real
  wall-clock lever for the load step** and is not yet done.
- **The bug below.**

### 3. `swiglu_limit` is lost in auto-derivation `[EST — reproduced]`
`Glm5NextTextExperts._apply_gate` reads `self.swiglu_limit`, but `LinearExperts2D.__init__`
stores that same value as `self.limit` (`MoEConfig.from_config` maps
`swiglu_limit → limit`). The derived class therefore has the value under the wrong name and
the **first forward pass raises `AttributeError`**.

`scripts/glm5_next_support.py::register()` subclasses the derived class and aliases
`self.swiglu_limit = self.limit`. Idempotent, must be called before `linearize_moe`.
Upstreamable either as that alias or as a proper `ARCH_TO_IMPORT_PATHS` entry.

## The MTP block is invisible to transformers `[EST]`

A meta-instantiated `Glm5NextForConditionalGeneration` has **313,890,426,878** parameters
against the checkpoint's **321,342,220,638**. The gap — **7,451,793,760** — is exactly the MTP
block at layer index 45 and its 288-expert MoE.

`linearize_moe` finds **42** MoE layers, not 43. **transformers does not instantiate the MTP
block at all**, so the llm-compressor path can neither see nor prune it.

> **Decision: exclude MTP from this run and archive its original tensors unmodified.**
> This *closes* R7 rather than tripping it — the risk was an *inconsistently* pruned MTP
> silently poisoning the downstream speculative-decoding project. An absent, clearly-documented
> MTP is recoverable; a half-pruned one is a trap.

## Measured throughput

One real-shape MoE layer (288 experts × 2048, **13.5 GiB** bf16) on Thor at MAXN:

| batch tokens | tok/s per layer | extrapolated full model (×42) |
|---:|---:|---:|
| 512 | 62,315 | ~1,484 |
| 2048 | **83,752** | **~1,994** |

MoE-only and random-weight, so **optimistic** — attention and sequential onloading will add.

### The number that reframes the schedule

REAP's saliency is a **conditional** mean over each expert's own active token set, so the
statistic that matters is **tokens per expert**, not corpus size. At 1,024 samples × 4,096
tokens = 4.2M tokens, each expert sees on average `4.2M × 8/288 ≈ 116k` tokens — far above any
plausible sufficiency floor, even for experts firing an order of magnitude below uniform.

**The 200M-token budget was sized against the REAP paper's sample *count* convention, not
against what the estimator needs.** The saliency pass is therefore plausibly **tens of minutes,
not 28 hours** — the L5 lever paying off far harder than projected. This is asserted, not
assumed: `s03` audits `min_tokens_per_expert` against a 2,000-token floor and warns loudly if
any expert falls below it. `[EXT, with a measured guard]`

## Pipeline architecture

`systemd --user` (`Linger=yes`, so it survives logout and SSH loss), `Restart=always`.
State in SQLite (`state/state.db`): `stages`, `metrics`, `events`, `kv`.

Stages resolve by dependency; a stage whose module does not exist yet parks as
`awaiting_impl` and is polled, which let source staging and corpus building start while later
stage code was still being written.

| Stage | Purpose |
|---|---|
| `s00_smoke` | structural validation, no real weights ✅ **passed** |
| `s01_source` | stage 328.3 GB FP8, resumable |
| `s01b_load` | probe load strategies against 117 GiB, record the winner |
| `s02_corpus` | build the calibration corpus (no deps → runs in parallel) |
| `s03_saliency` | REAP saliency + structural prune; dumps **raw accumulators** |
| `s04_sweep` | re-rank every ratio from cached scores (seconds, not hours) |
| `s05_heal` | first-moment correction — **non-critical** |
| `s06_emit` | healed FP8 + adapters ← **primary deliverable** (soft dep on heal) |
| `s07_quantize` | NVFP4 per-component policy |
| `s08_document` | tensor-level output format for the kernel work |

### Two decisions embedded in the code

**`moe_calibrate_all_experts` is `False` in s03 and `True` in s07.** Same flag, opposite correct
values: REAP needs the *real* routing distribution (a forced all-expert pass would corrupt the
conditional means), while quantisation genuinely does want every expert observed.

**s03 dumps raw saliency accumulators, which stock REAP does not.** `report_path` records only
the retained-expert list at the chosen sparsity — that cannot be re-ranked. Persisting
`sum_saliency` and `count` per layer is what makes the L3 sweep cost seconds, and it preserves
the numerator/denominator the quantile-blended arm would need.

## Healing scope was cut, deliberately

Full layer-local distillation is **not tractable** here: a ~165B student against a 117 GiB
envelope needs heavy offload for a backward pass, and the teacher is another 328 GB.

What is both principled and free is a **first-moment correction**, computable entirely from the
cached saliency with no teacher and no forward pass:

REAP keeps the **highest**-saliency experts. `norm_topk_prob=True` already renormalises gates
over the surviving support, so lost *gate* mass is compensated — but the retained experts have
systematically larger `‖f_j‖` than the deleted ones, so the pruned layer's expected output is
biased **high**, by exactly

```
    Σ_j c_j m_j / Σ_j c_j          (all experts)
    ─────────────────────────────
    Σ_{keep} c_j m_j / Σ_{keep} c_j   (retained)
```

Applying that ratio to each retained expert's `down_proj` restores the scale **in weight
space** — persistent, no runtime support, no config edit. It matters most for **mHC (R5)**,
whose Sinkhorn-normalised matrices conserve feature means and are therefore exactly sensitive
to a first-order scale shift in what feeds the residual streams.

`s05` refuses to apply a gain outside `[0.5, 1.0]`, since that would indicate a saliency bug
rather than a correction.

> **This is a first-moment correction, not distillation. It does not address rare-knowledge
> erosion (R1).** Labelled as such in the model card. `[EXT]`

## Saliency A/B arm B is deferred, not dropped

The stock tracker accumulates only `sum_saliency` and `count`, so **per-token quantiles were
never retained** and the `0.6·mean + 0.4·p99` arm cannot be computed post hoc. Running it needs
a modified `_expert_hook` keeping a sketch, plus a second calibration pass. Given arm B is
unvalidated and the main path was at risk, it was not worth destabilising. Recorded in the
`s04` artifact verdict so it stays visible.

---

## The residency problem, and why layer streaming is tractable

**Nothing can hold this model.** 314 GB of FP8 weights against 122 GiB of RAM; and once the
306 GiB source is staged, only ~170 GB of disk remains — not enough for accelerate to
disk-offload another 328 GB. All three of RAM, VRAM (unified, same 117 GiB) and spare disk are
individually insufficient.

Two ways out, probed in order by `s01b_load`.

### Path A — mmap-backed CPU placement (probed first)

`from_pretrained(..., device_map="cpu", dtype="auto")` on a **safetensors** checkpoint returns
tensors *memory-mapped from the file*. Nothing is written, so the pages are clean and the
kernel evicts them under pressure: a 328 GB model "on CPU" then costs address space rather
than resident memory, and llm-compressor's sequential pipeline pages in exactly the layer it is
onloading. `dtype="auto"` is load-bearing — anything that forces an FP8→BF16 conversion turns
the mapping into a 642 GB copy. `[EXT — plausible, and cheap to test, which is why it is
probed rather than assumed]`

### Path B — custom layer-streaming saliency (fallback, and confirmed feasible)

Reading `Glm5NextTextModel.forward` settles the question that decides this:

```python
hidden_states = inputs_embeds.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1).contiguous()
for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
    hidden_states, topk_indices = decoder_layer(hidden_states, attention_mask=..., ...)
```

**The entire inter-layer state is one tensor plus `topk_indices`.** mHC's four residual streams
live in the `hc_mult` axis of that single tensor and are manipulated *inside* each decoder
layer, not across the loop. There is no hidden cross-layer bookkeeping to replicate.

So a custom pass is mechanical: instantiate on `meta`, materialise layer *i*'s weights from the
mmap'd shards, run `decoder_layer`, accumulate saliency by hook, free, repeat. Peak residency is
one layer — 7.25 GB FP8, 14.5 GB dequantised — against 117 GiB. `[EST — read from source]`

**Chunk by samples, sweeping all layers per chunk**, rather than the reverse. The `hc_mult=4`
expansion makes activations 4× larger than intuition suggests: 1,024 samples × 4,096 tokens ×
4 × 4,096 × 2 B ≈ **137 GB**, which fits in neither RAM nor the remaining disk. Holding
activations per layer and re-reading them would cost ~6.3 TB of writes at 487 MB/s — hours.
Re-reading *weights* instead costs 333 GB per chunk at 3.4 GB/s ≈ 98 s, and much of it is
served from page cache. At 64-sample chunks that is ~26 min of I/O for the whole pass.

**Read is 7× faster than write on this box (3.4 GB/s vs 487 MB/s), so the design principle is:
re-read weights freely, never re-write activations.**
