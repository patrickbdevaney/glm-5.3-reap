# 60 — FP8 → NVFP4 on Thor SM110a

## Ordering: the directive's assumption needs one correction

Working assumption was **prune → heal → quantize**. That ordering is right, but the *source*
is FP8, not BF16, so the real pipeline is:

```
FP8 source (328.3 GB)
  → REAP prune          [pure FP8 tensor deletion, lossless, no dequant]
  → heal                [LoRA/full-FT on frozen FP8 base; adapters kept separate]
  → merge + quantize    [dequant FP8 → BF16 → apply adapters → NVFP4]
```

**Pruning never leaves FP8.** Experts are per-expert tensors with their own block scales
([10-target-model.md](10-target-model.md)), so deleting an expert is deleting six tensors and
renumbering. No dequantisation, no reconstruction, no numerical change whatsoever to the
surviving experts. `[EST]`

This is worth stating plainly because it removes an entire class of risk the directive was
budgeting for: there is **no prune-time quantisation error**, and the pruned checkpoint is
bit-identical to the source on every retained weight.

## Is FP8 → NVFP4 a harmful double quantisation?

Less than it looks. `[EXT, reasoned from the formats]`

| | FP8 block | NVFP4 block |
|---|---|---|
| element format | E4M3 (3 mantissa bits) | E2M1 (1 mantissa bit) |
| representable magnitudes per octave | 8 | 2 |
| elements per scale | **16,384** (128×128) | **16** |
| scale format | F32 | FP8 E4M3 + F32 per-tensor global |

The two grids differ in opposite directions: FP8 has a **much finer value grid** but a
**much coarser scale granularity**; NVFP4 is the reverse. NVFP4's 16-element blocks track
local magnitude far better than 128×128 blocks do, which partly *compensates* for the
coarser element format. The dominant error term is E2M1's 1-bit mantissa, which would be
incurred identically when quantising from a true BF16 master. The extra loss attributable to
having gone through E4M3 first is second-order.

Empirical anchor: NVIDIA's DeepSeek-R1 numbers put **NVFP4 within 1% of FP8** on key language
tasks. `[VEN]` Vendor claim — not independently reproduced here, and not on a hybrid model.

**Known NVFP4 weakness to mitigate:** with a block scaled so its max is 6, any value between
4 and 6 must snap to 4 or 6 — a value of 5 takes a large relative error. Block-scale
*initialisation* is therefore a real quality lever, and there is active literature on it:
**Four Over Six** (adaptive block scaling, arXiv 2512.02010), **SOAR** (scale optimisation for
reconstruction, arXiv 2605.12245), **RaZeR** (redundant zero remapping, arXiv 2501.04052).
The directive's ScaleSweep reference (arXiv 2606.07618) belongs to the same family. Treat
improved scale search as an **optional refinement after a baseline NVFP4 run**, not as part
of the critical path. `[EST that the lever exists; [OPEN] which method wins here]`

## Per-component precision policy

Modelled on the GLM-5.2 precedent — **NVFP4 on MoE, FP8 on attention, >70% size reduction
with GPQA maintained** `[VEN]`.

| Component | Params | Policy | Why |
|---|---:|---|---|
| Routed experts | 311.67 B | **NVFP4 (W4A4)** | 96.99% of mass — the only thing worth compressing |
| Shared expert | 1.08 B | NVFP4 | same MoE path; on every token |
| Attention (MLA + DSA) | 6.20 B | **FP8** | 1.9% of mass; MLA latents are sensitive |
| KDA linear-attn state (`A_log`, `dt_bias`, gates, conv) | ⊂ attention | **keep BF16 — do not quantise** | recurrence: error compounds along the sequence rather than averaging |
| Vision tower | 0.56 B | **keep BF16 — do not quantise** | 0.18% of mass; first-class capability; upstream skips it for exactly this reason |
| mHC (`hc_*`, `mapping_proj`) | 0.018 B | **keep BF16** | 0.006% of mass; Sinkhorn-normalised, numerically delicate |
| Routers (`mlp.gate`) | 0.05 B | **keep BF16** | routing decisions are argmax-sensitive; free to protect |
| `embed_tokens` / `lm_head` | 1.27 B | BF16 (`lm_head` always ignored) | standard |

**Everything worth protecting is affordable to protect.** The entire non-expert model is 3%
of the parameters; keeping all of it at BF16 costs ~19 GB vs ~10 GB at FP8. That is the
whole reason the 50% target has room.

## Recomputed memory table (replaces the directive's working estimates)

Weights only. NVFP4 = 4.5 bpw effective (4-bit element + FP8 scale per 16 elements).
Thor usable envelope ≈ **117 GiB**.

| Prune | Experts kept | NVFP4 experts | **A: rest BF16** | **B: rest FP8** | Verdict |
|---:|---:|---:|---:|---:|---|
| 0% | 311.67 B | 175.3 GB | 191.9 GB (178.7 GiB) | 185.0 GB (172.3 GiB) | — |
| 30% | 218.17 B | 122.7 GB | 139.3 GB (**129.7 GiB**) | 132.4 GB (**123.3 GiB**) | **does not fit** |
| 40% | 187.00 B | 105.2 GB | 121.8 GB (**113.4 GiB**) | 114.9 GB (**107.0 GiB**) | fits, ~4–10 GiB left |
| **50%** | **155.84 B** | **87.7 GB** | **104.3 GB (97.1 GiB)** | **97.3 GB (90.6 GiB)** | **target — 20–26 GiB headroom** |
| 55% | 140.25 B | 78.9 GB | 95.5 GB (88.9 GiB) | 88.6 GB (82.5 GiB) | comfortable |

The directive's estimates (50% → ~168 B → ~92 GB) were optimistic by 5–12 GB. **The
conclusion is unchanged: 50% is the right target and it fits with real headroom.** 30% still
does not fit and 40% still leaves nothing. `[EST]`

### KV cache is essentially free — the hybrid pays off

Only 11 of 45 layers keep a growing cache, and they are MLA with `kv_lora_rank 512` and
`qk_rope_head_dim 0`, so ~1 KB/token/layer at BF16 → **~11 KB/token**. At 128K context that
is **~1.4 GB**. The 34 KDA layers hold *fixed-size* recurrent state
(64 heads × 128 × 128 × 2 B ≈ 2.1 MB/layer → **~71 MB constant, context-independent**). `[EXT,
computed from config]`

Long context on Thor is bounded by weights, not by cache. That is a large practical win and a
direct consequence of the hybrid design the directive worried about.

## Thor SM110a: what it actually supports

- **SM110a > SM100**, so vLLM does **not** fall back to weight-only. Both weight-only and
  full **W4A4** NVFP4 are available. The llm-compressor warning about `< SM100` does not
  apply to Thor. `[EST]`
- NVIDIA state Thor supports NVFP4 and that recent vLLM containers deliver up to 3.5×
  over launch-day performance. `[VEN]`

### Prior art on this exact box — two traps already paid for

From this operator's own earlier Thor work (see `memory/dflash-122b-marlin-moe-crash.md`,
`memory/mistral-small4-thor-stack.md`):

1. **The Marlin FP4 MoE kernel faults in-kernel at 256-expert scale on Thor. The cutlass MoE
   backend handles it cleanly and is the only NVFP4 MoE backend that loads large MoE models
   on Thor.** GLM-5.3-Flash has **288 experts**, and **144 after a 50% prune — still well
   past the scale where Marlin was observed to fault.** Plan for cutlass from the start.
   `[EST — reproduced on this hardware]`
2. **MLA on Thor requires `TRITON_MLA`; the FLASHINFER backend is invalid for MLA.**
   GLM-5.3-Flash's 11 full-attention layers are MLA. `[EST — reproduced on this hardware]`

These are downstream serving concerns, not blockers for producing the checkpoint, but they
constrain the **output format**, which is the point of directive §6.10.

## Output format

**`compressed-tensors`** is the right handoff. It is llm-compressor's native export, vLLM
reads it directly, and it records the per-component precision map explicitly in
`config.json` — which is exactly what the downstream custom-CUDA-kernel work needs in order
to know, per tensor, what layout to expect. `[EST]`

`[OPEN]` — whether the later hand-written kernels would prefer a flat pre-swizzled layout
over `compressed-tensors`' canonical one. That is a decision for the kernel project, and it
can transform from `compressed-tensors` at that time. Do not pre-optimise the export for a
consumer that does not exist yet.
