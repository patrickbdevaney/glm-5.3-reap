# Porting glm5_next to llama.cpp — scope, from the code rather than from the paper

Written 2026-08-29. Everything below is read out of `modeling_glm5_next.py` and our own
checkpoint headers, not inferred from the architecture description.

## Adapter or fork? — **fork**

llama.cpp has no plugin path for a new architecture. Adding one touches, at minimum:
`src/llama-arch.{h,cpp}` (enum + tensor name mapping), `src/llama-model.cpp` (loader + graph
builder), `gguf-py/gguf/constants.py`, and `convert_hf_to_gguf.py`. There is no way to express
this as an out-of-tree adapter, so the deliverable is a fork (and, if it works, an upstream PR).

## What already exists — three of the four hard pieces

| piece | status in llama.cpp |
|---|---|
| KDA linear attention (34 of 45 layers) | **present** — `LLM_ARCH_KIMI_LINEAR`, `kda` |
| DSA sparse attention | **present** — `LLM_ARCH_GLM_DSA` |
| sigmoid-router MoE, `noaux_tc`, shared experts | **present** — `LLM_ARCH_GLM4_MOE` |
| **mHC hyper-connections** | **absent** — zero references in the tree |
| ViT / vision | separate `mmproj` path, standard |

So this is a port, not a from-scratch architecture. mHC is the new work.

## mHC: cheap in FLOPs, invasive in structure

**Cost.** 6 tensors per layer × 45 layers = **270 tensors, 35.39 M params, 67.5 MiB bf16 —
0.0214% of the model.** Per site the compute is a `[24 × 16384]` GEMV: ~393 K MACs, against ~18 B
for the MoE. mHC is free.

| tensor | shape | dtype | per layer |
|---|---|---|---|
| `hc_attn_fn`, `hc_ffn_fn` | `[24, 16384]` | BF16 | 2 |
| `hc_attn_base`, `hc_ffn_base` | `[24]` | F32 | 2 |
| `hc_attn_scale`, `hc_ffn_scale` | `[3]` | F32 | 2 |

`24 = (2 + H)·H` and `16384 = H·D` with `H = hc_mult = 4`, `D = 4096`.

**Why it is still invasive.** The residual stream is `[B, S, H, D]` for the whole stack, not
`[B, S, D]`. Embeddings are expanded to H identical streams at entry
(`inputs_embeds.unsqueeze(2).expand(-1,-1,H,-1)`) and collapsed by an *unweighted mean* at exit
(`Glm5NextTextHyperHead`). Every layer reads and writes four streams. That is a change to the
graph's spine, which is why it cannot be bolted on beside an existing GLM arch.

## The exact math, per HC site (two per layer: attn and ffn)

```
flat            = unweighted_rms_norm(streams.flatten(2))            # [B,S,H*D], in F32
pre_w,post_w,cw = linear(flat, fn).split([H, H, H*H])                # fn: [24, H*D]
pre             = sigmoid(pre_w * scale[0] + base[0:H]) + eps        # [B,S,H]
post            = 2 * sigmoid(post_w * scale[1] + base[H:2H])        # [B,S,H], range [0,2]
comb            = softmax(cw.view(H,H) * scale[2] + base[2H:], -1) + eps
comb            = sinkhorn(comb, iters=20)                           # doubly stochastic
x               = sum_h(pre[h] * streams[h])                         # collapse -> [B,S,D]
y               = sublayer(layernorm(x))                             # attn or MoE
streams'        = post[...,None] * y[...,None,:] + matmul(comb^T, streams)
```

The last line is an outer product `post ⊗ y` added to a mixing of the four streams by `comb^T`.
`hc_sinkhorn_iters = 20`, `hc_eps = 1e-6`, read from config.

Note the first Sinkhorn step is column-only (the loop runs `iters - 1` full row+column passes
after an initial column normalisation) — an off-by-one here silently changes the mixing matrix, so
it is worth porting literally rather than from the description.

## ggml op coverage — no new kernels needed, with one exception

`rms_norm`, `mul_mat`, `sigmoid`, `soft_max`, `mul`, `add`, `sum_rows`, `div`, `cont`/`permute`
all exist. The mHC forward is expressible in existing ops.

**The exception is Sinkhorn, and it is a graph-size problem rather than a math problem.** 20
iterations × 2 normalisations × 2 sites × 45 layers = **~3,600 extra graph nodes per forward**,
each doing trivial work on a `[4,4]` matrix. That is pure overhead and would plausibly dominate
decode. The port should fuse it: one custom op computing all 20 iterations for a whole layer's
worth of tokens in a single kernel. Small, self-contained, and the only genuinely new kernel in
the plan.

## Conversion: FP8 → GGUF must stream

`convert_hf_to_gguf.py` reads BF16/F16 safetensors. Ours are **FP8 E4M3 with 128×128
`weight_scale_inv` blocks**, and a BF16 intermediate of the pruned model is
**165.5 B × 2 = 331 GB** against 117 GiB free. Dequantising to disk is not an option.

We already have the pieces to avoid it: `dequant_fp8_block` and the streaming `ShardReader`. The
converter needs a glm5_next reader that dequantises per tensor and writes the target quant
directly, never materialising the BF16 model.

## Which quants to ship — narrow, not a ladder

165.5 B params, **94.2% in experts**. KV is nearly free here: only 11 of 45 layers grow with
context (the other 34 are KDA with a fixed 0.07 GiB state), and those 11 are MLA caching a 512-dim
latent — **1.38 GiB at 128k context in bf16, 0.69 GiB in fp8**. Weights are the entire budget.

| quant | bpw | size | headroom vs 122 GiB |
|---|---|---|---|
| Q4_K_M | 4.85 | 93.4 GiB | ~25 GiB |
| Q4_K_S | 4.55 | 87.6 GiB | ~31 GiB |
| IQ4_XS | 4.25 | 81.9 GiB | ~37 GiB |
| Q3_K_M | 3.91 | 75.3 GiB | ~43 GiB |
| *(ours, NVFP4)* | ~4.5 | *98.2 GiB* | *~19 GiB* |

Two observations. First, **Q4_K_M is smaller than our NVFP4** despite a higher nominal bpw,
because NVFP4 leaves the non-expert 9.64 B params in BF16 (16.5 GiB). That is headroom available
to us independently of GGUF.

Second, **stop at ~4 bpw.** We already removed half the experts, so survivors carry more unique
information per weight; an IQ2 of a REAP is 2.7 bpw *and* pruned, which is worse than either arm
alone and abandons the argument that beats an unpruned 1.78 bpw build.

## Ordering, and the dependency that is easy to miss

Inference first, then GGUF, then the CUDA server — but note **a GGUF cannot be validated without
a working reference**. Top-1 agreement against the unpruned teacher is how we would tell a good
conversion from a broken one, and the teacher is gone; what remains is our own FP8/NVFP4 artifact
and the cached teacher capture. So "land inference" is not merely first in sequence, it is the
oracle every later step is measured against.
