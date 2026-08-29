# A runtime shaped by the model, not by the hardware

Written 2026-08-29. The argument here is not aspirational — it comes out of numbers this project
already measured, and those numbers happen to be hardware-independent.

## The observation

Everything that moved decode performance on this model turned out to be a property of **the
model**, not of Thor:

| lever | size of the win | what it depends on |
|---|---|---|
| dense tail dominates AR decode | 76% of per-token traffic | 8 of 144 experts fire; attention always does |
| union-aware expert gather | 33% of expert traffic at block 8 | routing overlap statistics |
| KV is nearly free | 1.38 GiB at 128k | 34 of 45 layers are linear-attention |
| attention at 4 bpw | AR roofline +76% | which tensors are argmax-sensitive |
| top-k is not trimmable | rank 8 still carries 6.1% | the gate distribution |

Not one of those is a statement about SM110, wave size, or LPDDR5X. They are statements about
GLM-5.3-Flash. The *roofline* is hardware; the *ratios* are not. A runner that encodes them wins
on any device, and a generic runtime that ignores them loses on any device.

Meanwhile the things that genuinely are hardware-shaped — tile sizes, tensor-core instruction
selection, shared-memory budget, whether memory is unified — are exactly the things a compiler is
good at and a human is slow at.

## The inversion this suggests

The usual split is *one runtime, many models*: llama.cpp, vLLM, SGLang each carry a generic engine
and add per-architecture glue. The model-specific structure above has nowhere to live in that
design, so it is left on the table.

The alternative is *one model, many backends*: ship, alongside the weights, a **model profile**
that states the schedule the model wants, and lower it to whichever backend is present.

What belongs in a profile, from this project:

```yaml
residual:      {streams: 4, mix: mhc-sinkhorn, iters: 20, terminal: column}   # mHC
attention:     {kda: 34, mla_dsa: 11, kv_per_token_bytes: 11264}
moe:           {experts: 144, top_k: 8, gather: union, shared: 1}
precision:     {experts: nvfp4, attn_proj: nvfp4,
                protect: [kda_gates, conv1d, dsa_indexer, router, norms, mhc]}
speculation:   {kind: block-diffusion, block: 8, taps: [5,14,24,33,42]}
fusions:       [mhc_sinkhorn, expert_union_gather, gate_scatter_accumulate]
```

Every line is measured, not guessed, and every line is portable. `protect:` in particular is the
kind of knowledge that is expensive to rediscover and trivial to carry: the DSA indexer decides
*which tokens are attended*, so it is argmax-sensitive exactly like the MoE router, and quantising
it is a silent quality loss no benchmark suite would attribute correctly.

## Why this is more tractable than it sounds

The portable layer already exists, twice over:

* **ggml** runs CUDA, Metal, Vulkan, SYCL, HIP and CPU from one graph. Adding an architecture is
  graph construction plus, at most, a handful of custom ops — the glm5_next port under way needs
  exactly **one** new kernel (fused Sinkhorn), because everything else is expressible in existing
  ops. That is the whole hardware-portability problem, solved by someone else.
* **Triton** compiles one schedule to NVIDIA and AMD, and increasingly CPU. Where a fused op needs
  real tiling — the union-gathered MoE GEMV — Triton is where the schedule should be written, not
  in hand-rolled PTX that pins the work to one vendor and one generation.

So "hardware-agnostic, model-optimised" is not a new engine. It is: **encode the model-specific
schedule once, express the few genuinely fused kernels in a portable IR, and let the backend do
tiles.**

## The honest costs

**Portable kernels leave performance on the table.** A hand-tuned SM110 kernel will beat Triton,
plausibly by 20–40% on the inner GEMV. That is real and should not be waved away.

The argument is about which factor is larger. Model-specific structure is worth roughly **3×** on
this model (union gather 1.33× on the expert path, attention precision 1.76× on the dense path,
speculation ~2.8× on top). Hand-tuning is worth ~1.3×. **A portable runner that knows the model
beats a hand-tuned one that does not** — and the hand-tuned one only exists for the device it was
written for.

**Quantisation is where portability actually breaks**, not compute. NVFP4 needs FP4 hardware;
Metal and Vulkan have none. So a portable runner needs a per-backend precision policy — NVFP4 on
Blackwell, Q4_K/IQ4_XS through ggml elsewhere, MXFP4 where OCP formats are supported — while the
*protect list* stays identical across all of them. That is precisely the split this document
argues for: the policy is model knowledge, the format is hardware.

**Speculation is a second model in the graph.** A profile has to describe the drafter's taps and
verification rule, not just the target's. DFlash 2 reads `[5,14,24,33,42]` and drafts blocks of 8;
that is model data, and any runner that wants the 2.8× has to carry it.

## Where this project would go first

1. **Finish the glm5_next ggml port.** It is the cheapest possible test of the thesis: one
   architecture, one new kernel, and it immediately runs on CUDA, Metal, Vulkan and CPU. If the
   union-aware gather can be expressed in ggml, the claim is demonstrated rather than argued.
2. **Publish the profile beside the weights.** Even as documentation-with-numbers it saves the
   next person the eighteen hours of sweeps that produced the `protect:` list.
3. **Write the fused ops in Triton, once**, and call them from both the ggml backend and the CUDA
   server rather than writing them twice.

The CUDA server remains worth building — it is where the ceiling is, and where roofline percentage
is measurable without a portability tax confusing the picture. But it should be the *reference*
implementation of a schedule that also exists portably, not the only place the schedule lives.

## The community-serving version of the claim

A REAP'd, healed, quantised checkpoint is a better artifact than its bpw suggests, and the reason
is knowledge that currently lives in this repository rather than in the weights: which experts
were kept and why, which tensors must not be quantised, what the routing overlap is, where the
drafter taps. Shipping that as a profile — with GGUF, AWQ and MXFP4 builds pointing at it — is a
more useful contribution than any single hardware-tuned server, because it is the part that does
not have to be rediscovered per device.
