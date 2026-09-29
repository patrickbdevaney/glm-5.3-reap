---
license: mit
base_model:
  - zai-org/GLM-5.3-Flash
tags:
  - gguf
  - llama.cpp
  - reap
  - moe
  - pruned
---

# GLM-5.3-Flash REAP-50 — GGUF

4-bit GGUF quantisations of a **50%-expert-pruned** GLM-5.3-Flash (321B → ~165B), built with
[REAP](https://github.com/CerebrasResearch/reap) saliency pruning and a measured output-scale
correction.

## ⚠️ Requires a patched llama.cpp — stock builds cannot load these files

GLM-5.3-Flash uses **mHC hyper-connections**: four parallel residual streams per layer, mixed by
a Sinkhorn-normalised matrix. Upstream llama.cpp has no such operator, so `llama.cpp` as shipped
will refuse these files.

**Easiest — clone the branch directly.** It is a fork of llama.cpp with the patch already applied,
tagged at the exact tree these files were built and validated against:

```bash
git clone --branch glm5-next-reap50-gguf-v1 https://github.com/patrickbdevaney/llama.cpp
cd llama.cpp

cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release   # or -DGGML_CUDA=OFF for CPU
cmake --build build -j --target llama-completion llama-server llama-quantize
```

The tag is the fixed reference; `glm5_next` is the moving branch if you want ongoing work.

**Or apply the patch yourself** — `glm5-next-llama.cpp.patch` is in this repository and applies
cleanly to upstream `761797ff`:

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp && git checkout 761797ff
curl -LO https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-GGUF/resolve/main/glm5-next-llama.cpp.patch
git am glm5-next-llama.cpp.patch

cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target llama-completion llama-server llama-quantize
```

The patch is 15 commits. Everything needed to load and run these files is in the first three; the
rest is DSA (DeepSeek sparse attention) work that is **gated off by default** and does not change
behaviour, the EOG conversion fix described above, and the three commits that make the MTP block
runnable (see below) — none of which alter ordinary decoding.

What the patch adds:

* `ggml_mhc_sinkhorn` — one new operator, CPU and CUDA. Fused rather than composed from existing
  ops because the alternative is ~180 graph nodes per site for reductions over a 4×4 matrix,
  roughly **16k nodes and kernel launches per token** across 45 layers × 2 sites.
* `LLM_ARCH_GLM5_NEXT` — the hybrid graph: 34 KDA linear-attention layers interleaved 3:1 with
  11 NoPE MLA layers, sigmoid-routed MoE with a shared expert, mHC at both sites of every layer.
* `Glm5NextVisionModel` + `clip.vision.swiglu_limit` for the vision tower.

It is a normal llama.cpp otherwise, and everything else upstream keeps working.

### Running it

```bash
# text
./build/bin/llama-completion -m GLM-5.3-Flash-REAP50-Q4_K_M.gguf -ngl 0 --no-repack \
    -n 256 -p "Q: What is 17 multiplied by 23?
A:"

# server (OpenAI-compatible)
./build/bin/llama-server -m GLM-5.3-Flash-REAP50-Q4_K_M.gguf --port 8080
```

**On unified-memory boards (Jetson Thor, and similar):** use `-ngl 0 --no-repack`. Offloaded
layers become *non-evictable* device buffers out of the same physical RAM, so `-ngl 99` on a
93 GiB model against 122 GB total will OOM the machine — measured, not theoretical. With
`-ngl 0 --no-repack` the file stays memory-mapped and pages remain evictable.

## ⚠️ Add two stop tokens, or the model will talk to itself

**Pass these flags to every tool you run these files with:**

```bash
--override-kv tokenizer.ggml.eot_token_id=int:154827 \
--override-kv tokenizer.ggml.eom_token_id=int:154829
```

GLM-5.3 has **three** end-of-generation tokens. `generation_config.json` lists
`eos_token_id = [154820, 154827, 154829]` — `<|endoftext|>`, `<|user|>` and `<|observation|>` —
but a GGUF header carries only one `eos` id, so the other two are lost in conversion and
llama.cpp ends up with `<|endoftext|>` as its only EOG token.

The model ends an assistant turn with `<|user|>`, not `<|endoftext|>`. So with the default files
nothing stops generation: the model answers, emits `<|user|>`, then **hallucinates a follow-up
question and answers that too**, until it hits `-n`. This is easy to miss, because the first
answer is usually right — in testing, a chart-reading question was answered correctly in the first
turn and *incorrectly* in the fabricated second one, and any harness that reads the tail of the
output grades the wrong turn.

llama.cpp folds `eos`, `eot` and `eom` into its EOG set, and neither `<|user|>` nor
`<|observation|>` is in its name-matching list, so the two overrides above map the missing ids onto
`eot`/`eom`. Verified: with them, `load: printing all EOG tokens` lists all three and generation
stops at the end of the answer. It is also markedly faster, since nothing is generated past the
answer.

The conversion-side fix — emit the extra ids as `eot`/`eom` so the header carries them — **is in
`glm5-next-llama.cpp.patch` as shipped here**. It applies to files built from now on; the GGUFs
already uploaded have the incomplete header baked in, so for *these* files the two flags above are
the fix. They cost nothing and change no weights.

## Source durability

The source exists in three independent places, so no single one going away strands these files:

1. **GitHub** — [`patrickbdevaney/llama.cpp`](https://github.com/patrickbdevaney/llama.cpp) branch
   `glm5_next`, tag `glm5-next-reap50-gguf-v1`.
2. **The patch here** — `glm5-next-llama.cpp.patch`, 15 commits, applies to upstream `761797ff`.
3. **`glm5-next-llama.cpp.bundle`** — a complete git bundle: full history, all 15 commits, and the
   upstream base commit included, depending on no repository staying reachable at all:

```bash
git clone --branch glm5_next glm5-next-llama.cpp.bundle llama.cpp
```

The bundle matters because options 1 and 2 both assume some remote keeps serving history. It
verifies as `records a complete history`, so it needs no prerequisite objects from anywhere — a
copy of that one file is a copy of the entire project.

## Context length: read this before planning a long-context workload

The config advertises a **1,048,576**-token window. **You will not get that here, and the reason
is structural.**

GLM-5.3-Flash interleaves 34 KDA linear-attention layers with 11 **DeepSeek Sparse Attention**
layers that select the top 2,048 tokens per query. llama.cpp does not implement DSA — the indexer
tensors load and are never used, so those 11 layers run **dense**. Dense attention is a *superset*
of what DSA would select (the model sees its top-2,048 plus extra low-relevance tokens, which
softmax down-weights), so quality degrades gracefully rather than breaking — but the cost is
O(n²) where the model was designed for O(n·2048).

Memory is not the limit: MLA compresses the KV cache to 512 dims across only 11 of 45 layers,
roughly 11 KB/token, so 128k would be ~1.4 GB of cache. **Compute is the limit.**

Measured retrieval (needle-in-a-haystack, greedy):

| context | needle found | median latency |
|---|---|---|
| 2,000 | 3/3 | 54 s |
| 8,000 | 3/3 | 88 s |
| 32,000 | 3/3 | 236 s |

For genuine long context, use an implementation with DSA.

## What the pruning cost

Teacher-forced evaluation against the unpruned model, **241,516 held-out tokens** the calibration
never saw:

| | FP8 (unquantised parent) |
|---|---|
| top-1 agreement | **0.8425** |
| ΔNLL vs teacher | +0.1756 |

Per-domain top-1 agreement on that parent:

| domain | agreement |
|---|---|
| agentic | 0.873 |
| math | 0.920 |
| science | 0.830 |
| finance | 0.755 |
| ballast | 0.580 |
| code | 0.919 |

**Damage is not uniform, and that is by design.** Code and maths hold above 0.91; generic
"ballast" prose drops to 0.58. REAP ranks experts by saliency pooled over a calibration mixture,
and the mixture was chosen so that what survives is what retrieval *cannot* repair. If your
workload is code, maths, or agentic tool use, that profile is favourable. If it is open-domain
trivia, it is not.

These are the **parent's** numbers. The additional cost of 4-bit quantisation on top has not yet
been measured with the same harness.

## Quantisation lineage

Built `FP8 → Q8_0 → 4-bit`. The Q8_0 intermediate was measured to add **+0.78%** to the 4-bit
error versus quantising from F32 directly (`scripts/gguf_intermediate_cost.py`) — the source is
block-FP8 with one scale per 128×128 tile, and Q8_0 carries one scale per 32 weights, so the
intermediate has finer scale resolution than the thing it represents. A BF16 intermediate would
have needed 308 GiB and bought back that 0.78%.

No importance matrix was used for these files.

Measured perplexity on the same held-out text the calibration never saw, **8 chunks of 512
tokens** — a corruption tripwire, not a quality benchmark:

| file | perplexity |
|---|---|
| GLM-5.3-Flash-REAP50-IQ3_M | 4.6253 |
| GLM-5.3-Flash-REAP50-IQ4_XS | 4.6002 |
| GLM-5.3-Flash-REAP50-Q3_K_M | 4.6502 |
| GLM-5.3-Flash-REAP50-Q4_K_M | 4.5959 |
| GLM-5.3-Flash-REAP50-Q4_K_S | 4.5833 |

Differences of this size between levels are **within the noise of an 8-chunk sample** and should
not be read as one quant beating another. If Q4_K_S measures below Q4_K_M here, that is sampling
variance, not evidence that the smaller file is better — the gate exists to catch a broken quant
(nan, or a number an order of magnitude off), and it is sized for that job only.

## The MTP block runs now — but it will not make you faster yet

Layer 45 is a full multi-token-prediction block (3.81B params, ~2 GiB at Q4_K_M), and every quant
here carries it. Stock llama.cpp loads those tensors and never executes them. The patched build
does: `make_mtp_draft.py` (in this repository) lifts `blk.45` out of any of these files into a standalone
~2.3 GiB draft model, and `-md` picks it up automatically — the draft announces its architecture,
so there is no extra flag.

```bash
python make_mtp_draft.py GLM-5.3-Flash-REAP50-IQ3_M.gguf \
    GLM-5.3-Flash-REAP50-IQ3_M-MTP-draft.gguf   # ~35 s, no requantisation
./build/bin/llama-speculative-simple -m GLM-5.3-Flash-REAP50-IQ3_M.gguf \
    -md GLM-5.3-Flash-REAP50-IQ3_M-MTP-draft.gguf -ngl 24 --draft-max 3
```

Make the draft from **the same file you are serving**. Hidden states differ measurably between
quants (IQ3_M vs IQ4_XS cosine 0.891, against 0.999 for a same-quant control), and the draft
consumes the target's hidden state.

**Two things had to be fixed for this to work at all, and one is not fixed.**

Speculation rejects a drafted token by removing it from the KV cache, and 34 of these 45 layers
are KDA linear attention, whose rolling state has no per-token history to rewind to —
`llama_memory_recurrent::seq_rm` refuses the removal. The patch checkpoints the recurrent state
(145.56 MiB, fixed by the model, not by context length) and replays the accepted tokens. Worth
knowing if you are on an older build: that failure was *silent*. `llama_decode` returned -1,
nothing checked it, and generation continued from stale logits while reporting an acceptance rate
for tokens it had never verified.

What is not fixed is the economics. Acceptance is real — **46.5% at depth 2, 58.5% at depth 3**,
with the block exactly as GLM shipped it — but end to end it is *slower* than not speculating:

| | baseline | depth 2 | depth 3 |
|---|---|---|---|
| IQ3_M, `-ngl 24`, Thor | 2.99 t/s | 1.73 t/s | 1.85 t/s |

The draft is not the problem (5.3 ms/token) and neither is the checkpoint (~11 ms). The target's
cost per token barely falls with batch size until 32 tokens — 334 ms at batch 1, still 233 ms at
batch 16, then 43 ms at batch 32 — because `op_offload_min_batch_size` defaults to 32, and below it
the layers `-ngl 24` leaves in system memory are computed on the CPU. A draft of depth K produces a
batch of K+1, which is always on the wrong side of that. So verifying costs nearly as much as
decoding, and the ceiling at depth 3 is 1.29x before any overhead. (`GGML_OP_OFFLOAD_MIN_BATCH`
lowers the threshold and segfaults below 32 — do not reach for it.)

**So: use this if you want to measure or improve the head, not to go faster today.** Speculative
decoding via `-md` with an ordinary separate draft model works normally and is subject to exactly
the same batch-size ceiling.

## Vision

`mmproj-GLM-5.3-Flash-REAP50-F16.gguf` (1.1 GB) carries the 24-block vision tower. Pass it
alongside the text model:

```bash
./build/bin/llama-mtmd-cli -m GLM-5.3-Flash-REAP50-Q4_K_M.gguf \
    --mmproj mmproj-GLM-5.3-Flash-REAP50-F16.gguf --image photo.jpg -p "Describe this image."
```

The tower is the GLM-4V family, so llama.cpp's existing `PROJECTOR_TYPE_GLM4V` graph applies. One
delta the patch adds: GLM-5.3 clamps SwiGLU at `swiglu_limit` in the **vision** tower as well as
the text stack, carried as `clip.vision.swiglu_limit`.

**Verified working.** On a synthetic image with checkable content (a black "42", a red square
top-left, a blue circle bottom-right) the pruned 4-bit model answers:

> 1. There is a red square in the top-left area.
> 2. There is a blue circle in the bottom-right area.
> 3. There is a black number "42" in the center…

Shapes, colours and positions all correct. REAP pruned only the language-model FFN experts — the
ViT was untouched — but the language model is what consumes the visual tokens, so this is the
evidence that the pruned model can still see.

### Measured

**ChartQA relaxed accuracy: 19/24 = 79%** (95% CI 60%–91%), measured on held-out rows of the calibration corpus using `GLM-5.3-Flash-REAP50-IQ3_M.gguf`.

1 of the 24 replies came back empty and is **counted as wrong**, so this is a floor rather than a point estimate.

n=24 is small and deliberately so -- this is a smoke-level characterisation, not a leaderboard run. The 95% Wilson interval is wide; read it, not the point estimate. Decode is CPU-only (-ngl 0) at ~1 tok/s, which bounds n.

Scoring is standard ChartQA relaxed accuracy (numeric answers correct within 5%, text on exact match) with one deviation: **years are graded exactly**. 5% of 2014 is ±100 years, so an unmodified relaxed match scores "2013" as correct against "2014" and every year question becomes a free mark. Ungrading that would have inflated the number above.

## Importance matrix

`imatrix-GLM-5.3-Flash-REAP50.dat` (252 MB) is included so you can
reproduce these quants or roll your own at a size not shipped here.

Every quant in this repo was made from a **Q8_0 intermediate**, which is *not* published — it is
165 GiB, larger than everything else here combined. Build it yourself from the published FP8
checkpoint — **[`GLM-5.3-Flash-REAP50-FP8-v2`](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-FP8-v2)**,
the pass-2 healed one, the same parent listed under Provenance below. (Note the `-v2`:
`GLM-5.3-Flash-REAP50-FP8` without it is the **pass-1** checkpoint and is *not* what these GGUFs
were built from.) This is exactly the step this repo used:

```bash
# GLM5_KEEP_MTP=1 keeps the MTP block, which is a real 46th layer in the checkpoint
GLM5_KEEP_MTP=1 python convert_hf_to_gguf.py /path/to/GLM-5.3-Flash-REAP50-FP8-v2 \
    --outfile GLM-5.3-Flash-REAP50-Q8_0.gguf --outtype q8_0

./build/bin/llama-quantize --allow-requantize --imatrix imatrix-GLM-5.3-Flash-REAP50.dat \
    GLM-5.3-Flash-REAP50-Q8_0.gguf out.gguf IQ3_XXS
```

Quantising from one of the 4-bit files instead would stack loss on loss; start from Q8_0.

It was computed over 347 chunks of held-out text — held out deliberately, so the importance
matrix is not tuned on the same data the quantisation is then judged against. Expert coverage
reached 98.6–99.3%: a handful of the 144 experts per layer are never routed to by any calibration
chunk, and their rows fall back to unweighted quantisation. That is inherent to importance
matrices on sparse MoE models, not specific to this one.

**One caveat if you enable DSA.** The matrix contains **no** indexer entries, because DSA is
gated off in these files and `indexer.attn_k`, `indexer.attn_q_b` and `indexer.proj` are never
executed during calibration — an importance matrix only records tensors the run actually touches.
`llama-quantize` reports `did not find weights for blk.N.indexer.*` for each of the 12 DSA layers.
For the files as shipped this costs nothing, since those tensors are dead weight. But anyone
enabling DSA on an IQ quant is running the *selector* through the worst-quantised weights in the
model, and the selector decides which tokens attention may see at all — a bad selector does not
degrade smoothly, it drops the right context and the model answers fluently from the wrong tokens.
Pin it at Q8_0 if you go there — `--tensor-type` matches with `regex_search` over the full tensor
name, so one pattern covers all 12 layers:

```bash
./build/bin/llama-quantize --allow-requantize --imatrix imatrix-GLM-5.3-Flash-REAP50.dat \
    --tensor-type indexer=q8_0 \
    GLM-5.3-Flash-REAP50-Q8_0.gguf out.gguf IQ4_XS
```

Measured cost, from the quantiser's own per-tensor accounting: the three quantised indexer
tensors are 7.04 MiB per layer at Q8_0 and 3.53 MiB at IQ4_XS, so pinning all 12 layers costs
**~42 MiB** (84.5 MiB instead of 42.4 MiB) on an ~82 GiB file. `indexer.kpool_gate` and
`indexer.kpool_ape` are F32 in the source and are passed through untouched at every quant level,
so the pooling half of the selector is already at full precision either way.

## Provenance

- Parent: [`GLM-5.3-Flash-REAP50-FP8-v2`](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-FP8-v2)
- NVFP4: [`GLM-5.3-Flash-REAP50-NVFP4-v2`](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-NVFP4-v2)
