# DeepSeek Sparse Attention in llama.cpp — mechanism and implementation plan

`[EXT]` design, `[MEAS]` numbers. Written after reading `Glm5NextTextIndexer` in transformers
5.16.1, because the mechanism is intricate enough that re-deriving it from the paper would be
slower than re-reading the source, and getting it subtly wrong is invisible.

## Why this is needed at all

llama.cpp loads the indexer tensors and never uses them — `LLM_ARCH_GLM_DSA` and `DEEPSEEK2`
already ship that way upstream, so those 11 layers run **dense**. Dense is a *superset* of what
DSA selects (the model sees its top-2048 plus extra low-relevance tokens, which softmax
down-weights), so quality degrades gracefully rather than breaking. The cost is O(n²) where the
model was designed for O(n·2048), which is what makes 128k+ impractical rather than merely slow.

**Memory is not the constraint.** MLA compresses KV to `kv_lora_rank = 512` across only 11 of 45
layers, ~11 KB/token — 128k is ~1.4 GB of cache. The indexer adds ~257 floats/token/layer (see
below). Compute is the whole problem.

## The free correctness invariant

`index_topk = 2048`. **When context ≤ 2048 the selection is a no-op — it selects everything**, so
a correct sparse implementation must produce *bit-identical* logits to the dense path. The
existing 16-token fixture (`tests/test-glm5-next-logits.cpp`) therefore becomes the DSA gate for
free, and any selection bug, gather misindex or cache-stride error shows up at 16 tokens rather
than at 128k where nothing is diagnosable. Build against this first.

## Mechanism, exactly

Config: `index_n_heads 32`, `index_head_dim 128`, `index_topk 2048`, `index_kpool 4`,
`index_kpool_compress true`, `index_kpool_always_select_tail true`.

Tensors per DSA layer (all already in the GGUF, currently inert):

| tensor | shape | note |
|---|---|---|
| `indexer.wq_b` | `[q_lora 1536, 32*128]` | query, from `q_a_layernorm(q_a_proj(x))` — reuses the MLA LoRA |
| `indexer.wk` | `[4096, 128]` | key, **single head** (MQA-style) shared across the 32 query heads |
| `indexer.k_norm` | `[128]` w + b | **LayerNorm**, eps 1e-6 — not RMSNorm, and it has a bias |
| `indexer.weights_proj` | `[4096, 32]` | per-head weighting of the scores |
| `indexer.kpool_gate` | `[128, 4096]` | per-token gate logits used to pool |
| `indexer.kpool_ape` | `[4, 128]` | learned position-within-pool bias |

### 1. Per-token state (this is what gets cached)

    k     = LayerNorm(wk(x))                  [128]
    gate  = kpool_gate(x)                     [128]
    valid = 1                                 [1]
    cached per token per DSA layer: 257 floats

### 2. Pooling — pools of 4, starting at the FIRST REAL TOKEN

Not at slot 0. Pool p covers tokens `first_key + 4p .. first_key + 4p + 3`, so left padding does
not shift the grouping. Then, **per channel**, a softmax over the 4 tokens in the pool:

    logits[t,d] = gate[t,d] + ape[t,d]        t in 0..3, ape is per-position-in-pool
    prob        = softmax(logits, dim=t)      masked to -inf for invalid slots
    pool_key[d] = sum_t prob[t,d] * k[t,d]

A pool is selectable only if **its final token is visible to the query** (causality) and all four
slots are valid.

### 3. Scoring and selection

    scores  = relu( (q @ pool_key^T) * head_dim^-0.5 )     [32 heads]
    weights = weights_proj(x) * n_heads^-0.5               [32]
    index_scores = weights @ scores                         [pools]
    select_k = index_topk // index_kpool = 512 pools
    selected = topk(index_scores, 512) -> expand to token indices, plus the tail pool

Note the `relu` before the head-weighted sum — scores are non-negative, which is not what a
softmax-attention intuition predicts.

### 4. Consumption

The reference turns the selected indices into an **additive sparse mask** for the eager and SDPA
paths. So a mask-based llama.cpp implementation is *faithful to the reference*, not a shortcut —
it reproduces behaviour exactly while leaving compute at O(n²).

## Implementation order

**Phase 1 — indexer forward, no selection.** Compute k/gate/valid, cache them, compute
`index_scores`. Validate the scores against transformers on the tiny fixture. Nothing changes in
attention yet, so nothing can regress.

*Cache placement*: widen the MLA K row from `kv_lora_rank` to `kv_lora_rank + 128` and slice, the
same trick kimi-linear uses to carry the rope tail alongside the compressed KV. Reuses the
existing cache machinery instead of adding a third stream. `gate` and `valid` are recomputed
from the cached `k`… **no — `gate` depends on `x`, which is gone after the token is consumed, so
it must be cached too.** Row becomes `kv_lora + 128 (k) + 128 (gate) + 1 (valid)`.

**Phase 2 — mask.** Turn selection into a KQ mask. Gate: exact logits match at ctx ≤ 2048, then
NIAH at 8k/32k. Correct but still O(n²).

**Phase 3 — gather.** Replace the mask with a real gather of the selected KV, so attention runs
over 2048 instead of n. Decode first: at 128k the full MLA read is ~1.4 GB/token versus ~22 MB
for top-2048, roughly **60× less KV traffic**. Prefill can stay dense — it is one-shot and
compute-bound.

**Phase 4 — pooled scoring cost.** The indexer itself reads its whole key cache to score. k-pooling
already cuts that 4×; measure before optimising further.

## What this does NOT need

No new ggml operator. `argsort`/`top_k`, `get_rows`, `soft_max`, `clamp` and the existing
attention path cover it — unlike mHC, which genuinely had no equivalent. The work is cache
plumbing and graph construction, not kernels.

## Phase 2b — where the indexer state lives `[EXT 2026-08-30]`

Decode needs the indexer key and gate of **every cached token**, and neither is recoverable after
the fact: the key is `LayerNorm(wk(x))` and the gate is `kpool_gate(x)`, both projections of the
hidden state, which is gone once the token is consumed. The compressed MLA latent already in the
cache is a *different* projection of the same `x` and cannot be inverted. So the state has to be
cached, and the only question is where.

**(a) Widen the K row on DSA layers, zero-pad Q.** `n_embd_k_gqa` already supports per-layer
variation, so the row becomes `kv_lora (512) + key (128) + gate (128) = 768`. Attention still
contracts the full row, so Q must be 768 wide — padded with zeros above 512, which makes the
extra dimensions contribute exactly nothing to the scores. The indexer then reads its state from
the same cache.

*Cost*: the QK product on 11 layers goes 512 → 768, **+50% on that matmul**. That sounds
backwards for a feature whose purpose is less work, and it would be, if it stayed dense. Under
Phase 3's gather, attention runs over 2048 tokens instead of `n` — at 128k that is a ~64×
reduction, so +50% on the row width is noise against it. It is only a bad trade if Phase 3 never
lands.

*Correction `[MEAS 2026-08-30]` — this claim was wrong when first written.* I asserted (a) was
contained in the graph builder. It is not. The cache row comes from

    n_embd_k_gqa(il) = n_embd_head_k(il) * n_head_kv(il)

and `n_embd_head_k(il)` resolves to one of exactly two values, `n_embd_head_k_swa` or
`n_embd_head_k_full`, selected by `is_swa(il)`. There is no general per-layer array. Widening the
row on DSA layers therefore requires a new per-layer mechanism in `llama_hparams` — which the
cache reads. So (a) touches shared cache-adjacent code after all, and its stated advantage over
(b) evaporates.

Reusing `is_swa` to mark DSA layers would avoid new hparams and is *tempting and wrong*: it is
consulted elsewhere for sliding-window behaviour this model does not have.

**(b) priced `[MEAS 2026-08-30]`.** Worse than (a), not better. The recurrent cache is sized by
`n_embd_r()`, which is **global, not per-layer** — it returns one figure for the whole model, set
by the 34 KDA layers. DSA layers are not recurrent and get no `r_l`/`s_l` allocation at all.
Using it would mean either marking DSA layers recurrent (wrong: layer typing is consulted
throughout) or adding a genuinely new cache type. That is more shared-code surface than (a), not
less.

**Revised standing**: there is no cheap path. Both options change code the cache reads.

* (a) costs one new per-layer field in `llama_hparams` plus +50% on the QK matmul for 11 layers.
  Bounded, understood, and the +50% disappears against Phase 3's ~64× reduction at 128k.
* (b) costs a new cache type. Unbounded until designed.

**Proceed with (a)**, with its cost stated plainly rather than justified by a containment
argument that turned out to be false.

### Worth saying out loud: is this worth doing here at all?

Dense already measures **3/3 needle retrieval at 32k** on the pruned 4-bit model, and upstream
ships `LLM_ARCH_GLM_DSA` and `DEEPSEEK2` dense today — so dense is the accepted standard, not a
shortfall. The case for DSA in llama.cpp rests entirely on contexts beyond ~64k, which this
runtime cannot reach at O(n²) anyway. Meanwhile the same work in a bespoke CUDA server has no
shared-code blast radius and is where the operator actually wants it.

This is not an argument to stop — it is the honest cost/benefit, recorded so the decision is
made deliberately rather than by momentum.

**(b) A separate cache stream.** Cleanest conceptually and no wasted QK width, but it means
touching `llama_kv_cache` — the machinery the working, published GGUFs depend on. Not worth the
blast radius for 256 floats per token.

**(c) Reuse the recurrent state cache (`r_l`/`s_l`).** The model already carries one for the 34
KDA layers. But DSA layers are not recurrent, and marking them so would change layer typing
everywhere it is consulted. Rejected.

**Decision: (a).** The +50% is a real cost, stated rather than hidden, and it is bounded by
Phase 3 landing. The alternative that avoids it costs a change to shared cache code, and this
project has already learned what happens when something subtly wrong ships in a path nobody
re-measures.

### Verification, unchanged

* `ctx <= index_topk` (2048): selection is a no-op, so sparse must equal dense **exactly**. The
  16-token fixture is the gate.
* Above it: NIAH at 8k/32k must still retrieve, against the dense baseline of 3/3 at each.

## Phase 2c is harder than Phase 3 in ggml, not easier `[MEAS 2026-08-30]`

The plan assumed mask-then-gather: build the additive mask first (correct, still O(n²)), then
replace it with a gather for speed. In ggml that ordering is backwards.

**The mask needs a per-token scatter, and ggml has no op for it.** The mask is `[n_kv, n_tokens]`,
and for each token `t` the selected positions differ, so writing it means scattering along `ne0`
independently per column. The candidates:

* `ggml_top_k` returns *indices*, "in no particular order" — no values, no mask.
* `ggml_argsort` gives an ordering, but extracting the k-th largest **value** per row (to
  threshold against) needs a per-row gather, which `ggml_get_rows` does not do — it gathers whole
  rows along `ne1`.
* `ggml_set_rows` is a scatter, but along `ne1`, with **one shared index vector** across rows.
  Transposing to `[n_tokens, n_kv]` does not help: the indices still differ per token.

So a faithful mask needs a new custom operator — the very thing this port was pleased not to need.

**The gather does not have this problem, because at decode `n_tokens == 1`.** One query, one set
of selected indices, and `ggml_get_rows` gathers the chosen KV rows directly. No scatter, no new
op. Prefill keeps the dense path, which is the standard split anyway: prefill is one-shot and
compute-bound, decode is where the KV traffic dominates.

**Revised order: skip 2c, implement Phase 3 decode-only.** That is also where the entire benefit
lives — at 128k the full MLA read is ~1.4 GB/token against ~22 MB for top-2048, roughly **60×**.
The mask would have bought correctness parity that dense already delivers (3/3 needle retrieval
at 32k, measured) at no speed gain and the cost of a custom op.

The `ctx <= index_topk` exact-match invariant still gates it: with `n_kv <= 2048` the gather
selects everything, so decode must match dense exactly.

### Phase 3 decode gather — the path is viable, with one detour

Index arithmetic in ggml looked like a blocker (`top_k` returns I32; there is no I32 add or
scale). It is not: `ggml_cpy` converts **I32 → F32 and F32 → I32**, so the pool-to-token
expansion goes through float and back.

    sel   = ggml_top_k(index_scores, index_topk / kpool)   // I32 [select_k]
    self  = cpy(sel, F32)                                  // to float
    first = scale(self, kpool)                             // pool p -> first token p*kpool
    tok   = add(repeat(reshape(first, 1, select_k)),        // broadcast
                arange(0, kpool))                          // [kpool, select_k]
    toki  = cpy(reshape(tok, select_k*kpool), I32)
    k_sel = ggml_get_rows(k_cache, toki)                   // gathers along ne1 == n_kv

`ggml_get_rows` gathers along `ne1`, and the K cache is `[n_embd_k_gqa, n_kv]`, so `n_kv` is
already the gathered axis. No transpose needed.

Remaining piece: attention must then run over the gathered rows rather than the cache, which
means calling `build_attn_mha` with the gathered K/V instead of `build_attn`. `mctx` is public on
`llm_graph_input_attn_kv`, so the builder can reach the cache to slice the indexer state out of
the widened rows.

**Decode only.** At `n_tokens == 1` there is one query and one index set. For prefill each token
selects differently, so a shared gather is impossible and the dense path stays — which is the
usual split regardless, since prefill is compute-bound and decode is KV-bandwidth-bound.

## imatrix and the DSA indexer (measured 2026-08-30)

The shipped imatrix contains **zero** indexer entries. That is correct and expected: DSA is gated
off, so `indexer.attn_k`, `indexer.attn_q_b` and `indexer.proj` are never touched in the forward
pass, and an importance matrix only records tensors the calibration run actually executes.

`llama-quantize` therefore reports `did not find weights for blk.N.indexer.*` for each of the 12
DSA layers (3, 7, 11, ... 43, and the MTP block at 45) — 36 warnings, plus `output.weight` and
`token_embd.weight`, which are standard. Only three of the seven indexer tensors per layer are
affected; `k_norm.weight`, `k_norm.bias`, `kpool_ape` and `kpool_gate` stay F32 and never needed
imatrix data.

**Consequence, and it is a real one for the IQ quants.** Those three tensors per layer are
quantized with no importance weighting, i.e. plain round-to-nearest. For the GGUFs as shipped
this costs nothing, because DSA is off and the tensors are dead weight. But anyone enabling DSA
on an IQ-quantized file is running the selector through the worst-quantized weights in the
model, and the selector's job is to decide which tokens attention may see at all. A bad selector
does not degrade output smoothly; it drops the right context and the model answers fluently from
the wrong tokens.

The fix is cheap and should land before DSA is enabled by default: the indexer is tiny --
`attn_k` 0.53 MiB, `attn_q_b` 6.38 MiB, `proj` 0.13 MiB per layer, about **77 MiB across all 12
layers**, well under 0.1% of a 92 GiB file. Pin them with `--tensor-type` at Q8_0 rather than
letting them fall to IQ4_XS, or collect imatrix data for them by running the calibration pass
once with DSA enabled.

## Phase 3: sparse attention over the cache (2026-08-30)

`llm_build_glm5_next::build_attn_dsa` now runs gathered attention against selected pools, gated
behind `hparams.dsa_enabled` and falling back to dense `build_attn` for every case it does not
handle.

The construction that makes it tractable is gathering at **pool** rather than token granularity.
Consecutive cache cells are contiguous, so the `[D, n_kv]` cache view can be re-viewed as
`[D*kpool, n_pools]` and one `ggml_get_rows` with pool ids gathers whole pools. This removes the
index arithmetic entirely — ggml has no ops for scaling I32 pool ids into token ids — and pool
granularity is DSA's own, since `select_k = indexer_top_k / kpool` (2048/4 = 512 pools).

Verified in `tests/test-dsa-pool-gather.cpp`: the K re-view, the V sub-view, the
transpose→get_rows→transpose mask gather, and the strided slot-0 pool mask. All four constructions
were mutation-tested (dropped transpose, wrong ne0 stride, shifted V offset, wrong re-view width)
and every mutant fails.

Three bail-outs return `nullptr` and take the dense path:

* **`n_pools <= select_k`** — the free DSA invariant. Every pool would be selected, so selection is
  a no-op and sparse must equal dense *exactly*. Routing to dense makes that identity structural
  rather than something a test has to catch.
* **`n_tokens != 1`** — prefill. `top_k` gives a per-token selection and one gathered K/V serves
  one selection; prefill needs block-sparse machinery this does not have. Decode is also where the
  win is, since prefill is compute-bound and decode is KV-bandwidth-bound.
* Row not widened, quantized cache, multi-head KV, ragged final pool.

All guards run **before** any graph mutation. If the function returned `nullptr` after expanding
its stores, the caller's dense `build_attn` would emit a second `cpy_k` into the same slots.

A pool-level causal mask is added to the scores before `top_k`, otherwise selection spends slots
on pools attention will mask to `-inf`. Under a causal mask a pool has a visible token iff its
first token is visible, so slot 0 of each pool is the exact pool mask.

### Position ordering is checked, not assumed

Pooling groups kpool consecutive *cache cells* and treats them as kpool consecutive *sequence
positions*. That holds for one sequence filling a fresh cache in order, and breaks under a shared
unified cache, a context shift or defragmentation — where attention stays correct (the gathered
mask travels with the gathered rows) but *selection* silently pools unrelated positions and drops
the context the answer needed. So the cache is asked: `llama_kv_cache::pos_ordered_prefix` returns
the length of the leading run where cell *i* holds position *i*, requires everything past it to be
empty, and returns 0 for any layout pooling cannot account for. Memoised per ubatch on the cache
context; the sparse path bails to dense on 0.

### The padded tail

`get_n_kv` pads `n_kv` up to a multiple of 256, so the live window ends mid-pool and a strict
"cell *i* is position *i* for all of n_kv" test would have been false almost always — leaving the
sparse path permanently dormant, which is safe and useless. The prefix length fixes that, and
splits the window into `n_full` complete pools of real tokens plus one partial pool.

That partial pool holds the **newest** 1..kpool-1 tokens — precisely the ones a decode step must
not lose — and it cannot be scored, because it is not a whole pool. It is therefore always
attended rather than selected: its offset is known on the host, so it is appended to the gather as
a plain `ggml_view_2d` on both K and the mask. No index tensor is needed (a host-known constant
cannot be baked into a graph whose buffers are not yet allocated), and `top_k` cannot pick it twice
because scoring only ever sees the `n_full` complete pools. The padding slots inside that partial
pool are hidden by the mask, which travels with it.

Covered by three further checks in `tests/test-dsa-pool-gather.cpp` (tail appended after the
selected pools, selected rows undisturbed, tail mask row paired with tail K rows), each killed by
an off-by-one-pool mutation.
