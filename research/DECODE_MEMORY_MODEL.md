# Decode memory model — what the CUDA server has to hit, and why

Measured 2026-08-29 from the shipped pass-2 NVFP4 checkpoint and the router cache. Decode on this
box is bandwidth-bound, so every number here is bytes moved per token; Thor is ~273 GB/s.

## The finding that reorders the optimisations

Only **8 of 144** experts are read per token, but the whole attention stack is. So:

| | dense (every token) | experts | total | roofline |
|---|---|---|---|---|
| AR decode, NVFP4 as shipped | 14.28 | 4.54 | **18.82 GiB** | 13.5 tok/s |
| AR, attention at NVFP4 | 6.27 | 4.54 | **10.81 GiB** | 23.5 tok/s |
| block-8, acc 5.8, attn NVFP4 | 1.08 | 4.24 | **5.32 GiB** | **48 tok/s** |

**AR decode is dense-bound (76% of traffic for 15% of the weights). Speculative decode is
expert-bound (80% of traffic).** Blocking amortises the always-resident weights across 8
positions but *not* the experts, because each position routes independently. The two modes want
different optimisations and the server needs both.

Excluded from every figure, because text decode does not read them: the vision tower (1.05 GiB)
and the embedding table (1.18 GiB — it is a row lookup, 8 KB/token). A server that keeps them in
the hot path is moving 2.2 GiB/token for nothing.

## Union-aware expert gather — a hard requirement, not an optimisation

A block of 8 tokens routes independently, so a naive verifier issues 8 × 8 = 64 expert reads per
layer. The actual number of **distinct** experts is lower, measured on the post-prune router over
three layers:

| block size B | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| distinct experts | 8.0 | 14.7 | 25.7 | **43.4** | 67.1 |
| if no reuse | 8 | 16 | 32 | 64 | 128 |
| **saving** | — | 8% | 19% | **33%** | 48% |

**Gathering the union once instead of per position saves 33% of expert traffic at block 8, for
zero accuracy cost.** At the block-8 figures above that is 2.1 GiB/token — larger than any other
single lever except attention quantisation, and unlike that one it costs nothing at all.

This is a **lower bound**. The router cache is strided (`arange(0, N, N//n)` at 20%), so
consecutive cached tokens are ~5 apart in the text; genuinely adjacent tokens will overlap more.
Re-measure on contiguous tokens once a server can produce them.

Implementation shape: build the block's routing table first, take `unique()` over the flattened
top-8 indices, gather those expert weights once, then scatter-accumulate each position's
contribution with its own gate. The MoE kernel must be written against the union, not looped over
positions — retrofitting this later means rewriting the inner loop.

## Levers, sorted by whether they cost anything

**Lossless — take all of them.**
1. Speculative decoding itself: one weight read amortised over a block. DFlash 2 is exact — greedy
   output matches the target bit-for-bit.
2. Union-aware expert gather: 33% of expert traffic at block 8 (above).
3. Do not touch vision or the full embedding table during text decode: 2.2 GiB/token.

**NOT near-lossless — measured, and declined. `[MEAS 2026-08-29]`**
4. Attention projections at NVFP4. Three arms built from the same corrected FP8 base, same mask,
   same healing, one variable moved, each scored on the same 241,516 held-out tokens:

   | arm | quantised | GiB | top-1 | Δ | McNemar z | milli-top1 / GiB |
   |---|---|---|---|---|---|---|
   | shipped v2 | — | 98.2 | 0.83857 | — | — | — |
   | `kda` | KDA q/k/v | 93.6 | 0.83627 | −0.00230 | 5.8 | **0.500** |
   | `post` | o_proj + MLA proj | 94.8 | 0.83306 | −0.00551 | 12.3 | 1.621 |
   | `all` | both | 90.2 | 0.83184 | −0.00673 | 14.5 | 0.841 |

   **The prediction above was wrong in both directions.** Expected cost was ~0.004 top-1 for the
   full set; it is 0.0067. And the protection argument was applied to the wrong tensors: q/k/v on
   the 34 linear-attention layers were expected to be the fragile ones, because they feed the
   delta-rule state where error propagates along the sequence — the same reasoning that protects
   the gates. Measured, they are the **most** quantisation-tolerant part of attention, 3.2× more
   byte-efficient than `o_proj` plus the MLA projections. The likely reason is that `o_proj`
   writes straight into the residual stream, where mHC's four streams carry its error forward
   unattenuated, while q/k/v error is partly washed out by score normalisation downstream; and the
   MLA projections are low-rank factorisations (`q_a→q_b`, `kv_a→kv_b`) where error in a
   compressed latent is amplified on expansion rather than averaged.

   **Damage is sub-additive.** `kda` alone costs 0.00230 and `post` alone 0.00551, summing to
   0.00781, but both together cost 0.00673 — 14% less than the sum. So "cost per GiB" is not a
   property of a tensor set; it depends on what else is already quantised. Any lever priced by
   subtracting two arms will misprice it: subtraction predicted `kda` at −0.00122 and it measured
   −0.00230, off by 1.9×.

   **Declined for the shipped artifact, on the project's own priority order.** Even the cheapest
   arm costs 42% of the entire pass-2 quality gain (+0.00545 from the mask, +0.00556 from fixing
   healing) and the ratio is what decides it: the *whole* justification is a **roofline estimate**,
   13.5 → 23.5 tok/s, and nothing has measured decode throughput on this box for this
   architecture. Trading measured intelligence for unmeasured speed is the wrong side of that
   trade. 98.2 GiB already leaves 18.8 GiB for KV cache, activations, the drafter and the CUDA
   context, so the 4.6 GiB is not needed to *fit* — only to go faster.

   **Correct sequencing: build the CUDA server, measure AR decode at 98.2 GiB, then revisit.** If
   the measured speedup approaches the roofline, `kda` at 93.6 GiB for −0.0023 becomes a
   defensible clearly-labelled speed variant. `post` is dominated and should never ship — it costs
   82% of the damage for 43% of the bytes.

   Protected throughout, and still correct: KDA gates (`f_a/f_b/g_a/g_b`), `b_proj` (the delta-rule
   beta term — earlier notes wrongly grouped it with q/k/v as safe), the short causal convs, DSA's
   `indexer`, and all norms. `nvfp4_attn_mode` in `s07_quantize` selects `none`/`kda`/`post`/`all`
   and derives `config.json`'s ignore list from the same setting, so the metadata cannot disagree
   with the weights.

**Lossy — declined, with the number that decides it.**
5. Reducing top-k. Gate mass by rank, post-prune, averaged over three layers:

   | rank | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
   |---|---|---|---|---|---|---|---|---|
   | share | 25.7% | 18.5% | 14.0% | 11.2% | 9.4% | 8.1% | 7.1% | 6.1% |

   Ranks 7–8 are 25% of expert traffic for **13.2%** of gate mass; ranks 5–8 are 50% of traffic
   for **30.7%**. That is not a trimmable tail, it is a genuinely distributed mixture — and rank 8
   still carries 6.1%. Dropping to top-6 buys ~1.1 GiB/token under speculation (~6% of AR) while
   discarding 13% of the MoE output of a model that has *already* lost half its experts. Same
   double-jeopardy as REAP-plus-IQ2, for less.

   The defensible variant is *adaptive* top-k — cut only where rank 8's gate is unusually low —
   scorable with the same reconstruction residual used for healing. Second-order behind 1–4;
   sequence it last.

## Honest ceiling for speculative decode

Not acceptance × AR. Verification re-reads the weights once per block, so the ceiling is
`block_traffic / acceptance`. At acceptance 5.8 and attention quantised that is 5.32 GiB/token,
i.e. **48 tok/s at 100% of roofline** — but attention is *not* quantised in the shipped artifact
(see lever 4), so the operative figure is the BF16-attention one, and every number in this section
is a roofline, not a measurement. Prior art on this box (DSpark CUDA decode) reached ~25% of
peak bandwidth, which would be ~12 tok/s; the gap between those two numbers is the actual
engineering target, and it is worth more than any further algorithmic change.
