# What is actually reachable on this hardware, and what is not

Written 2026-08-28, mid-pipeline. Sorted by whether it needs a new sweep, the unpruned teacher, or
neither — because that, not cleverness, is what decides feasibility here.

## Tier 1 — free, needs neither a new sweep nor the teacher

These use accumulators and the router cache we already have. Minutes to hours of compute.

### 1.1 Per-expert healing fitted in closed form  ← the single best available improvement

Current healing is **one scalar per layer**. But the sweep captured `out_sum[j]` — the summed
**vector** `Σ g_j·f_j` for every expert — and the router cache gives the post-prune gate
rescaling per expert. So the correction can be fitted as a least-squares problem:

    minimise  ‖ Σ_{all j} out_sum[j]  −  Σ_{j∈keep} a_j · r_j · out_sum[j] ‖²      over a_j

where `r_j` is expert j's mean post/pre-prune gate ratio. Closed form, no backward pass, no
teacher pass, no re-download. The current method is the degenerate case `a_j = const`.

This is the one place we can plausibly go **above** published REAP practice, which uses a scalar
or nothing. It is also directly testable: compare the residual `‖teacher_mean − student_mean‖`
under scalar versus per-expert. If it does not improve the residual, we keep the scalar and have
lost an hour.

**Must happen before `s05_heal`.** After that the gain is baked into block scales (though it is
invertible, so this is a scheduling preference, not a deadline).

### 1.2 A balanced-retention mask, to close the ballast hole

REAP ranks on saliency **pooled** across domains, which is why retention ranges 0.487–0.747. With
per-bucket accumulators we can instead pick the mask that maximises the **minimum** per-domain
retention, or any weighted compromise. Cost: seconds to compute; it changes the mask before
surgery. Whether it is desirable is a judgement — it trades pooled quality for evenness, and the
RAG argument says the current shape may already be the right one.

### 1.3 Router-aware staged greedy re-scoring

One-shot ranking ignores that removing an expert renormalises the top-8 over a different support.
The router cache makes staged re-scoring possible: drop the worst expert, recompute gates, re-rank,
repeat. ~1–2 h. The one criterion idea not yet ruled out by the tie-band analysis.

## Tier 2 — needs the unpruned teacher, which surgery deletes

Feasible, but only before `s04b_surgery` or after a ~3 h re-download.

- **Per-layer output matching against real teacher activations** (a stronger version of 1.1 that
  uses per-token data rather than accumulated sums).
- **A 40% ratio arm.** Surgery consumes the source, so a second ratio costs a re-download. Worth
  it only if the evaluation says 50% is too aggressive.

## Tier 3 — needs the student to actually run, and is the highest-information item

- **Absolute generative benchmarks** — HumanEval, GSM8K, MBPP, BFCL — run on the *student* alone.
  This needs no teacher (110 s/token there), because the NVFP4 student fits Thor. It is the only
  thing that answers "is it still smart" rather than "how far did it move".
  **Gated on inference working at all**, which is still unproven for this architecture on this box.

## Out of scope — not cleverness-limited, hardware-limited

| | why |
|---|---|
| teacher generation of any kind | 328 GB ÷ 3.0 GB/s ≈ **110 s per decoded token** |
| self-generated or difficulty-filtered calibration | requires the above |
| full-model distillation healing | backward pass through a 165B student |
| 16k-token calibration | needs sparse-DSA + chunkwise-KDA kernels first |
| non-uniform per-layer expert allocation | **not compute** — `num_local_experts` is one scalar |
| expert merging (REAM/EEP) | tractable but REAP's own thesis is that merging loses to pruning for MoE; a step sideways at best |

## Does any of it need another full REAP run on the teacher?

**No.** Everything in Tier 1 reuses the 18-hour sweep's output. Tier 2 needs the teacher *present*,
not re-swept. Only a change to the calibration data itself — longer sequences, different domains,
regenerated responses — would require a new sweep, and every such change is in the out-of-scope
table for other reasons.

## Recommended order

1. **1.1 per-expert healing** — before `s05_heal`, measurable, plausibly above SOTA.
2. Let the pipeline finish and **measure** (dNLL, top-1 agreement, per-domain).
3. **Tier 3 benchmarks** once inference exists — the largest information gain of anything here.
4. Revisit 1.2 / 1.3 / a 40% arm **only if** the evaluation says the mask or the ratio is the
   limiting factor. Not before; that is how a project spends a week improving the wrong number.
