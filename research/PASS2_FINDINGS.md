# Pass-2 findings — corrections first, then the research

Full synthesis: [`PASS2_FINDINGS_raw.md`](PASS2_FINDINGS_raw.md) (13 agents, 1.44M subagent
tokens, 35 min). **Read the corrections below before acting on it** — the agents inspected the
filesystem *while `s07_quantize` was still running*, so two of their four gates describe a state
that no longer exists.

## Corrections to the raw synthesis `[MEAS 2026-08-27 21:40]`

| raw claim | actual |
|---|---|
| **G0-4**: "`s07_quantize` did not finish — 58/62 shards, no `config.json`. You have no loadable student today." | **Wrong, stale.** 62/62 shards, `config.json` present, `s08_document` done, pipeline reports `all stages terminal`. NVFP4 is **96.0 GiB**, 18,270 packed tensors = exactly 42 × (144+1) × 3. |
| **G0-1**: "the source weights are gone" — framed as a loss | **Correct but not a loss.** `s04b` deletes each source shard immediately after writing its survivors; that is the documented R10 mechanism that keeps the prune inside the disk envelope. Re-staging is ~55 min and already automated. |
| **G0-1**: recommends deleting the FP8 tree to make room | **Do not.** It is the master and the only local copy; it *is* uploaded (`patrickbdevaney/GLM-5.3-Flash-REAP50-FP8`), but deleting the local copy to free 157 GiB should be a deliberate decision, not a build step. 191 GiB is currently free. |
| raw suggests "unlink-as-you-go surgery" as a new idea | **Already implemented** — that is exactly what `s04b_surgery` does, with per-shard verification before each unlink. |

## G0-2 — the one substantive finding, and it stands `[OPEN]`

The synthesis independently re-derived `s05`'s healing gain from the saliency accumulators and
reproduced it to 3 decimal places (median 0.6933 vs the shipped 0.6964). Two things follow, and
they point in opposite directions:

**Confirmed correct.** It verified in the installed model source that `g_j` as captured by
`scripts/stream_saliency.py` excludes `e_score_correction_bias` (selection-only) and uses the
renormalized, `routed_scaling_factor`-scaled gate — i.e. the coefficient that actually
multiplies `f_j`. **Pass 1's saliency is faithful to REAP as this architecture implements it.**
That was the single most important thing to get right, and it is right.

**Open risk.** `norm_topk_prob=True` makes the gates sum to the same mass before *and* after
pruning, so gate mass is already conserved by renormalization. Our per-layer gain is derived
from **pre-prune** conditional means and may therefore over-correct: the shrink is directionally
right (REAP keeps higher-‖f‖ experts, so the pruned layer's output is biased high), but its
**magnitude is not verified**, and it was applied to the block scales of all 6,048 retained
experts.

This cannot be settled from the accumulators alone — it needs the unpruned teacher, which pass 2
re-stages anyway. **Highest-priority check in pass 2:** measure the actual per-layer output-norm
ratio teacher-vs-student on held-out data and compare it to 0.696, rather than trusting a
first-moment derivation. If it disagrees, the correction is re-fittable cheaply — it lives
entirely in F32 block scales.

## What the research adds that we did not already know

Taking only what survived adversarial verification and is not stale:

1. **Free accumulators on the pass we already run** — per-domain, per-modality and second-moment
   statistics plus a top-40 router-logit cache, at ~zero marginal cost (+37 GB disk). Converts
   one saliency pass into ~8 offline criteria and an exact mixture sweep. This is the single
   best idea in the report: it removes ~5 h of would-be extra passes.
2. **Split-half keep-set-overlap as a stopping rule** — tells you whether the calibration budget
   was sufficient instead of guessing at a token floor. Directly addresses pass 1's weakest
   point (501/12,096 expert slots decided on <2,000 tokens).
3. **Router-aware staged greedy re-scoring** — one-shot ranking ignores that removing an expert
   renormalizes top-8 over a different support. Needs the logit cache from (1).
4. **Paired teacher-vs-student teacher-forced ΔNLL + flip rates** on the 8% held-out split as
   the eval. Cheap, interpretable, and the thing pass 1 skipped entirely.
5. **~25.5 experts per layer sit within ±5% of the 50% cut** — the mask is genuinely uncertain
   near the boundary, which is what makes a better criterion worth running.

## Confirmed closed

- **MTP exclusion is correct**, not an open question. Stop re-litigating it.
- **Saliency capture is faithful** to REAP as implemented here (see G0-2 above).
