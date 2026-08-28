# Authorized end-to-end sequence

Operator authorization 2026-08-28: run the whole pipeline autonomously, delete the base unpruned
FP8 source, upload the improved artifacts to **new** HF repos, skip other quantizations.

## The one ordering constraint that cannot be relaxed

**The teacher must be captured before the source is deleted, because the source IS the teacher.**
`s04b_surgery` unlinks each source shard right after writing its survivors — that is the R10
mechanism keeping the prune inside the disk envelope, and it means surgery *is* the source
deletion. There is no separate delete step, and nothing that needs the unpruned model can run
after it.

So P9.5 (score the teacher, persist with taps, evaluate pass 1 as the A/B baseline) is not
optional and is not reorderable. `s09_eval` enforces it: it refuses to run with neither a teacher
nor a cached capture, rather than discovering the problem halfway through.

## Sequence

| # | step | writes | consumes |
|---|---|---|---|
| P4 | saliency sweep, 5.5M tokens | accumulators, router cache, snapshots | — |
| P5–P7 | gates: heal re-fit, split-half, criterion shootout | json | — |
| **P9.5** | **score teacher → `artifacts/eval/teacher.pt`; evaluate pass-1 student** | teacher capture + pass-1 baseline | **must precede P10** |
| P10 | `s04b_surgery` → `output/pruned-fp8` | pruned FP8 | **deletes the 306 GiB source, shard by shard** |
| P11 | `s05_heal` → `s06_emit` → `output/glm-5.3-flash-reap50-fp8-pass2` | final FP8 | — |
| P13 | evaluate pass-2 student vs the **cached** teacher | the A/B | — |
| P14 | `s07_quantize` → `output/glm-5.3-flash-reap50-nvfp4-pass2`; upload both | HF | — |

Target repos (new, so pass 1 survives as the published baseline):
`patrickbdevaney/GLM-5.3-Flash-REAP50-FP8-v2` and `…-NVFP4-v2`.

Local pass-1 FP8 is deletable only after P9.5 has scored it.

## The comparison the operator raised, and why we will be able to answer it

Unsloth's `UD-IQ1_S` of the **unpruned** model is ~100 GB and reportedly retains ~73% of top-1
accuracy. Ours is ~96 GiB. Same footprint, opposite strategy:

| | experts kept | bits/param | route to ~100 GB |
|---|---|---|---|
| UD-IQ1_S | **288/288** (all) | ~2.5 bpw over 321B | quantize hard, prune nothing |
| ours | **144/288** | 4.5 bpw over ~165B | prune half, quantize gently |

These are the two ends of the same budget, and which wins is genuinely open — aggressive
sub-3-bit quantization degrades MoE sharply, but 50% expert pruning is not free either.

**It is empirically answerable, and `s09_eval` now reports the metric that answers it.** "Retains
X% of top-1 accuracy" is exactly `1 - flip_rate` measured against the unpruned model — which is
precisely our teacher. So `top1_agreement` from P13 is directly comparable to their 73%, provided
both are measured against the same unpruned reference. Ours is.

Caveat to keep honest: a like-for-like claim needs the same eval set and the same reference
checkpoint. Ours is a held-out slice of our own 7-domain permissive corpus, not whatever Unsloth
measured on. The number is comparable in *kind*; treat a gap under a few points as inconclusive.

## Downstream, after the weights land

1. Fine-tune the DFlash 2 drafter against the REAP (`DRAFTER_PLAN.md`), private repo — upstream is
   `cc-by-nc-nd-4.0`.
2. Pure-CUDA OpenAI-compatible server: prefill, AR decode, speculative decode, FP8/INT4 KV.
   The multi-week item — KDA and mHC are new kernel surface, and MLA measured 41% of the time
   budget on the prior port, not MoE.
