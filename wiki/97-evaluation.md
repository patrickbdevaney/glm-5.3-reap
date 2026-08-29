# Evaluation — what the paired eval measures, and one bug that inverted its answer

## The bug: image-placeholder positions were being scored `[MEAS 2026-08-28 21:30]`

`s09_eval` masked scored positions with `gold != pad_token_id` and nothing else. For an image-text
record that keeps **every image-placeholder position in the loss** — and those are exactly the
positions whose *embedding was replaced by an image feature* before layer 0. The model is asked to
predict the literal placeholder token id from a hidden state that is a picture, at positions the
training objective masks out and the model is therefore never trained to emit.

Split by whether the gold token is a placeholder:

| | n | teacher NLL | student NLL | dNLL | flip |
|---|---|---|---|---|---|
| vision, placeholder | 96,597 | **16.94** | 15.19 | −1.752 | 0.732 |
| vision, real text | 532 | 10.45 | 8.49 | −1.958 | 0.547 |
| everything else | 240,984 | **0.997** | 1.200 | +0.203 | 0.162 |

A teacher NLL of 16.94 is the model saying "I have no idea", which is the correct answer to a
question that should never have been asked. Those positions were **99.5% of the vision bucket and
29% of every scored token**, and they carried the headline numbers with them:

| metric | reported | corrected |
|---|---|---|
| dNLL_mean | −0.359 | **+0.198** |
| top1_agreement | 0.674 | **0.837** |
| teacher_nll | 5.567 | **1.018** |
| topk_KL | 4.108 | **0.695** |
| tap_drift @5 | 0.293 | **0.170** |

**The tell was the sign.** A pruned student cannot beat its teacher by 0.36 nats while disagreeing
with it on a third of tokens. Nothing else about the run looked wrong — it completed, the token
counts matched, no sample was skipped, no warning fired.

### Where the fix had to go, and where it must not

The obvious fix is to narrow the mask at **capture** time. That would have been a worse bug.

The teacher capture costs 93 minutes and `s04b_surgery` **deletes its source**, so it is reused
across every student. `compare()` aligns the two captures positionally and truncates to
`min(len)`. Narrowing the capture mask would produce a fresh student capture of 241,516 tokens
compared against a cached teacher of 338,113 — by `[:n]` truncation, silently, against unrelated
positions. Same failure class, new costume, and this time undetectable because the *sign* would
look plausible.

So the capture mask stays "not padding" and nothing more, and `compare()` rebuilds the gold-token
array (deterministically, from the same `load_heldout`) and drops the placeholder positions there.
Every capture stays mutually alignable regardless of when it was taken. `scripts/eval_recompare.py`
re-runs the corrected comparison over cached captures and calls `s09_eval.compare` rather than
reimplementing it — two copies of this arithmetic is how the corrected number and the shipped
number drift apart.

## What the corrected evaluation says — pass-1 baseline, FP8

Teacher: unpruned GLM-5.3-Flash. Student: the published pass-1 REAP-50 FP8, re-healed.
241,516 held-out tokens the calibration never saw.

**Overall: top-1 agreement 0.837, dNLL +0.198, top-k KL 0.695.**

| domain | predicted retention | measured top-1 | dNLL |
|---|---|---|---|
| agentic | 0.747 | 0.863 | +0.182 |
| code | 0.728 | **0.921** | +0.049 |
| science | 0.720 | 0.829 | +0.138 |
| math | 0.713 | **0.919** | +0.026 |
| finance | 0.651 | 0.741 | +0.220 |
| general / ballast | 0.487 | **0.572** | +1.021 |
| vision | 0.682 | *(532 tokens — unmeasured)* | |

### The saliency framework predicted this

Per-domain retention was computed from the calibration accumulators **before** any of these tokens
were scored. Against measured top-1 agreement it gives **Pearson r = 0.942**, Spearman ρ = 0.714.

That is the strongest validation this project has of REAP itself: a quantity derived from routing
statistics on calibration text predicted per-domain behaviour on held-out text, including the size
of the gap. It also confirms the model card's central claim empirically rather than by argument —
**ballast is both the least-retained domain (0.487) and the least-agreeing (0.572)**, and it is the
one domain retrieval can repair. Code and maths, which retrieval cannot supply, sit at 0.92.

### Comparable to aggressively-quantised GGUFs, with care

`1 − flip_rate` against the unpruned model is the same quantity releases quote as "retains X% of
top-1 accuracy". Our REAP-50 measures **0.837**; the Unsloth `UD-IQ1_S` class of build is quoted
around **0.73** at a similar memory footprint. Two caveats keep this honest: the comparison is
across different calibration/eval text, and it is *our* pass-1 model, before pass-2's mask and
per-expert healing. It is indicative, not a head-to-head.

## The gap this exposed: vision is UNMEASURED

After excluding placeholders the vision bucket is **532 tokens across 28 records**. The held-out
image-text records average ~3,450 placeholder positions against ~19 real text tokens — they are
image-plus-short-caption records, so there is almost no text to score.

**R3 is therefore not resolved by this evaluation**, and "unmeasured" is a different statement from
"measured and fine". What *is* known: vision's per-domain saliency retention is 0.682, mid-pack
among seven domains and far above ballast, so the mask did not strip vision-serving experts. What
is not known is whether the surviving ones are enough. Closing this needs an image-text held-out
set with substantial text responses, and — because surgery consumes the source — a teacher that no
longer exists on this box. `by_domain` now carries a `sufficient` flag so a bucket this small can
never again be read as a result.

## Standing caveat

All of the above is **teacher-forced**: dNLL and top-1 agreement on ground-truth prefixes. It
measures how far the student moved from the teacher, not whether the student is *smart*. Only
generative benchmarks answer that, and they are gated on inference working at all — see
`research/ZENITH_ON_THOR.md` Tier 3.

## Pass 2 measured end-to-end: the proxies did not translate `[MEAS 2026-08-28 23:33]`

Pass 2 changed two things against the published pass-1 checkpoint — a re-ranked mask (+2.5% on
reconstruction residual) and per-expert healing (+8.6% on the same). Both were measured, both were
real, and neither showed up:

| metric | pass 1 | pass 2 | Δ | σ |
|---|---|---|---|---|
| top-1 agreement | 0.8370 | 0.8369 | −0.0001 | 0.00075 |
| ΔNLL mean | 0.1979 | 0.1940 | −0.0039 | |
| top-k KL | 0.6948 | 0.6939 | −0.0009 | |

Per domain it is a redistribution: agentic **+0.0079** (5.8 σ, real), science **−0.0069** (4.1 σ,
real), ballast −0.0054 (2.3 σ), code and maths inside noise.

**Why, honestly.** Even after healing the relative reconstruction residual is 0.27. That error is
dominated by information deleted with the experts, not by the correction quality, so removing 8%
of it leaves the argmax where it already was — the tokens that were going to flip had already
flipped. The continuous metric (ΔNLL) moved ~2% in the right direction; the discrete one has no
resolution at that scale.

This is the counter-lesson to §"the saliency framework predicted this" above. Per-domain retention
predicted *where* the damage lands at r = 0.942 — a strong, useful result. The reconstruction
residual did **not** predict end-to-end movement between two masks that agree on 91% of experts.
A proxy can be well-founded, measured on held-out data, and still be measuring something the
downstream metric cannot see.

**Confound, stated plainly.** Mask and healing changed together, so neither is individually
credited or blamed. Separating them is cheap because healing is a multiply on F32 block scales and
therefore invertible: a pass-2-mask + scalar-healing arm costs one re-heal (~15 min) and one eval
(~80 min). That ablation is the honest next measurement, and it is the one that would decide
whether per-expert healing earns its place or is simply harmless.

**One regression worth tracking.** Tap drift at the DFlash2 tap layers moved the wrong way at the
deep taps — layer 33 +0.0092, layer 42 +0.0166, layer 14 −0.0037. Small, but those are exactly the
features the drafter consumes, so it is the one number here that bears on downstream speculative
decoding rather than on generation quality.
