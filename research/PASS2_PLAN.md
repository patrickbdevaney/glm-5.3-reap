# Pass 2 — implementation plan

Derived from [`PASS2_FINDINGS.md`](PASS2_FINDINGS.md) (corrections) and the raw synthesis §5.
Sequenced so **every hour of compute is preceded by the cheap check that could invalidate it.**

Pass 1's defect was never the criterion — the saliency capture was verified faithful. It was that
we decided 12,096 expert slots on **1.1% of the corpus we built** and then never measured the
result. Pass 2 fixes sampling and adds measurement, in that order of confidence.

## Budget

~38–55 Thor-h critical path. Stop-points are real; if the budget dies, stopping after P6 still
yields a strictly better artifact than pass 1, and stopping after P13 yields **the first evaluated
REAP this project has produced** — which is the actual point.

## Disk plan `[MEAS 2026-08-27 21:50]`

The synthesis said "delete both output trees." That was written against a stale view. Corrected:

| tree | size | status | disposition |
|---|---|---|---|
| `output/…-fp8` | 157 GiB | **verified on HF**: 62 shards, 157.0 GiB, config+index | delete local **before P12**, not now |
| `output/…-nvfp4` | 96 GiB | uploading | delete local **after upload verifies**; reproducible from FP8 in 0.6 h |
| `offload/` | 17 GiB | dead accelerate scratch | **deleted** ✔ |
| docker | 30.7 GB | pre-approved prune | **reclaimed** ✔ |
| `source/GLM-5.3-Flash` | 9 files | intact metadata | **keep** — lets re-download skip work |

Free: 190 → **302 GiB**. Source needs ~306 GiB, so NVFP4's 96 GiB must come out before P1
completes. Both deletions are **escalations** under the directive (destructive disk ops on model
weights) and are queued for approval, not taken unilaterally. Both artifacts are published and
reproducible; neither is a one-way door.

## Stages

Numbered P0–P14. **CP** = critical path.

| # | Stage | What | h | CP |
|---|---|---|---|---|
| P0 | Verify + free | ✔ offload, ✔ docker; queue NVFP4 + FP8 local deletion for approval | 0.2 | **CP** |
| P1 | Re-stage source | 62 shards, detached, per-file size-compare retry loop (the pass-1 fix) | 1.5–3 | **CP** |
| P2 | Instrument `stream_saliency.py` | 6 accumulators + log-histogram + top-40 fp16 logit cache, replicated per domain-bucket and per modality | 0.5 dev | **CP** |
| P3 | Chunk `s03` | 0.5M tokens/chunk (~21 GiB activations), 45 layers/chunk, resumable accumulation | 0.5 dev | **CP** |
| P4 | **Saliency pass, 5.5M tokens** | 11 chunks × ~82 min, detached, watch the 61 GiB headroom | 14–17 | **CP** |
| P5 | **G0-2: re-fit the healing gain** | `scripts/heal_refit.py` - measured per-layer output-norm ratio vs the 0.696 first-moment value. **Runs after CHUNK 1, not after all 12**: if the correction is wrong that is worth knowing 13 h early. Harness validated - it reproduces the shipped 0.6964 to within 0.9%. | 0.5 | **CP** |
| P6 | Split-half overlap gate | keep-set overlap at 50%; **gate >0.95** or the token budget was too small | 0.2 | **CP** |
| P7 | Criterion shootout | 8 offline masks from the accumulators + pairwise overlap; pick 2. Includes **arm B (quantile-blended 0.6*mean + 0.4*p99)**, which pass 1 had to defer because the tracker kept only sum and count - the log-histogram accumulator now makes it available without a new pass. | 1 | high-value |
| P8 | AIMER cross-check | calibration-free second opinion, streamed | 0.5–1 | opt |
| P9 | Staged greedy re-scoring | R=8–16, router-aware, needs P2's logit cache | 1–2 | opt |
| P10 | Materialise mask A | surgery 0.29 + heal 0.33 + quantize 0.60 | 1.3 | **CP** |
| P11 | Materialise mask B | serially, deleting between arms | 1.3–2.6 | opt |
| P12 | Paired eval harness | lockstep teacher/student streamed forward, B=4, teacher-forced only | 4–8 dev | **CP** |
| P13 | **Eval pass** | held-out per-domain ΔNLL + flip/KL + PopQA + multimodal slice + BFCL non-live AST | 6–15 | **CP** |
| P14 | Decide + ship | ratio, criterion, healing gain → final FP8 + NVFP4 → HF | 2 | **CP** |

## The three things that actually change the outcome

1. **P4 — 10× the calibration tokens.** 501/12,096 slots were decided on <2,000 tokens and ~25.5
   experts/layer sit within ±5% of the cut. This is where the mask is genuinely undetermined.
2. **P5 — the healing gain.** Currently a first-moment derivation applied to 6,048 experts'
   block scales, directionally right but of unverified magnitude, on top of a renormalization
   the router already performs. Measure it instead of deriving it.
3. **P13 — any evaluation at all.** Pass 1 shipped unmeasured. If the budget forces a choice
   between more tokens and any eval, **choose the eval.**

## Ruled out — do not re-litigate

MTP block (closed, correctly excluded) · non-uniform allocation (C5, expert count is one scalar) ·
expert merging (needs full FP8 requant per merge) · intra-expert pruning (measured selectivity
×1.005 vs ×1.286) · router KD healing (backward through a 165B student) · self-generated
calibration or **any** teacher generation (~110 s/token) · re-fitting `e_score_correction_bias` ·
16k sequences (DSA is already dense at 2048) · `device_map="auto"` anywhere (this is what killed
s07) · running anything as a child of a Claude Code Bash call (`setsid`, always).

## Ordering rationale

P5 and P6 are ~0.7 h combined and sit **before** every materialisation. If the split-half gate
fails, more tokens are needed and materialising anything is premature. If the healing re-fit
disagrees with 0.696, every downstream artifact would inherit the error — which is precisely how
pass 1 shipped. Cheap checks first, in front of the expensive irreversible ones.
