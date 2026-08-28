# STATE — terse recovery point

Updated 2026-08-28 01:45 EDT. Read this first after any context loss.

## Where we are

**Pass 1 is COMPLETE and published.** Pass 2 (the optimised REAP) is in stage P1.

| artifact | where | status |
|---|---|---|
| FP8 master, 157 GiB, 62 shards | `output/glm-5.3-flash-reap50-fp8` + HF `…-REAP50-FP8` | **RE-HEALED 2026-08-28** with the measured gain (6,048 scales x median 1.3121) and re-uploaded. Local copy deletes only after **P9.5** has scored it — it is the A/B baseline. |
| NVFP4, 96 GiB, 62 shards | HF `patrickbdevaney/GLM-5.3-Flash-REAP50-NVFP4` | verified byte-for-byte, **local copy deleted** to make room for the source |
| pipeline infra | `github.com/patrickbdevaney/glm-5.3-reap` | ~95 commits pushed |

## Running right now

**`s03_saliency`, pass 2** — detached, `logs/p2_saliency.log`. 10 chunks x 0.5M tokens = 5.5M.
Measured ~109 min/chunk, so **~18 h**, finishing late 2026-08-28. Resumable per chunk
(`state/s03_chunks.json` + `SS.load_accumulators`); a kill costs one chunk.

Calibration inputs, vs pass 1: **2,280 text + 406 image-text** (pass 1: 217 + 39).

- `glm53-memguard` user service: **active**.
- `glm53-reap` user service: **stopped, still enabled**. It has `Restart=always,RestartSec=60` and
  was relaunching the finished pass-1 graph every minute. Start it only when the pass-2 graph is
  ready; `s04b_surgery` DELETES source shards, so it must never run behind an unreviewed sweep.

Source: 62/62 shards, byte-exact (328,337,455,672). Disk 79 GiB free.

## Pass-2 plan

`research/PASS2_PLAN.md` — P0–P14, ~38–55 Thor-h. `research/PASS2_FINDINGS.md` has the corrections
(two of the research's four gates were stale; it inspected the box mid-`s07`).

Done: **P0** disk, **P1** re-stage, **P2** instrumented saliency, **P3** chunked sweep.
In flight: **P4** the sweep.

Written and tested, waiting on data:
- `scripts/heal_refit.py` (**P5**) — run after **chunk 1**, not chunk 10. Reproduces the shipped
  0.6964 to within 0.9%, so the measured number will be trustworthy.
- `scripts/split_half.py` (**P6**) — gate on >0.95 keep-set overlap.
- `scripts/criterion_shootout.py` (**P7**) — 8 criteria offline; 2 on pass-1 dumps, 8 on pass-2.
- `scripts/stages/s09_eval.py` (**P12**) — paired teacher/student, per-domain dNLL, flips,
  top-k KL, and drift at the DFlash tap layers. 13 tests pass.
- `research/DRAFTER_PLAN.md`, `research/CLOUD_COUNTERFACTUAL.md` — deliverables in progress.

## Hard-won facts (do not rediscover)

- `n_group == 1`, `topk_group == 1` → the grouped-topk path is an identity. Post-prune routing is
  plain top-8 over survivors, so it is exactly replayable offline (`scripts/router_replay.py`).
- The gate is `scores.gather(...)`, which **excludes** `e_score_correction_bias`; the bias is
  selection-only. Pass 1's saliency capture is faithful. Verified in the installed model source.
- `ffn_hc` collapses the `hc_mult` axis **before** `self.mlp`, so the MLP sees `[B, S, H]` and the
  valid-mask is 1:1 with expert rows. A mismatch now raises instead of misaligning silently.
- `norm_topk_prob` conserves gate mass under pruning (measured: pre 2.5000, post 2.5000). This is
  why the pass-1 healing gain (0.696) is `[OPEN]` — directionally right, magnitude unverified.
- `num_local_experts` is a **scalar** → non-uniform per-layer allocation is unloadable. Settled.
- Do not run long work as a child of a Claude Code Bash call. `setsid`, always.
- `pkill -f pipeline.py` matches the calling shell's own command line and kills the session. Use
  `pgrep -f 'pipel[i]ne\.py'`.

## Traps found, fixed, still worth knowing

- **`mm_done` kv said 1802 multimodal samples; the disk had 128.** The rest were deleted by a
  later reclaim. `collect_multimodal` now counts files and ignores the counter, and indexes past
  the highest surviving shard so a top-up cannot overwrite one. Pass 1 calibrated vision on **39**
  samples. This is R3, the named escalation risk.
- Pass 1 accumulated **padding tokens** as real tokens — worst for the shortest documents. Fixed.
- Pass 1 discarded the domain bucket, so the mixture could not be re-weighted without a new pass.
  Now carried through to per-bucket accumulators.
- `s07` dequantises 53 non-expert FP8 tensors to BF16 (+1.45 GiB, quality-neutral). Queued as a
  pass-2 change; see `wiki/60-quantization.md`.

## Pass-2 config

`calib_tokens=5.5M`, `calib_chunk_tokens=0.5M` → **12 chunks**, 2,686 samples, `MAX_LEN=2048`,
`BATCH=2`. Pass 1 measured 83.2 min for 256 samples → **~14.5 h** projected, +~18 min of extra
weight re-reads for chunking. A kill costs one chunk (`state/s03_chunks.json` + `load_accumulators`).

Text buckets all have headroom. Multimodal wants 403 and has 128 → the top-up is why it matters.

## Tests

`.venv/bin/python tests/test_pass2_saliency.py` — 18 checks: exact agreement with the real router,
a naive per-token reference for every accumulator, and a bit-exact resume round-trip.
