# The counterfactual: what a REAP of this model looks like without the hardware bound

**Status: OPEN — accumulating evidence.** Finalised after P13/P14, when pass 2 has produced the
first evaluated artifact. Entries are added as they are *measured*, not recalled.

**Scope.** What changes if the single-Thor constraint is lifted — one modest GPU node, hours not
weeks, tens-to-low-hundreds of dollars, not thousands. Explicitly **not** in scope: better data or
better REAP theory. Our corpus (48.3M permissive tokens across 7 domains) and our criterion
(faithful REAP, verified against the model source) are already good, and would be re-used verbatim.

## The distinction that makes this document useful

Every limitation this project hit falls into one of three classes, and only class **A** is what
the counterfactual is about. Keeping them apart is the whole value here — it is easy to blame
hardware for something a config field caused.

| class | meaning | a GPU cluster helps? |
|---|---|---|
| **A — hardware-bound** | blocked by 122 GiB unified memory, one device, or unified-memory pathology | **yes, directly** |
| **B — software/runtime-bound** | blocked by a config field, a missing kernel, or an unported feature | **no** — same wall on 8×H100 |
| **C — settled** | measured and closed; not a limitation | no, and none needed |

---

## A — hardware-bound. These are the counterfactual.

| # | blocked capability | what bounds it on Thor | what it would buy | evidence |
|---|---|---|---|---|
| A1 | **Teacher generation of any kind** | 328 GB ÷ 3.0 GB/s ≈ **110 s per decoded token**; the model cannot be held at all | self-generated calibration, difficulty filtering, on-policy data — the entire family every drafter paper calls mandatory | `[MEAS]` pass 1 |
| A2 | **Layer-local distillation healing** | needs a backward pass through a 165B student; even holding it for forward is streaming-only | replaces the first-moment block-scale scalar with actual repair. Plausibly the **largest single quality unlock** | `[MEAS]` pass 1 |
| A3 | **Expert merging (REAM/EEP)** | every merge forces a full FP8 128×128 requant; ~2 materialisation passes + 10–20 h dev | REAM is C5-compatible and published *at exactly 50%* — a cost objection, never a validity one | `[EXT]` REAP paper |
| A4 | **Generative evaluation** | one Thor, 17.9 tok/s AR wall; n>1 sampling on AIME/HumanEval/BFCL-live is out of reach | measures the capabilities actually claimed (agentic, coding) rather than teacher-forced ΔNLL proxies | `[MEAS]` |
| A5 | **Parallel ratio/criterion arms** | each materialisation is ~1.3 h **serial**, and disk forces deleting one arm before building the next | 40/50/60% and 3–4 criteria evaluated **concurrently**, decided on measurement | `[MEAS]` pass 1 |
| A6 | **Drafter retraining at scale** | regeneration needs the target *served*; capture is prefill-bound | DFlash 2 retrained against the REAP, the 2.42× decode win | see `DRAFTER_PLAN.md` |
| A7 | **Calibration budget** | 5.5M tokens ≈ 13 h; the full 48.3M corpus is ~4.7 days | **possibly nothing.** P6's split-half gate decides this, and it may show 5.5M was already sufficient | *pending P6* |

A7 is listed deliberately as a **candidate non-unlock**. The honest version of this document has to
include the things a cluster would *not* have fixed, or it is just a wish list.

## B — NOT hardware. A cluster changes nothing here.

| # | limitation | actual cause |
|---|---|---|
| B1 | **Non-uniform per-layer expert allocation** | `num_local_experts` is a **single scalar** read by both `Glm5NextTextExperts` and the router. Measured *better* (worst-layer retained mass 0.491 → 0.649) and **still unloadable**. Needs a runtime patch, not GPUs. |
| B2 | **16k-token calibration** | `index_topk=2048` means DSA already runs dense at 2048; needs sparse-DSA + chunkwise KDA kernels |
| B3 | **MTP / DFlash serving** | needs vLLM or SGLang support on the target arch; a porting problem |
| B4 | **GGUF export** | `llama.cpp` has no `glm5_next` converter — KDA, mHC and DSA are all novel |
| B5 | **The mixture acting in tokens, not samples** | a corpus-spec bug, fixed offline for free by per-bucket accumulators |
| B6 | **The multimodal corpus loss** | a resume counter trusted over the filesystem |

B1 is the sharpest example: the most tempting "if only we had more compute" item on the list is
not a compute problem at all, and we already paid ~1.5 h to learn that.

## C — settled, and would be settled identically anywhere

- REAP saliency capture is **faithful** to this architecture — the gate excludes
  `e_score_correction_bias`; verified in the installed model source.
- `n_group == 1` → grouped-topk is an identity → post-prune routing is exactly replayable offline.
- `norm_topk_prob` conserves gate mass under pruning (2.5000 → 2.5000).
- Expert-granular selectivity ×1.286 vs block-granular ×1.005 — intra-expert pruning is dead on
  measurement, not on cost.

## Pareto sketch — to be priced once pass 2 lands

The shape of the argument, not yet the numbers: a **single 8-GPU node for 1–2 days** holds the
321B FP8 (306 GiB) with room for activations, which is the one threshold that unlocks A1, A2, A4
and A5 *simultaneously* — they are all downstream of "can hold the model and run a backward pass".
That is the knee of the curve. Beyond it, more GPUs buy wall-clock, not capability.

To fill in after P13: the measured ΔNLL of pass 2, so the counterfactual's headline claim
(distillation healing beats a first-moment scalar by X) is stated against a real baseline rather
than asserted.

## Running evidence log

- `[2026-08-27]` A5: pass-1 materialisation measured at surgery 0.29 h + heal 0.33 h + quantize
  0.60 h = **1.3 h/arm**, strictly serial, with the FP8 master needing 157 GiB resident throughout.
- `[2026-08-27]` B5: realised token mixture vs intended sample mixture — code **5.7%** against a
  21% quota; fixed offline, no compute required.
- `[2026-08-27]` B6: `mm_done` claimed 1802 samples, disk held 128; pass 1 calibrated vision on
  **39** samples. A cluster would have calibrated vision on 39 samples just as happily.
