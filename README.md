# glm-5.3-reap

End-to-end pipeline that takes **`zai-org/GLM-5.3-Flash`** (321B params, MoE, natively
multimodal) from the published weights to a **REAP-pruned FP8 checkpoint**, on a single
**NVIDIA Jetson AGX Thor** — 122 GiB of unified memory, no cloud, no second machine.

The output is a pruned FP8 checkpoint that can then be quantised to NVFP4 (included) or
converted to GGUF (downstream, not included).

## Weights: which version is which, and where

Three selection passes have shipped. **Each is a different mask, not a re-run of the same one.**
Older weights are never overwritten — a new pass lands in its own subdirectory.

| | method | healing | where |
|---|---|---|---|
| **v1** | REAP scalar criterion | scalar | `*-FP8-v2` / `*-NVFP4-v2` — superseded, see note |
| **v2** | refined REAP mask | **per-expert** | repo **root** of each weight repo |
| **v3** | **HOPE** — interaction-aware, `protect_frac=0.04` | per-expert | subdir **`pass3-hope/`** |

| artifact | repo |
|---|---|
| FP8 (160.6 GiB) | [`patrickbdevaney/GLM-5.3-Flash-REAP50-FP8-v2`](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-FP8-v2) |
| NVFP4 (93.6 GiB) | [`patrickbdevaney/GLM-5.3-Flash-REAP50-NVFP4-v2`](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-NVFP4-v2) |
| GGUF | [`patrickbdevaney/GLM-5.3-Flash-REAP50-GGUF`](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-GGUF) |

> The repo names carry a `-v2` suffix from the pass-2 release and are now just names. The
> **subdirectory** tells you the version: root = v2, `pass3-hope/` = v3.

All three keep **144 of 288 experts** (REAP50). What differs is *which* 144.

### What each pass actually bought — measured, not claimed

**v1 → v2.** A better mask on its own terms: reconstruction residual 0.2730 → 0.2663 (−2.5%),
winning in 30 of 42 layers, with the two masks agreeing on 91.0% of experts. End to end, against
241,516 held-out tokens and the same cached teacher, it was **zero to within noise**:

| metric | v1 | v2 | Δ |
|---|---|---|---|
| top-1 agreement | 0.8370 | 0.8369 | **−0.0001** (SE 0.00075) |
| ΔNLL mean | 0.1979 | 0.1940 | −0.0039 (−2.0%) |
| top-k KL | 0.6948 | 0.6939 | −0.0009 |

Per domain it was a *redistribution*, not a lift — agentic +0.0079 (5.8 σ), science −0.0069
(4.1 σ), ballast −0.0054 (2.3 σ). And it is **confounded**: v2 changed the mask and the healing
method together, so neither can be credited alone.

The honest lesson is about the proxy. Even after healing the relative reconstruction residual is
**0.27** — the intermediate error is dominated by information genuinely deleted with the experts,
so shaving 8% off it leaves the argmax where it already was. Tokens whose top-1 was going to flip
had already flipped.

**v2 → v3.** Two specific defects in v2, each addressed:

1. **A per-expert ranking cannot see interactions.** [arXiv 2609.18916] shows scalar REAP is
   exactly HOPE with the off-diagonal of `F` zeroed: two experts that duplicate each other are
   cheap to drop together, two that complement each other are not. v3 minimises `pᵀFp` — the
   output error a prune set actually causes — over the same accumulators, with no extra forward
   passes.
2. **Nothing protected a thin domain.** v2 ran with no domain floor and lost ballast. v3 sets
   `protect_frac = 0.04`, **measured** off this run's accumulators at the knee of worst-domain
   retention against interaction cost — not guessed, and not inherited from MiMo's 0.08, which
   sits where marginal efficiency is 0.43.

v3 is calibrated on **866,678,064 routed slots** (42 layers × 10 chunks) and heals to a median
gain of 0.766 across 42 layers with none skipped.

> **v3 has no bio coverage.** The bio domain saw **0** of 866,678,064 routed slots, because every
> bio source is excluded by licence or clinical scope (`camel-ai/biology`, `tattabio/OG`,
> `PubMedQA`). Selection scored the 7 live domains. This is a property of the corpus, not a bug,
> and no amount of recomputation changes it.

> **Long-context assurance for v3 is at 8192 tokens, not the intended 16384.** A single
> 16384-token forward needs >93 GiB — attention is quadratic in sequence length and one forward is
> indivisible, so no amount of blocking or memory reclaim helps. 8192 is still 4× the 2048 tokens
> the mask was calibrated at. See [wiki/31-pass3-hope-method.md](wiki/31-pass3-hope-method.md).

## Why this is not a normal compression job

| | |
|---|---|
| Model on disk | **328.3 GB** (FP8 E4M3, 128×128 block scales — *not* BF16) |
| Routed experts | **311.7B params = 96.99% of the model** |
| Host memory | 122 GiB unified (CPU and GPU share one pool) |
| Free disk | ~110–160 GiB after staging the source |

The model **cannot be placed** — not in RAM, not via disk offload, not by any
`device_map`. So the pipeline never loads it. It streams one decoder layer at a time from
mmap'd shards, and performs the prune as tensor surgery on safetensors directly.

## Pipeline

```
s00_smoke      structural validation, no real weights
s01_source     stage 328.3 GB FP8, resumable, byte-verified
s01b_load      decide placement strategy by arithmetic (verdict here: "stream")
s02_corpus     build the calibration corpus (runs in parallel with staging)
s03_saliency   REAP saliency by layer streaming -> raw accumulators per layer
s04_sweep      re-rank every prune ratio from cached scores; go/no-go gate
s04b_surgery   tensor-level prune; deletes each source shard once its survivors are written
s05_heal       first-moment output-scale correction (applied to block scales)
s06_emit       healed FP8 + adapters  <-  PRIMARY DELIVERABLE
s07_quantize   NVFP4A16 via compressed-tensors
s08_document   per-tensor layout + precision map for downstream kernel work
```

Run it:

```bash
bash scripts/build_env.sh                       # torch cu130 + transformers 5.16.1 + llm-compressor@main
systemctl --user enable --now glm53-memguard.service   # see "Memory" below - not optional
systemctl --user enable --now glm53-reap.service
.venv/bin/python scripts/status.py              # one-screen view, safe any time
```

State lives in `state/state.db` (SQLite: stages, metrics, events, kv). Every stage is
idempotent and resumable; the orchestrator retries with backoff and survives SSH loss under
`systemd --user` with lingering enabled.

## Memory: why `glm53-memguard.service` is mandatory

Measured on this box (and independently by the DSpark project, 2026-08-20): **a touched
2048 MiB `cudaMalloc` charges 42 MiB to the calling cgroup.** Tegra unified memory comes from
the driver's allocator, not the page allocator, so:

- the OOM killer scores by RSS and **never selects the real consumer** — failures leave *no*
  `oom-kill` line in dmesg
- the pages are driver-pinned: not page cache, not swappable, not reclaimable
- `memory.max` cannot bound it, and this kernel has no PSI, so `systemd-oomd` cannot run

`memguard.sh` polls `MemAvailable` at 2 Hz, drops page cache first (this workload's dominant
term is reclaimable mmap cache), and only then kills — and only ever this project's own
`run_stage.py` children. It never picks the largest RSS, because RSS is precisely the number
proven not to reflect who holds the memory.

Two thresholds were tuned **down**, against intuition: the level floor sits *below* the measured
healthy plateau (2–3 GiB available), because a floor above it kills working runs.

## Measured, against the unpruned teacher

Teacher-forced paired evaluation, **241,516 held-out tokens** the calibration never saw, scored
against the unpruned model on identical inputs. Figures below are the **pass-1** REAP-50 FP8
(the published checkpoint); pass 2 supersedes it.

| | |
|---|---|
| **Top-1 agreement** | **0.837** |
| ΔNLL (student − teacher), mean / median | +0.198 / +0.001 |
| Top-k KL (teacher ‖ student) | 0.695 |

| domain | predicted retention | measured top-1 | ΔNLL |
|---|---|---|---|
| code | 0.728 | **0.921** | +0.048 |
| math | 0.713 | **0.919** | +0.026 |
| agentic | 0.747 | **0.863** | +0.182 |
| science | 0.720 | **0.829** | +0.138 |
| finance | 0.651 | **0.741** | +0.220 |
| general / ballast | 0.487 | **0.572** | +1.021 |
| vision | 0.682 | *532 tokens — unmeasured* | |

**Pass 2 measures 0.8425 — `+0.0055` over pass 1, 9.3σ paired** — and that number took two corrections
to arrive at. Pass 2 changed the mask *and* the healing method (per-layer scalar → per-expert
least-squares vector) together, and first measured 0.8369: a wash. An ablation isolating the two
`[MEAS 2026-08-29]` found they were two real effects of opposite sign that had cancelled:

| | mask | healing | top-1 |
|---|---|---|---|
| pass 1 (published) | pass 1 | scalar | 0.8370 |
| **pass 2 (ships)** | **pass 2** | **scalar** | **0.8425** |
| pass 2, as first built | pass 2 | per-expert | 0.8369 |

Per-expert healing was **11.8σ worse end-to-end** (McNemar, paired on identical tokens) despite reducing held-out reconstruction residual
in 41 of 42 layers — it is `c_j = (gate before pruning)/(gate after)`, so it suppresses exactly the
experts the post-prune router leans on hardest. It has been reverted off the checkpoint; the
shipped correction is the per-layer scalar. Full analysis in `research/HEALING_ABLATION.md`, and
it is the fourth time on this project that an end-to-end arm caught something five layers of
internal validation did not.

**Per-domain retention was computed from routing statistics before any of these tokens were
scored, and predicts the measured agreement at Pearson r = 0.942.** That is the strongest
validation here of REAP itself — and it confirms empirically what the prune was designed to do:
the damage lands on generic ballast (0.572), the one capability retrieval can repair, while code,
maths and agentic behaviour — which retrieval cannot supply — hold above 0.86.

Two things this is **not**. It is teacher-forced agreement, not capability: it measures how far
the student moved, not whether it is smart. And vision is *unmeasured*, not measured-and-fine —
the held-out image-text records carry ~19 real text tokens each against ~3,450 image placeholders.
See `wiki/97-evaluation.md`, including the bug that made every one of these numbers wrong until
2026-08-28.

## Hard-won facts

- **The published checkpoint is FP8, not BF16.** The 642 GB BF16 repo elsewhere on the Hub is a
  dequantised upcast carrying no extra information. Pruning in FP8 is *lossless* on every
  retained weight — experts are per-expert tensors with their own `weight_scale_inv`.
- **The attention stack is Kimi Linear's**: 34 KDA linear-attention layers interleaved 3:1 with
  11 MLA+DSA layers. That gives REAP a published precedent on this architecture family.
- **A KDA forward costs ~13 GiB of transient memory per 2048-token sequence**, linear in batch.
  This bounds calibration batch size independently of the weights, and is a concrete argument
  for hand-written kernels in any serving work.
- **transformers does not instantiate the MTP block** (layer 45). It is excluded and left
  unmodified upstream rather than inconsistently pruned.
- **Image-placeholder positions must be excluded from any loss you compute.** Their embedding is
  replaced by an image feature before layer 0 and the training objective masks them; scoring them
  put teacher NLL at 16.94 against 1.00 on real text and inverted the sign of every headline
  evaluation number. See `wiki/97-evaluation.md`.
- **Expert outputs are near-orthogonal and mostly token-dependent**: mean |cos(μ_i,μ_j)| = 0.091,
  and only 3.4% of an expert's output energy is its mean. Together these say a *fixed* rescaling
  cannot do better than one coefficient per expert — which is what healing now fits.
- `llm-compressor` needs a shim for `glm5_next` (`scripts/glm5_next_support.py`): auto-derivation
  loses `swiglu_limit`, which `_apply_gate` reads.

## Layout

```
scripts/            pipeline.py (orchestrator), stages/, memguard.sh, status.py, guard.py
scripts/stages/     one module per stage, each with run() -> dict
wiki/               append-only knowledge base; 00-log.md is the running record
research/           FINDINGS.md (Phase 0), tensor inventory, and the negative results:
                    HEALING_ABLATION.md is the one worth reading
PLAN.md             implementation plan and its revisions
systemd/            the two user services
```

`wiki/` is the substantive artifact: every claim is tagged `[EST]` established / `[VEN]` vendor
claim / `[EXT]` our extrapolation / `[OPEN]` no source exists, and corrections are appended
rather than overwritten.

## Not included

Weights. Calibration corpora. Anything under `corpus/`, `output/`, `source/`,
`artifacts/saliency/`.

## Licence

Pipeline code: MIT. `zai-org/GLM-5.3-Flash` is MIT, and the calibration corpus is
permissive-licence-only by design so a derivative can stay MIT.
