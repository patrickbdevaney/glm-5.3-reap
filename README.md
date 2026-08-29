# glm-5.3-reap

End-to-end pipeline that takes **`zai-org/GLM-5.3-Flash`** (321B params, MoE, natively
multimodal) from the published weights to a **REAP-pruned FP8 checkpoint**, on a single
**NVIDIA Jetson AGX Thor** — 122 GiB of unified memory, no cloud, no second machine.

The output is a pruned FP8 checkpoint that can then be quantised to NVFP4 (included) or
converted to GGUF (downstream, not included).

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
research/           FINDINGS.md (Phase 0) + full tensor inventory of the source
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
