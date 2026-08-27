# 00 — Chronological log

Append-only. Newest entries at the bottom. One entry per working session or discrete finding.

---

## 2026-08-27 — Session 1: Phase 0 deep research

**Scope:** directive §3 (deep research) and §4 (housekeeping). No weights downloaded, no
implementation started.

### Host reconnaissance
- Confirmed running **on** the Jetson AGX Thor itself (`/proc/device-tree/model`), L4T R38 rev 4.0,
  kernel 6.8.12-tegra, CUDA 13.0.48, 122 GiB RAM, sm_110a.
- **Disk: 936 G total, 592 G used, 298 G available.**
- Measured NVMe **3.4 GB/s read, 487 MB/s write**; HF download **103 MB/s single stream,
  107 MB/s across 8 parallel streams** → link-saturated, parallelism buys nothing.
- **Flywheel loop confirmed stopped:** crontab still has the hourly entry, but a
  `FLYWHEEL_STOP` sentinel makes every tick a no-op since 2026-08-26. Box is dedicated.
  `dsv4-oomsentry.service` still running (passive OOM backstop — harmless, left alone).
- `glm-5.3-reap` existed as an empty git repo with no commits. Now scaffolded.

### Model verification
- Fetched `config.json` and **all 62 safetensors headers via HTTP range requests** — full
  tensor inventory (76,108 tensors) without downloading a single weight. Saved to
  `research/glm53_tensors.json`.
- **The release is FP8, not BF16.** `quantization_config: {quant_method: fp8, fmt: e4m3,
  weight_block_size: [128,128], activation_scheme: dynamic}`. On disk **328.3 GB**, not 640 GB.
- `unsloth/GLM-5.3-Flash` (BF16, 642 GB) confirmed a **dequantised upcast** —
  `unsloth/GLM-5.3-Flash-FP8` has a dtype profile identical to `zai-org`'s.
- **Routed experts = 311,672,586,240 params = 96.99% of the model.**
- mHC = 17.7 M (0.006%). Vision tower = 563.6 M (0.18%), dense ViT, **no MoE**.
  Routers = 50.7 M. MTP block at layer index 45 with its own 288-expert MoE (7.43 B).
- **Directive's "not MLA+DSA" is wrong** — 11 full layers are MLA+DSA, interleaved 3:1 with
  34 KDA linear layers. That is Kimi Linear's stack.
- Shards are ordered **lexicographically by tensor name**, so numeric layer order ≠ shard order.

### Research
- REAP (arXiv 2510.13999, ICLR 2026): saliency `S_j = mean over active tokens of g_j·‖f_j‖₂`.
  **No data above 50%. Uniform per-layer allocation, no ablation.** Calibration for ≥110B is
  **12,228 samples @ 16,384 tokens**.
- Read `llm-compressor` `modifiers/pruning/reap/base.py` (327 lines, release 0.13.0):
  `REAPPruningModifier` introspects MoE structure generically via `get_moe_attrs`, prunes a
  uniform fraction per layer, validates `top_k` reachability, warns against
  `moe_calibrate_all_experts`.
- **Found `cerebras/Kimi-Linear-REAP-35B-A3B-Instruct`** — Cerebras's own REAP-30 on a
  KDA 3:1 hybrid. Near-lossless; **LongBench v2 flat**; worst regression **FRAMES −3.4**.
  This is the single most important find of the session.
- Traced the Nemotron3 hybrid-fragility claim to arXiv 2607.01444 (biomedical QA) —
  the variable is **expert-pool size**, not hybridness.
- Verified calibration dataset availability against the HF API:
  `nvidia/Nemotron-CC-Math` is **gated (401)**; `lmms-lab/DocVQA` 307-redirects; all others resolve.

### Decisions
- FP8-native throughout (operator-confirmed mid-session).
- Stream the source by **layer-ordered range requests**; never stage 328 GB.
- Recommend **dropping RLVR** — ~460 days single-stream at this model size.
- **Fully fine-tune mHC + routers** (68.4 M combined) rather than LoRA them.

### Open at end of session
- Operator sign-off on: RLVR removal, the zero-risk 52 GiB reclaim, corpus size increase to 12,288.
- `glm5_next` × llm-compressor smoke test (R4) — highest-value cheap early check.
- Finance/quant calibration source (R9).

---
