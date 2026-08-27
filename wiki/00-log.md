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

## 2026-08-27 — Session 1 addendum: operator decisions + disk state change

### Operator decisions (all as recommended)
1. **RLVR dropped** from this run; long-horizon coherence pursued via the 22% agentic
   calibration share + LoRA SFT. Documented as a future option. → directive §6.7 amended.
2. **Primary deliverable = healed FP8 base + separately-stored adapters**, merged at NVFP4
   quantisation time. No BF16 export stage. → directive §6.8 amended. Saves ~306 GiB.
3. Zero-risk reclaim approved (docker dangling layers + `thor-vllm-cache`).

### Disk state changed outside this session
Between the Phase 0 audit and the reclaim step, free space moved **298 GiB → 453 GiB**
(used 592 G → 436 G). **Not done by this session.** Observed changes:

| Path | Before | After |
|---|---:|---:|
| `model-backups` | 62 G | **removed** |
| `laguna-s1-cuda-server/models` | 70 G | 444 M |
| `thor-vllm-cache` | 31 G | 25 G |

### Reclaim outcome — approval partially not exercised, deliberately
- `docker image prune` → **reclaimed 0 B.** The 21 GiB that `docker system df` reports as
  RECLAIMABLE is **unreferenced *tagged* images**, not dangling layers; freeing it requires
  `docker image prune -a`, which would delete **all five** images (incl. the 68 G
  `vllm-dflash-thor:dllm` and 50 G jetson-thor vLLM). That is not zero-risk and was **not**
  done. `[EST]` — corrects the "+21 GiB zero-risk" figure in [20](20-host-thor.md)/[90](90-open-risks.md).
- **`thor-vllm-cache` (25 G) deliberately NOT deleted.** The approval was granted to buy
  margin; with 453 GiB free the margin is no longer needed, and deleting a regenerable
  compile cache costs recompilation time for zero benefit. Trivially reversible if wanted.

### Consequence for the plan
Peak requirement was **~251 GiB**; free is now **453 GiB** — margin ~202 GiB.

**The streaming design is retained anyway**, and not out of caution: it is strictly better
than staging. It costs ~52 min/pass of download that would otherwise be ~52 min/pass of
download *plus* 306 GiB of disk, and it keeps peak footprint at ~15 GB during the saliency
pass. Staging the source would buy nothing and cost 306 GiB. `[EXT]`

The extra headroom is instead spent on things that were previously unaffordable: keeping the
pruned FP8 checkpoint resident *alongside* the NVFP4 output for validation, and retaining
sweep artifacts at more than one prune ratio simultaneously.

---

## 2026-08-27 — Session 2: build-out and autonomous launch

**Operator instruction:** run the whole thing end-to-end without hand-back, surviving SSH
collapse, preserving work at each step and publishing intermediates to HF.

### Environment
Built `.venv` (uv). Resolved the transformers/llmcompressor conflict by installing
llm-compressor from **git main** (`0.13.1.dev51`) — the pypi release pins
`transformers<=5.14.1`, and `glm5_next` exists only at **≥5.16.1** (verified against the
GitHub tree at four tags). torch **2.13.0+cu130**, `arch_list` includes **sm_110**, reports
`NVIDIA Thor`. torchvision silently upgraded torch 2.10→2.13; re-validated everything and
pinned both.

### Power
Box was in **120 W mode with no clock lock**. Applied MAXN + `jetson_clocks`:
GPU ceiling 1386→**1575 MHz**, **EMC 2750→4266 MHz**. The memory controller was at 64% of
peak on a bandwidth-bound workload. Reversible.

### Stage 0 — PASSED
Found and fixed three real `glm5_next` × llm-compressor gaps, and discovered the MTP block is
invisible to transformers (313.89B meta params vs 321.34B in the checkpoint — the gap is
exactly MTP's 7.45B). Measured ~1,994 tok/s full-model estimate. Verified the processor emits
**256 image tokens** for a 448×448 image, so the R3 assertion holds. See
[96-implementation.md](96-implementation.md).

### Pipeline
Nine stages under `systemd --user` (`Linger=yes`), SQLite state, per-stage retry, resume across
kills, HF publishing. Launched; source staging in progress.

### The residency problem
314 GB of FP8 against 122 GiB RAM, with only ~170 GB of disk left once the source is staged —
RAM, unified VRAM and spare disk are each individually insufficient. Two paths:
**A** mmap-backed CPU placement (probed by `s01b_load`), **B** custom layer streaming.

Reading `Glm5NextTextModel.forward` confirmed **B is mechanical, not a reimplementation**: the
whole inter-layer state is one tensor plus `topk_indices`. Path B was pre-written
(`scripts/stream_saliency.py`) rather than left as an overnight dead-end, and its FP8 block
dequantisation was **validated against real downloaded shards** — scale shape `(32,16)` for a
`(4096,2048)` weight, block (0,0) exact.

### Scope changes taken autonomously
- **Healing narrowed** to a first-moment correction computable from cached saliency alone
  (no teacher, no forward pass). Gradient healing of a 165B student is not tractable here.
  Marked non-critical; `s06` takes it as a soft dependency so a failed heal cannot block the
  deliverable.
- **Saliency A/B arm B deferred** — the stock tracker keeps only `sum` and `count`, so
  quantiles cannot be recovered post hoc. Recorded in the `s04` verdict rather than dropped
  silently.
- **MTP excluded and archived** rather than inconsistently pruned.

### Open at end of session
- `s01b_load` outcome decides Path A vs Path B.
- Whether the ~1,024-sample calibration budget clears the per-expert sufficiency floor
  (asserted in `s03`, not assumed).

---

## 2026-08-27 — Session 2b: the run goes autonomous, and what broke

`s01_source` DONE in **23.1 min** (62 shards, 328,337,455,672 bytes, byte-exact). 40 GB of
download staging reclaimed after verification.

### The decisive result: the model cannot be placed

`s01b_load` verdict: **`stream`**.

| | |
|---|---:|
| model | 305.8 GiB |
| RAM available | 116.2 GiB |
| free disk | 162.4 GiB |

Neither RAM placement nor disk offload can hold it. **Path B is not a contingency, it is the
path.** `[EST]`

**Near-miss worth recording.** The first version of `s01b` simply *tried*
`device_map="cpu", dtype="auto"`, on the theory that safetensors would stay mmap-backed and the
kernel would page it. It does not — transformers materialises into RAM. The probe reached
**119 GB RSS with 5 GB available** and had to be killed; an OOM there would have taken the
orchestrator and the corpus build with it. `s01b` now decides by arithmetic and never attempts
a placement it knows cannot fit.

### Five real bugs, all found by running rather than by reading

1. **Disk precheck was absolute, not remaining.** Failed a download that was 44% complete
   because it demanded 320 GiB free rather than 320 − already-downloaded.
2. **The orchestrator loop was single-threaded.** The multi-hour corpus build *blocked* the
   source download; total time became the sum rather than the max. Long stages now run as
   detached children.
3. **Difficulty-band starvation.** The band mix was treated as a hard constraint, so once a
   band filled every subsequent row in it was skipped — a source whose rows all land in one
   band scans forever. CoderForge (128K-context trajectories, uniformly "hard") produced zero
   samples for ten minutes. Bands are now a preference, relaxed after a bounded scan.
4. **JSON-string columns.** CoderForge stores an entire trajectory as a JSON *string* in
   `messages`, so `isinstance(msgs, list)` was False and the extractor returned `None` —
   **0 accepted from 35,000 scanned rows.** Fixed generically.
5. **Zombie children read as alive.** Popen children are never `wait()`ed, so on exit they
   become zombies — and `os.kill(pid, 0)` **succeeds** on a zombie, because the PID entry
   survives until reaped. The orchestrator reported a stage healthy for three minutes after it
   had died. `_pid_alive` now reads `/proc/<pid>/stat` and treats state `Z` as dead, and the
   loop reaps children each iteration.

### Streaming path built and validated

- **Checkpoint conversion mapping applied by hand.** Building layers manually means applying
  `get_checkpoint_conversion_mapping("glm5_next")` manually. Without it, `load_state_dict`
  silently missed 11 params in a KDA layer and 6 in an MoE layer — and those were **mHC**
  (`hc_attn_*` → `attn_hc.*`) and the **KDA forget gate** (`dt_bias`, `A_log`, `f_a/f_b_proj`),
  plus a `q/k/v_conv1d` fusion. Leaving them missing would have left randomly-initialised
  weights in exactly the two components most sensitive to pruning, and the saliency would have
  looked plausible while being meaningless. `[EST]`
- **Each layer is now built once, not once per batch.** Activations are sized to fit in RAM
  (512 × 2048 × hc_mult 4 × 4096 × 2 B ≈ 34 GB), so the pass costs 45 layer builds rather than
  ~8,000 — the difference between ~6 minutes and ~18 hours of weight re-reads.
- Layer build verified: KDA layer 0.54 GiB in 0.9 s, MoE layer **13.78 GiB in 6.2 s**, no
  missing params.

### Surgery designed around the disk constraint (R10)

`s04b` prunes at the **tensor level** — no model object — and **deletes each source shard once
its survivors are written**, so free space grows monotonically through the pass. Safe because
the source is public and re-downloadable in 23 min, and a completed-shard ledger makes it
resumable.

Two correctness details: routers are sliced to the retained set (a router still emitting 288
logits over 144 experts is simply wrong), and an expert with **zero routed tokens is ranked
`-inf`, not 0** — its conditional mean is *undefined*, not low, and ranking it 0 would let an
unobserved expert outrank a genuinely weak observed one.

### Corpus build: three more source-level failures

- **Config auto-correction became a hazard.** The self-healing loader substitutes `configs[0]`
  on a missing config. `nvidia/Nemotron-VLM-Dataset-v2` has **46 configs whose first
  alphabetically is `wiki_de`** — German Wikipedia *text*. It silently selected that for the
  vision bucket. Now: explicit configs for all 15 multimodal sources, and substitution only
  when *no* config was requested; an explicit config that fails to resolve raises. `[EST]`
- **Script-based datasets are dead.** `datasets` 5.x dropped loading-script support, which
  removes `ibm-research/finqa`, `dreamerdeo/finqa`, `bigcode/commitpackft` and `ncbi/pubmed`.
- **`kensho/DocFinQA` overflows Arrow's int32 offsets** on its ~123k-word contexts. Derated
  rather than dropped — it is still the only genuine long-context source in the corpus.

Bucket weights now deliberately sum to >1 so that failing sources have fallback capacity.

### Corpus state at 03:03

| bucket | have | quota |
|---|---:|---:|
| agentic | 2949 | 2949 ✅ |
| code | 2580 | 2580 ✅ |
| math | 1843 | 1843 ✅ |
| science | 1229 | 1229 ✅ |
| ballast | 860 | 860 ✅ |
| multimodal | 1104 | 1843 |
| finance | 245 | 983 |

Multimodal layout verified on disk: `pixel_values (832, 1176)` 2-D, `image_grid_thw (1, 3)`,
**208 image tokens** in the record's `input_ids`. The R3 defence is real, not nominal.

### Streaming REAP validated on real weights (03:35)

Ran two streamed layers on real checkpoint weights with a synthetic batch:

| layer | type | forward | output | finite |
|---|---|---|---|---|
| 0 | `linear_attention` (KDA) | 1.77 s | (2, 512, 4, 4096) | ✅ |
| 3 | `deepseek_sparse_attention` (MLA+DSA) + MoE | 0.53 s | (2, 512, 4, 4096) | ✅ |

**Saliency accumulated: 279/288 experts fired, 8,192 expert-token assignments
(1,024 tokens × top-8, exactly right), mean saliency 4.14.** `[EST]`

That is the end-to-end proof that the whole approach works: build a layer from mmap'd FP8
shards, dequantise, run the real module, capture `g_j · ‖f_j‖` before the gate scales it,
accumulate, free. The hc_mult=4 residual stream propagates correctly between layers.

**Throughput note.** REAP needs far fewer tokens than the corpus budget implies. The per-expert
floor of 2,000 tokens requires only `2000 × 288 / 8 ≈ 72,000` tokens *total*; the planned
512 × 2048 = 1.05M tokens gives ~29k per expert, roughly 14× the floor. If the measured
per-layer ETA proves too slow, the sample count can be cut substantially before the estimator
degrades — the stage reports both, so the decision is made from measurement.

### Corpus complete (03:35) — s03 saliency running

| bucket | collected | quota | |
|---|---:|---:|---|
| agentic | 2949 | 2949 | ✅ |
| code | 2580 | 2580 | ✅ |
| math | 1843 | 1843 | ✅ |
| science | 1229 | 1229 | ✅ |
| ballast | 860 | 860 | ✅ |
| multimodal | 1802 | 1843 | 98% |
| finance | 686 | 983 | 70% |

**48,283,959 text tokens.** Finance is the one materially short bucket, for the reasons in
[85](85-corpus-sources.md): two of its best sources are script-based (dead in `datasets` 5.x)
and DocFinQA overflows Arrow's int32 offsets. It recovered from 245 to 686 once the
replacement sources were named correctly.

`s03` drew **431 text + 81 image-text = 512** calibration samples and is streaming layers.

### Recipe bugs caught before they could cost hours

Two in `s07`, both found by constructing the objects rather than waiting for the stage:

1. **`config_groups` takes `QuantizationScheme` objects, not `{"scheme": "NVFP4"}` dicts** —
   pydantic rejects the dict form outright. `preset_name_to_scheme(name, targets=...)` is the
   correct constructor, and the resolved schemes were verified to be exactly the intended
   policy: experts 4-bit float / `tensor_group` / **group_size 16** (NVFP4's blocks) with
   dynamic local activations; attention 8-bit float / channel weights / token-dynamic. `[EST]`
2. **`device_map="cpu"` cannot work for the pruned model either** — ~156 GiB against 116 GiB of
   RAM, the same wall that forced streaming in `s03`. By that point surgery has deleted the
   source as it wrote, so ~300 GiB of disk is free and accelerate can offload. `s07` now
   measures free space and refuses with a clear message if surgery did not free it.

### s03 first timing, a self-inflicted OOM, and two sync bugs

**Layer 0 (dense): 7.4 min → ETA 324 min.** Layers 0–2 have no MoE (`first_k_dense_replace: 3`),
so that number did not yet include the 288-expert loop.

**Self-inflicted OOM.** While s03 was running with a 21.3 GiB working set, a 13.78 GiB layer-build
validation was started alongside it. The kernel OOM-killer took the *pipeline stage*, not the
ad-hoc job — `dmesg`: `Killed process 223145 (python) ... task_memcg=.../glm53-reap.service`.
Twelve minutes of a multi-hour pass lost to an entirely avoidable mistake. `scripts/guard.py`
now refuses to start ad-hoc work while `s03`/`s04b`/`s05`/`s07` are marked running. `[EST]`

**Two synchronisation bugs in the saliency hook**, both invisible in the small validation run
because they cost per *expert per batch*:

1. `for expert_idx in hit:` iterated a **CUDA** tensor in Python, which syncs on every element
   access — 288 syncs per batch.
2. The accumulator update called `.cpu()` per expert, purely to add a scalar to a host tensor —
   a full device sync for telemetry.

At 135 batches that is ~38,880 syncs each, ~78k per MoE layer, across 42 MoE layers. `hit` is
now pulled to host once and the accumulators live on the GPU until dump.

Re-validated after the fix on a real MoE layer: **3,461 tok/s**, 281/288 experts fired,
`tokens = 32,768` for 4,096 tokens × top-8 — exact.

> **Lesson worth keeping:** the sync cost was invisible at validation scale and only appeared at
> 135 batches. Timing a hot loop at realistic batch counts, not just at correctness scale, would
> have caught it before the first pass rather than during it.
