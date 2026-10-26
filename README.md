# glm-5.3-reap

GLM-5.3-Flash (321B parameters, mixture of experts, natively multimodal) pruned to 144 of 288 routed experts per
layer, then quantized to NVFP4 and GGUF so it runs on one 128 GB NVIDIA Jetson AGX Thor. The entire pipeline ran on
that single Thor, with no cloud and no second machine. The checkpoint never fits in memory, so every stage streams
one decoder layer at a time and performs the prune as tensor surgery on the safetensors files.

## Status (2026-10-26)

Research release. The recipe was carried forward, with pre-registered gates, into
[deepseek-v4.1-flash-reap-thor](https://github.com/patrickbdevaney/deepseek-v4.1-flash-reap-thor). The weights are
usable for evaluation and local serving. They are not a drop-in replacement for the unpruned model: top-1 agreement
with it is 0.84 on held-out text, and general-knowledge text drops to 0.58.

**Use v2.** Three selection passes shipped. Each is a different expert mask, and none overwrote another.

| Pass | Selection | Healing | Where | End-to-end evidence |
|---|---|---|---|---|
| **v2 (recommended)** | Refined REAP mask | Per-layer scalar, measured by router replay | Root of each weight repo | top-1 **0.8425**, FP8 (241,516 tokens) |
| v1 | REAP scalar criterion | Per-layer scalar | `GLM-5.3-Flash-REAP50-FP8` / `-NVFP4` (no `-v2`) | top-1 0.8370, FP8 |
| v3 | HOPE (interaction-aware), `protect_frac = 0.04` | First-moment fallback (see below) | `pass3-hope/` subdirectory | **None.** Never evaluated end to end |

> **Correction (2026-10-24).** Earlier versions of this README and of the model cards described v3 as per-expert
> healed. That was wrong, for three reasons:
> - The heal stage's keep-set guard detected that the measured gains had been fitted to the v2 keep-set. A broad
>   `except Exception` downgraded that to a warning, so v3 shipped with a first-moment per-layer fallback (median
>   gain 0.766) and no per-expert tensors.
> - v3 has never been scored end to end, so nothing shows that it beats v2.
> - The v3 NVFP4 also quantizes the linear-attention (KDA) q/k/v projections. That mode measured −0.0023 top-1
>   (z = 5.8) on v2 and had been deliberately left out of the v2 release.
>
> The planned fix is to re-heal v3 with a guard that fails instead of warning, then run a paired v3-vs-v2 evaluation.

| Artifact | Repository | Size |
|---|---|---|
| FP8 master | [`GLM-5.3-Flash-REAP50-FP8-v2`](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-FP8-v2) | 160.6 GiB |
| NVFP4 (compressed-tensors, vLLM) | [`GLM-5.3-Flash-REAP50-NVFP4-v2`](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-NVFP4-v2) | v2 98.2 GiB, v3 93.6 GiB |
| GGUF (from the v2 FP8) + vision `mmproj` | [`GLM-5.3-Flash-REAP50-GGUF`](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-GGUF) | IQ3_M to Q4_K_M |
| CUDA server for the NVFP4 build | [`glm-5.3-flash-cuda-server`](https://github.com/patrickbdevaney/glm-5.3-flash-cuda-server) | — |

The `-v2` suffix in the repository names dates from the pass-2 release and is now only a name. The subdirectory
identifies the pass.

## Results

Teacher-forced, paired evaluation against the unpruned model on identical inputs. It uses **241,516 held-out
tokens** that calibration never saw, with image-placeholder positions excluded. Top-1 is agreement with the teacher's
argmax. KL is over the teacher's top-k. Paired differences use McNemar tests on the same tokens.

| Checkpoint | top-1 | top-k KL | ΔNLL mean / median |
|---|---|---|---|
| **v2 FP8** | **0.8425** | **0.650** | 0.176 / 0.001 |
| v2 NVFP4 (released) | 0.8386 | 0.673 | 0.184 / 0.002 |
| v2 NVFP4 with KDA attention quantized (the v3 NVFP4 mode) | 0.8363 | 0.686 | 0.191 / 0.002 |
| v1 FP8 | 0.8370 | 0.695 | 0.198 / 0.002 |
| v2 mask with per-expert healing (reverted) | 0.8369 | 0.694 | 0.194 / 0.001 |

Per domain (v2 FP8):

| Domain | Tokens | top-1 | ΔNLL |
|---|---|---|---|
| Math | 66,558 | 0.920 | +0.022 |
| Code | 15,821 | 0.920 | +0.052 |
| Agentic | 69,202 | 0.873 | +0.137 |
| Science | 45,034 | 0.830 | +0.125 |
| Finance | 22,215 | 0.755 | +0.181 |
| General knowledge ("ballast") | 22,154 | 0.580 | +0.989 |
| Vision | 532 | not measured (too few text tokens) | — |

- **Retention is predictable from routing statistics.** Per-domain retention computed from routing statistics,
  before any token was scored, predicts measured agreement at Pearson r = 0.942 (pass 1). The prune put its damage
  on general-knowledge text, the capability retrieval can most easily restore. Code, math and agentic behaviour,
  which retrieval cannot supply, stay at or above 0.87.
- **Long-context retrieval is intact.** On the Q4_K_M GGUF, needle-in-a-haystack retrieval succeeded at 2K, 8K and
  32K tokens and at depths 0.1, 0.5 and 0.9: 9 of 9 cases
  ([`artifacts/niah_GLM-5.3-Flash-REAP50-Q4_K_M.json`](artifacts/niah_GLM-5.3-Flash-REAP50-Q4_K_M.json)).
- **Serving speed.** The NVFP4 build reaches 11.69 tok/s decode on Thor in the companion CUDA server, with an NVFP4
  dense overlay that costs +1.56% perplexity.

### What each pass bought

**v1 → v2: two real effects of opposite sign that cancelled.** Pass 2 changed the mask and the healing method
together. The first measurement showed no change (0.8369 vs 0.8370). An ablation that isolated the two found:

| Change, other factor held fixed | Δ top-1 | Paired z |
|---|---|---|
| Pass-2 mask vs pass-1 mask (scalar healing) | **+0.0055** | 9.3 |
| Per-layer scalar vs per-expert healing (pass-2 mask) | **+0.0056** | 11.8 |

Per-expert healing reduced held-out reconstruction residual in 41 of 42 layers, yet it was worse end to end. It fits
`c_j = gate before pruning / gate after`, so it suppresses exactly the experts the post-prune router relies on most.
v2 ships the per-layer scalar ([`research/HEALING_ABLATION.md`](research/HEALING_ABLATION.md)). This is the fourth
time on this project that an end-to-end arm caught what layer-level validation missed. Reconstruction residual is not
a sufficient proxy for selection or healing.

**v2 → v3: unproven.** v3 targets two real gaps in v2.
- **Interactions.** Scalar REAP is HOPE with the off-diagonal of `F` zeroed (arXiv 2609.18916). v3 minimises `pᵀFp`
  over the same accumulators, with no extra forward passes.
- **Protecting thin domains.** v3 sets `protect_frac = 0.04`, measured at the knee of worst-domain retention against
  interaction cost.

It is calibrated on 866,678,064 routed slots. Because of the healing fallback and the missing evaluation above, its
value is a hypothesis, not a result ([`wiki/31-pass3-hope-method.md`](wiki/31-pass3-hope-method.md)).

## Method

| Fact | Value |
|---|---|
| Source checkpoint | 328.3 GB, FP8 E4M3 with 128×128 block scales (not BF16) |
| Routed experts | 311.7B parameters, 96.99% of the model |
| Host memory | 122 GiB unified (CPU and GPU share one pool) |
| Free disk during the run | ~110–160 GiB after staging the source |

The checkpoint cannot be placed in memory or offloaded with any `device_map`. Pruning in FP8 is lossless on every
retained weight, because each expert is its own tensor with its own `weight_scale_inv`.

```
s00_smoke      structural validation, no real weights
s01_source     stage 328.3 GB FP8, resumable, byte-verified
s01b_load      placement decided by arithmetic (verdict: stream)
s02_corpus     calibration corpus, permissive licences only
s03_saliency   REAP saliency by layer streaming -> per-layer, per-domain accumulators
s04_sweep      re-rank every prune ratio from cached scores; go/no-go gate
s04b_surgery   tensor-level prune; each source shard deleted once its survivors are written
s05_heal       output-scale correction folded into block scales
s06_emit       healed FP8  <-  primary deliverable
s07_quantize   NVFP4A16 via compressed-tensors
s08_document   per-tensor layout and precision map for kernel work
```

Every stage is idempotent and resumable. State lives in SQLite (`state/state.db`), and the orchestrator retries with
backoff under `systemd --user`, so it survives SSH loss.

## Engineering findings

- **Unified-memory OOM is invisible to the kernel's accounting.** A touched 2,048 MiB `cudaMalloc` charges 42 MiB to
  the calling cgroup. The OOM killer therefore never selects the real consumer, and `memory.max` cannot bound it.
  [`scripts/memguard.sh`](scripts/memguard.sh) polls `MemAvailable` at 2 Hz, drops page cache first, and kills only
  this project's own stage processes. It never kills by largest RSS, because RSS is the number shown not to reflect
  who holds the memory ([`wiki/94-memguard-killed-the-wrong-process.md`](wiki/94-memguard-killed-the-wrong-process.md)).
- **Image-placeholder positions must be excluded from every loss.** Scoring them put teacher NLL at 16.94 against
  1.00 on real text, and it inverted the sign of every headline number until 2026-08-28
  ([`wiki/97-evaluation.md`](wiki/97-evaluation.md)).
- **Attention is Kimi-Linear-style:** 34 KDA linear-attention layers interleaved 3:1 with 11 MLA+DSA layers. A KDA
  forward needs ~13 GiB of transient memory per 2,048-token sequence. That bounds calibration batch size, and a
  single 16,384-token forward needs more than 93 GiB.
- **Expert outputs are near-orthogonal and mostly token-dependent.** The mean |cos(μᵢ, μⱼ)| is 0.091, and only 3.4%
  of an expert's output energy is its mean. A fixed rescaling can do no better than one coefficient per expert.
- **Framework support.** `transformers` does not instantiate the MTP block (layer 45). The GGUF keeps it and the
  safetensors path leaves it unmodified. `llm-compressor` needs a shim for `glm5_next`
  ([`scripts/glm5_next_support.py`](scripts/glm5_next_support.py)).

## Getting started

**llama.cpp (GGUF).** Stock llama.cpp cannot load these files: GLM-5.3 uses hyper-connections, four residual streams
mixed by a Sinkhorn-normalised matrix. Use the fork branch the files were validated against:

```bash
git clone --branch glm5-next-reap50-gguf-v1 https://github.com/patrickbdevaney/llama.cpp
cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=ON && cmake --build llama.cpp/build -j --target llama-server
llama.cpp/build/bin/llama-server -m GLM-5.3-Flash-REAP50-Q4_K_M.gguf --port 8080
```

The GGUF model card lists the required flags, memory limits on 128 GB machines, the vision `mmproj`, and the MTP
draft model.

**Rebuilding the pipeline.** Requires a Jetson AGX Thor or a comparable ≥ 122 GiB unified-memory system, plus about
500 GB of disk (the 328 GB source plus working space).

```bash
bash scripts/build_env.sh                               # torch cu130, transformers 5.16.1, llm-compressor@main
systemctl --user enable --now glm53-memguard.service    # required, see Engineering findings
systemctl --user enable --now glm53-reap.service
.venv/bin/python scripts/status.py                      # one-screen view, safe at any time
```

## Compatibility

| Runtime | State |
|---|---|
| llama.cpp fork `glm5-next-reap50-gguf-v1` on Jetson AGX Thor | Tested: text, vision `mmproj`, MTP draft |
| Companion CUDA server (NVFP4) on Jetson AGX Thor | Tested: 11.69 tok/s decode |
| vLLM with compressed-tensors NVFP4 | The format targets it; not tested in this repository |
| Upstream llama.cpp, other runtimes | Not supported (no hyper-connection operator) |

## Limitations

- **General knowledge** drops to 0.58 top-1, the largest loss of any domain. This is by design: it is the
  capability retrieval can repair.
- **Vision is unmeasured,** not measured and fine. The held-out image-text records carry ~19 text tokens each against
  ~3,450 image placeholders.
- **No biology coverage in v3.** Every biology source was excluded by licence or clinical scope, so the bio domain
  saw 0 of 866,678,064 routed slots.
- **v3 long-context assurance is at 8,192 tokens,** not the intended 16,384, because one 16,384-token forward exceeds
  memory ([`wiki/99-long-context-assurance.md`](wiki/99-long-context-assurance.md)).
- **Evaluation scope.** All quality numbers are agreement with the teacher, not downstream task benchmarks.
  Agreement measures how far the student moved, not how capable it is.
- **Speculative decoding** with the native MTP head (72.3% acceptance) is a net slowdown on both the GGUF and the
  CUDA server. The cost is the MoE batch curve, not the head
  (see the [CUDA server](https://github.com/patrickbdevaney/glm-5.3-flash-cuda-server)'s speculation section).

## Repository map

| Path | Contents |
|---|---|
| [`scripts/`](scripts/) | Orchestrator (`pipeline.py`), one module per stage (`stages/`), `memguard.sh`, `status.py` |
| [`wiki/`](wiki/) | Append-only knowledge base. Claims are tagged `[EST]` established, `[VEN]` vendor claim, `[EXT]` our extrapolation, `[OPEN]` no source exists; corrections are appended, never overwritten. `00-log.md` is the running record |
| [`research/`](research/) | Phase-0 findings, tensor inventory, pass-2 plan and findings, and the negative results; `HEALING_ABLATION.md` is the one to read |
| [`artifacts/`](artifacts/) | Evaluation JSONs, paired tests, masks, criterion shoot-out, NIAH results |
| [`gguf_README.md`](gguf_README.md) | Source of the GGUF model card |
| [`PLAN.md`](PLAN.md), [`STATE.md`](STATE.md) | Implementation plan with revisions; pipeline state |
| [`systemd/`](systemd/) | The two user services |
| [`reports/`](reports/), [`research_notes/`](research_notes/) | The 1T-class compression research that led to the DeepSeek-V4.1 project |

Not included: weights, calibration corpora, and anything under `corpus/`, `output/`, `source/` or
`artifacts/saliency/`.

## License

- **Pipeline code:** MIT ([`LICENSE`](LICENSE)).
- **Weights:** MIT, inherited from [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash).
- **Calibration corpus:** permissive licences only, by design, so that the derivative can stay MIT.

## Citation

```bibtex
@misc{devaney2026glm53reap,
  title  = {glm-5.3-reap: GLM-5.3-Flash pruned to 144 experts for a single Jetson AGX Thor},
  author = {Devaney, Patrick},
  year   = {2026},
  url    = {https://github.com/patrickbdevaney/glm-5.3-reap}
}
```

This work builds on REAP (router-weighted expert activation pruning, Lasby et al., 2025, arXiv:2510.13999), HOPE
(arXiv:2609.18916), Z.ai's GLM-5.3-Flash, llm-compressor, and llama.cpp.
