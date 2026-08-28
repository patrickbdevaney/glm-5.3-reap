# STATE — read this first after any context loss

## Running right now
| what | where | notes |
|---|---|---|
| `s07_quantize` | `logs/s07_quantize.log` | tensor-level NVFP4, streams shard-by-shard, resumable by output-file presence |
| FP8 backup | `logs/backup_fp8.log` | → public `patrickbdevaney/GLM-5.3-Flash-REAP50-FP8`, resumable, 40 attempts |
| deep research | workflow `wi8gva0n1` | 6 areas → adversarial verify → synthesis |

Check with: `.venv/bin/python scripts/status.py` and `.venv/bin/python scripts/preflight.py`

## Artifacts that exist
- `output/glm-5.3-flash-reap50-fp8` — **157 GiB, healed, verified. THE MASTER.** Everything
  else derives from it. Uploading to HF.
- `output/glm-5.3-flash-reap50-nvfp4` — being written now, ~89–96 GiB expected.
- `artifacts/saliency/*.pt` — 42 layers of REAP accumulators, 83 min of compute. **Pushed to
  git.** The one thing that would really hurt to lose.
- `corpus/shards/text/*.pt` — 48.3M tokens tokenised, survives. `corpus/shards/multimodal` —
  2 shards, 128 image-text records.
- Source `source/GLM-5.3-Flash` — **consumed by surgery**. Re-download is ~55 min if needed.

## Pass 2 (queued, not started)
Sequence agreed with the operator:
1. NVFP4 finishes → upload both FP8 and NVFP4 as **provisional** repos
2. Deep research lands (`research/DEEP_RESEARCH_PROMPT.md` is the brief) →
   **write the synthesis straight to `research/PASS2_FINDINGS.md`, do not hold it in context**
3. Implementation plan → `PASS2_PLAN.md`
4. Run the optimised REAP. Goal is the best achievable pass, not an increment.

Known gaps in pass 1 that pass 2 must fix (my own assessment, `wiki/00-log.md` has detail):
- used **0.52M of 48.3M** calibration tokens (1.1%); 501/12,096 expert slots below the
  2,000-token floor, min 190
- quantile-blended saliency (arm B) designed, never run — and I owned the hook, so it was free
- layer-local distillation ruled out early, then the streaming machinery made it feasible; never
  revisited
- **zero evaluation**; `HELDOUT_FRACTION` is in the spec and unused
- MAX_LEN 2048 truncated everything; DocFinQA (123k-word contexts) dropped entirely

## Hard-won facts (do not rediscover)
- Released weights are **FP8 E4M3**, not BF16. Pruning in FP8 is lossless on retained weights.
- **The model cannot be loaded.** Work one layer/tensor at a time, always.
- KDA forward ≈ **13 GiB per 2048-token sequence**, linear in batch → bounds batch size.
- Tegra unified memory is invisible to the OOM killer; `MemAvailable` under-reports by ~100 GiB.
  Never treat it as a health signal. `glm53-memguard.service` handles it.
- `config.num_local_experts` is a **scalar** → per-layer expert counts are unloadable.
- `scripts/nvfp4_tensor.py` is verified **bit-identical** to compressed-tensors' compressor.
