# GLM-5.3-Flash REAP + Heal — Working Wiki

Append-only knowledge base for this project. **Everything** learned from web research,
paper reading, repo inspection, or on-box experimentation lands here — not in terminal
scrollback, not only in a chat reply.

## How to use this wiki

- **Never rewrite history.** Correct a claim by appending a new dated entry that supersedes
  it and editing the old line to say `~~superseded~~ → see [YYYY-MM-DD]`.
- Every factual claim carries a **source**: arXiv ID, repo path + commit, HF model ID, or
  `MEASURED` (with the command that produced it).
- Tag every claim with a confidence marker:
  - `[EST]` — established: reproduced, measured, or stated in a peer-reviewed/primary source
  - `[VEN]` — vendor/marketing claim, unverified
  - `[EXT]` — our own extrapolation or inference
  - `[OPEN]` — known unknown, no source exists
- New topic → new numbered file, add a line to the index below.

## Index

| File | Topic |
|---|---|
| [00-log.md](00-log.md) | Chronological append-only log of every session, finding, and experiment |
| [10-target-model.md](10-target-model.md) | GLM-5.3-Flash: verified architecture and parameter accounting |
| [20-host-thor.md](20-host-thor.md) | Jetson AGX Thor: measured capability and storage envelope |
| [30-reap.md](30-reap.md) | REAP method, published results, toolkit landscape |
| [40-hybrid-fragility.md](40-hybrid-fragility.md) | The central risk: pruning hybrid linear-attention + mHC models |
| [50-multimodal.md](50-multimodal.md) | Vision preservation under expert pruning |
| [60-quantization.md](60-quantization.md) | FP8 → NVFP4 path, Thor SM110a support |
| [70-healing.md](70-healing.md) | Recovery training: LoRA, distillation, RLVR |
| [80-calibration.md](80-calibration.md) | Calibration corpus design and evidence |
| [90-open-risks.md](90-open-risks.md) | Open risks, contradictions with the directive, decision log |
| [91-capability-coverage.md](91-capability-coverage.md) | What calibration protects: long-context and non-English gaps |
| [92-long-context-program.md](92-long-context-program.md) | Holding the REAP at native 1M: arithmetic, saturation, NIAH/LongPPL plan |
| [93-seqlen-cost-and-memory.md](93-seqlen-cost-and-memory.md) | Sequence length is ~free at fixed tokens; the memory wedge and its two fixes |
| [94-memguard-killed-the-wrong-process.md](94-memguard-killed-the-wrong-process.md) | Two wedges: a second enabled service, and a guard whose whitelist made it kill the healthy job |
| [95-memory-lookahead.md](95-memory-lookahead.md) | Predicting peak memory before allocating; why kill and cgroups both failed |
| [98-router-kd-global.md](98-router-kd-global.md) | Global router KD: scope, cost, sequencing |

## Status

Phase 0 (deep research) — in progress. No implementation started. No weights downloaded.
