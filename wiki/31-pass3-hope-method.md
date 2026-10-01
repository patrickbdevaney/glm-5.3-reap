# 31 — Pass 3: the HOPE method, end to end

*Completed 2026-10-01. Supersedes pass 2's scalar REAP selection. Shipped as `pass3-hope/` in the
HF weight repos; pass 2 remains at repo root, untouched.*

## What changed, and why

Pass 2 selected experts by ranking each one alone and taking the top-k. [arXiv 2609.18916] shows
that is exactly HOPE with the off-diagonal of `F` zeroed — two experts that duplicate each other
are cheap to drop together, two that complement each other are not, and a per-expert ranking
cannot tell the difference. Pass 2 measured what that cost: **ballast dNLL 0.989, top-1 agreement
0.580**, against 0.916 for code. That hole is the reason for pass 3.

Pass 3 minimises `pᵀFp` — the output error a prune set actually causes, interactions included —
over the same accumulators, with no extra forward passes.

## Result

| | value |
|---|---|
| ratio | 0.5 — **144 of 288 experts** kept, 43 MoE layers |
| routers sliced | 86 |
| tensors | 38,956 kept / 37,152 dropped |
| FP8 checkpoint | 160.6 GiB |
| **NVFP4 checkpoint** | **93.6 GiB**, 62 shards, 57,808 tensors |
| Thor envelope | 117 GiB — **fits with 23 GiB spare** |
| heal gain | median 0.766 (0.667–0.853), 42 layers, 0 skipped, 6,048 experts scaled |

Calibration: 42 layers × 10 chunks, **866,678,064 routed slots**.

## protect_frac was measured, not guessed

`conf/protect_frac_override.txt` held `0.0` with an explicit note that it was *"a PLACEHOLDER, not
a decision"* — the right value is a property of this run's accumulators, which did not exist when
it was written. They exist now, so `scripts/sweep_protect_frac.py` measured the curve:

| pf | worst_ret | icost | | step | marginal gain/cost |
|---|---|---|---|---|---|
| 0.000 | 0.5399 | 0.1183 | | 0.000→0.005 | 4.52 |
| 0.020 | 0.5506 | 0.1215 | | 0.010→0.020 | 3.03 |
| **0.040** | **0.5580** | **0.1270** | | **0.030→0.040** | **1.41** ← last step that pays |
| 0.060 | 0.5621 | 0.1347 | | 0.040→0.060 | 0.54 |
| 0.080 | 0.5667 | 0.1452 | | 0.060→0.080 | 0.43 |
| 0.120 | — | — | | INFEASIBLE: protection exceeds the prunable population |

**0.04** buys +0.0181 worst-domain retention for +0.0088 interaction cost.

> **Read the knee MARGINALLY.** Cumulative gain/cost falls monotonically by construction, so its
> maximum is always the smallest nonzero pf — 0.005 here — whatever the curve's shape. The sweep
> script reported that as a "knee" on its first run and it was an artifact of the statistic, not a
> property of the data. It no longer does.

MiMo's 0.08 was not assumed and does not transfer: it sits where marginal efficiency is 0.43. This
model has 288 experts over 8 domains (7 live), a different corpus and a different `F`. The curve
IS monotone near zero here, unlike MiMo where `pf=0.01` was worse than `pf=0` — the grid was dense
below 0.02 to test that rather than assume it.

## bio has zero coverage, structurally

**0 of 866,678,064 routed slots.** Not a partial pass: every bio source is in
`corpus_spec.EXCLUDED` — `camel-ai/biology` (licence_nc), `tattabio/OG` (licence_sharealike),
`qiaojin/PubMedQA` and `bigbio/pubmed_qa` (scope_clinical). No recomputation populates it.

Selection scores the 7 live domains. **The shipped model must not claim bio capability.**

`assert_domains_live` refuses to rank against a zero-mass domain, and its advice — "wait for those
buckets' chunks" — is right for an incomplete pass and useless here. The only way past it was a
blanket `--allow-dead-domains` that would also wave through a genuinely partial pass, so the two
cases are now distinguished: `conf/structurally_empty_domains.txt` lists what cannot be populated,
with reasons, and anything dead and unlisted still raises.

Per-domain retention at the shipped setting: finance 0.686, math 0.687, agentic 0.706, code 0.722,
vision 0.724. Worst domain is **general** (17.58% of corpus) at every pf — not a thin domain.

## Long-context assurance is at 8192, not 16384

`capture_teacher_routing.py` runs before surgery because surgery deletes the source shards: once
it runs the unpruned teacher is gone short of a 328 GB re-download, and this is the only
long-context measurement that becomes impossible afterwards.

Captured: 42 MoE layers × 16 rows × 8192 tokens, top-8 int16 — **5,505,024 tokens of routing**,
expert ids spanning 0–287.

It was intended at 16384. One 16384-token forward **died starting from 93.0 GiB of free memory**.
The MoE FFN is not the cause — routed activations at 16k are ~1.5 GiB — attention is quadratic in
sequence length and that is what binds. No block size or reclaim strategy helps, because the
indivisible unit of work is a single forward pass. At 8192 the same forward uses ~57 GiB and
completes with ~44 GiB spare, and 8192 is still 4× the 2048 tokens the mask was calibrated at.

## Two bugs worth remembering

**The capture was never capturing routing.** It read the layer's second return value as expert
ids. That value is `prev_topk_indices`, the sparse-attention index passthrough, and is `None` for
these layers — so nothing was written for any MoE layer while it logged "layer N captured" each
time. Only the orchestrator's `"exited 0 but wrote nothing"` check caught it. Without that guard
the full capture would have printed 45 success lines and handed surgery an empty directory. Fixed
by patching the router's `forward`, the mechanism `stream_saliency` already used.

**`sorted(glob("*.pt"))[0]` stopped being a layer dump.** `artifacts/saliency` holds 42 per-layer
dumps *and* `accumulators.pt`, which s04_sweep writes. Fifteen call sites assumed every `*.pt`
match was a per-layer dump — true until s04_sweep ran. `s04b_surgery` died on
`KeyError: 'num_experts'` immediately after completing the selection. `common.layer_dumps()` now
returns per-layer dumps only, sorted by layer index — which also fixes a quieter bug, since
lexicographic order puts layer 10 before layer 3.

## Memory engineering that made the pass possible

See [96-the-real-oom-cause.md](96-the-real-oom-cause.md) and [90-data-handling.md](90-data-handling.md).
In short: `LAYER_COST_GIB` was 5.76, the **weight** size of a layer. The true host footprint of
sweeping one is **~53 GiB**, because `torch.cuda.empty_cache()` returns essentially nothing on this
integrated board and device memory comes from system RAM via nvmap. Sizing against 5.76 is why
blocks of 9, then 5, then 2 layers all died.

The fix that carried 90 layers: **one layer per process, and reclaim on the memory floor rather
than on a counter.** Both halves matter and both were got wrong first:

- a fixed every-25-batches reclaim could not see a 40 GiB collapse that happened inside one
  interval (s03 layer 4);
- a conditional 30 GiB floor never fired at all when one unit of work cost 57 GiB (the capture,
  which stalled at block 23/45 refusing to start a forward it could not finish).

A floor below the cost of one unit of work is useless. Where a unit is expensive, reclaim
unconditionally: at ~2 s against a multi-second forward it is cheap enough, and "every time" has
no threshold to get wrong.
