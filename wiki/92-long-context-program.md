# 92 — Making the REAP hold at native 1M context

*Opened 2026-09-26. Requirement stated directly: the REAP must work "as optimally as possible
with a kv cache context length of the total native 1 million context for niah, longppl and
multi hop long horizon agentic contextual use".*

This page is the plan and the arithmetic. It supersedes nothing in
[91-capability-coverage.md](91-capability-coverage.md); it is what to do about it.

---

## 1. The load-bearing fact: 1M context FITS on Thor  `[EST]`

I expected this to be the blocker. It is not. MEASURED from `config.json` — GLM's DSA layers use
MLA, which caches a 512-dim compressed latent rather than full K/V, and `qk_rope_head_dim = 0`
so there is no separate rope cache:

| component | at 1,048,576 tokens |
|---|---|
| DSA/MLA KV cache — 11 layers × 512 latent × bf16 | **11.00 GiB** |
| KDA recurrent state — 34 layers × 64 heads × 128 × 128 | **0.066 GiB**, *constant in context* |
| total context state | **11.07 GiB** |
| REAP50 NVFP4 weights | 98.20 GiB |
| **resident at full 1M** | **109.27 GiB of a 117 GiB envelope** |

For contrast, un-compressed KV for those same 11 layers would be **704 GiB**. MLA is the entire
reason this is possible.

**Consequence: NIAH and LongPPL at genuine 1M are runnable on hardware already owned.** No cloud
spend, no hosted eval. The margin is ~7.7 GiB, which is thin but real, and the KDA majority costs
nothing extra as context grows.

---

## 2. Why calibrating *at* 1M is the wrong goal  `[EXT]`

Saliency holds hidden states per row: `S × 4096 × 2 B`.

| S | per row, per copy |
|---|---|
| 2,048 | 0.016 GiB |
| 16,384 | 0.125 GiB |
| 131,072 | 1.0 GiB |
| 1,048,576 | **8.0 GiB** |

At 1M a single row costs 8 GiB, so a statistically meaningful number of rows is impossible. Brute
force is out. But it is also **unnecessary**, and the reason is structural:

- **DSA sets `index_topk = 2048`.** Those layers attend to at most 2,048 selected keys *no matter
  how long the context is*. Their input distribution is bounded by construction.

  > **CORRECTION 2026-09-26.** This page originally added "so it never reaches the quadratic
  > regime". That is **false for memory**. `index_topk` caps what is ATTENDED; the indexer must
  > still SCORE every key to choose the top 2,048, so the score matrix is `[B, heads, L, L]` —
  > 32 GiB at L=16,384 with 64 heads in bf16, before softmax and copies. This wedged the box
  > entering the first DSA layer. The *routing-distribution* argument for saturation is
  > unaffected — bounded attended-set still bounds what the router sees — but the **cost**
  > argument was wrong. See [95-memory-lookahead.md](95-memory-lookahead.md). `[EST]`
- **KDA carries a fixed-size recurrent state** (64 × 128 × 128 per layer). A fixed-size state has
  bounded capacity; its distribution reaches a steady state.

Both attention mechanisms in this model have **bounded state**. So the hidden-state distribution
the MoE router reads should *saturate* at some length far below 1M. Prediction with a sign, which
is what makes it a test rather than a hope:

> Per-expert saliency should change with S up to some saturation length S*, and stop changing
> above it. Calibrating at S* is then equivalent to calibrating at 1M, at a fraction of the cost.

The goal is therefore **find S\*, calibrate at or above it, and verify at true 1M** — not
calibrate at 1M.

---

## 3. The cheap direct test nobody has run  `[OPEN]`

Everything above is about choosing a *new* mask. There is a much cheaper question about the mask
we have already shipped, and the instrumentation for it already exists from the router-KD work
(`_affected_rows` in `router_kd_run.py`):

> Run long-context tokens through the **existing** mask and measure the fraction of routed slots
> that hit a pruned expert, **as a function of token position**.

- Flat in position → long-context routing is unaffected; the shipped artifact is fine and this
  whole program is insurance.
- Rising with position → the shipped model degrades with context, and we would know it without
  recalibrating anything.

On MiMo this measured **1.13 % of slots / 8.71 % of tokens** overall — but that number was taken
at ≤4,095 tokens, so it says nothing about position. This is hours of work, needs no new
calibration, and can falsify the entire concern in either direction. **Run it first.**

---

## 4. Evals: the rulers that are not self-confirming  `[EXT]`

[`../research/REAP_METHOD_AND_FINETUNE_VIABILITY_2026-09-26.md`](../research/REAP_METHOD_AND_FINETUNE_VIABILITY_2026-09-26.md)
establishes that worst-domain retention and `pᵀFp` are **rigged rulers** — each is maximised by
construction by the selector that optimises it. NIAH, LongPPL and multi-hop agentic traces are
none of those things: they are external, and they measure the capability directly.

- **NIAH** — synthetic by construction, so no corpus problem. Sweep needle depth × context
  length up to 1M. Cheapest of the three.
- **LongPPL** — needs *real* long documents. The current corpus caps at 16,384 and cannot supply
  them; sourcing long books/repos is a prerequisite, not a detail.
- **Multi-hop long-horizon agentic** — the one the user actually daily-drives, and the hardest to
  score. Likely paired teacher/student on real trajectories rather than a pass/fail metric.

All three run against the paired teacher/student protocol already in `s09_eval`.

---

## 5. Honest limits  `[EST]`

1. **If 50 % pruning genuinely removes capacity that long context needs, no calibration recovers
   it.** Calibration chooses *which* experts to keep, not how many. The only lever would be a
   lower global ratio.
2. **GLM cannot spend that lever selectively.** `Glm5NextTextExperts.__init__` reads ONE scalar
   `config.num_local_experts` applied to every layer, so a ragged per-layer budget is unloadable
   by vLLM and the GGUF converters. It is a global ratio or nothing — and going past 50 % is an
   escalation, per standing constraint.
3. **The corpus cannot currently test above 16,384 tokens.** Extending the ladder to find S*
   requires genuinely long documents first. Concatenating existing samples would manufacture fake
   long-range structure and answer the wrong question.
4. **MiMo cannot be recalibrated at all right now.** The 166 GB source was deleted; disk stands
   at 129 GB free. Re-download is a disk decision as well as a time one. What *can* be done on
   MiMo without it is §3, which only needs the accumulators and the mask.

---

## 6. Order of work

1. **§3 affected-rate vs position**, on the existing mask. Cheapest, and can end the concern.
2. **Finish the S ladder** (`exp_seqlen_saliency.py`, running 2048/8192/16384 + control).
3. **Source long documents**, extend the ladder to 64K/128K, locate S*.
4. **Build NIAH at 1M** — §1 says it fits.
5. Only then decide `calib_max_len` for pass 3.

LongPPL and the agentic suite follow; they are the more expensive rulers and the ladder result
determines how much they need to cover.

---

## 7. "Retrieve" and "use" are not the same risk  `[EXT]`

The requirement names *"niah, longppl and multi hop long horizon agentic"*. Those split cleanly
along what REAP actually modifies, and the split is worth stating because it is easy to read the
§1–2 architecture argument as covering both. It does not.

**REAP prunes experts. It does not touch attention.** No KDA parameter, no DSA parameter, no KV
projection, no cache layout changes. The machinery that *locates* a needle at position 700,000 is
bit-identical to the unpruned model's.

- **Retrieval (NIAH) is structurally the safer half.** It is mostly an attention property, and
  attention is untouched. Expect it to hold, and treat a NIAH pass as confirming the cheap half.
- **Use — reasoning over what was retrieved — is where the risk actually lives.** That is expert
  work, and half the experts are gone.

So a clean NIAH result would be genuine but **weak** evidence: it is the outcome most protected by
what REAP leaves alone. LongPPL and multi-hop agentic are the informative rulers.

### The failure mode the architecture argument does not cover

§2 argues that bounded state (DSA `index_topk=2048`, fixed-size KDA state) makes routing
statistics saturate, so a mask calibrated short is probably the *right mask* for long context.
Grant that entirely. It still leaves untouched:

> A per-token quality loss too small to see at 500 tokens can be decisive across 100,000 tokens
> of multi-hop reasoning, because errors compose.

"Right mask" and "enough capacity" are different claims. This is the same error-composition
argument that makes the layer-local KD objective an approximation
([98-router-kd-global.md](98-router-kd-global.md) §2) — local adequacy does not imply global
adequacy, and nothing measured so far speaks to the global case.

### Where architecture is a risk, not a reassurance

On MiMo, 39 of 48 layers are SWA-128 and **cannot integrate long-range information at all**. All
long-range integration is concentrated in **9 layers**. Bounded state does not dilute the risk
there — it concentrates it. Pruning experts in those 9 layers is disproportionately consequential,
and the layer-7 affected rate of 18.18 % of slots is a reason to look there first, not a reason to
relax. `[OPEN]`
