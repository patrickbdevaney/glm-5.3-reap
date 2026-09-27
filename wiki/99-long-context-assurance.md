# 99 — How to actually assure long-context capability

*2026-09-27. Everything measured so far is about the MASK — whether we kept the right experts.
Nothing yet measures the PRUNED MODEL at long context. These are different claims and only the
first has evidence.*

## What "assure" can and cannot mean

It cannot mean proof. It can mean: enumerate the ways 50% expert pruning could break long context,
and measure each one. There are exactly three, and they are not equally likely.

| mechanism | exposed? | why |
|---|---|---|
| **Retrieval** — locating content at position 700,000 | **structurally safe** | REAP prunes experts and touches **no attention parameter**. No KDA weight, no DSA weight, no KV projection. The machinery that finds a needle is bit-identical to the unpruned model. |
| **Routing drift** — the mask being wrong at long positions | **open, testable, cheap** | the router reads hidden state, which does evolve with context. If routing at 500K differs from routing at 2K, a mask calibrated at 2K is the wrong mask there. |
| **Capacity** — reasoning over what was retrieved | **open, expensive** | half the experts are gone. A per-token loss invisible at 500 tokens can be decisive across 100,000 tokens of multi-hop reasoning, because errors compose. |

[93](93-seqlen-ladder-results.md) partly closes the middle row: a 4x sequence-length change moved
the keep-set (0.9745) far less than resampling the same length did (0.8380). But that spans
2k->8k and is about *saliency*, not about what the pruned model does at 500K.

---

## The urgent part: one test needs the teacher, and surgery destroys it

`s04b_surgery` **deletes source shards as it writes survivors**. At 24 min/block and 41 blocks
remaining, that is **~16 hours away**. After it, the unpruned GLM teacher is gone and cannot be
recovered without re-downloading 328 GB.

### Test 1 — affected-rate versus token POSITION  `[OPEN]`

> Run long-context tokens through the **unpruned** model and record, per position, what fraction
> of its top-8 routed slots would land on an expert the mask prunes.

- **flat in position** -> routing is position-invariant, the mask is as good at 500K as at 2K,
  and the middle row above is closed.
- **rising with position** -> the mask degrades with context, and we know it *before* shipping
  rather than from a user report.

On MiMo this measured **1.13% of slots / 8.71% of tokens** overall — but at <=4,095 tokens, so it
says nothing about position. Layer 7 alone ran at 18.18%.

**The sequencing dissolves the same way the global router KD did** ([98](98-router-kd-global.md)):
the teacher is only needed as a *source of routing decisions*, not as a live model. Capture them
once, use them forever:

```
42 MoE layers x tokens x top-8 expert ids, int16
   128K tokens : 0.08 GiB
     1M tokens : 0.66 GiB
```

**0.66 GiB buys the teacher's complete routing behaviour at full context, permanently.** Once
captured, affected-rate versus position can be computed offline against *any* candidate mask,
including masks that do not exist yet, with no GPU and no teacher.

Cost: one long-context forward of the unpruned model, and it must run **after `s04_sweep`
produces a mask and before `s04b_surgery` starts deleting**. That window is narrow because
`s04_sweep` is fast. The capture itself does not need the mask — only the *analysis* does — so
**the capture can run any time in the next 16 hours** and should not wait for the mask.

> **Recommendation: insert the capture before surgery.** It is ~1-2 h of GPU against a 328 GB
> irreversible loss, and it is the only test here that becomes impossible later. `[EXT]`

**MiMo cannot have this.** Its 166 GB teacher was deleted, so MiMo's long-context assurance is
limited to the teacher-free tests below, on the shipped artifact.

---

## The teacher-free tests, after `s07_quantize`

All three run on the pruned+quantised model, need no teacher, and fit the box —
[92](92-long-context-program.md) measures 1M context at **109.27 GiB** of a 117 GiB envelope for
GLM and **115.01 GiB** for MiMo.

### Test 2 — NIAH, to 1M

Cheapest of the three and synthetic, so no corpus problem: sweep needle depth x context length.

**Read it correctly.** Retrieval is the row REAP leaves structurally untouched, so a pass is
**weak** evidence — it confirms the half that was never at risk. A *failure* would be very
informative, because it would mean something is wrong that the mechanism does not predict.

### Test 3 — LongPPL

The informative ruler for the capacity row. Needs **real long documents**, which the corpus does
not have: `MAX_TOKENS = 16_384`. Sourcing them is a prerequisite, not a detail. Concatenating
existing samples manufactures fake long-range structure and answers the wrong question.

### Test 4 — multi-hop long-horizon agentic

Closest to the actual use ("agentic daily driving"), hardest to score, and the only one that
exercises error composition over a long horizon directly. Likely paired teacher/student on real
trajectories rather than a pass/fail number — and paired scoring against the GLM teacher also
needs capture before surgery, or a cached-logit equivalent.

---

## Order

1. **Capture teacher routing at long context — before surgery, ~16 h window.** Irreversible if
   missed.
2. Affected-rate vs position, offline, against the pass-3 mask.
3. NIAH to 1M after quantisation — cheap, confirms the safe half.
4. LongPPL once long documents are sourced.
5. Multi-hop agentic.

Steps 2-5 are all recoverable later. **Only step 1 has a deadline.** `[EST]`
