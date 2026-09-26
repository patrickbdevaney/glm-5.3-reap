# 98 — Global router KD: scope, cost, and the sequencing that makes it possible

*Opened 2026-09-26, after the layer-local objective was measured and rejected. Scoping only —
nothing here has run.*

## 1. Why the local objective is out of work, and why that does not close the question

MEASURED 2026-09-26 (MiMo, `scripts/gate_router_kd_stratified.py`, 7/7): under **layer-local
output matching**, no sampling rule × budget × step beats the teacher's sliced router. Damage
falls monotonically as the step approaches *change nothing* — lr 1e-3: −299 %, 1e-4: −62 %,
1e-5: −7.5 %, 3e-6: −2.2 % on a held-out population. The 47/47 guard was correct. `[EST]`

The token budget was never the cause: only **1.13 % of routed slots** and 8.71 % of tokens hit a
pruned expert, because REAP keeps what the router picks most. Layers 1–2 sit at 0.0000 %.

That result is about **one objective**, not about router repair. The stage's own docstring
already conceded the limit:

> local matching cannot see how errors compose across layers, so it is an approximation, not the
> published method.

Layer-local asks each layer to reproduce its own output given the *teacher's* input. It is
teacher-forced at every boundary, so it can never observe that a small layer-7 deviation is
amplified by layer 20, nor trade a worse layer-7 output for a better final distribution. The
published method (arXiv 2603.02217) distils against the **full model's next-token
distribution**. That is a different loss with a different optimum, and it is untried. `[EST]`

---

## 2. The sequencing problem, and its resolution

Stated constraint: router KD needs the unpruned teacher, but `s04b_surgery` **deletes source
shards as it goes**, and Thor cannot hold teacher and student simultaneously (306 GB FP8 source
against a 117 GiB envelope — CLAUDE.md §5 forbids a second full-precision copy anyway).

Read naively this forces KD to run before or during surgery, which is the worst possible time.

**It does not.** The teacher is only ever needed as a *target*, never as a live model:

1. **Before surgery** — one streaming forward pass of the unpruned teacher over the KD token
   subset, caching **top-K next-token logprobs to disk**. The teacher is then never needed again
   and surgery may delete it.
2. **After surgery** — train the pruned student's routers against the cached targets. Student
   only. No teacher resident, no second copy, no conflict with surgery.

Cache size is what makes this work. Full logits are impossible: vocab 154,880 × 2 B = 310 KB per
token, 15 TB over the corpus. Top-K=64 is 64 ids (int32) + 64 logprobs (fp16) = **384 B/token**:

| KD token budget | teacher cache on disk |
|---|---|
| 2.4 M tokens (300 steps × 8,192) | **0.9 GB** |
| 8 M tokens | 3.1 GB |
| full 48.3 M corpus | 18.6 GB |

Top-64 of 154,880 carries essentially all the mass of a converged LM's next-token
distribution; the tail is what KD temperature would flatten anyway. `[EXT]`

---

## 3. Cost on this box  `[EXT]`

Trainable parameters — routers only, all 42 MoE layers:

```
288 experts × 4096 hidden = 1,179,648 per layer × 42 = 49.5 M params
Adam fp32 moments: 49.5 M × 8 B = 396 MB        -- trivial
```

Activation memory with gradient checkpointing at layer boundaries, seq 2048:

```
45 layers × 2048 × 4096 × 2 B = 755 MB          -- trivial
```

Neither is the cost. **The cost is streaming 306 GB of weights twice per step** — forward, then
again in reverse for backward, because the gradient path from the final logits back to router L
must traverse every layer above it.

```
306 GB × 2 ÷ ~2.5 GB/s NVMe  ≈  4 min/step  (+ recompute)
300 steps × 8,192 tokens      ≈  25-30 h
```

Same order as the saliency pass, no cloud spend, fits the hardware. It is affordable; it is not
cheap, and it would sit on the GPU for a day that the pass-3 chain also wants.

---

## 4. Expected value, stated honestly  `[OPEN]`

Argument for: it is the actual published method, it is the one objective that can see error
composition, and the local result cannot speak to it.

Argument against: the local study found the teacher's sliced router already near-optimal at every
step size. If the sliced router is near-optimal *globally* too, global KD buys approximately
nothing for 25–30 h.

Caveat carried forward from the local study: its fixture's experts were an elementwise map of the
hidden state. Real expert functions may leave the teacher's router further from optimal than that
fixture did — which is an argument the global result could differ, and the reason this is `[OPEN]`
rather than closed.

**Priority: below the sequence-length test** in [91-capability-coverage.md](91-capability-coverage.md).
That test is a partial pass, costs hours not days, and can invalidate the *mask itself* — a
strictly larger prize than a router correction on a mask that may be calibrated for the wrong
context regime. Fix what selects the experts before polishing what routes to them. `[EXT]`

---

## 5. Build order, if approved

1. `gate_teacher_cache.py` — top-K cache round-trips, and KL against a full-logit oracle on a
   toy model agrees to float tolerance. Gate before the expensive pass (CLAUDE.md §2: gate on
   real weights, never against a dead engine).
2. `s03c_teacher_logits.py` — streaming teacher pass, resumable at chunk granularity, writes the
   cache. **Must complete before `s04b_surgery` deletes the shards.**
3. `s07b_router_kd_global.py` — student-only KD, routers trainable, KL against cache.
4. Accept per layer only on the **held-out population** — the acceptance bug the local stage
   had (judging on its own training rows) must not be re-introduced.
5. Final gate is the existing paired teacher/student dNLL/flip on `teacher.pt`, not the KD loss.
   A KD loss that improves while dNLL worsens is a self-confirming ruler, same failure mode as
   [80-calibration.md](80-calibration.md) §"both our rulers are rigged".
