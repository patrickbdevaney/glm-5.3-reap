# What is actually reachable on this hardware, and what is not

Written 2026-08-28, mid-pipeline. Sorted by whether it needs a new sweep, the unpruned teacher, or
neither — because that, not cleverness, is what decides feasibility here.

## Tier 1 — free, needs neither a new sweep nor the teacher

These use accumulators and the router cache we already have. Minutes to hours of compute.

### 1.1 Per-expert healing fitted in closed form  — **DONE, 2026-08-28**

Healing was **one scalar per layer**. That scalar is the degenerate case of a least-squares
problem: choose a coefficient per *retained* expert minimising

    Σ_t ‖ y_t − ŷ_t ‖²        y_t = Σ_{i∈S_t} g_i f_i(x_t)      (unpruned)
                              ŷ_t = Σ_{j∈S'_t} c_j g'_j f_j(x_t) (pruned, rescaled)

with `S`/`S'` the pre- and post-prune top-8 and `g`/`g'` the `norm_topk_prob` gates.

**Why it is computable from one sweep.** Decompose `f_i(x) = μ_i + ε_i(x)`. Assume only that
residuals of *different* experts are uncorrelated — strictly weaker than the full orthogonality
the shipped scalar already assumes, since the cross terms `μ_i·μ_k` are kept exactly. Then
`E[f_i·f_j] = μ_i·μ_j + δ_ij v_i` and the normal equations are

    A = (P ⊙ G) + diag(N ⊙ v)      P_jk = Σ_t 1[j,k∈S'_t] g'_j g'_k
    b_j = Σ_i C_ji G_ij + C_jj v_j  C_ji = Σ_t 1[j∈S'_t] g'_j 1[i∈S_t] g_i

every term of which comes from artefacts we already had: `μ_i` from `out_sum`/`gate_sum`,
`E‖f_i‖²` from `norm_sq_by_bucket`/`count`, and `P`,`C`,`N` from replaying routing on the router
cache. Under full orthogonality `A` is diagonal and the solution collapses to `c_j = C_jj / N_j`
— *the norms cancel entirely*. That assumption-light form is what ships.

`scripts/heal_perexpert.py`; `tests/test_heal_perexpert.py` checks the quadratic form against a
brute-force simulation, which is what catches an index-order bug in `accumulate`.

**Measured, on held-out tokens (50/50 interleaved split, fit on one half, scored on the other).**
Relative residual `Σ‖y−ŷ‖² / Σ‖y‖²`, mean over 42 layers:

| correction | mean rel. residual |
|---|---|
| none | 0.3332 |
| P5's shipped scalar | 0.2914 |
| **per-expert (magnitude-preserving)** | **0.2663** |
| per-expert (pure least squares) | 0.2471 |

**41 of 42 layers improve**; median reduction vs the shipped scalar +4.8%, best +43.4%. 40 layers
ship a vector, 2 keep the scalar, no coefficient hits the clamp bounds.

**The objective is not the obvious one, and this matters.** Pure LS is MSE-optimal but
*scale-biased*: it shrinks ŷ below y whenever the two are imperfectly correlated (regression
attenuation), landing at a median coefficient of **0.75** where P5's magnitude-matching scalar
sits at **0.909**. Minimising per-layer MSE is the wrong objective for a residual network — a
systematic 17% attenuation of the MoE pathway compounds multiplicatively over 42 layers, and
mHC's Sinkhorn-normalised connection matrices were fitted against the *original* output scale.
So the LS solution is rescaled to satisfy `E‖ŷ‖² = E‖y‖²`, keeping the per-expert *structure* —
the actual innovation — while leaving the global scale exactly where P5 validated it. A layer
only ships a vector if it beats **both** the LS-consistent baseline by 2% and P5's shipped scalar
outright, so no holdout fluctuation can make the checkpoint worse.

### 1.1b Orthogonality, finally measured

P5's docstring promised a `--check-orthogonality` that was never implemented. It falls out of
1.1 for free, because the Gram matrix has to be built anyway:

- **off-diagonal mass of A: 1.9%** (median over layers)
- **mean |cos(μ_i, μ_j)| between distinct experts: 0.091**

So the near-orthogonality that both the shipped scalar and `c_j = C_jj/N_j` rest on is real, not
assumed. The full solve including every cross term agrees with the diagonal form to four decimal
places (0.25258 vs 0.25258), which is the clean confirmation.

### 1.1c The mean/residual split — this bounds every richer hypothesis class

Also free from the same accumulators:

    ‖μ_i‖² / E‖f_i‖²  =  0.034   (median over 42 layers; range 0.016–0.181)

**96.6% of an expert's output energy is token-dependent residual, not its mean.** A rescaling —
scalar, per-channel diagonal, low-rank, anything that multiplies the expert's output by a fixed
operator — can only act on structure a fixed operator sees. With 3.4% of the energy in the mean
and residuals measurably near-orthogonal across experts, a per-channel refinement provably
collapses back toward `c_j = C_jj/N_j`. This is the number that closes off Tier 2's per-token
matching (below) without having to run it.

### 1.2 A balanced-retention mask, to close the ballast hole

REAP ranks on saliency **pooled** across domains, which is why retention ranges 0.487–0.747. With
per-bucket accumulators we can instead pick the mask that maximises the **minimum** per-domain
retention, or any weighted compromise. Cost: seconds to compute; it changes the mask before
surgery. Whether it is desirable is a judgement — it trades pooled quality for evenness, and the
RAG argument says the current shape may already be the right one. **Deliberately not applied**:
the retention profile we have is the one the RAG argument asks for, and evening it out would
spend the best-retained buckets (agentic, code, science, maths — the ones retrieval cannot
restore) to buy back ballast that retrieval can.

### 1.3 Router-aware staged greedy re-scoring

One-shot ranking ignores that removing an expert renormalises the top-8 over a different support.
The router cache makes staged re-scoring possible: drop the worst expert, recompute gates,
re-rank, repeat.

**1.1 supplies the missing piece here: an objective a candidate mask can be *scored* against.**
`Σ‖y−ŷ‖²/Σ‖y‖²` is computable for any keep-set on held-out tokens, and `Σ‖y‖²` depends only on
*pre*-prune routing, so it is a common denominator across masks. `heal_perexpert.py --keep-set`
scores an arbitrary mask in ~8 minutes. That turns mask selection from an argument about proxies
into a measurement.

### 1.4 Mask comparison, measured — pass 2 vs pass 1

The first use of that objective is the question we could previously only answer by proxy. Same
fit procedure, same held-out tokens, both masks keeping 144 of 288:

| mask | mean rel. residual (per-expert healed) | mean rel. residual (no healing) |
|---|---|---|
| **pass 2 (shipped)** | **0.2663** | **0.3332** |
| pass 1 (published) | 0.2730 | 0.3405 |

**Pass 2 reduces reconstruction error by 2.5%, winning in 30 of 42 layers** (per-layer spread
−5.8% to +10.1%; the two masks agree on 91.0% of experts). This independently corroborates the
saliency-mass proxy measured earlier (+1.52%) — same sign, same order of magnitude, derived from
a completely different quantity. The second sweep bought a real, if modest, improvement.

## Tier 2 — needs the unpruned teacher, which surgery deletes

- **Per-layer output matching against real teacher activations** — a version of 1.1 fitted on
  per-token data rather than accumulated sums. **Not worth pursuing, and 1.1 is what shows why.**
  Two measurements bound the gain: the exact-cross-term solve agrees with the model-based one to
  four decimals (§1.1b), so the modelling approximation is costing ~nothing; and 96.6% of expert
  output energy is token-dependent residual (§1.1c), so the remaining 0.27 relative residual is
  not approximation error — it is *information deleted with the 144 experts*. No rescaling of the
  survivors recovers it. Fitting it more exactly moves the third decimal. Recovering it at all
  means changing the surviving experts' weights, i.e. distillation, which is out of scope below.

- **A 40% ratio arm.** **Out of scope — memory, not compute.** Measured from the actual
  checkpoint: routed experts are **90.3%** of the parameters, and the 50%-prune NVFP4 build is
  96.0 GiB. Keeping 173 of 288 instead of 144 scales that share by 1.201:

  | ratio | kept/288 | NVFP4 weights | headroom vs 117 GiB |
  |---|---|---|---|
  | 50% (shipped) | 144 | 96.0 GiB | 21.0 GiB |
  | 45% | 158 | 104.4 GiB | 12.6 GiB |
  | 40% | 173 | 113.5 GiB | **3.5 GiB** |

  3.5 GiB has to hold the KV cache, activations, the CUDA context *and* the 1.171B DFlash2
  drafter. It does not fit in any serving sense. The earlier framing — "worth it only if the
  evaluation says 50% is too aggressive, costs a ~3 h re-download" — was wrong: the re-download
  was never the binding constraint. **45% is the only arm that is even arguable**, and it spends
  8.4 GiB of headroom to move retention from 0.50 to 0.55 of the expert budget. That trade should
  be made against Tier 3 evidence, not before it.

## Tier 3 — needs the student to actually run, and is the highest-information item

- **Absolute generative benchmarks** — HumanEval, GSM8K, MBPP, BFCL — run on the *student* alone.
  This needs no teacher (110 s/token there), because the NVFP4 student fits Thor. It is the only
  thing that answers "is it still smart" rather than "how far did it move".
  **Gated on inference working at all**, which is still unproven for this architecture on this box.

## Out of scope — not cleverness-limited, hardware-limited

| | why |
|---|---|
| teacher generation of any kind | 328 GB ÷ 3.0 GB/s ≈ **110 s per decoded token** |
| self-generated or difficulty-filtered calibration | requires the above |
| full-model distillation healing | backward pass through a 165B student |
| 16k-token calibration | needs sparse-DSA + chunkwise-KDA kernels first |
| non-uniform per-layer expert allocation | **not compute** — `num_local_experts` is one scalar |
| expert merging (REAM/EEP) | tractable but REAP's own thesis is that merging loses to pruning for MoE; a step sideways at best |
| a 40% ratio arm | **memory, not compute**: 113.5 GiB of NVFP4 weights leaves 3.5 GiB for KV cache, activations, CUDA context and the drafter (see Tier 2) |
| per-token teacher output matching | bounded to the third decimal by the two measurements in §1.1b/§1.1c — the residual that remains is deleted information, not approximation error |

## Does any of it need another full REAP run on the teacher?

**No.** Everything in Tier 1 reuses the 18-hour sweep's output. Tier 2 needs the teacher *present*,
not re-swept. Only a change to the calibration data itself — longer sequences, different domains,
regenerated responses — would require a new sweep, and every such change is in the out-of-scope
table for other reasons.

## Recommended order

1. ~~Implement 1.1 and run it before `s05_heal`~~ — **done 2026-08-28**; 40 of 42 layers ship a
   per-expert vector, mean reconstruction error 0.2914 → 0.2663 against P5's scalar.
2. Let the pipeline finish and **measure** — `s09_eval`'s ΔNLL, top-1 agreement and per-domain
   drift are the first absolute numbers this project will have.
3. Generative benchmarks once inference exists (Tier 3). Highest information per unit of work,
   and nothing above tells us whether the model is *smart*.
4. Revisit 1.2 / a 45% arm **only if** step 2 or 3 says the mask or the ratio is what is limiting.

That order matters more than it looks. Steps 1 and 4 are both mask/weight work, and doing 4 first
is how a project spends a week improving a number that was never the binding one.
