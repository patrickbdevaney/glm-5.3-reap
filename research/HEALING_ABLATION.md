# Per-expert healing is worse than the scalar it replaced

`[MEAS 2026-08-29]` — the ablation the pass-2 writeup promised, and it came back against the
technique. Two published conclusions are wrong as a result; both are corrected below.

## What was unattributed

Pass 2 changed two things against pass 1 — the **mask** (a second, longer saliency sweep) and the
**healing method** (per-layer scalar → per-expert least-squares vector) — and measured top-1
agreement of 0.83693 against pass 1's 0.83703. A wash. `wiki/30-reap.md` recorded the honest
version of that: "pass 2 changed the mask and the healing method together, so neither can be
credited or blamed individually."

It also noted the separation was cheap, because **healing is a multiply on F32 block scales and
is therefore invertible**. `scripts/heal_ablation.py` exploits that without touching disk: it
re-evaluates the *shipped* pass-2 checkpoint with

    multiplier_j = scalar_gain_layer / c_j

injected at layer-materialisation time (`s03_saliency.HEAL_OVERRIDE`), so the weights that reach
the forward pass are exactly the ones the per-layer scalar would have produced. Same checkpoint,
same 241,516 held-out tokens, same cached teacher — one variable moves.

## Result

| metric | per-expert (shipped) | **per-layer scalar** | Δ |
|---|---|---|---|
| **top-1 agreement** | 0.83693 | **0.84238** | **+0.00545** |
| ΔNLL vs teacher | 0.19396 | **0.17601** | −0.01795 |
| top-k KL | 0.69388 | **0.65248** | −0.04141 |
| student NLL | 1.21183 | **1.19388** | −0.01795 |
| flip rate | 0.16307 | **0.15762** | −0.00545 |

Binomial s.e. on 241,516 tokens at p≈0.84 is **0.00075**, so +0.00545 is 7.3σ on the *unpaired*
bound. That bound is conservative, because both arms score the identical token set — see the
paired test below, which puts it at **11.8σ**.

Every sufficiently-sampled domain improves, none regresses:

| domain | tokens | per-expert | scalar | Δ |
|---|---|---|---|---|
| agentic | 69,202 | 0.8707 | 0.8739 | +0.0032 |
| math | 66,558 | 0.9182 | 0.9197 | +0.0015 |
| science | 45,034 | 0.8216 | 0.8303 | +0.0087 |
| finance | 22,215 | 0.7409 | 0.7535 | +0.0126 |
| ballast | 22,154 | 0.5671 | 0.5801 | +0.0130 |
| code | 15,821 | 0.9187 | 0.9194 | +0.0007 |
| *vision* | *532* | *0.3891* | *0.4417* | *(under-sampled, not scored)* |

A uniform, same-signed improvement across seven independent buckets is not what a noise
fluctuation looks like.

### Confirmed on the reverted weights, not just the override

The table above is the load-time ablation. After `heal_revert_to_scalar.py` wrote the correction
into the checkpoint, the same evaluation was re-run against the weights that actually ship:

| | top-1 | ΔNLL | top-k KL |
|---|---|---|---|
| ablation predicted (override at load) | 0.84238 | 0.17601 | 0.65248 |
| **measured on the reverted checkpoint** | **0.84249** | **0.17563** | **0.65030** |

The 1.1e-4 gap is the difference between one multiply and a divide-then-multiply in F32 — the
revert applies `scalar/c_j` to weights already carrying `c_j`, the override reconstructed the same
product at load. Nothing in the pipeline is sensitive at that scale; what matters is that the
persistent artifact reproduces the measurement that justified it.

### The paired test, which is the correct one

Both arms score the same 241,516 tokens against the same cached teacher, so tokens where both
arms agree with the teacher — or where both miss — carry no information about which arm is better
and do not belong in the denominator. McNemar on the discordant pairs:

| | count |
|---|---|
| both arms agree with the teacher | 196,293 |
| neither agrees | 32,203 |
| **per-expert right, scalar wrong** | **5,839** |
| **scalar right, per-expert wrong** | **7,181** |
| discordant (all the signal) | 13,020 — **5.39%** of tokens |

    delta = +0.00556      paired s.e. = 0.00047      chi2 = 138.1      z = 11.8

The paired s.e. is 0.00047 against the unpaired 0.00075, so the correct test is **~1.6× sharper**
and the result is **11.8σ**. Note also that per-expert healing is not uniformly worse — it wins
5,839 tokens the scalar loses. It simply loses 7,181 more than it wins. This is a shifted
distribution, not a broken checkpoint, which is exactly why the aggregate null in the original
pass-1/pass-2 comparison was able to hide it.

## Why it lost — the coefficient is a promotion penalty

The shipped coefficient is `c_mag = to_magnitude(c_diag)`, and under the near-orthogonality that
was separately measured (off-diagonal mass 1.9%) the diagonal solution is exactly

    c_j  =  ( gate mass expert j received BEFORE pruning ) / ( gate mass it receives AFTER )

`wiki/70-healing.md` states the intended reading plainly: an expert *promoted into* the top-8 by
pruning "is correctly shrunk — it is doing work it never did before."

**That is the bug, and it is a conceptual one, not an arithmetic one.** The promoted expert is
doing work it never did before *because the expert that used to do that work has been deleted*.
It is the only thing left standing in that slot. Damping it by its promotion ratio does not
restore the deleted expert's contribution — it removes the substitute's contribution too, and
leaves the residual stream with a hole where an imperfect approximation would have been.

The distribution says how hard this bites. Over 5,760 coefficients:

| | value |
|---|---|
| min | 0.2743 |
| p1 / p10 / median / p90 | 0.4767 / 0.6920 / 0.8727 / 1.0036 |
| layer scalars, for comparison | 0.8335 – 0.9495 (median 0.9096) |
| experts scaled **below** their layer's scalar | 3,574 / 5,760 = **62.0%** |
| suppressed >1.5× vs the scalar | 238 |
| suppressed >2× vs the scalar | 45 |

The worst-hit expert is attenuated **3.3×** relative to what the scalar would have given it — and
by construction those are precisely the experts the post-prune router leans on hardest.

### This is regression attenuation again, one level down

The writeup already caught attenuation once. Pure least squares landed at a median coefficient of
0.75 against the scalar's 0.909, so the solution was rescaled to satisfy `E‖ŷ‖² = E‖y‖²` —
"keep the per-expert *structure*, which is the actual innovation, and leave the global scale
exactly where P5 validated it."

The rescale fixes the **aggregate** symptom with one number per layer. Attenuation is
**per-expert**. After rescaling, total layer energy is right and its *allocation* is still tilted
away from promoted experts toward incumbents. The technique diagnosed the disease and then
treated only its layer-mean.

Stated as a rule the correction reads: *the more the post-prune router relies on an expert, the
harder we suppress it.* That is a defensible thing to do when minimising `‖y_unpruned − ŷ‖²`
against unpruned targets. It is the wrong thing to do to a model that has to run without them.

## The proxy failed, and it failed confidently

The reconstruction residual `Σ‖y−ŷ‖²/Σ‖y‖²` said the opposite, measured on **held-out** tokens
with a 50/50 interleaved split, improving in **41 of 42 layers**:

| correction | mean rel. residual | end-to-end top-1 |
|---|---|---|
| none | 0.3332 | — |
| per-layer scalar | 0.2914 | **0.84238** |
| per-expert (shipped) | **0.2663** | 0.83693 |

**Better reconstruction, worse model.** The residual is not a broken measurement — it is a
correct measurement of the wrong thing, in two ways that compound:

1. **MSE buys variance reduction with signal.** 96.6% of an expert's output energy is
   token-dependent residual (`‖μ‖²/E‖f‖² = 0.034`). Shrinking a mismatched expert reliably lowers
   squared error, because a damped wrong answer is closer to the target than a loud wrong answer.
   Next-token prediction does not accept that trade: the token-dependent part *is* the
   information, and 3.3× attenuation discards it wholesale. Squared error is indifferent to
   whether the energy it removed was noise or signal.
2. **A per-layer objective does not compose.** Each layer is fitted against *unpruned* inputs. At
   inference every layer receives *pruned* inputs from 44 predecessors. The fit is open-loop,
   validated in a regime the deployed model is never in, and there are 45 layers plus mHC's four
   residual streams for a small consistent tilt to accumulate through.

A hold-out set defends against overfitting the *objective*. Nothing about a hold-out defends
against the objective being the wrong one. That distinction is the reusable lesson here.

## The second correction: the pass-2 mask was real

The ablation retroactively de-confounds the pass-1/pass-2 comparison, because pass 1 shipped
scalar healing. Comparing like for like — **both arms scalar-healed**, same held-out tokens:

| | mask | healing | top-1 agreement |
|---|---|---|---|
| pass 1 (published) | pass 1 | scalar | 0.83703 |
| **pass 2** | **pass 2** | **scalar** | **0.84238** |
| pass 2 as shipped | pass 2 | per-expert | 0.83693 |

**The pass-2 mask is worth +0.00545, not the ~0.0000 previously reported.** McNemar on the same
two captures: 9,304 tokens pass 1 gets right and pass 2 does not, 10,621 the other way, 19,925
discordant — `chi2 = 86.9`, **z = 9.3**, paired s.e. 0.00058. Per-expert healing cost almost
exactly what the new mask gained, and the two cancelled to a null that was then written up as
"the mask improvement did not translate." It translated. It was masked.

Both paired tests are in `artifacts/eval/healing_paired_test.json` and
`artifacts/eval/mask_paired_test.json`; `scripts/heal_paired_test.py` recomputes the first.

This also settles, with an end-to-end number, the question `wiki/30-reap.md` could previously only
answer by two proxies (saliency mass +1.524%, reconstruction residual −2.5%). Both had the right
sign. Both understated the size.

## What was done about it

1. `scripts/heal_revert_to_scalar.py` multiplies each retained expert's `down_proj.weight_scale_inv`
   by `scalar_gain / c_j` on the pass-2 FP8 checkpoint — the same arithmetic the ablation ran in
   memory, made persistent. 5,760 tensors across 40 layers, multiplier range 0.7902–3.3029.
   Idempotent under a `{target, keep_set_sha, op}` fingerprint, and verified afterwards against
   `artifacts/preheal_probe.json`: post-revert scale ÷ pre-heal scale must equal the **layer
   scalar**, which for the probed experts differs from `c_j` by 6–27%.
2. NVFP4 rebuilt from the corrected FP8 base — the previously-published NVFP4 descends from the
   per-expert-healed weights and is superseded.
3. Both v2 repositories republished with corrected cards. The correction is invertible at zero
   quality cost, so there is no argument for shipping the inferior arm with a caveat attached.

## What survives

The per-expert machinery is not wasted, and three of its by-products stand independently of the
verdict:

* **Orthogonality, finally measured** — off-diagonal mass 1.9%, mean `|cos(μ_i,μ_j)|` 0.091, full
  solve vs diagonal form agreeing to four decimals. This validates the assumption the *scalar*
  rests on, which is now the shipped correction.
* **The mean/residual split** — `‖μ‖²/E‖f‖² = 0.034`. It closed off per-token output matching
  before that work was started, and it is half the explanation for why this ablation went the way
  it did.
* **A scorable objective for candidate masks** — `heal_perexpert.py --keep-set` compares two
  keep-sets on a common denominator in ~8 minutes. Given what is above, its verdicts now need
  end-to-end confirmation before they are acted on, but as a *cheap filter* it had the right sign
  on the pass-1/pass-2 question.

## Method note

This is the fourth time on this project that a locally impeccable derivation optimised something
subtly beside the point: the first-moment healing gain (over-corrected 30.8% by ignoring
`norm_topk_prob` renormalisation), the eval scoring (counted image placeholders as text and made
the pruned student beat its teacher), the healing ledger (made healing a silent no-op), and now
this.

The pattern is consistent enough to name: **every one was caught by an end-to-end measurement and
none by the derivation's own validation.** The per-expert fit had a unit test against a
brute-force simulation, a held-out split, a clamp-bound check, a two-candidate ship gate, and an
orthogonality report — five layers of internal validation, all passing, all measuring the wrong
quantity. The ablation cost one eval run.

The operational rule that follows: **a weight-space change ships only after an end-to-end arm,
never on a reconstruction proxy alone.** It is affordable precisely when the change is invertible,
which weight-space rescalings always are — so the cost of finding out is one eval, and the cost of
not finding out is a published checkpoint.
