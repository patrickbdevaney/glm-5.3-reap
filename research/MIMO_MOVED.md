# MiMo-V2.6-Flash work moved out of this repo, 2026-09-23

This repo is the GLM-5.3-Flash REAP project. The MiMo-V2.6-Flash pipeline now lives in
**`~/xiaomi-2.6-flash-REAP`** (github.com/patrickbdevaney/xiaomi-2.6-flash-REAP): the feasibility
survey, the SOTA methodology sweep, `hope_fmatrix.py`, `mimo_saliency.py`, `mimo_corpus_spec.py`,
`pull_mimo.sh` and their gates.

`scripts/corpus_spec.py` is VENDORED there so that repo runs standalone. **This copy stays
authoritative for the shipped GLM-5.3-Flash-REAP50 artifact** — MiMo overrides `TOKEN_TARGET` in
its own spec rather than editing the mixture here, so the provenance of the GLM REAP cannot be
disturbed by MiMo work and the two cannot silently diverge.

Findings from that work that apply to THIS repo regardless:

- **HOPE (arXiv 2609.18916) subsumes REAP** — REAP is provably HOPE with the pairwise interaction
  terms zeroed, and at 40–50% it beat REAP in every agentic experiment (mean +2.8%, up to +6.1%).
  Any future GLM pass should collect the F-matrix; it is the one statistic that cannot be
  recovered from a finished pass.
- **`criterion_shootout.py` may have dismissed the winner.** Its `frequency` criterion,
  `sum(g*||f||)`, is `(0,1,1)` in the unified family of arXiv 2606.15716 — which wins at 50% when
  calibration is capability-aligned, as ours is. The shootout measured only mask *overlap* and
  labelled it "a control that SHOULD look different"; it was never materialised or evaluated, and
  it was the one criterion that diverged (overlap 0.755, 35 experts/layer).
- **The upstream REAP logit-renorm fix (2026-03-11) is a no-op for us.** `stream_saliency.py:218`
  reads the gate from the model's router, which already renormalises under `norm_topk_prob`.
- **Never judge a REAP by an averaged score** — at high sparsity, calibration strategies span 2.85
  points of averaged accuracy but 51.9 points of code retention (arXiv 2606.03328). `mask_eval`'s
  `by_domain` decomposition is the only valid readout.
