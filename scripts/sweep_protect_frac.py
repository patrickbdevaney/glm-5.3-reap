"""Measure protect_frac and read the knee off the curve.

conf/protect_frac_override.txt has held 0.0 since 2026-09-26 with an explicit note that it is
"a PLACEHOLDER, not a decision": the right value is a property of THIS run's accumulators, which
did not exist then. They exist now (s03 complete, 42 layers, 10 chunks), so this is the
measurement that placeholder was waiting for.

WHAT IS BEING TRADED. protect_frac reserves the top slice of every DOMAIN separately, so a thin
domain keeps the experts only it uses. Pass 2 shipped protect_frac=0 and lost ballast
(dNLL 0.989, top-1 0.580 against 0.916 for code) -- a global ranking cannot see an expert that
only one 5%-of-corpus domain needs. But protection is not free: every protected expert is one the
greedy cannot drop, so interaction cost (p^T F p, HOPE's own objective) rises with it.

The knee is where worst-domain retention stops improving fast enough to justify the interaction
cost it costs.

NOT MONOTONE NEAR ZERO. On MiMo, pf=0.01 was WORSE than pf=0, because too few protected experts
constrain the greedy without covering anything. So the grid starts dense near zero rather than
assuming the curve rises from the origin, and MiMo's 0.08 is NOT assumed to transfer: this model
has 288 experts over 8 domains (7 live), a different corpus and a different F.

bio is dead by LICENCE, not by an incomplete pass: every bio source is in corpus_spec.EXCLUDED
(camel-ai/biology licence_nc, tattabio/OG licence_sharealike, PubMedQA scope_clinical). No amount
of recomputation populates it, so the sweep scores the 7 live domains.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import reap_select  # noqa: E402

ACC = ROOT / "artifacts" / "saliency" / "accumulators.pt"
RATIO = float(__import__("os").environ.get("SWEEP_RATIO", "0.5"))
GRID = [0.0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.12, 0.16]


def main() -> int:
    if not ACC.exists():
        print(f"missing {ACC}"); return 1
    out = ROOT / "artifacts" / "_pf_sweep"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"ratio {RATIO}, HOPE selection, 7 live domains (bio excluded by licence)\n")
    print(f"{'pf':>6} {'worst_dom':>10} {'worst_ret':>10} {'mean_ret':>9} "
          f"{'icost':>8} {'icost_max':>9}")
    for pf in GRID:
        try:
            r = reap_select.run(ACC, out / f"pf_{pf}.json", RATIO, "hope", "reap_1_1_1",
                                protect_frac=pf, allow_dead=True)
        except ValueError as e:
            # Protection can exceed the prunable population: protecting k experts per domain
            # across 7 live domains can leave fewer than n_prune unprotected. That is a hard
            # feasibility ceiling, not a tuning failure -- stop the grid and say where it is.
            print(f"{pf:>6.3f}  INFEASIBLE: protection exceeds the prunable population "
                  f"({type(e).__name__}). Grid stops here.")
            break
        rows.append((pf, r["worst_retention"], r["mean_retention"], r["interaction_cost"]))
        print(f"{pf:>6.3f} {r['worst_domain']:>10} {r['worst_retention']:>10.4f} "
              f"{r['mean_retention']:>9.4f} {r['interaction_cost']:>8.4f} "
              f"{r['interaction_cost_max']:>9.4f}")

    # MARGINAL efficiency, not cumulative. Cumulative gain/cost falls monotonically by
    # construction and would name the smallest nonzero pf the "knee" every time; the knee is
    # where the NEXT increment stops paying for itself.
    print("\nmarginal gain per unit interaction cost (knee = last step with ratio >= 1):")
    print(f"{'step':>14} {'d_worst':>9} {'d_icost':>9} {'ratio':>7}")
    knee = rows[0][0]
    for (p0, w0, m0, i0), (p1, w1, m1, i1) in zip(rows, rows[1:]):
        dw, di = w1 - w0, i1 - i0
        ratio = dw / di if di > 0 else float("inf")
        print(f"{p0:>6.3f}->{p1:<6.3f} {dw:>+9.4f} {di:>+9.4f} {ratio:>7.2f}")
        if ratio >= 1.0:
            knee = p1
    print(f"\nKNEE (marginal) = {knee}")

    base_w, base_i = rows[0][1], rows[0][3]
    print(f"\ncumulative against pf=0 (worst {base_w:.4f}, icost {base_i:.4f}), for reference only --")
    print("cumulative gain/cost falls monotonically by construction, so its maximum is always the")
    print("smallest nonzero pf and is NOT the knee:")
    print(f"{'pf':>6} {'d_worst':>9} {'d_icost':>9} {'ratio':>8}")
    for pf, w, m, ic in rows[1:]:
        dw, di = w - base_w, ic - base_i
        print(f"{pf:>6.3f} {dw:>+9.4f} {di:>+9.4f} {(dw/di if di>0 else float('inf')):>8.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
