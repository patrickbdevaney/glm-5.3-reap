"""P6 - the split-half stopping rule: was the calibration budget actually enough?

Pass 1's dominant defect was sampling, not the criterion: 501/12,096 expert slots were decided on
under 2,000 tokens, and ~25.5 experts per layer sat within +-5% of the cut. But "2,000 tokens" was
a guessed floor, not a measurement. This measures the thing that actually matters instead.

Split the calibration into two independent halves, rank each one separately, and ask how much the
two keep-sets agree. If half the data and the other half of the data choose nearly the same
experts, the budget is sufficient and more tokens would not move the decision. If they disagree,
the mask is still being determined by sampling noise and the extra tokens are the cheapest
available quality.

Reads the cumulative per-chunk snapshots written by stream_saliency.dump_light: half A is the
snapshot at the midpoint chunk, half B is the final snapshot minus that one.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
SNAP = ROOT / "artifacts" / "saliency_snapshots"


def _keep(sal: torch.Tensor, cnt: torch.Tensor, ratio: float) -> set[int]:
    cond = torch.where(cnt > 0, sal / cnt.clamp(min=1), torch.zeros_like(sal))
    k = int(round(sal.numel() * (1.0 - ratio)))
    return set(int(i) for i in cond.argsort(descending=True)[:k])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratio", type=float, default=0.5)
    ap.add_argument("--gate", type=float, default=0.95)
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "split_half.json"))
    a = ap.parse_args()

    chunks = sorted(d for d in SNAP.glob("chunk_*") if d.is_dir())
    if len(chunks) < 2:
        raise SystemExit(f"need >=2 chunk snapshots, found {len(chunks)}")
    mid = chunks[len(chunks) // 2 - 1]
    last = chunks[-1]
    print(f"half A = {mid.name} (cumulative), half B = {last.name} - {mid.name}")

    rows, overlaps = [], []
    for fa in sorted(mid.glob("*.pt")):
        fb = last / fa.name
        if not fb.exists():
            continue
        A, B = torch.load(fa, weights_only=False), torch.load(fb, weights_only=False)
        sa, ca = A["sum_saliency"].double(), A["count"].double()
        # Half B is the DIFFERENCE of cumulative snapshots - the two halves must be disjoint or
        # the overlap is measured against itself and comes out spuriously high.
        sb, cb = B["sum_saliency"].double() - sa, B["count"].double() - ca
        if float(cb.sum()) <= 0:
            continue
        ka, kb = _keep(sa, ca, a.ratio), _keep(sb, cb, a.ratio)
        ov = len(ka & kb) / max(1, len(ka))
        overlaps.append(ov)
        rows.append({"layer": A["layer"], "overlap": ov,
                     "tokens_a": float(ca.sum()), "tokens_b": float(cb.sum()),
                     "min_tokens_per_expert_b": float(cb.min())})

    if not overlaps:
        raise SystemExit("no comparable layers")
    t = torch.tensor(overlaps)
    res = {"ratio": a.ratio, "gate": a.gate, "layers": len(rows),
           "overlap_mean": float(t.mean()), "overlap_min": float(t.min()),
           "overlap_p10": float(t.quantile(0.10)),
           "layers_below_gate": int((t < a.gate).sum()), "per_layer": rows}
    Path(a.out).write_text(json.dumps(res, indent=2))

    print(f"\nlayers            : {len(rows)}")
    print(f"keep-set overlap  : mean {res['overlap_mean']:.4f}  p10 {res['overlap_p10']:.4f}  "
          f"min {res['overlap_min']:.4f}")
    print(f"below gate {a.gate}: {res['layers_below_gate']}/{len(rows)} layers")
    worst = sorted(rows, key=lambda r: r["overlap"])[:5]
    print("\nworst layers:")
    for r in worst:
        print(f"  {r['overlap']:.4f}  {r['layer']}  (min tokens/expert in half B: "
              f"{r['min_tokens_per_expert_b']:.0f})")
    if res["overlap_mean"] >= a.gate:
        print(f"\nPASS - two independent halves choose the same experts {100*res['overlap_mean']:.1f}% "
              "of the time. More calibration would not move the mask; spend the budget on "
              "evaluation instead.")
    else:
        print(f"\nFAIL - the mask is still moving with the data ({100*res['overlap_mean']:.1f}% "
              f"< {100*a.gate:.0f}%). More tokens are the cheapest quality available; materialising "
              "now bakes in sampling noise.")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
