"""Per-domain saliency retention: which domains did the prune actually cost?

The evaluation measures dNLL per domain, but only after two multi-hour scoring passes. This
answers a sharper, cheaper question from data the sweep already produced:

    for each domain, what fraction of the expert output IT relies on survived the prune?

REAP ranks experts on saliency pooled across all domains, so a domain whose useful experts are
rare overall can lose disproportionately - and that is exactly what a "hole" is. Because pass 2
accumulates saliency per bucket, this is a sum over the keep-set, computable in seconds:

    retention[b] = sum over KEPT experts of  saliency_b[j]   /   sum over ALL experts of saliency_b[j]

A domain retaining materially less than the others has had more of what it uses deleted. This is
predictive - available before any evaluation - and it says WHERE to look rather than only that
something is wrong.
"""
from __future__ import annotations
import argparse, json, statistics as st
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
SAL = ROOT / "artifacts" / "saliency"
KEEP = ROOT / "artifacts" / "reap_retained_experts.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "domain_holes.json"))
    a = ap.parse_args()
    keep = json.loads(KEEP.read_text())
    per_layer, buckets = [], None
    for f in sorted(SAL.glob("*.pt")):
        d = torch.load(f, weights_only=False)
        ln = d["layer"]
        if ln not in keep or "sum_by_bucket" not in d:
            continue
        buckets = d["buckets"]
        sb = d["sum_by_bucket"].double()                 # [n_buckets, n_experts]
        k = torch.tensor(sorted(keep[ln]))
        tot = sb.sum(dim=1)                              # each domain's total gated output
        kept = sb[:, k].sum(dim=1)
        row = {"layer": ln}
        for i, b in enumerate(buckets):
            if float(tot[i]) > 0:
                row[b] = float(kept[i] / tot[i])
        per_layer.append(row)

    agg = {}
    for b in buckets:
        vals = [r[b] for r in per_layer if b in r]
        if vals:
            agg[b] = {"mean": st.mean(vals), "min": min(vals),
                      "worst_layer": min(per_layer, key=lambda r: r.get(b, 9))["layer"],
                      "layers": len(vals)}
    overall = st.mean([v["mean"] for v in agg.values()])
    Path(a.out).write_text(json.dumps({"per_domain": agg, "per_layer": per_layer}, indent=2))

    print(f"{'domain':10s} {'retained':>9s} {'worst layer':>12s} {'vs mean':>9s}")
    for b, v in sorted(agg.items(), key=lambda kv: kv[1]["mean"]):
        rel = 100 * (v["mean"] - overall) / overall
        flag = "  <-- HOLE" if rel < -3 else ("  <-- weak" if rel < -1 else "")
        print(f"{b:10s} {v['mean']:9.4f} {v['worst_layer'].split('.')[-2]:>12s} {rel:+8.2f}%{flag}")
    print(f"\nmean retention across domains: {overall:.4f}")
    spread = max(v['mean'] for v in agg.values()) - min(v['mean'] for v in agg.values())
    print(f"spread best-to-worst          : {100*spread:.2f} points")
    if spread < 0.03:
        print("\nNo domain-specific hole: the prune cost every domain about the same share of "
              "the expert output it uses. Regression should be broad, not concentrated.")
    else:
        print("\nUneven: the domains flagged above lost materially more of the expert output they "
              "rely on, and are where the evaluation should be read most carefully.")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
