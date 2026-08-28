"""P7 - emit every candidate mask the accumulators support, and compare them.

Pass 1 could compute exactly ONE criterion, because the tracker kept one summed number and one
count per expert. Every alternative needed another 14-hour pass, so the criterion was never a
choice - it was whatever the tracker happened to record. Pass 2's accumulators make eight
criteria available offline, at zero marginal compute.

The point is NOT to declare a winner from these numbers. It is to find out whether the choice
matters at all. If every criterion picks nearly the same experts, the question is closed for
free and the budget belongs to evaluation. If they diverge, we have candidate masks to
materialise and decide with measurement (P10/P13) instead of by assertion.

Criteria
--------
  reap          mean(g*||f||) over the expert's own tokens   -- stock REAP, the baseline
  quantile      0.6*mean + 0.4*p99, p99 from the log-histogram -- pass 1 deferred this for want
                of per-token quantiles; the histogram now supplies them
  var_aware     mean + lambda*std of g*||f||                 -- rewards consistently-useful
                experts over ones with a few huge activations
  norm_only     mean ||f|| alone, gate removed               -- separates "this expert computes
                something big" from "the router likes it"
  gate_only     mean g alone                                 -- the pure router-preference view
  frequency     sum(g*||f||), NOT the mean                   -- the frequency-weighted ranking
                REAP exists to avoid; included as a control that SHOULD look different
  mix_sample    per-bucket means re-weighted to the intended SAMPLE mixture
  mix_codemath  per-bucket means re-weighted toward code+math
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
SALIENCY = ROOT / "artifacts" / "saliency"


def _p_from_hist(hist: torch.Tensor, lo: float, hi: float, bins: int, q: float) -> torch.Tensor:
    """Approximate a per-expert quantile of g*||f|| from its log-histogram."""
    c = hist.double().cumsum(-1)
    tot = c[:, -1:].clamp_min(1)
    idx = (c / tot < q).sum(-1).clamp(max=bins - 1)
    centres = lo + (idx.double() + 0.5) * (hi - lo) / bins
    return torch.pow(10.0, centres)


def criteria_for(rec: dict, lam: float = 0.5) -> dict[str, torch.Tensor]:
    sal = rec["sum_saliency"].double()
    cnt = rec["count"].double().clamp_min(1)
    mean = sal / cnt
    out = {"reap": mean, "frequency": sal}

    if "sq_by_bucket" in rec:
        sq = rec["sq_by_bucket"].sum(0).double()
        var = (sq / cnt - mean.pow(2)).clamp_min(0)
        out["var_aware"] = mean + lam * var.sqrt()
    if "norm_sum_by_bucket" in rec:
        out["norm_only"] = rec["norm_sum_by_bucket"].sum(0).double() / cnt
    if "gate_sum_by_bucket" in rec:
        out["gate_only"] = rec["gate_sum_by_bucket"].sum(0).double() / cnt
    if "hist" in rec and "hist_range" in rec:
        lo, hi, bins = rec["hist_range"]
        out["quantile"] = 0.6 * mean + 0.4 * _p_from_hist(rec["hist"], lo, hi, bins, 0.99)
    if "sum_by_bucket" in rec:
        sb = rec["sum_by_bucket"].double()
        cb = rec["cnt_by_bucket"].double().clamp_min(1)
        per = sb / cb                                   # [buckets, experts] conditional means
        names = rec.get("buckets", [])
        def mix(w: dict[str, float]) -> torch.Tensor:
            acc = torch.zeros_like(mean)
            tot = 0.0
            for i, b in enumerate(names):
                if b in w and float(rec["cnt_by_bucket"][i].sum()) > 0:
                    acc += w[b] * per[i]
                    tot += w[b]
            return acc / max(tot, 1e-9)
        out["mix_sample"] = mix({"agentic": .24, "code": .21, "math": .15, "vision": .15,
                                 "science": .10, "finance": .08, "general": .07})
        out["mix_codemath"] = mix({"code": .35, "math": .30, "agentic": .15, "vision": .10,
                                   "science": .05, "finance": .03, "general": .02})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratio", type=float, default=0.5)
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "criterion_shootout.json"))
    a = ap.parse_args()

    files = sorted(SALIENCY.glob("*.pt"))
    if not files:
        raise SystemExit("no saliency dumps")
    keeps: dict[str, dict[str, set]] = {}
    for f in files:
        rec = torch.load(f, weights_only=False)
        ln = rec["layer"]
        for name, score in criteria_for(rec).items():
            k = int(round(score.numel() * (1.0 - a.ratio)))
            keeps.setdefault(name, {})[ln] = set(
                int(i) for i in score.argsort(descending=True)[:k])

    names = sorted(keeps)
    base = "reap"
    rows = {}
    for n in names:
        if n == base:
            continue
        ov = [len(keeps[n][ln] & keeps[base][ln]) / max(1, len(keeps[base][ln]))
              for ln in keeps[base] if ln in keeps[n]]
        t = torch.tensor(ov)
        rows[n] = {"overlap_vs_reap_mean": float(t.mean()),
                   "overlap_vs_reap_min": float(t.min()),
                   "differs_by_experts": round((1 - float(t.mean())) * len(next(iter(keeps[base].values()))))}

    pair = {}
    for x, y in itertools.combinations(names, 2):
        ov = [len(keeps[x][ln] & keeps[y][ln]) / max(1, len(keeps[x][ln]))
              for ln in keeps[x] if ln in keeps[y]]
        pair[f"{x}|{y}"] = round(float(torch.tensor(ov).mean()), 4)

    res = {"ratio": a.ratio, "layers": len(files), "criteria": names,
           "vs_reap": rows, "pairwise_mean_overlap": pair}
    Path(a.out).write_text(json.dumps(res, indent=2))

    print(f"criteria available: {len(names)}  layers: {len(files)}  ratio: {a.ratio:.0%}\n")
    print(f"{'criterion':14s} {'overlap vs reap':>16s} {'min':>8s}  {'experts differing':>18s}")
    for n in sorted(rows, key=lambda k: -rows[k]["overlap_vs_reap_mean"]):
        r = rows[n]
        print(f"{n:14s} {r['overlap_vs_reap_mean']:16.4f} {r['overlap_vs_reap_min']:8.4f}"
              f" {r['differs_by_experts']:18d}")
    spread = 1 - min(r["overlap_vs_reap_mean"] for r in rows.values()) if rows else 0.0
    print()
    if spread < 0.05:
        print(f"Every criterion agrees within {100*spread:.1f}% of REAP's keep-set. The choice of "
              "criterion does not matter at this ratio - close the question and spend the budget "
              "on evaluation instead.")
    else:
        print(f"Criteria disagree by up to {100*spread:.1f}% of the keep-set. Materialise the two "
              "most divergent plausible masks and decide with measured dNLL, not by argument.")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
