"""Stage 4 - prune-ratio sweep and gate, entirely from cached saliency.

Lever L3: building a pruned model at ratio r does not require materialising a checkpoint.
The raw accumulators dumped in stage 3 let every ratio be re-ranked in seconds, so this stage
costs minutes rather than the 4-12 h a checkpoint-per-ratio sweep would.

30% is deliberately NOT evaluated for a decision: it is provably over the 117 GiB envelope
(123.3 GiB), so its quality answers a question we cannot act on. It is reported for the curve.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import ROOT, ARTIFACTS, log, metric, kv_get, publish  # noqa: E402

STAGE = "s04_sweep"
SALIENCY = ROOT / "artifacts" / "saliency"
RATIOS = [0.30, 0.40, 0.50, 0.55]
ENVELOPE_GIB = 117.0

# From the exact accounting in wiki/10-target-model.md
ROUTED_EXPERT_PARAMS = 311_672_586_240
NON_EXPERT_BF16 = 6_926_096_640
NON_EXPERT_FP8 = 2_743_074_816
NVFP4_BPW = 4.5


def _size_gib(ratio: float, rest_fp8: bool) -> float:
    experts = ROUTED_EXPERT_PARAMS * (1 - ratio) * NVFP4_BPW / 8
    rest = NON_EXPERT_BF16 * (1 if rest_fp8 else 2) + NON_EXPERT_FP8
    return (experts + rest) / 2**30


def run() -> dict:
    import torch
    files = sorted(SALIENCY.glob("*.pt"))
    if not files:
        raise RuntimeError("no saliency accumulators found; stage 3 must run first")

    per_layer = []
    for f in files:
        d = torch.load(f, weights_only=False)
        s_sum = d["sum_saliency"].double()
        cnt = d["count"].double()
        mean = torch.where(cnt > 0, s_sum / cnt.clamp(min=1), torch.zeros_like(s_sum))
        per_layer.append({"layer": d["layer"], "mean": mean, "count": cnt})

    sweep = {}
    for r in RATIOS:
        keep_mass, sal_mass, layers = [], [], 0
        for L in per_layer:
            n = L["mean"].numel()
            k = n - int(n * r)
            order = torch.argsort(L["mean"], descending=True)
            keep = order[:k]
            tot_c, tot_s = L["count"].sum(), (L["mean"] * L["count"]).sum()
            if tot_c > 0:
                keep_mass.append(float(L["count"][keep].sum() / tot_c))
                sal_mass.append(float((L["mean"][keep] * L["count"][keep]).sum() / tot_s))
            layers += 1
        rm = sum(keep_mass) / max(len(keep_mass), 1)
        sm = sum(sal_mass) / max(len(sal_mass), 1)
        # REAP's saliency is a CONDITIONAL mean over each expert's own active token set, so
        # it is frequency-invariant: selecting the top-saliency experts does not preferentially
        # retain high-traffic ones. Expected retained routing mass is therefore ~= (1 - ratio),
        # NOT ~1.0. Gating raw routing mass against a 0.90 threshold would flag a perfectly
        # healthy 50% prune as RED. What matters is (a) how much of the layer's total output
        # contribution survives - saliency mass - and (b) whether retained experts carry MORE
        # traffic than an average expert, which is routing mass relative to expectation.
        rm_ratio = rm / max(1 - r, 1e-9)
        entry = {
            "routing_mass_retained": round(rm, 4),
            "routing_mass_vs_expected": round(rm_ratio, 3),
            "saliency_mass_retained": round(sm, 4),
            "min_layer_routing_mass": round(min(keep_mass), 4) if keep_mass else None,
            "size_gib_rest_fp8": round(_size_gib(r, True), 1),
            "size_gib_rest_bf16": round(_size_gib(r, False), 1),
            "fits_envelope": _size_gib(r, True) < ENVELOPE_GIB,
            "layers": layers,
        }
        # Gate against the RANDOM-PRUNING NULL, not an absolute constant. Removing a
        # fraction r of experts at random retains (1 - r) of the total g*||f|| contribution by
        # construction, so "saliency mass retained" alone says nothing without that reference:
        # 0.729 at a 50% prune sounds poor and is in fact 1.46x better than random.
        # Absolute thresholds here were guesses; this ratio is anchored.
        concentration = sm / max(1 - r, 1e-9)
        entry["concentration_vs_random"] = round(concentration, 3)
        entry["gate"] = ("green" if concentration > 1.40
                         else "amber" if concentration > 1.15 else "red")
        sweep[f"{r:.2f}"] = entry
        metric(STAGE, "routing_mass_retained", rm, tag=f"{r:.2f}")
        metric(STAGE, "saliency_mass_retained", sm, tag=f"{r:.2f}")
        log(f"ratio {r:.0%}: saliency mass {sm:.3f} = x{concentration:.2f} vs random "
            f"({entry['gate']}), routing mass {rm:.3f} (x{rm_ratio:.2f} vs expected), "
            f"{entry['size_gib_rest_fp8']:.0f} GiB, fits={entry['fits_envelope']}", STAGE)

    chosen = kv_get("chosen_ratio", 0.50)
    c = sweep[f"{chosen:.2f}"]
    verdict = {
        "chosen_ratio": chosen,
        "gate": c["gate"],
        "note": ("30% is reported for the curve only - it is provably over the 117 GiB "
                 "envelope, so its quality cannot change the decision."),
        "saliency_arm": ("A (stock REAP conditional mean). Arm B (quantile-blended "
                         "0.6*mean+0.4*p99) is NOT available: the stock tracker accumulates "
                         "only sum and count, so per-token quantiles were never retained. "
                         "Running arm B requires modifying REAP's _expert_hook to keep a "
                         "sketch, and a second calibration pass. Deferred, not silently "
                         "dropped."),
    }
    if c["gate"] == "red":
        msg = (f"GATE RED at {chosen:.0%}: retained saliency mass is only "
               f"x{c['concentration_vs_random']} better than random pruning. This is the "
               f"disproportionate-damage signal the directive asks to stop on.")
        log(msg, STAGE, "ERROR")
        if not kv_get("override_red_gate", False):
            # An autonomous pipeline must not push past its own go/no-go gate.
            raise RuntimeError(msg + " Halting. Set kv override_red_gate=true to proceed.")
    else:
        log(f"gate {c['gate'].upper()} at {chosen:.0%}: saliency mass "
            f"{c['saliency_mass_retained']:.3f}, routing mass "
            f"x{c['routing_mass_vs_expected']} vs expected", STAGE)
    res = {"sweep": sweep, "verdict": verdict}
    out = ARTIFACTS / "s04_sweep.json"
    out.write_text(json.dumps(res, indent=2, default=str))
    publish(out, "artifacts", "stage04/s04_sweep.json", stage=STAGE)
    return res
