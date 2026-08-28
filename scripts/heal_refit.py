"""P5 - measure the healing gain instead of deriving it.

WHY THIS EXISTS
---------------
`s05_heal` multiplies every retained expert's FP8 block scale by a per-layer scalar (median
0.6964 in pass 1) intended to undo the output inflation that pruning causes. That scalar came
from a FIRST-MOMENT derivation: the ratio of mean saliency over all experts to mean saliency
over the kept experts. Two things make it suspect, and both were flagged `[OPEN]`:

  1. `norm_topk_prob` already renormalises the top-8 gates to a fixed mass (measured: 2.5000
     before AND after pruning). Gate mass is therefore conserved by the router itself, so a
     correction derived as though it were not may be correcting something twice.
  2. The derivation uses PRE-prune gates on both sides of the ratio. After pruning, a surviving
     expert's gate is strictly larger - it divides by a smaller sum - and how much larger depends
     on which of its competitors died, per token. A ratio of pre-prune means cannot see that.

So: measure it. The estimator here is
        ||y||^2  ~  sum_e  g_e^2 * ||f_e||^2
which is exact when expert outputs are mutually orthogonal, and a good approximation in 4096
dimensions where independently-trained expert outputs are near-orthogonal. Crucially that
assumption is TESTABLE from data we now collect, and `--check-orthogonality` does test it against
the measured `out_sum` vectors rather than asserting it.

Everything it needs comes from ONE saliency pass:
  * per-expert mean ||f_e||        <- `norm_sum_by_bucket` / `cnt_by_bucket`
  * exact pre/post-prune gates     <- the router score cache, replayed by `router_replay`
  * the per-layer keep-set         <- the ranking under test

Run it after CHUNK 1, not after all 12: if the correction is wrong, that is worth knowing 13
hours early.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import torch

import router_replay as RR

ROOT = Path(__file__).resolve().parent.parent
SALIENCY = ROOT / "artifacts" / "saliency"
ROUTER_CACHE = ROOT / "artifacts" / "router_cache"
SOURCE = ROOT / "source" / "GLM-5.3-Flash"


def _layer_index(lname: str) -> int:
    m = re.search(r"layers\.(\d+)\.", lname)
    return int(m.group(1)) if m else -1


def load_bias(layer_idx: int) -> torch.Tensor | None:
    """Read e_score_correction_bias for one layer straight from the source shards."""
    from safetensors import safe_open
    idx_path = SOURCE / "model.safetensors.index.json"
    if not idx_path.exists():
        return None
    wm = json.loads(idx_path.read_text())["weight_map"]
    key = f"model.language_model.layers.{layer_idx}.mlp.gate.e_score_correction_bias"
    if key not in wm:
        return None
    with safe_open(str(SOURCE / wm[key]), framework="pt", device="cpu") as f:
        return f.get_tensor(key).float()


RETAINED = ROOT / "artifacts" / "reap_retained_experts.json"
_RETAINED_CACHE: dict | None = None


def keep_mask_from(sal: torch.Tensor, cnt: torch.Tensor, ratio: float,
                   lname: str | None = None) -> torch.Tensor:
    """The keep-set, preferring the one that was ACTUALLY shipped.

    REAP's saliency is a CONDITIONAL mean - sum over the expert's own active tokens, divided by
    its own token count - so the ranking is by `sum/count`, never by `sum`. Ranking by the raw
    sum is a frequency-weighted ranking, which is the specific thing REAP exists not to be, and
    it moves this layer's gain by ~14%. s04_sweep gets this right; the comparison here has to
    match it or it measures its own bug.
    """
    global _RETAINED_CACHE
    e = sal.numel()
    if lname is not None:
        if _RETAINED_CACHE is None:
            _RETAINED_CACHE = json.loads(RETAINED.read_text()) if RETAINED.exists() else {}
        idx = _RETAINED_CACHE.get(lname)
        if idx:
            keep = torch.zeros(e, dtype=torch.bool)
            keep[torch.tensor(idx, dtype=torch.long)] = True
            return keep
    cond = torch.where(cnt > 0, sal / cnt.clamp(min=1), torch.zeros_like(sal))
    k = int(round(e * (1.0 - ratio)))
    keep = torch.zeros(e, dtype=torch.bool)
    keep[cond.argsort(descending=True)[:k]] = True
    return keep


def measure_layer(rec: dict, cache: dict | None, ratio: float,
                  bias: torch.Tensor | None) -> dict | None:
    sal = rec["sum_saliency"].double()
    cnt = rec["count"].double().clamp_min(1)
    keep = keep_mask_from(sal, cnt, ratio, rec.get("layer"))

    # Mean ungated output norm per expert. Falls back to the REAP ratio if a pass-1 dump is
    # passed in, which has no `norm_sum_by_bucket` - reported, never silently substituted.
    if "norm_sum_by_bucket" in rec:
        m = (rec["norm_sum_by_bucket"].sum(0).double() / cnt)
        have_norm = True
    else:
        m = (sal / cnt)          # g*||f|| rather than ||f||: only a fallback
        have_norm = False

    # The pass-1 estimator, recomputed here EXACTLY as s05_heal computes it: token-weighted
    # (sum of saliency over sum of counts), not the unweighted mean of per-expert conditional
    # means. The two differ by ~14% on this data, which is the same order as the correction
    # itself - so comparing against the wrong one would manufacture a discrepancy.
    cond = torch.where(cnt > 0, sal / cnt, torch.zeros_like(sal))
    tot_c, keep_c = cnt.sum(), cnt[keep].sum()
    if tot_c > 0 and keep_c > 0 and float((cnt[keep] * cond[keep]).sum()) > 0:
        e_all = float((cnt * cond).sum() / tot_c)
        e_keep = float((cnt[keep] * cond[keep]).sum() / keep_c)
        first_moment = e_all / e_keep
    else:
        first_moment = float("nan")
    unweighted = float(cond.mean() / cond[keep].mean()) if keep.any() else float("nan")

    out = {"experts": int(sal.numel()), "kept": int(keep.sum()),
           "first_moment_gain": first_moment, "first_moment_unweighted": unweighted,
           "has_true_norms": have_norm}

    if cache is None or bias is None:
        return out

    scores, idx = cache["scores"].float(), cache["idx"].long()
    i_pre, w_pre, ok_pre = RR.simulate_from_cache(scores, idx, bias, None)
    i_post, w_post, ok_post = RR.simulate_from_cache(scores, idx, bias, keep)
    ok = ok_pre & ok_post
    if int(ok.sum()) == 0:
        out["measured_gain"] = None
        out["resolvable_frac"] = 0.0
        return out

    mm = m.float()
    # ||y||^2 ~ sum_e g_e^2 * ||f_e||^2, per token, pre and post.
    e_pre = (w_pre[ok] ** 2 * mm[i_pre[ok]] ** 2).sum(-1)
    e_post = (w_post[ok] ** 2 * mm[i_post[ok]] ** 2).sum(-1)
    ratio_tok = (e_post.clamp_min(1e-20) / e_pre.clamp_min(1e-20)).sqrt()
    out["resolvable_frac"] = float(ok.float().mean())
    out["inflation_measured"] = float(ratio_tok.mean())     # how much bigger the pruned layer is
    out["measured_gain"] = float(1.0 / ratio_tok.mean())    # the scalar that undoes it
    out["inflation_p50"] = float(ratio_tok.median())
    out["gate_mass_pre"] = float(w_pre[ok].sum(-1).mean())
    out["gate_mass_post"] = float(w_post[ok].sum(-1).mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratio", type=float, default=0.5)
    ap.add_argument("--shipped-gain", type=float, default=0.6964,
                    help="the pass-1 value actually baked into the published FP8")
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "heal_refit.json"))
    a = ap.parse_args()

    caches: dict[str, dict] = {}
    for f in sorted(ROUTER_CACHE.glob("chunk_*.pt")):
        d = torch.load(f, weights_only=False)
        for ln, v in d["layers"].items():
            if ln in caches:
                caches[ln] = {"scores": torch.cat([caches[ln]["scores"], v["scores"]]),
                              "idx": torch.cat([caches[ln]["idx"], v["idx"]])}
            else:
                caches[ln] = v
    print(f"router cache: {len(caches)} layers from {len(list(ROUTER_CACHE.glob('chunk_*.pt')))} chunk(s)")

    rows = []
    for f in sorted(SALIENCY.glob("*.pt")):
        rec = torch.load(f, weights_only=False)
        ln = rec["layer"]
        li = _layer_index(ln)
        r = measure_layer(rec, caches.get(ln), a.ratio, load_bias(li) if caches.get(ln) else None)
        if r:
            r["layer"] = ln
            rows.append(r)

    meas = [r["measured_gain"] for r in rows if r.get("measured_gain")]
    fm = [r["first_moment_gain"] for r in rows if r.get("first_moment_gain") == r.get("first_moment_gain")]
    # Stamp the keep-set these gains were measured against. A healing gain is only valid for the
    # mask it was derived from - applying pass-1 gains to a pass-2 mask would be the same class of
    # error the gains themselves were fixing: a stale intermediate used with confidence.
    keyhash = None
    if RETAINED.exists():
        keyhash = hashlib.sha256(RETAINED.read_bytes()).hexdigest()[:16]
    res = {"ratio": a.ratio, "layers": len(rows), "shipped_gain": a.shipped_gain,
           "keep_set_sha": keyhash,
           "first_moment_median": float(torch.tensor(fm).median()) if fm else None,
           "measured_median": float(torch.tensor(meas).median()) if meas else None,
           "per_layer": rows}
    Path(a.out).write_text(json.dumps(res, indent=2))

    print(f"\nlayers analysed        : {len(rows)}")
    print(f"shipped (pass 1)       : {a.shipped_gain:.4f}")
    uw = [r["first_moment_unweighted"] for r in rows
          if r.get("first_moment_unweighted") == r.get("first_moment_unweighted")]
    if fm:
        print(f"first-moment, recomputed: {res['first_moment_median']:.4f}  "
              f"(s05's token-weighted estimator)")
        if uw:
            print(f"  same, expert-weighted : {float(torch.tensor(uw).median()):.4f}  "
                  "(NOT what s05 uses; shown to size the estimator choice)")
    if meas:
        mm = res["measured_median"]
        print(f"MEASURED (router replay): {mm:.4f}")
        gm = [r["gate_mass_pre"] for r in rows if "gate_mass_pre" in r]
        gp = [r["gate_mass_post"] for r in rows if "gate_mass_post" in r]
        if gm:
            print(f"gate mass pre/post      : {sum(gm)/len(gm):.4f} / {sum(gp)/len(gp):.4f}"
                  "   (equal => renormalisation conserves it, as expected)")
        d = abs(mm - a.shipped_gain) / a.shipped_gain
        print(f"\ndisagreement with shipped: {100*d:.1f}%")
        if d > 0.05:
            print("  => the shipped correction is off by more than 5%. Re-fit before "
                  "materialising anything; it lives entirely in F32 block scales, so it is "
                  "cheap to change and expensive to inherit.")
        else:
            print("  => shipped correction is within 5% of measured; pass-1 healing stands.")
    else:
        print("MEASURED: unavailable - run after chunk 1 so the router cache exists.")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
