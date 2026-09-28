"""Measure the QUALITY of a candidate expert mask WITHOUT surgery, healing or quantisation.

WHY THIS EXISTS
---------------
Every mask comparison we have made so far is a PROXY: reconstruction error of the dropped
out_sum, retained routing mass, per-domain retained saliency. `mass` beats `mean` on all of
them, and none of them is quality. Worse, reconstruction error is itself frequency-weighted
(total contribution = frequency x per-token contribution), so it structurally FAVOURS `mass`
and on its own is circular. Deciding a multi-day surgery run on it would be unjustified.

This scores a candidate mask directly, in nats/token, by applying it AT RUNTIME to the
unpruned checkpoint: restrict the router's candidates to the keep-set, re-select top-8 among
survivors, renormalise the gate, and apply that mask's healing gain. That is exactly what the
surgically pruned model computes, so the measured dNLL is the mask's real cost.

THE TEACHER IS AN ARM OF THE SAME PASS, not the cached capture.
s09_eval records a real bug where two captures of different lengths were compared by `[:n]`
truncation, i.e. "silently against unrelated positions", reporting a pruned student BEATING
its teacher. Running the unmasked arm in the same sweep over the same rows makes that failure
structurally impossible: every arm sees identical tokens in identical order.

COST. Weights dominate (328 GB streamed once, peak residency ONE decoder layer), so arms are
nearly free: the layer is built once and every arm forwards through it before it is freed.
N masks cost ~one pass, not N passes.

ROUTING SEMANTICS (from stream_saliency.patch_router_for_cache, which is authoritative):
  selection = topk(scores + e_score_correction_bias)   <- bias REORDERS candidates
  gate      = scores, renormalised over the surviving top-8, x routed_scaling_factor
Masking must therefore be applied to the SELECTION score, while the gate is read from
`scores` alone. Getting this backwards silently changes which experts run.
"""
from __future__ import annotations
import argparse, gc, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
ARTIFACTS = ROOT / "artifacts"
NE = 288

# ---------------------------------------------------------------- mask construction
def build_mask(criterion: str) -> dict[str, list[int]]:
    """Keep-set per layer for a named criterion, from the banked pass-2 accumulators."""
    import torch
    if criterion == "shipped":
        return json.loads((ARTIFACTS / "reap_retained_experts.json").read_text())
    out = {}
    for f in sorted((ARTIFACTS / "saliency").glob("*.pt")):
        d = torch.load(f, weights_only=False)
        c = d["count"].double()
        s = d["sum_saliency"].double()
        m = torch.where(c > 0, s / c.clamp(min=1), torch.zeros_like(c))
        key = {"mean": m, "mass": s}[criterion]          # `s` IS m*c
        rank = torch.where(c > 0, key, torch.full_like(m, float("-inf")))
        k = NE // 2
        out[d["layer"]] = sorted(int(i) for i in torch.argsort(rank, descending=True)[:k])
    return out


def healing_gains(mask: dict[str, list[int]]) -> dict[str, float]:
    """REAP's scalar output-scale correction, re-fitted per mask (s05_heal formula).

        gain = E[g||f||] over ALL experts / E[g||f||] over RETAINED

    Retained experts have systematically larger ||f|| than deleted ones, and the router
    renormalises its gates over the survivors, so without this the layer's output norm is
    biased upward. Each candidate mask needs its OWN gain -- reusing the shipped one would
    confound the mask comparison with a mis-fitted scale.
    """
    import torch
    g = {}
    for f in sorted((ARTIFACTS / "saliency").glob("*.pt")):
        d = torch.load(f, weights_only=False); L = d["layer"]
        if L not in mask:
            continue
        c = d["count"].double(); s = d["sum_saliency"].double()
        keep = torch.tensor(mask[L])
        all_mean = s.sum() / c.sum().clamp(min=1)
        kep_mean = s[keep].sum() / c[keep].sum().clamp(min=1)
        g[L] = float(all_mean / kep_mean.clamp(min=1e-12))
    return g


# ---------------------------------------------------------------- runtime masking
def patch_router_for_masking():
    """Restrict routing to a keep-set, exactly as surgery would.

    The keep-set travels ON THE ROUTER MODULE (`_mask_keep`/`_mask_gain`), not in a global
    "current layer" context. A global would be an ordering hazard: one missed reset and a
    layer silently evaluates with its neighbour's mask, which is invisible in the output and
    would corrupt the whole comparison.

    Masking drives the SELECTION score to -inf for pruned experts and then defers to the
    model's own forward, so top-k, group masking, gate renormalisation and
    routed_scaling_factor all stay on the deployed code path. Re-implementing that arithmetic
    here is precisely how a divergence from the real pruned model would creep in.
    """
    import torch
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextTopkRouter
    if getattr(Glm5NextTextTopkRouter, "_mask_patched", False):
        return
    orig = Glm5NextTextTopkRouter.forward

    def forward(self, hidden_states):
        keep = getattr(self, "_mask_keep", None)
        if keep is None:                      # teacher arm: untouched forward
            return orig(self, hidden_states)
        bias = self.e_score_correction_bias
        drop = torch.ones(bias.shape[-1], dtype=torch.bool, device=bias.device)
        drop[keep.to(bias.device)] = False
        saved = bias.data.clone()
        try:
            bias.data[drop] = float("-inf")   # removes from SELECTION; `scores` (the gate) untouched
            rl, tw, ti = orig(self, hidden_states)
        finally:
            bias.data.copy_(saved)
        return rl, tw * getattr(self, "_mask_gain", 1.0), ti

    Glm5NextTextTopkRouter.forward = forward
    Glm5NextTextTopkRouter._mask_patched = True


def attach_mask(layer, li: int, mask, gains):
    """Tag this layer's router with its keep-set. Returns False if the layer has no router."""
    import torch
    router = None
    for m in layer.modules():
        if type(m).__name__ == "Glm5NextTextTopkRouter":
            router = m
            break
    if router is None:
        return False
    if mask is None:
        router._mask_keep, router._mask_gain = None, 1.0
        return True
    name = f"model.language_model.layers.{li}.mlp"
    if name not in mask:                      # dense layer, or MTP: leave untouched
        router._mask_keep, router._mask_gain = None, 1.0
        return True
    router._mask_keep = torch.tensor(mask[name], dtype=torch.long)
    router._mask_gain = float(gains.get(name, 1.0))
    return True


def selfcheck():
    """Offline gate: masks and gains must reproduce values we already know independently."""
    import torch
    shipped = build_mask("shipped")
    mean = build_mask("mean")
    ks = [L for L in mean if L in shipped]
    same = sum(len(set(mean[L]) & set(shipped[L])) for L in ks)
    tot = sum(len(shipped[L]) for L in ks)
    mass = build_mask("mass")
    errs = {}
    for name, mk in (("mean", mean), ("mass", mass)):
        e = []
        for f in sorted((ARTIFACTS / "saliency").glob("*.pt")):
            d = torch.load(f, weights_only=False); L = d["layer"]
            if L not in mk: continue
            o = d["out_sum"].double()
            m = torch.zeros(NE, dtype=torch.bool); m[torch.tensor(mk[L])] = True
            e.append(float(o[~m].sum(0).norm() / o.sum(0).norm()))
        errs[name] = sum(e) / len(e)
    gm, gs = healing_gains(mean), healing_gains(mass)
    print(f"mean-criterion reproduces shipped mask: {same}/{tot} = {same/tot:.2%}   (expect 100%)")
    print(f"recon err   mean {errs['mean']:.5f} (expect 0.26974)   mass {errs['mass']:.5f} (expect 0.19594)")
    print(f"healing gain  mean mean={sum(gm.values())/len(gm):.4f}  mass mean={sum(gs.values())/len(gs):.4f}")
    ok = abs(same/tot - 1.0) < 1e-9 and abs(errs['mean']-0.26974) < 2e-5 and abs(errs['mass']-0.19594) < 2e-5
    print("SELFCHECK", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# ---------------------------------------------------------------- streaming driver
SRC = ROOT / "source" / "GLM-5.3-Flash"

def run_arms(arm_names, out_path):
    """Score each arm on the SAME held-out rows and compare every mask to the teacher arm.

    Reuses s09_eval.score_checkpoint unchanged -- it is the proven streaming path (peak
    residency one decoder layer) and it already produces exactly the per-token statistics
    `compare` consumes. The only injection is `_build_layer`, which score_checkpoint imports
    at CALL time, so replacing the module attribute attaches our mask without forking any of
    the streaming, vision or final-projection code.

    The teacher arm is scored in this same run over these same rows, so `compare`'s positional
    alignment is exact by construction rather than by the length coincidence s09_eval warns
    about.
    """
    import torch
    import stages.s09_eval as E
    import stages.s03_saliency as S3
    from common import log

    if not (SRC / "model.safetensors.index.json").exists():
        raise SystemExit(f"unpruned source not staged at {SRC} - run stage s01_source first")

    patch_router_for_masking()
    rows, mm = E.load_heldout()
    log(f"mask_eval: {len(rows)} text + {len(mm)} image-text held-out rows, arms={arm_names}",
        "mask_eval")

    specs = {}
    for n in arm_names:
        if n == "teacher":
            specs[n] = (None, {})
        else:
            mk = build_mask(n)
            specs[n] = (mk, healing_gains(mk))

    orig_build = S3._build_layer
    caps = {}
    try:
        for name in arm_names:
            mask, gains = specs[name]
            def build(cfg, i, reader, dtype, _m=mask, _g=gains):
                layer = orig_build(cfg, i, reader, dtype)
                attach_mask(layer, i, _m, _g)
                return layer
            S3._build_layer = build
            t0 = time.time()
            caps[name] = E.score_checkpoint(SRC, rows, mm, name)
            log(f"mask_eval: arm {name} scored {caps[name]['nll'].numel()} tokens "
                f"in {(time.time()-t0)/60:.1f} min", "mask_eval")
    finally:
        S3._build_layer = orig_build

    if "teacher" not in caps:
        raise SystemExit("no teacher arm - nothing to compare against")
    T = caps["teacher"]
    out = {"teacher_tokens": int(T["nll"].numel()), "arms": {}}
    for name, cap in caps.items():
        if name == "teacher":
            continue
        r = E.compare(T, cap)
        out["arms"][name] = r
        log(f"mask_eval: {name}  dNLL {r['dNLL_mean']:+.4f}  top1 {r['top1_agreement']:.4f}",
            "mask_eval")
    Path(out_path).write_text(json.dumps(out, indent=1))
    # The decision this whole tool exists to make.
    names = [n for n in out["arms"]]
    if len(names) == 2:
        a, b = names
        da, db = out["arms"][a]["dNLL_mean"], out["arms"][b]["dNLL_mean"]
        win = a if da < db else b
        print(f"\nVERDICT: {win} is better by {abs(da-db):.4f} nats/token "
              f"({a} {da:+.4f} vs {b} {db:+.4f})")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true",
                    help="offline: verify masks/gains against independently known values")
    ap.add_argument("--arms", default="teacher,mean,mass")
    ap.add_argument("--out", default=str(ARTIFACTS / "mask_eval.json"))
    a = ap.parse_args()
    if a.selfcheck:
        sys.exit(selfcheck())
    print("streaming eval requires the unpruned source checkpoint and the GPU; "
          "run scripts/mask_eval_run.sh once s01_source completes and the GPU is free.")
