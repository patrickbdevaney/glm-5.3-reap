"""Tier-1.1 - PER-EXPERT healing, fitted in closed form. No teacher, no forward pass.

WHAT THIS GENERALISES
---------------------
`s05_heal` (corrected by P5) multiplies every retained expert in a layer by ONE scalar chosen so
the layer's expected output magnitude matches the unpruned layer's. That scalar is the degenerate
case of a much better-posed question:

    choose c in R^{144} minimising   sum_t || y_t - yhat_t ||^2
    y_t    = sum_{i in S_t}  g_{i,t}  f_i(x_t)              <- unpruned layer output
    yhat_t = sum_{j in S'_t} c_j g'_{j,t} f_j(x_t)          <- pruned layer, per-expert rescaled

with S_t / S'_t the pre- and post-prune top-8 and g / g' the corresponding `norm_topk_prob`
gates. This is a least-squares problem in c, so it has a closed form - IF the expectations
E[f_i . f_j] are known.

WHY IT IS COMPUTABLE FROM ONE SWEEP
-----------------------------------
Decompose each expert's output around its own mean:  f_i(x) = mu_i + eps_i(x),  E[eps_i] = 0.
Assume only that residuals of DIFFERENT experts are uncorrelated - E[eps_i . eps_k] = 0 for
i != k. This is strictly weaker than the full orthogonality that the shipped scalar already
assumes, and unlike it the cross terms mu_i . mu_k are kept exactly. Then

    E[f_i . f_j] = mu_i . mu_j + delta_ij v_i          v_i = E||f_i||^2 - ||mu_i||^2

and the normal equations A c = b are

    A = (P .* G) + diag(N .* v)        P_jk = sum_t 1[j,k in S'_t] g'_j g'_k
    b_j = sum_i C_ji G_ij + C_jj v_j   C_ji = sum_t 1[j in S'_t] g'_j 1[i in S_t] g_i
                                       N_j  = P_jj
    G_ij = mu_i . mu_j

Every one of those comes from artefacts we already have:
  * mu_i         <- `out_sum` (the summed GATED expert output vector) / `gate_sum`
  * E||f_i||^2   <- `norm_sq_by_bucket` / `cnt_by_bucket`
  * P, C, N      <- replaying pre- and post-prune routing from the router score cache

THE ASSUMPTION-LIGHT SOLUTION
-----------------------------
Drop the cross terms (full orthogonality, i.e. exactly what the shipped scalar assumes) and A
becomes diagonal. Then A_jj = N_j E||f_j||^2 and b_j = C_jj E||f_j||^2, so

    c_j = C_jj / N_j                     (the norms CANCEL)

i.e. "scale expert j by the gate it used to receive over the gate it now receives". It needs no
norm data at all, which makes it the robust variant: it cannot be wrong for a reason that lives
in `out_sum`. An expert promoted INTO the top-8 by pruning contributes to N_j but not to C_jj,
so it is correctly shrunk - it is doing work it never did before.

HONESTY
-------
Both solutions are MODEL-BASED. So: the tokens are split 50/50, c is fitted on one half, and the
residual is scored on the other, against (a) no correction, (b) the optimal scalar under the same
model, (c) the scalar actually shipped by P5. Per-expert is only recommended if it beats the
optimal scalar OUT OF SAMPLE by more than a set margin. `--report-orthogonality` measures how
true the orthogonality assumption actually is, which the P5 docstring promised and never
delivered.
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
RETAINED = ROOT / "artifacts" / "reap_retained_experts.json"

TOP_K = 8
TOK_CHUNK = 32768          # tokens per replay block; caps the [T,E] scatter at ~38 MB
# c multiplies an F32 block scale. The shipped scalar sits near 0.91; a per-expert coefficient
# far outside this band is a fit artefact (a near-singular row), not a correction.
C_LO, C_HI = 0.25, 1.60


def _layer_index(lname: str) -> int:
    m = re.search(r"layers\.(\d+)\.", lname)
    return int(m.group(1)) if m else -1


def load_bias(layer_idx: int) -> torch.Tensor | None:
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


def accumulate(scores: torch.Tensor, idx: torch.Tensor, bias: torch.Tensor,
               keep: torch.Tensor, kept_ids: torch.Tensor, n_exp: int,
               sel: torch.Tensor) -> dict:
    """Replay routing over `sel` tokens and accumulate the sufficient statistics.

    Returns P [K,K], C [K,E], Q [E,E] and the resolvable fraction, with K = len(kept_ids).
    Everything is float64: P and C are sums over ~280k tokens of products of gates that are
    O(0.3), and the solve differences the results, so f32 accumulation loses the signal.
    """
    K = kept_ids.numel()
    loc = torch.full((n_exp,), -1, dtype=torch.long)
    loc[kept_ids] = torch.arange(K)
    P = torch.zeros(K, K, dtype=torch.float64)
    C = torch.zeros(K, n_exp, dtype=torch.float64)
    Q = torch.zeros(n_exp, n_exp, dtype=torch.float64)
    n_ok = n_tot = 0
    for s in range(0, sel.numel(), TOK_CHUNK):
        t = sel[s:s + TOK_CHUNK]
        sc, ix = scores[t].float(), idx[t].long()
        i_pre, w_pre, ok_pre = RR.simulate_from_cache(sc, ix, bias, None, top_k=TOP_K)
        i_post, w_post, ok_post = RR.simulate_from_cache(sc, ix, bias, keep, top_k=TOP_K)
        ok = ok_pre & ok_post
        n_tot += int(ok.numel())
        n_ok += int(ok.sum())
        if not bool(ok.any()):
            continue
        ip, wp = i_pre[ok].long(), w_pre[ok].double()
        iq, wq = loc[i_post[ok].long()], w_post[ok].double()
        # Every token contributes an outer product over its own top-8. Building the dense
        # [t, K] one-hot would be 32768*144*8 - flatten to index_put_ pairs instead.
        n = ip.shape[0]
        # P: post x post
        a = iq.repeat_interleave(TOP_K, 1).reshape(-1)
        b_ = iq.repeat(1, TOP_K).reshape(-1)
        v = (wq.repeat_interleave(TOP_K, 1) * wq.repeat(1, TOP_K)).reshape(-1)
        P.index_put_((a, b_), v, accumulate=True)
        # C: post x pre
        a = iq.repeat_interleave(TOP_K, 1).reshape(-1)
        b_ = ip.repeat(1, TOP_K).reshape(-1)
        v = (wq.repeat_interleave(TOP_K, 1) * wp.repeat(1, TOP_K)).reshape(-1)
        C.index_put_((a, b_), v, accumulate=True)
        # Q: pre x pre (only needed for the ||y||^2 constant in the holdout score)
        a = ip.repeat_interleave(TOP_K, 1).reshape(-1)
        b_ = ip.repeat(1, TOP_K).reshape(-1)
        v = (wp.repeat_interleave(TOP_K, 1) * wp.repeat(1, TOP_K)).reshape(-1)
        Q.index_put_((a, b_), v, accumulate=True)
        del a, b_, v, ip, wp, iq, wq
    return {"P": P, "C": C, "Q": Q, "n": n, "resolvable": n_ok / max(n_tot, 1)}


def moments(rec: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(mu [E,H], E||f||^2 [E], v [E]) from the saliency accumulators."""
    cnt = rec["count"].double().clamp_min(1)
    gat = rec["gate_sum_by_bucket"].sum(0).double().clamp_min(1e-12)
    mu = rec["out_sum"].double() / gat[:, None]        # gate-weighted mean of f_i
    ef2 = rec["norm_sq_by_bucket"].sum(0).double() / cnt
    v = ef2 - (mu * mu).sum(-1)
    # mu is gate-weighted while E||f||^2 is not, so v can go slightly negative for an expert
    # whose gate correlates with its norm. Floor it rather than let a negative variance flip
    # the sign of a diagonal entry.
    v = v.clamp_min(0.05 * ef2)
    return mu, ef2, v


def residual(c: torch.Tensor, A: torch.Tensor, b: torch.Tensor, ssy: float) -> float:
    return float(ssy - 2.0 * (b * c).sum() + c @ A @ c)


def build_normal(st: dict, mu: torch.Tensor, v: torch.Tensor,
                 kept_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float]:
    P, C, Q = st["P"], st["C"], st["Q"]
    muk = mu[kept_ids]
    G_kk = muk @ muk.T                                   # [K,K]
    G_ke = muk @ mu.T                                    # [K,E]
    N = torch.diagonal(P)
    Cjj = C[torch.arange(kept_ids.numel()), kept_ids]
    vk = v[kept_ids]
    A = P * G_kk + torch.diag(N * vk)
    b = (C * G_ke).sum(-1) + Cjj * vk
    ssy = float((Q * (mu @ mu.T)).sum() + (torch.diagonal(Q) * v).sum())
    return A, b, ssy


def solve_layer(rec: dict, cache: dict, bias: torch.Tensor, keep_idx: list[int],
                shipped: float | None, margin: float, ridge_grid: list[float],
                objective: str = "magnitude") -> dict:
    n_exp = int(rec["count"].numel())
    kept_ids = torch.tensor(sorted(keep_idx), dtype=torch.long)
    keep = torch.zeros(n_exp, dtype=torch.bool)
    keep[kept_ids] = True
    K = kept_ids.numel()

    scores, idx = cache["scores"], cache["idx"]
    T = scores.shape[0]
    # Deterministic 50/50 split. Interleaved, not a prefix: the cache is written chunk by chunk
    # in corpus order, so a contiguous split would fit on one set of domains and score on
    # another and report the domain gap as an overfitting gap.
    all_t = torch.arange(T)
    fit_t, hold_t = all_t[0::2], all_t[1::2]

    mu, ef2, v = moments(rec)
    st_f = accumulate(scores, idx, bias, keep, kept_ids, n_exp, fit_t)
    st_h = accumulate(scores, idx, bias, keep, kept_ids, n_exp, hold_t)
    # The FALLBACK for a skipped layer is heal_refit's scalar, which is computed from exactly
    # these tokens with no resolvability gate at all. Being stricter here than the thing we fall
    # back to would discard the better estimate in favour of the worse one on identical data.
    # Gate only where the sample is too small to fit 144 coefficients, and mark it.
    low_conf = st_f["resolvable"] < 0.5 or st_h["resolvable"] < 0.5
    if st_f["resolvable"] < 0.30 or st_h["resolvable"] < 0.30:
        return {"skipped": "under 30% of cached tokens resolve post-prune",
                "resolvable": st_f["resolvable"]}

    A_f, b_f, ssy_f = build_normal(st_f, mu, v, kept_ids)
    A_h, b_h, ssy_h = build_normal(st_h, mu, v, kept_ids)

    ones = torch.ones(K, dtype=torch.float64)
    # (a) optimal single scalar under the SAME model - the fair baseline for "does per-expert
    # actually buy anything", as opposed to beating a scalar that was fitted differently.
    s_opt = float((b_f @ ones) / (ones @ A_f @ ones))
    # (b) assumption-light diagonal solution: c_j = C_jj / N_j, norms cancel.
    N_f = torch.diagonal(st_f["P"]).clamp_min(1e-12)
    Cjj_f = st_f["C"][torch.arange(K), kept_ids]
    c_diag = (Cjj_f / N_f).clamp(C_LO, C_HI)

    # (c) full solve, ridge-regularised TOWARDS the optimal scalar (not towards zero): with 144
    # free parameters and correlated columns, shrinking to the scalar is the prior we actually
    # believe. lambda picked on the holdout.
    scale = float(torch.diagonal(A_f).mean())
    best = None
    for lam in ridge_grid:
        Ar = A_f + torch.eye(K, dtype=torch.float64) * (lam * scale)
        br = b_f + (lam * scale) * s_opt * ones
        try:
            c = torch.linalg.solve(Ar, br)
        except Exception:
            continue
        c = c.clamp(C_LO, C_HI)
        r = residual(c, A_h, b_h, ssy_h)
        if best is None or r < best[1]:
            best = (c, r, lam)
    if best is None:
        return {"skipped": "normal equations unsolvable"}
    c_full, r_full, lam_star = best

    # MAGNITUDE-PRESERVING variant. Pure least squares is MSE-optimal but scale-biased: it
    # shrinks yhat below y whenever the two are imperfectly correlated (regression attenuation),
    # which is why c_LS lands near 0.75 where P5's magnitude-matching scalar lands near 0.91.
    # Minimising per-layer MSE is not obviously the right objective for a residual network: a
    # systematic attenuation of the MoE pathway compounds multiplicatively over 42 layers, and
    # mHC's Sinkhorn-normalised connection matrices were fitted against the ORIGINAL output
    # scale. So rescale the LS solution to satisfy E||yhat||^2 = E||y||^2 - keeping the relative
    # per-expert structure, which is the actual innovation, while leaving the global scale where
    # the pipeline has already validated it.
    def to_magnitude(cv):
        q = float(cv @ A_f @ cv)
        if q <= 0:
            return cv
        return (cv * (ssy_f / q) ** 0.5).clamp(C_LO, C_HI)

    c_mag = to_magnitude(c_diag)
    c_mag_full = to_magnitude(c_full)

    r_none = residual(ones, A_h, b_h, ssy_h)
    r_sopt = residual(ones * s_opt, A_h, b_h, ssy_h)
    r_diag = residual(c_diag, A_h, b_h, ssy_h)
    r_ship = residual(ones * shipped, A_h, b_h, ssy_h) if shipped else None
    r_mag = residual(c_mag, A_h, b_h, ssy_h)
    r_mag_full = residual(c_mag_full, A_h, b_h, ssy_h)
    # The scalar that preserves magnitude, for an apples-to-apples baseline against c_mag.
    s_mag = float((ssy_f / (ones @ A_f @ ones)) ** 0.5)
    r_smag = residual(ones * s_mag, A_h, b_h, ssy_h)

    # Orthogonality report: how much of A actually lives off the diagonal. This is the
    # assumption both the shipped scalar and c_diag rest on, and it has never been measured.
    Ad = torch.diagonal(A_f).abs().sum()
    Ao = A_f.abs().sum() - Ad
    muk = mu[kept_ids]
    nrm = muk.norm(dim=-1).clamp_min(1e-12)
    cosm = (muk @ muk.T) / (nrm[:, None] * nrm[None, :])
    cosm.fill_diagonal_(0.0)
    off_cos = float(cosm.abs().sum() / (K * (K - 1)))

    # Pick the variant to ship. Per-expert must beat the optimal scalar out of sample by
    # `margin`; between the two per-expert forms prefer the diagonal one on a tie, because it
    # rests on strictly fewer assumptions.
    if objective == "magnitude":
        base_name, base_c, base_r = "scalar_mag", ones * s_mag, r_smag
        cands = [("per_expert_mag", c_mag, r_mag),
                 ("per_expert_mag_full", c_mag_full, r_mag_full)]
        tie = ("per_expert_mag", c_mag, r_mag)
    else:
        base_name, base_c, base_r = "scalar_opt", ones * s_opt, r_sopt
        cands = [("per_expert_diag", c_diag, r_diag),
                 ("per_expert_full", c_full, r_full)]
        tie = ("per_expert_diag", c_diag, r_diag)
    # Never ship a per-expert vector that is worse than the scalar ACTUALLY BEING REPLACED.
    # The margin above is measured against the LS-consistent baseline, but what a layer really
    # falls back to is P5's shipped scalar - and on one layer in 42 that scalar is the better of
    # the two. Gate on both so a per-layer holdout fluctuation cannot make the checkpoint worse.
    thresh = base_r * (1.0 - margin)
    if r_ship is not None:
        thresh = min(thresh, r_ship)
    win = min((x for x in cands if x[2] <= thresh), key=lambda x: x[2], default=None)
    if win is None:
        chosen, cvec = base_name, base_c
    elif win[0].endswith("_full") and tie[2] <= win[2] * 1.005:
        # Prefer the assumption-light form on a tie: it rests on nothing that lives in `out_sum`.
        chosen, cvec = tie[0], tie[1]
    else:
        chosen, cvec = win[0], win[1]

    def rel(r):
        return None if r is None else float(r / ssy_h)

    return {
        "experts": n_exp, "kept": K,
        "resolvable_fit": st_f["resolvable"], "resolvable_hold": st_h["resolvable"],
        "objective": objective, "low_confidence": bool(low_conf),
        "scalar_opt": s_opt, "scalar_mag": s_mag,
        "shipped_scalar": shipped, "ridge_lambda": lam_star,
        "holdout_rel_residual": {"none": rel(r_none), "scalar_opt": rel(r_sopt),
                                 "scalar_mag": rel(r_smag), "shipped": rel(r_ship),
                                 "per_expert_diag": rel(r_diag), "per_expert_full": rel(r_full),
                                 "per_expert_mag": rel(r_mag),
                                 "per_expert_mag_full": rel(r_mag_full)},
        "gain_vs_scalar": {"per_expert_diag": float(1.0 - r_diag / r_sopt),
                           "per_expert_full": float(1.0 - r_full / r_sopt),
                           "per_expert_mag": float(1.0 - r_mag / r_smag)},
        "gain_vs_shipped": (float(1.0 - r_mag / r_ship) if r_ship else None),
        "orthogonality": {"offdiag_mass_frac": float(Ao / (Ao + Ad)),
                          "mean_abs_cos_mu": off_cos},
        "chosen": chosen,
        "c_stats": {"min": float(cvec.min()), "p50": float(cvec.median()),
                    "max": float(cvec.max()), "mean": float(cvec.mean()),
                    "clamped": int(((cvec <= C_LO + 1e-9) | (cvec >= C_HI - 1e-9)).sum())},
        "kept_ids": [int(x) for x in kept_ids],
        "c": [float(x) for x in cvec],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--margin", type=float, default=0.02,
                    help="required out-of-sample residual reduction over the optimal scalar")
    ap.add_argument("--ridge", default="0.0,0.001,0.01,0.05,0.2,1.0")
    ap.add_argument("--objective", default="magnitude", choices=("magnitude", "ls"),
                    help="magnitude: LS shape, rescaled to preserve E||y||^2 (the criterion P5 "
                         "already validated). ls: pure least squares, MSE-optimal but "
                         "scale-attenuating.")
    ap.add_argument("--layers", default="", help="comma-separated layer indices, for a pilot")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--keep-set", default=str(RETAINED),
                    help="score an ALTERNATIVE keep-set under the same objective. `ssy` depends "
                         "only on PRE-prune routing, so the relative residual is directly "
                         "comparable across masks - which makes this a reconstruction-error "
                         "comparison of two masks rather than a saliency-mass proxy.")
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "heal_perexpert.json"))
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    keep_path = Path(a.keep_set)
    retained = json.loads(keep_path.read_text())
    refit = {}
    rf = ROOT / "artifacts" / "heal_refit.json"
    if rf.exists() and keep_path.resolve() == RETAINED.resolve():
        refit = {r["layer"]: r.get("measured_gain")
                 for r in json.loads(rf.read_text()).get("per_layer", [])}

    caches: dict[str, dict] = {}
    for f in sorted(ROUTER_CACHE.glob("chunk_*.pt")):
        d = torch.load(f, weights_only=False, map_location="cpu")
        for ln, vv in d["layers"].items():
            if ln in caches:
                caches[ln] = {"scores": torch.cat([caches[ln]["scores"], vv["scores"]]),
                              "idx": torch.cat([caches[ln]["idx"], vv["idx"]])}
            else:
                caches[ln] = vv
    print(f"router cache: {len(caches)} layers, "
          f"{next(iter(caches.values()))['scores'].shape[0]} tokens")

    want = {int(x) for x in a.layers.split(",") if x.strip()} if a.layers else None
    grid = [float(x) for x in a.ridge.split(",")]
    rows = []
    for f in sorted(SALIENCY.glob("*.pt")):
        rec = torch.load(f, weights_only=False, map_location="cpu")
        ln = rec["layer"]
        li = _layer_index(ln)
        if want is not None and li not in want:
            continue
        keep_idx = retained.get(ln)
        cache, bias = caches.get(ln), load_bias(li)
        if not keep_idx or cache is None or bias is None:
            print(f"  {ln}: skipped (keep={bool(keep_idx)} cache={cache is not None} "
                  f"bias={bias is not None})")
            continue
        r = solve_layer(rec, cache, bias, keep_idx, refit.get(ln), a.margin, grid, a.objective)
        r["layer"] = ln
        rows.append(r)
        if "skipped" in r:
            print(f"  {ln}: SKIP {r['skipped']}")
        else:
            hr = r["holdout_rel_residual"]
            print(f"  L{li:>2} chosen={r['chosen']:<16} "
                  f"rel-resid none={hr['none']:.4f} scal={hr['scalar_opt']:.4f} "
                  f"diag={hr['per_expert_diag']:.4f} mag={hr['per_expert_mag']:.4f} "
                  f"| offdiag={r['orthogonality']['offdiag_mass_frac']:.3f} "
                  f"cos={r['orthogonality']['mean_abs_cos_mu']:.3f}")
        del rec

    good = [r for r in rows if "skipped" not in r]
    res = {
        "method": "per-expert least-squares healing (closed form, no teacher)",
        "margin": a.margin,
        "keep_set": keep_path.name,
        "keep_set_sha": hashlib.sha256(keep_path.read_bytes()).hexdigest()[:16],
        "layers": len(good),
        "chosen_counts": {k: sum(1 for r in good if r["chosen"] == k)
                          for k in ("scalar_opt", "scalar_mag", "per_expert_diag",
                                    "per_expert_full", "per_expert_mag",
                                    "per_expert_mag_full")},
        "objective": a.objective,
        "per_layer": rows,
    }
    if good:
        def med(fn):
            xs = sorted(fn(r) for r in good)
            return xs[len(xs) // 2]
        res["median_holdout_rel_residual"] = {
            k: med(lambda r, k=k: r["holdout_rel_residual"][k])
            for k in ("none", "scalar_opt", "scalar_mag", "shipped", "per_expert_diag",
                      "per_expert_full", "per_expert_mag")
            if all(r["holdout_rel_residual"].get(k) is not None for r in good)}
        res["median_gain_vs_scalar"] = {
            k: med(lambda r, k=k: r["gain_vs_scalar"][k])
            for k in ("per_expert_diag", "per_expert_full", "per_expert_mag")}
        res["median_offdiag_mass_frac"] = med(
            lambda r: r["orthogonality"]["offdiag_mass_frac"])
        res["median_abs_cos_mu"] = med(lambda r: r["orthogonality"]["mean_abs_cos_mu"])
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(f"\nlayers solved      : {len(good)}")
    print(f"chosen             : {res.get('chosen_counts')}")
    if good:
        print(f"median rel-residual: {res['median_holdout_rel_residual']}")
        print(f"median gain vs opt : {res['median_gain_vs_scalar']}")
        print(f"off-diagonal mass  : {res['median_offdiag_mass_frac']:.4f} "
              f"(mean |cos(mu_i,mu_j)| = {res['median_abs_cos_mu']:.4f})")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
