"""Check the per-expert normal equations against a brute-force simulation.

`build_normal` claims that for any coefficient vector c,

    sum_t ||y_t - yhat_t||^2  ==  ssy - 2 b.c + c^T A c

That identity is pure algebra given the modelling assumption f_i(x) = mu_i + eps_i with
independent zero-mean residuals. So: sample data that satisfies the assumption EXACTLY, run the
real routing replay, and compare the quadratic form against the residual computed the long way.
If the index bookkeeping in `accumulate` (local vs global expert ids, post x pre in C, the
pre x pre Q used for ||y||^2) is wrong anywhere, this diverges immediately - and it is precisely
that bookkeeping, not the algebra, that a healing bug would hide in.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import heal_perexpert as HP  # noqa: E402
import router_replay as RR   # noqa: E402


def test_normal_equations_match_bruteforce():
    torch.manual_seed(0)
    E, K, H, T = 12, 6, 48, 3000
    HP.TOP_K = 4
    RR_TOPK = 4

    scores = torch.rand(T, E).half()
    idx = torch.arange(E).expand(T, E).contiguous().to(torch.int32)  # full candidate list
    bias = torch.randn(E) * 0.05
    kept_ids = torch.arange(E)[torch.randperm(E)[:K]].sort().values
    keep = torch.zeros(E, dtype=torch.bool)
    keep[kept_ids] = True

    mu = torch.randn(E, H, dtype=torch.float64) * 0.7
    v = (torch.rand(E, dtype=torch.float64) * 0.5 + 0.1)

    sel = torch.arange(T)
    st = HP.accumulate(scores, idx, bias, keep, kept_ids, E, sel)
    A, b, ssy = HP.build_normal(st, mu, v, kept_ids)

    # Brute force. Residuals are drawn per (expert, token) with the exact variance v_i, so the
    # simulated data satisfies the model rather than approximating it.
    sc = scores.float()
    i_pre, w_pre = RR.simulate(sc, bias, None, top_k=RR_TOPK)
    i_post, w_post = RR.simulate(sc, bias, keep, top_k=RR_TOPK)
    loc = torch.full((E,), -1, dtype=torch.long)
    loc[kept_ids] = torch.arange(K)

    c = torch.rand(K, dtype=torch.float64) * 1.2 + 0.3
    eps = torch.randn(T, E, H, dtype=torch.float64) * (v / H).sqrt()[None, :, None]
    f = mu[None, :, :] + eps                                        # [T,E,H]

    y = (f.gather(1, i_pre.long()[:, :, None].expand(-1, -1, H))
         * w_pre.double()[:, :, None]).sum(1)
    yh = (f.gather(1, i_post.long()[:, :, None].expand(-1, -1, H))
          * (w_post.double() * c[loc[i_post.long()]])[:, :, None]).sum(1)
    brute = float(((y - yh) ** 2).sum())
    quad = HP.residual(c, A, b, ssy)

    # Monte-Carlo error is O(1/sqrt(T*H)) on a quantity of size `brute`; 2% is comfortably
    # outside the noise but far tighter than any index-order bug would survive.
    assert abs(quad - brute) / brute < 0.02, f"quad={quad:.3f} brute={brute:.3f}"


def test_diagonal_solution_is_the_gate_ratio():
    """Under exact orthogonality the solve must reduce to c_j = C_jj / N_j, norms cancelling."""
    torch.manual_seed(1)
    E, K, H, T = 10, 5, 64, 2000
    HP.TOP_K = 4
    scores = torch.rand(T, E).half()
    idx = torch.arange(E).expand(T, E).contiguous().to(torch.int32)
    bias = torch.randn(E) * 0.05
    kept_ids = torch.arange(E)[torch.randperm(E)[:K]].sort().values
    keep = torch.zeros(E, dtype=torch.bool)
    keep[kept_ids] = True

    st = HP.accumulate(scores, idx, bias, keep, kept_ids, E, torch.arange(T))
    mu = torch.zeros(E, H, dtype=torch.float64)            # zero means => pure residual, orthogonal
    v = torch.rand(E, dtype=torch.float64) * 3 + 0.5       # deliberately unequal norms
    A, b, ssy = HP.build_normal(st, mu, v, kept_ids)
    c_solve = torch.linalg.solve(A, b)

    N = torch.diagonal(st["P"])
    Cjj = st["C"][torch.arange(K), kept_ids]
    assert torch.allclose(c_solve, Cjj / N, rtol=1e-8, atol=1e-10)


fails = []


def check(name, cond, extra=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name} {extra}")
    if not cond:
        fails.append(name)


if __name__ == "__main__":
    print("per-expert healing:")
    for fn in (test_normal_equations_match_bruteforce, test_diagonal_solution_is_the_gate_ratio):
        try:
            fn()
            check(fn.__name__, True)
        except AssertionError as e:
            check(fn.__name__, False, str(e))
    print(f"\n{len(fails)} failure(s)" if fails else "\nall checks passed")
    raise SystemExit(1 if fails else 0)
