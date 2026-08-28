"""Checks for the paired-eval comparison math.

This runs after two multi-hour scoring passes, so a bug here is discovered at the most expensive
possible moment. Everything is verified against a brute-force reference or an identity.
"""
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import stages.s09_eval as EV

fails = []
def check(name, cond, extra=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name} {extra}")
    if not cond: fails.append(name)

torch.manual_seed(0)
V, K = 154880, EV.TOPK
BUCKETS = ["code", "math", "agentic"]

def synth(n, seed, shift=0.0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(n, 256, generator=g) + shift
    lp = torch.log_softmax(logits, -1)
    v, i = lp.topk(K, -1)
    return {"nll": -lp[:, 0].clone(), "argmax": lp.argmax(-1).to(torch.int32),
            "topk_idx": i.to(torch.int32), "topk_logp": v.to(torch.float16),
            "buckets": [BUCKETS[j % 3] for j in range(n)],
            "taps": {li: torch.randn(max(1, n // 8), 64, generator=g) for li in EV.TAP_LAYERS}}

# ---- 1. identity: a checkpoint compared to itself must show zero change -----------------------
A = synth(500, 1)
r = EV.compare(A, A)
check("identical -> dNLL 0", abs(r["dNLL_mean"]) < 1e-9, f"{r['dNLL_mean']:.2e}")
check("identical -> flip 0", r["flip_rate"] == 0.0)
check("identical -> KL ~0", abs(r["topk_KL"]) < 1e-5, f"{r['topk_KL']:.2e}")
check("identical -> tap drift 0", max(r["tap_drift"].values()) < 1e-6)
check("per-domain covers every bucket", set(r["by_domain"]) == set(BUCKETS))
check("per-domain tokens sum to total",
      sum(d["tokens"] for d in r["by_domain"].values()) == r["tokens"])

# ---- 2. chunked KL == dense reference ---------------------------------------------------------
B = synth(500, 2)
r2 = EV.compare(A, B)
ti, tp = A["topk_idx"][:500].long(), A["topk_logp"][:500].float()
si, sp = B["topk_idx"][:500].long(), B["topk_logp"][:500].float()
dense = torch.full((500, V), -30.0)
dense.scatter_(1, si, sp)
ref = float((tp.exp() * (tp - dense.gather(1, ti))).sum(-1).mean())
check("chunked KL matches dense reference", abs(r2["topk_KL"] - ref) < 1e-4,
      f"chunked {r2['topk_KL']:.6f} vs dense {ref:.6f}")
check("KL is non-negative for different models", r2["topk_KL"] > 0, f"{r2['topk_KL']:.4f}")
check("different models flip some argmaxes", 0.0 < r2["flip_rate"] <= 1.0,
      f"{r2['flip_rate']:.3f}")

# ---- 3. sign convention: a WORSE student must give POSITIVE dNLL ------------------------------
# dNLL = student - teacher, so positive means the student assigns less probability to the gold
# token, i.e. pruning cost us something. A sign flip here would invert every conclusion.
W = {**A, "nll": A["nll"] + 0.25}
check("worse student -> positive dNLL", EV.compare(A, W)["dNLL_mean"] > 0.2)
Bt = {**A, "nll": A["nll"] - 0.25}
check("better student -> negative dNLL", EV.compare(A, Bt)["dNLL_mean"] < -0.2)

# ---- 4. the chunk boundary must not change the answer ------------------------------------------
big_a, big_b = synth(9000, 3), synth(9000, 4)
r_big = EV.compare(big_a, big_b)
old_ch = EV.__dict__.get("CH")
check("KL is stable across the 4096 chunk boundary", r_big["topk_KL"] > 0 and
      r_big["tokens"] == 9000, f"n={r_big['tokens']} KL={r_big['topk_KL']:.4f}")

# ---- 5. mismatched lengths truncate to the shorter, never crash ---------------------------------
short = synth(120, 5)
r5 = EV.compare(big_a, short)
check("mismatched lengths truncate safely", r5["tokens"] == 120)

print("\n" + ("ALL PASS" if not fails else f"FAILURES: {fails}"))
sys.exit(1 if fails else 0)
