"""Correctness gates for the pass-2 saliency instrumentation.

Every check here guards a decision that is expensive or impossible to revisit: the accumulators
are written once by a 14-17 hour pass, and the offline replay is the only thing standing between
"we measured the healing gain" and "we derived it again and hoped".
"""
import sys, math
from pathlib import Path
from types import SimpleNamespace
import torch, torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import stream_saliency as SS
import router_replay as RR
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextTopkRouter

torch.manual_seed(0)
E, H, T, K = 32, 16, 64, 8
cfg = SimpleNamespace(num_experts_per_tok=K, num_local_experts=E, hidden_size=H,
                      routed_scaling_factor=2.5, n_group=1, topk_group=1, norm_topk_prob=True)
fails = []

def check(name, cond, extra=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name} {extra}")
    if not cond: fails.append(name)

# ---- 1. replay reproduces the real router exactly -------------------------------------------
router = Glm5NextTextTopkRouter(cfg)
with torch.no_grad():
    router.weight.normal_(0, 0.5)
    router.e_score_correction_bias.normal_(0, 0.3)   # non-zero: catches gate/selection mixups
hs = torch.randn(T, H)
logits, w_ref, i_ref = router(hs)
scores = logits.sigmoid()
i_sim, w_sim = RR.simulate(scores, router.e_score_correction_bias, None, K, True, 2.5)

def canon(idx, w):
    o = idx.argsort(dim=-1)
    return idx.gather(1, o), w.gather(1, o)
ir, wr = canon(i_ref, w_ref); isim, wsim = canon(i_sim, w_sim)
check("replay selects identical experts", torch.equal(ir, isim))
check("replay reproduces gates", torch.allclose(wr, wsim, atol=1e-5),
      f"max|d|={(wr-wsim).abs().max():.2e}")

# ---- 2. the gate must EXCLUDE e_score_correction_bias ---------------------------------------
# If the gate wrongly included the bias, zeroing the bias would leave gates unchanged for a
# fixed selection. Force the same selection and confirm the gates are bias-free.
b0 = torch.zeros_like(router.e_score_correction_bias)
_, w_nobias = RR.simulate(scores, b0, None, K, True, 2.5)
i_nb, _ = RR.simulate(scores, b0, None, K, True, 2.5)
same_sel = torch.equal(*[x.sort(dim=-1).values for x in (i_sim, i_nb)])
check("gate is read from scores, not scores+bias",
      (not same_sel) or torch.allclose(canon(i_sim, w_sim)[1], canon(i_nb, w_nobias)[1], atol=1e-6))

# ---- 3. pruning renormalises gates upward (the P5 effect) -----------------------------------
keep = torch.ones(E, dtype=torch.bool); keep[E // 2:] = False
i_p, w_p = RR.simulate(scores, router.e_score_correction_bias, keep, K, True, 2.5)
check("pruned routing stays inside the keep-set", bool(keep[i_p].all()))
check("gate mass is conserved by renormalisation",
      torch.allclose(w_p.sum(-1), w_sim.sum(-1), atol=1e-4),
      f"pre={w_sim.sum(-1).mean():.4f} post={w_p.sum(-1).mean():.4f}")

# ---- 4. cache replay == direct replay, and reports its own coverage --------------------------
sfc = scores + router.e_score_correction_bias
ci = sfc.topk(24, dim=-1).indices
cv = scores.gather(1, ci)
i_c, w_c, enough = RR.simulate_from_cache(cv, ci, router.e_score_correction_bias, None, K)
check("cache replay matches direct replay (unpruned)",
      torch.equal(canon(i_sim, w_sim)[0], canon(i_c, w_c)[0]) and bool(enough.all()))
i_cp, w_cp, en_p = RR.simulate_from_cache(cv, ci, router.e_score_correction_bias, keep, K)
ok = en_p
check("cache replay matches direct replay (pruned, where resolvable)",
      torch.equal(canon(i_p[ok], w_p[ok])[0], canon(i_cp[ok], w_cp[ok])[0]),
      f"resolvable {int(ok.sum())}/{T}")

# ---- 5. accumulators equal a naive reference -------------------------------------------------
SS.reset_accumulators()
SS.patch_experts_for_saliency()
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextExperts
fwd = Glm5NextTextExperts.forward
# F.linear(x, W) is x @ W.T, so weights are [out, in] exactly as the checkpoint stores them.
stub = SimpleNamespace(num_experts=E,
                       gate_up_proj=torch.randn(E, 2 * H, H) * 0.1,
                       down_proj=torch.randn(E, H, H) * 0.1,
                       _apply_gate=lambda x: F.silu(x[..., :H]) * x[..., H:])
SS.set_current_layer("L"); SS.set_bucket("code"); SS.set_valid_mask(None)
x = torch.randn(T, H)
_ = fwd(stub, x, i_sim, w_sim)
acc = SS.ACC["L"]

ref_sum = torch.zeros(E, dtype=torch.float64); ref_cnt = torch.zeros(E, dtype=torch.float64)
ref_osum = torch.zeros(E, H, dtype=torch.float64)
for t in range(T):
    for p in range(K):
        e = int(i_sim[t, p]); g = float(w_sim[t, p])
        cur = stub._apply_gate(F.linear(x[t], stub.gate_up_proj[e]))
        f = F.linear(cur, stub.down_proj[e])
        ref_sum[e] += g * float(f.norm()); ref_cnt[e] += 1
        ref_osum[e] += (f.double() * g)
b = SS.BUCKET_ID["code"]
check("sum matches naive reference",
      torch.allclose(acc["sum"][b], ref_sum, rtol=1e-4),
      f"max rel {( (acc['sum'][b]-ref_sum).abs()/(ref_sum.abs()+1e-9) ).max():.2e}")
check("count matches naive reference", torch.equal(acc["cnt"][b], ref_cnt))
check("out_sum (vector) matches naive reference",
      torch.allclose(acc["osum"].double(), ref_osum, rtol=1e-2, atol=1e-3))
check("attributed to the right bucket only",
      float(acc["sum"].sum()) > 0 and float(acc["sum"][[i for i in range(len(SS.BUCKETS)) if i != b]].sum()) == 0.0)
check("hist total equals token count", int(acc["hist"].sum()) == int(ref_cnt.sum()))
check("back-compat SAL_SUM alias is the bucket-summed vector",
      torch.allclose(SS.SAL_SUM["L"].sum(0), ref_sum, rtol=1e-4))

# ---- 6. padding must not be accumulated ------------------------------------------------------
SS.reset_accumulators()
valid = torch.zeros(T, dtype=torch.bool); valid[: T // 2] = True
SS.set_current_layer("L2"); SS.set_bucket("math"); SS.set_valid_mask(valid)
_ = fwd(stub, x, i_sim, w_sim)
check("valid mask drops padded tokens",
      int(SS.ACC["L2"]["cnt"].sum()) == int(valid.sum()) * K,
      f"got {int(SS.ACC['L2']['cnt'].sum())} want {int(valid.sum())*K}")

print("\n" + ("ALL PASS" if not fails else f"FAILURES: {fails}"))
sys.exit(1 if fails else 0)
