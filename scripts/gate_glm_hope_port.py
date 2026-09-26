"""Gate the HOPE/selector port onto GLM before a 328 GB, ~18 h pass depends on it.

The off-diagonal of F is the one statistic pass 3 cannot recompute. If it is wrong, nothing
downstream says so: `solve_prune_set` will happily minimise p^T F p against a wrong F and
return a confident, plausible, differently-wrong mask. So it is checked against a brute-force
per-token oracle, through the REAL patched forward -- not a reimplementation of it.

What is and is not being gated here: the oracle shares `_apply_gate` and the projections with
the code under test, because the claim is not "GLM's MLP math is right" (pass 2 established
that) but "the co-activation accumulation over that math is right". The oracle walks tokens and
slot pairs one at a time; the implementation walks experts and scatters. Those are genuinely
different traversals of the same quantity, which is what makes the comparison worth anything.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as TF

sys.path.insert(0, str(Path(__file__).resolve().parent))

OK = [0, 0]


def check(name, cond, extra=""):
    OK[1] += 1
    OK[0] += bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' -- ' + extra) if extra else ''}")


# --------------------------------------------------------------------------------------------
# A tiny experts module with the interface the patched forward uses.
# --------------------------------------------------------------------------------------------
E, H, INTER, K = 12, 16, 24, 4


class TinyExperts:
    def __init__(self, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.num_experts = E
        self.gate_up_proj = torch.randn(E, 2 * INTER, H, generator=g) * 0.1
        self.down_proj = torch.randn(E, H, INTER, generator=g) * 0.1

    def _apply_gate(self, x):
        a, b = x.chunk(2, dim=-1)
        return TF.silu(a) * b


def oracle_F(mod, x, idx, gates, valid):
    """Per-token, per-slot-pair reference. Deliberately the slow, obvious traversal."""
    n, k = idx.shape
    s = np.zeros((n, k))
    for t in range(n):
        for j in range(k):
            e = int(idx[t, j])
            cur = mod._apply_gate(TF.linear(x[t:t + 1], mod.gate_up_proj[e]))
            f = TF.linear(cur, mod.down_proj[e])
            s[t, j] = float(gates[t, j]) * float(f.float().norm())
    fs = np.zeros((E, E)); fc = np.zeros((E, E))
    for t in range(n):
        if valid is not None and not bool(valid[t]):
            continue
        for a in range(k):
            for b in range(k):
                fs[idx[t, a], idx[t, b]] += s[t, a] * s[t, b]
                fc[idx[t, a], idx[t, b]] += 1.0
    return fs, fc


def run_forward(SS, mod, x, idx, gates, valid, bucket="code", layer="model.language_model.layers.7.mlp"):
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextExperts
    SS.reset_accumulators()
    SS.set_current_layer(layer)
    SS.set_bucket(bucket)
    SS.set_valid_mask(valid)
    Glm5NextTextExperts.forward(mod, x, idx, gates)
    return SS.FACC[layer]


def main():
    import stream_saliency as SS
    SS.patch_experts_for_saliency()

    g = torch.Generator().manual_seed(7)
    N = 20
    mod = TinyExperts()
    x = torch.randn(N, H, generator=g)
    idx = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(N)])
    gates = torch.rand(N, K, generator=g)

    print("[1] F accumulation vs a per-token oracle")
    fa = run_forward(SS, mod, x, idx, gates, None)
    fs_ref, fc_ref = oracle_F(mod, x, idx, gates, None)
    fs = fa.sum[0].numpy(); fc = fa.cnt[0].numpy()
    check("f_sum matches the oracle", np.allclose(fs, fs_ref, rtol=1e-5, atol=1e-8),
          f"max|d|={np.abs(fs - fs_ref).max():.3e}")
    check("f_cnt matches the oracle", np.array_equal(fc, fc_ref))
    check("F is symmetric", np.allclose(fs, fs.T))
    check("F is not diagonal (the off-diagonal carries signal)",
          np.abs(fs - np.diag(np.diag(fs))).max() > 0,
          f"max off-diag={np.abs(fs - np.diag(np.diag(fs))).max():.4f}")

    print("[2] padding is excluded from F, not merely from the scalars")
    valid = torch.zeros(N, dtype=torch.bool); valid[: N // 2] = True
    fa_v = run_forward(SS, mod, x, idx, gates, valid)
    fs_v_ref, fc_v_ref = oracle_F(mod, x, idx, gates, valid)
    check("f_sum honours the valid mask", np.allclose(fa_v.sum[0].numpy(), fs_v_ref, rtol=1e-5, atol=1e-8))
    check("f_cnt honours the valid mask", np.array_equal(fa_v.cnt[0].numpy(), fc_v_ref))
    check("masking actually changed F (the mask is not a no-op)",
          not np.allclose(fa_v.sum[0].numpy(), fs))

    print("[3] buckets: the pass-3 corpus names reach the right rows")
    check("ballast -> general", SS.bucket_id("ballast") == SS.BUCKET_ID["general"])
    check("multimodal -> vision", SS.bucket_id("multimodal") == SS.BUCKET_ID["vision"])
    try:
        SS.bucket_id("world_knowledge"); bad = True
    except KeyError:
        bad = False
    check("an unknown bucket raises instead of silently becoming the catch-all", not bad)
    run_forward(SS, mod, x, idx, gates, None, bucket="ballast")
    acc = SS.ACC["model.language_model.layers.7.mlp"]
    m = acc["sum"].cpu().numpy()
    check("ballast mass lands in general's row only",
          m[SS.BUCKET_ID["general"]].sum() > 0 and m[SS.BUCKET_ID["vision"]].sum() == 0)

    print("[4] dump/load round-trip -- a killed chunk must not lose F")
    with tempfile.TemporaryDirectory() as td:
        run_forward(SS, mod, x, idx, gates, None)
        before = SS.FACC["model.language_model.layers.7.mlp"].sum[0].clone()
        SS.dump(Path(td))
        SS.reset_accumulators()
        check("reset really cleared F", not SS.FACC)
        n = SS.load_accumulators(Path(td), torch.device("cpu"))
        check("load restored the layer", n == 1)
        check("F survived the round-trip exactly",
              torch.equal(SS.FACC["model.language_model.layers.7.mlp"].sum[0], before))

        print("[5] bridge into the selector's format")
        import glm_acc_bridge
        d = glm_acc_bridge.build(Path(td))
        li = 7
        check("f_sum is indexed by ABSOLUTE layer number, not dump order",
              d["f_sum"].shape[0] == li + 1 and torch.equal(d["f_sum"][li], before.cpu()))
        check("bridge carried the bucket list", d["buckets"] == SS.BUCKETS)
        check("bridge renamed the per-bucket keys", set(d["acc"][
              "model.language_model.layers.7.mlp"]) >= {"sum", "sq", "cnt", "nrm", "gat"})

    print("[6] the bridge refuses F-less dumps rather than zero-filling them")
    with tempfile.TemporaryDirectory() as td:
        SS.reset_accumulators()
        SS.set_current_layer("model.language_model.layers.7.mlp")
        SS.set_bucket("code"); SS.set_valid_mask(None)
        from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextExperts
        Glm5NextTextExperts.forward(mod, x, idx, gates)
        SS.dump(Path(td))
        rec = torch.load(next(Path(td).glob("*.pt")), weights_only=False)
        rec.pop("f_sum"); rec.pop("f_cnt")
        torch.save(rec, next(Path(td).glob("*.pt")))
        import glm_acc_bridge
        try:
            glm_acc_bridge.build(Path(td)); refused = False
        except glm_acc_bridge.MissingF:
            refused = True
        check("a pass with no HOPE matrix is refused, not silently ranked against F=0", refused)

    print("[7] layer_index survives both naming conventions")
    from reap_select import layer_index
    check("GLM  model.language_model.layers.31.mlp -> 31",
          layer_index("model.language_model.layers.31.mlp") == 31)
    check("MiMo model.layers.31.mlp -> 31", layer_index("model.layers.31.mlp") == 31)
    try:
        layer_index("model.embed_tokens"); raised = False
    except ValueError:
        raised = True
    check("a path with no layer number raises", raised)

    print(f"\n{OK[0]}/{OK[1]} checks passed")
    sys.exit(0 if OK[0] == OK[1] else 1)


if __name__ == "__main__":
    main()
