"""Standalone mHC reference + test vectors for the llama.cpp port.

WHY AN ORACLE FIRST
-------------------
Everything else in glm5_next already exists in llama.cpp (KDA via `LLM_ARCH_KIMI_LINEAR`, DSA via
`LLM_ARCH_GLM_DSA`, sigmoid-router MoE via `LLM_ARCH_GLM4_MOE`). mHC does not, so mHC is the port,
and mHC is also the part that fails silently: it is 24 numbers per token per site driving a
Sinkhorn projection, and a wrong iteration order or a misplaced epsilon still produces a
plausible doubly-stochastic matrix and a model that is subtly, unfixably wrong.

So: implement it once in plain torch with no transformers dependency, prove it matches the
reference module bit-for-bit on real weights from our checkpoint, then dump input/output vectors
the C++ implementation can be tested against directly. The C++ port then has a ground truth
instead of a paper.

THE THREE PLACES IT GOES WRONG
------------------------------
1. **Sinkhorn order.** The loop is: softmax(-1), +eps, then a COLUMN normalisation, then
   `iters - 1` full (row, column) passes. Reading it as `iters` symmetric passes gives a
   different matrix. `hc_sinkhorn_iters = 20`.
2. **Precision.** `flat` is normalised and projected in F32 even when the streams are BF16. The
   mixing weights come from a 24-wide vector; rounding it to BF16 before the softmax visibly
   moves `comb`.
3. **`post` is not a probability.** It is `2 * sigmoid(...)`, range [0, 2] — it scales the
   sublayer output, and clamping it to [0, 1] anywhere silently halves the block contribution.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "vendor" / "mhc"


def unweighted_rms_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps).to(x.dtype)


def mhc_forward(streams: torch.Tensor, fn: torch.Tensor, base: torch.Tensor,
                scale: torch.Tensor, hc: int, sinkhorn_iters: int, eps: float,
                rms_eps: float = 1e-5):
    """Reference mHC. streams [.., H, D] -> (post [.., H], comb [.., H, H], collapsed [.., D])."""
    flat = unweighted_rms_norm(streams.flatten(start_dim=-2).float(), rms_eps)
    mix = torch.nn.functional.linear(flat, fn.float())
    pre_w, post_w, comb_w = mix.split([hc, hc, hc * hc], dim=-1)
    pre_b, post_b, comb_b = base.float().split([hc, hc, hc * hc])
    pre_s, post_s, comb_s = scale.float().unbind(0)

    pre = torch.sigmoid(pre_w * pre_s + pre_b) + eps
    post = 2 * torch.sigmoid(post_w * post_s + post_b)          # range [0,2], NOT a probability
    comb_logits = comb_w.view(*comb_w.shape[:-1], hc, hc) * comb_s + comb_b.view(hc, hc)
    comb = torch.softmax(comb_logits, dim=-1) + eps
    # Column first, then (iters-1) full passes. Order is load-bearing.
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    collapsed = (pre.unsqueeze(-1) * streams).sum(dim=-2).to(streams.dtype)
    return post, comb, collapsed


def apply_site(streams, sublayer_out, post, comb):
    """The residual update: post (x) y  +  comb^T @ streams."""
    return (post.to(streams.dtype).unsqueeze(-1) * sublayer_out.unsqueeze(-2)
            + torch.matmul(comb.to(streams.dtype).transpose(-1, -2), streams))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--layer", type=int, default=10)
    ap.add_argument("--tokens", type=int, default=4)
    a = ap.parse_args()

    from common import kv_get
    from transformers import AutoConfig
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextHyperConnection
    from safetensors import safe_open

    ck = Path(a.ckpt) if a.ckpt else ROOT / "output" / str(kv_get("emit_name") or "")
    cfg = AutoConfig.from_pretrained(ck).text_config
    hc, iters, eps = cfg.hc_mult, cfg.hc_sinkhorn_iters, cfg.hc_eps
    D = cfg.hidden_size
    wm = json.loads((ck / "model.safetensors.index.json").read_text())["weight_map"]

    ok = True
    vectors = {}
    for site in ("attn", "ffn"):
        keys = {n: f"model.language_model.layers.{a.layer}.hc_{site}_{n}"
                for n in ("fn", "base", "scale")}
        w = {}
        for n, k in keys.items():
            with safe_open(str(ck / wm[k]), framework="pt") as f:
                w[n] = f.get_tensor(k)
        torch.manual_seed(0)
        streams = torch.randn(1, a.tokens, hc, D, dtype=torch.bfloat16)

        # reference module, loaded with the real weights
        mod = Glm5NextTextHyperConnection(cfg)
        mod.fn.data = w["fn"].clone()
        mod.base.data = w["base"].float().clone()
        mod.scale.data = w["scale"].float().clone()
        with torch.no_grad():
            p_ref, c_ref, x_ref = mod(streams)
        p_mine, c_mine, x_mine = mhc_forward(streams, w["fn"], w["base"], w["scale"],
                                             hc, iters, eps, cfg.rms_norm_eps)

        def rel(x, y):
            return float((x.float() - y.float()).norm() / y.float().norm().clamp_min(1e-12))
        rp, rc, rx = rel(p_mine, p_ref), rel(c_mine, c_ref), rel(x_mine, x_ref)
        good = max(rp, rc, rx) < 1e-5
        ok &= good
        print(f"  layer {a.layer} {site:<4}  post {rp:.2e}  comb {rc:.2e}  collapsed {rx:.2e}"
              f"   {'MATCH' if good else 'MISMATCH'}")
        # doubly-stochastic check on the Sinkhorn output
        rs = c_mine.sum(-1).flatten(); cs = c_mine.sum(-2).flatten()
        print(f"                  comb row sums {rs.min():.6f}..{rs.max():.6f}   "
              f"col sums {cs.min():.6f}..{cs.max():.6f}")
        vectors[site] = {
            "streams": streams.float().flatten().tolist(),
            "post": p_mine.float().flatten().tolist(),
            "comb": c_mine.float().flatten().tolist(),
            "collapsed": x_mine.float().flatten().tolist(),
        }

    meta = {"layer": a.layer, "tokens": a.tokens, "hc_mult": hc, "hidden_size": D,
            "sinkhorn_iters": iters, "hc_eps": eps, "rms_norm_eps": cfg.rms_norm_eps,
            "checkpoint": ck.name,
            "note": "streams are [1, tokens, hc_mult, hidden_size] row-major float32"}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "mhc_test_vectors.json").write_text(json.dumps({"meta": meta, **vectors}))
    print(f"\nwrote {OUT/'mhc_test_vectors.json'}  ({(OUT/'mhc_test_vectors.json').stat().st_size/1e6:.1f} MB)")
    print("all sites match" if ok else "MISMATCH - do not port until this is understood")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
