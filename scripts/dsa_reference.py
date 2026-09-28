"""Indexer oracle + test vectors for the DSA port, mirroring scripts/mhc_reference.py.

The mHC port worked because it had ground truth before a line of C++ was written. DSA is worse:
the selection is a two-stage pooled top-k whose failure mode is a *plausible* set of indices, and
a wrong pooling offset or a missing relu still produces a model that runs and is quietly worse.

So: reimplement the indexer in plain torch, prove it matches transformers' Glm5NextTextIndexer on
real weights, then dump inputs and outputs the C++ can be tested against directly.

THE THREE PLACES IT WILL GO WRONG
  1. Pooling starts at the FIRST REAL TOKEN, not slot 0, so left padding does not shift pool
     boundaries. Getting this wrong misaligns every pool by a constant.
  2. `relu` sits between the per-head scores and the head-weighted sum. Scores are non-negative,
     which is not what softmax-attention intuition predicts, and dropping it changes which pools
     win.
  3. The pool key is a PER-CHANNEL softmax over the 4 tokens in the pool of
     (gate + ape) - not a mean, and not a softmax over pools.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "vendor" / "dsa"


def pooled_states(k, gate, valid, kpool, ape):
    """(keys, gate, valid) -> (pool_keys, pool_indices, pool_valid). Mirrors get_pooled_states."""
    B, S, D = k.shape
    n_pools = (S + kpool - 1) // kpool
    first = torch.where(valid.any(-1), valid.long().argmax(-1),
                        torch.full((B,), S, dtype=torch.long))
    offs = torch.arange(n_pools * kpool).view(1, n_pools, kpool)
    idx = first[:, None, None] + offs
    b = torch.arange(B)[:, None, None]
    safe = idx.clamp(0, S - 1)

    gk, gg, gv = k[b, safe], gate[b, safe], valid[b, safe]
    gv = gv & (idx < S)
    pool_valid = gv.all(-1)
    idx = idx.masked_fill(~gv, -1)

    logits = gg.float() + ape.float()[None, None]
    logits = logits.masked_fill(~gv[..., None], float("-inf"))
    prob = torch.nan_to_num(logits.softmax(dim=2)).to(gk.dtype)
    pool_keys = (prob * gk).sum(dim=2)
    return pool_keys, idx, pool_valid


def indexer_scores(x, q_resid, w, cfg):
    """Returns index_scores [B, S, n_pools] - the quantity the C++ must reproduce."""
    B, S = x.shape[:2]
    hd, nh, kpool = cfg["index_head_dim"], cfg["index_n_heads"], cfg["index_kpool"]

    q = F.linear(q_resid, w["wq_b"]).view(B, S, nh, hd)
    k = F.layer_norm(F.linear(x, w["wk"]), (hd,), w["k_norm_w"], w["k_norm_b"], eps=1e-6)
    gate = F.linear(x, w["kpool_gate"])
    valid = torch.ones(B, S, dtype=torch.bool)

    pool_keys, pool_idx, pool_valid = pooled_states(k, gate, valid, kpool, w["kpool_ape"])

    scores = torch.matmul(q.float(), pool_keys.transpose(-1, -2).float().unsqueeze(1))
    scores = F.relu(scores * hd ** -0.5)                     # relu is load-bearing
    weights = F.linear(x, w["weights_proj"]).float() * (nh ** -0.5)
    index_scores = torch.matmul(weights.unsqueeze(-2), scores).squeeze(-2)

    # CAUSALITY. A pool is selectable by the query at position s only if its LAST token is
    # visible to s. Omitting this is not cosmetic - early queries can otherwise select pools
    # from their own future, and the masked entries are what makes the reference comparison
    # nan rather than merely wrong.
    pool_end = pool_idx[..., -1].clamp(0, S - 1)                     # [B, n_pools]
    q_pos = torch.arange(S).view(1, S, 1)
    visible = pool_end[:, None, :] <= q_pos                          # [B, S, n_pools]
    valid_candidates = visible & pool_valid[:, None, :]
    index_scores = index_scores.masked_fill(~valid_candidates,
                                            torch.finfo(index_scores.dtype).min)
    return index_scores, pool_keys, pool_idx, pool_valid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "output" / "glm-5.3-flash-reap50-fp8-pass2"))
    ap.add_argument("--layer", type=int, default=3)          # first DSA layer
    ap.add_argument("--tokens", type=int, default=24)
    ap.add_argument("--validate", action="store_true")
    a = ap.parse_args()

    from safetensors import safe_open
    ck = Path(a.ckpt)
    cfg_all = json.loads((ck / "config.json").read_text())
    cfg = cfg_all.get("text_config", cfg_all)
    wm = json.loads((ck / "model.safetensors.index.json").read_text())["weight_map"]
    D = cfg["hidden_size"]

    pre = f"model.language_model.layers.{a.layer}.self_attn."
    names = {
        "wq_b": pre + "indexer.wq_b.weight", "wk": pre + "indexer.wk.weight",
        "k_norm_w": pre + "indexer.k_norm.weight", "k_norm_b": pre + "indexer.k_norm.bias",
        "weights_proj": pre + "indexer.weights_proj.weight",
        "kpool_gate": pre + "indexer.index_kpool_compress_gate",
        "kpool_ape": pre + "indexer.index_kpool_compress_ape",
    }
    w = {}
    for n, key in names.items():
        with safe_open(str(ck / wm[key]), framework="pt") as f:
            w[n] = f.get_tensor(key).float()

    torch.manual_seed(0)
    x = torch.randn(1, a.tokens, D) * 0.05
    q_resid = torch.randn(1, a.tokens, cfg["q_lora_rank"]) * 0.05

    scores, pool_keys, pool_idx, pool_valid = indexer_scores(x, q_resid, w, cfg)
    print(f"layer {a.layer}: {a.tokens} tokens -> {scores.shape[-1]} pools "
          f"(kpool={cfg['index_kpool']}, topk={cfg['index_topk']})")
    print(f"  index_scores  mean {scores.mean():.6f}  min {scores.min():.6f}  max {scores.max():.6f}")
    print(f"  pool_valid    {pool_valid.sum().item()}/{pool_valid.numel()}")

    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({"x": x, "q_resid": q_resid, "weights": w, "cfg": {
        k: cfg[k] for k in ("index_head_dim", "index_n_heads", "index_kpool",
                            "index_topk", "hidden_size", "q_lora_rank")},
        "index_scores": scores, "pool_keys": pool_keys,
        "pool_indices": pool_idx, "pool_valid": pool_valid},
        OUT / "dsa_case.pt")
    print(f"\nwrote {OUT / 'dsa_case.pt'}")
    dump_binary()
    if a.validate:
        ok = validate(a)
        raise SystemExit(0 if ok else 1)


def validate(a):
    """Prove the reimplementation matches transformers' Glm5NextTextIndexer.

    The module returns top-k INDICES, not scores, and at short context every pool is selected
    (select_k = min(topk // kpool, n_pools)), so comparing its output would compare nothing.
    Capture the intermediate `index_scores` instead - that is the quantity the C++ reproduces.
    """
    import torch
    from safetensors import safe_open
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig
    from transformers.models.glm5_next import modeling_glm5_next as M

    ck = Path(a.ckpt)
    cfg_all = json.loads((ck / "config.json").read_text())
    cfg = Glm5NextTextConfig(**{k: v for k, v in cfg_all.get("text_config", cfg_all).items()
                                if k != "architectures"})
    case = torch.load(OUT / "dsa_case.pt", weights_only=False)
    w, x, q_resid = case["weights"], case["x"], case["q_resid"]

    mod = M.Glm5NextTextIndexer(cfg, a.layer).eval()
    with torch.no_grad():
        mod.wq_b.weight.copy_(w["wq_b"]); mod.wk.weight.copy_(w["wk"])
        mod.k_norm.weight.copy_(w["k_norm_w"]); mod.k_norm.bias.copy_(w["k_norm_b"])
        mod.weights_proj.weight.copy_(w["weights_proj"])
        mod.index_kpool_compress_gate.copy_(w["kpool_gate"])
        mod.index_kpool_compress_ape.copy_(w["kpool_ape"])

    grabbed = {}
    real_topk = torch.Tensor.topk
    def spy(self, k, dim=-1, **kw):
        if self.dim() == 3 and self.shape[:2] == x.shape[:2]:
            grabbed["scores"] = self.clone()
        return real_topk(self, k, dim=dim, **kw)
    torch.Tensor.topk = spy
    try:
        mask = torch.ones(*x.shape[:2], dtype=torch.bool)
        with torch.no_grad():
            mod(hidden_states=x, q_resid=q_resid, attention_mask=mask, past_key_values=None)
    finally:
        torch.Tensor.topk = real_topk

    if "scores" not in grabbed:
        print("could not capture index_scores from the module"); return False
    ref = grabbed["scores"]
    mine = case["index_scores"]
    n = min(ref.shape[-1], mine.shape[-1])
    # Compare only unmasked entries: both sides fill invalid pools with finfo.min, and including
    # those makes the norm inf.
    lo = torch.finfo(ref.dtype).min / 2
    m = (ref[..., :n] > lo) & (mine[..., :n] > lo)
    print(f"  comparing {int(m.sum())}/{m.numel()} unmasked score entries")
    r = float((mine[..., :n][m] - ref[..., :n][m]).norm() / ref[..., :n][m].norm().clamp_min(1e-12))
    ok = r < 1e-5
    print(f"index_scores rel error vs transformers: {r:.3e}   {'MATCH' if ok else 'MISMATCH'}")
    return ok


def dump_binary():
    """Flat binary of the case, so the ggml test can read it without a torch dependency.

    Layout (all little-endian f32 unless noted):
      i32 S, D, hd, nh, kpool, q_lora, n_pools
      x            [S, D]
      q_resid      [S, q_lora]
      wq_b         [nh*hd, q_lora]
      wk           [hd, D]
      k_norm_w     [hd]      k_norm_b [hd]
      weights_proj [nh, D]
      kpool_gate   [hd, D]
      kpool_ape    [kpool, hd]
      index_scores [S, n_pools]   <- expected output
    """
    import struct, torch
    c = torch.load(OUT / "dsa_case.pt", weights_only=False)
    w, cfg = c["weights"], c["cfg"]
    x, q = c["x"][0], c["q_resid"][0]
    sc = c["index_scores"][0]
    S, D = x.shape
    hd, nh, kp, ql = cfg["index_head_dim"], cfg["index_n_heads"], cfg["index_kpool"], cfg["q_lora_rank"]
    P = sc.shape[-1]
    p = OUT / "dsa_case.bin"
    with open(p, "wb") as f:
        f.write(struct.pack("<7i", S, D, hd, nh, kp, ql, P))
        for t in (x, q, w["wq_b"], w["wk"], w["k_norm_w"], w["k_norm_b"],
                  w["weights_proj"], w["kpool_gate"], w["kpool_ape"], sc):
            f.write(t.contiguous().float().numpy().astype("<f4").tobytes())
    print(f"wrote {p}: S={S} D={D} hd={hd} nh={nh} kpool={kp} pools={P}")


if __name__ == "__main__":
    main()
