"""Dump Sinkhorn inputs/outputs so the ggml op can be tested against the validated oracle.

`vendor/mhc/mhc_test_vectors.json` carries streams and the final post/comb/collapsed, but not
the intermediate `comb_logits` the fused op actually consumes. This recomputes them with the
reference path and writes a flat binary the C test reads, so the op is checked in isolation -
a transcription bug in the normalisation order shows up here rather than as a slightly wrong
model 90 layers later.
"""
from __future__ import annotations
import json, struct, sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from mhc_reference import unweighted_rms_norm  # noqa: E402

V = json.loads((ROOT / "vendor" / "mhc" / "mhc_test_vectors.json").read_text())
meta = V["meta"]
hc, D = meta["hc_mult"], meta["hidden_size"]
T, iters, eps = meta["tokens"], meta["sinkhorn_iters"], meta["hc_eps"]

from safetensors import safe_open
ck = ROOT / "output" / meta["checkpoint"]
wm = json.loads((ck / "model.safetensors.index.json").read_text())["weight_map"]

out = open(ROOT / "vendor" / "mhc" / "sinkhorn_case.bin", "wb")
out.write(struct.pack("<iiif", hc, T * 2, iters, eps))     # hc, n_slices, iters, eps
logits_all, comb_all = [], []
for site in ("attn", "ffn"):
    w = {}
    for n in ("fn", "base", "scale"):
        k = f"model.language_model.layers.{meta['layer']}.hc_{site}_{n}"
        with safe_open(str(ck / wm[k]), framework="pt") as f:
            w[n] = f.get_tensor(k)
    streams = torch.tensor(V[site]["streams"], dtype=torch.float32).view(1, T, hc, D).bfloat16()
    flat = unweighted_rms_norm(streams.flatten(start_dim=-2).float(), meta["rms_norm_eps"])
    mix = torch.nn.functional.linear(flat, w["fn"].float())
    _, _, comb_w = mix.split([hc, hc, hc * hc], dim=-1)
    _, _, comb_b = w["base"].float().split([hc, hc, hc * hc])
    comb_s = w["scale"].float().unbind(0)[2]
    logits = comb_w.view(*comb_w.shape[:-1], hc, hc) * comb_s + comb_b.view(hc, hc)
    logits_all.append(logits.reshape(-1, hc, hc))
    comb_all.append(torch.tensor(V[site]["comb"], dtype=torch.float32).view(-1, hc, hc))

L = torch.cat(logits_all).contiguous()
C = torch.cat(comb_all).contiguous()
out.write(L.numpy().astype("<f4").tobytes())
out.write(C.numpy().astype("<f4").tobytes())
out.close()
print(f"wrote {ROOT/'vendor'/'mhc'/'sinkhorn_case.bin'}: {L.shape[0]} slices of {hc}x{hc}, "
      f"iters={iters} eps={eps}")
print(f"reference comb row-sums {C.sum(-1).flatten()[:4].tolist()}")
print(f"reference comb col-sums {C.sum(-2).flatten()[:4].tolist()}  (column-stochastic)")
