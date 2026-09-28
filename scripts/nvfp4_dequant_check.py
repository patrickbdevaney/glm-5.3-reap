"""Is the hand-rolled NVFP4 dequant in _build_layer actually right?

`stages/s03_saliency._build_layer` reconstitutes NVFP4 weights itself rather than going through
compressed-tensors, so that a layer can be materialised one at a time without ever building a
model - the only way anything of this size runs on this box. Every hidden state captured from the
NVFP4 checkpoint inherits that arithmetic, and a draft head trained on subtly wrong teacher states
would be wrong in a way no downstream check would attribute back here.

The quantise direction is already covered: nvfp4_tensor.verify_against_library() demands bit
identity against compressed-tensors' own compressor. This closes the other direction, against the
library's own decompress() on tensors taken from the shipped checkpoint rather than on synthetic
input.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import torch
from safetensors import safe_open
from compressed_tensors.compressors.nvfp4.base import NVFP4PackedCompressor
from compressed_tensors.compressors.nvfp4.helpers import unpack_fp4_from_uint8
from compressed_tensors.quantization import (
    QuantizationArgs, QuantizationScheme, QuantizationStrategy, QuantizationType,
)

CKPT = Path("output/glm-5.3-flash-reap50-nvfp4-pass2")


def handrolled(pk, sc, gs, dtype=torch.bfloat16):
    """Verbatim the arithmetic in _build_layer.fetch()'s NVFP4 branch."""
    sc = sc.to(torch.float32)
    gs = gs.to(torch.float32)
    out_f, in_f = pk.shape[0], pk.shape[1] * 2
    q = unpack_fp4_from_uint8(pk, out_f, in_f, dtype=torch.float32)
    g = q.reshape(out_f, in_f // 16, 16)
    w = g * (sc.reshape(out_f, in_f // 16, 1) / gs)
    return w.reshape(out_f, in_f).to(dtype)


def main():
    idx = json.loads((CKPT / "model.safetensors.index.json").read_text())["weight_map"]
    bases = sorted({k[: -len("weight_packed")] for k in idx if k.endswith("weight_packed")})
    print(f"{len(bases)} NVFP4 tensors in the checkpoint")

    # A spread rather than one tensor: attention and MoE quantise to different shapes and a
    # scale-broadcast bug can be shape-specific.
    picks = [b for b in bases if "self_attn" in b][:2] + \
            [b for b in bases if "experts.0." in b][:2] + \
            [b for b in bases if "shared_expert" in b][:1] + \
            [b for b in bases if ".45." in b][:2]
    if not picks:
        sys.exit("no tensors matched the sample selection")

    args = QuantizationArgs(num_bits=4, type=QuantizationType.FLOAT, symmetric=True,
                            strategy=QuantizationStrategy.TENSOR_GROUP, group_size=16)
    scheme = QuantizationScheme(targets=["Linear"], weights=args)

    worst = 0.0
    for b in picks:
        shard = idx[b + "weight_packed"]
        with safe_open(CKPT / shard, framework="pt") as f:
            pk = f.get_tensor(b + "weight_packed")
            sc = f.get_tensor(b + "weight_scale")
            gs = f.get_tensor(b + "weight_global_scale")

        mine = handrolled(pk, sc, gs).to(torch.float32)

        sd = {"weight_packed": pk, "weight_scale": sc, "weight_global_scale": gs}
        theirs = dict(NVFP4PackedCompressor.decompress(sd, scheme))
        theirs = theirs["weight"].to(torch.float32)

        if mine.shape != theirs.shape:
            print(f"SHAPE MISMATCH {b}: {tuple(mine.shape)} vs {tuple(theirs.shape)}")
            worst = float("inf"); continue

        denom = theirs.abs().max().clamp_min(1e-12)
        rel = (mine - theirs).abs().max() / denom
        exact = torch.equal(mine, theirs)
        worst = max(worst, float(rel))
        print(f"{b[-58:]:58s} {tuple(mine.shape)!s:>18s} "
              f"max|d|/max|w| {float(rel):.3e} {'EXACT' if exact else ''}")

    print()
    if worst == 0.0:
        print("PASS: bit-identical to compressed-tensors on every sampled tensor.")
    elif worst < 1e-6:
        print(f"PASS: agrees to {worst:.2e} relative - float reassociation, not a formula error.")
    else:
        print(f"FAIL: disagrees by {worst:.2e} relative. _build_layer's NVFP4 branch is wrong,")
        print("      and every hidden state captured from this checkpoint would be wrong with it.")
        sys.exit(1)


if __name__ == "__main__":
    main()
