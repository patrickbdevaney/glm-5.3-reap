"""Does routing through a Q8_0 intermediate cost anything the 4-bit step does not already?

The BF16/F32 intermediate is the textbook path but needs 308 GiB against 262 GiB free, so the
build goes FP8 -> Q8_0 -> Q4_K. The question is whether that middle step is free in practice.
Three errors on real expert weights, all against the exact F32 dequantisation of the FP8 source:

    A  Q8_0 alone            - what the intermediate costs by itself
    B  Q4_K direct from F32  - the ideal path
    C  Q4_K via Q8_0         - what actually ships

If C is indistinguishable from B, the intermediate is free and the disk constraint costs nothing.
If C is materially worse, the plan has to change (delete the local NVFP4 and use BF16).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path.home() / "glm5-llama.cpp" / "gguf-py"))
from gguf import quants, GGMLQuantizationType as T  # noqa: E402

CK = ROOT / "output" / "glm-5.3-flash-reap50-fp8-pass2"


def dequant_fp8_block(w: torch.Tensor, s: torch.Tensor, bs: int = 128) -> torch.Tensor:
    o, i = w.shape
    wf = w.to(torch.float32)
    sr = s.to(torch.float32)
    # The scale grid is ceil(o/bs) x ceil(i/bs); expand it and crop, rather than assume the
    # weight dims are exact multiples of the block size (they are not for every projection).
    sr = sr.repeat_interleave(bs, 0).repeat_interleave(bs, 1)[:o, :i]
    assert sr.shape == wf.shape, f"scale grid {tuple(sr.shape)} vs weight {tuple(wf.shape)}"
    return wf * sr


def rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def main():
    from safetensors import safe_open
    wm = json.loads((CK / "model.safetensors.index.json").read_text())["weight_map"]
    # A spread of tensor kinds: routed expert projections and a dense one, from different depths.
    names = [k for k in wm if k.endswith(".weight") and (k + "_scale_inv") in wm]
    picks = [n for n in names if ".mlp.experts.0." in n][:3] + \
            [n for n in names if ".mlp.experts.77." in n][:2] + \
            [n for n in names if "shared_experts" in n][:1]
    print(f"{'tensor':<52}{'A Q8_0':>10}{'B Q4_K':>10}{'C via Q8':>10}{'C/B':>8}")
    rows = []
    for n in picks:
        with safe_open(str(CK / wm[n]), framework="pt") as f:
            w = f.get_tensor(n)
            s = f.get_tensor(n + "_scale_inv")
        W = dequant_fp8_block(w, s).numpy().astype(np.float32)

        q8 = quants.dequantize(quants.quantize(W, T.Q8_0), T.Q8_0).astype(np.float32)
        q4_direct = quants.dequantize(quants.quantize(W,  T.Q4_K), T.Q4_K).astype(np.float32)
        q4_via8   = quants.dequantize(quants.quantize(q8, T.Q4_K), T.Q4_K).astype(np.float32)

        a, b, c = rel(q8, W), rel(q4_direct, W), rel(q4_via8, W)
        rows.append((a, b, c))
        short = n.replace("model.language_model.layers.", "L").replace(".weight", "")
        print(f"  {short:<50}{a:>10.5f}{b:>10.5f}{c:>10.5f}{c/b:>8.4f}")

    A, B, C = (np.mean([r[i] for r in rows]) for i in range(3))
    print(f"\nmean   Q8_0 alone {A:.5f} | Q4_K direct {B:.5f} | Q4_K via Q8_0 {C:.5f}")
    print(f"the intermediate adds {100*(C/B - 1):.2f}% to the 4-bit error")
    print("VERDICT:", "intermediate is free" if C/B < 1.02 else
          "intermediate costs real quality - use BF16 instead")


if __name__ == "__main__":
    main()
