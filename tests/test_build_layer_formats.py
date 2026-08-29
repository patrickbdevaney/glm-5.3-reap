"""Build one MoE decoder layer from BOTH checkpoint formats and check nothing is missing.

The NVFP4 read path shipped broken and was not caught until an evaluation crashed 45 minutes in
with `mat1 and mat2 to have the same dtype: BFloat16 != float` - an error that names neither the
tensor nor the format. The cause: NVFP4 stores `weight_packed` / `weight_scale` /
`weight_global_scale` and NO bare `weight`, so iterating raw tensor names fed the module a
parameter it does not have. `load_state_dict(strict=False)` reported it as missing, the module
kept its uninitialised f32 buffer, and the run continued.

The check that would have caught it in seconds is: after building a layer, is `missing` empty?
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

fails = []


def check(name, cond, extra=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name} {extra}")
    if not cond:
        fails.append(name)


def probe(ckpt: Path, label: str, layer_idx: int = 3):
    from transformers import AutoConfig
    from stages.s03_saliency import _build_layer
    from stream_saliency import ShardReader
    cfg = AutoConfig.from_pretrained(ckpt).text_config
    reader = ShardReader(ckpt)
    layer = _build_layer(cfg, layer_idx, reader, torch.bfloat16)
    bad = [n for n, p in layer.named_parameters() if p.dtype not in (torch.bfloat16, torch.float32)]
    # Every floating parameter must have been REPLACED by a real tensor. `to_empty` leaves f32
    # garbage, so any f32 float parameter in a bf16 build is one that was never loaded.
    stale = [n for n, p in layer.named_parameters()
             if p.is_floating_point() and p.dtype == torch.float32]
    check(f"{label}: no unexpected dtypes", not bad, str(bad[:3]))
    check(f"{label}: no parameter left at to_empty f32", not stale, str(stale[:3]))
    e = layer.mlp.experts.down_proj if hasattr(layer.mlp, "experts") else None
    if e is not None:
        finite = bool(torch.isfinite(e[0]).all())
        nonzero = float(e[0].abs().float().mean()) > 0
        check(f"{label}: expert down_proj finite and non-trivial", finite and nonzero)
    reader.release()
    del layer


def numeric_agreement(fp8: Path, nvfp4: Path, layer_idx: int = 3):
    """The NVFP4 read path can be structurally fine and numerically wrong - a mis-applied group
    or global scale still yields a full, finite tensor. Dequantise the same expert from both
    checkpoints and compare. NVFP4 is E2M1 with a per-16 FP8 scale, so a few percent relative
    error is expected; an order of magnitude is a scale bug."""
    from transformers import AutoConfig
    from stages.s03_saliency import _build_layer
    from stream_saliency import ShardReader
    cfg = AutoConfig.from_pretrained(fp8).text_config
    out = {}
    for label, d in (("fp8", fp8), ("nvfp4", nvfp4)):
        r = ShardReader(d)
        L = _build_layer(cfg, layer_idx, r, torch.bfloat16)
        out[label] = L.mlp.experts.down_proj[:4].detach().float().clone()
        r.release()
        del L
    a, b = out["fp8"], out["nvfp4"]
    rel = float((b - a).norm() / a.norm())
    cos = float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0))
    check(f"NVFP4 dequant matches FP8 (rel err {rel:.4f}, cos {cos:.5f})",
          rel < 0.15 and cos > 0.985)


if __name__ == "__main__":
    print("build a decoder layer from each checkpoint format:")
    from common import kv_get
    for name, label in ((str(kv_get("nvfp4_name") or ""), "NVFP4"),
                        (str(kv_get("emit_name") or ""), "FP8")):
        d = ROOT / "output" / name
        if not (d / "model.safetensors.index.json").exists():
            print(f"  SKIP  {label} ({d} absent)")
            continue
        probe(d, label)
    fp8d = ROOT / "output" / str(kv_get("emit_name") or "")
    nvd = ROOT / "output" / str(kv_get("nvfp4_name") or "")
    if (fp8d / "model.safetensors.index.json").exists() and \
       (nvd / "model.safetensors.index.json").exists():
        numeric_agreement(fp8d, nvd)
    print(f"\n{len(fails)} failure(s)" if fails else "\nall checks passed")
    raise SystemExit(1 if fails else 0)
