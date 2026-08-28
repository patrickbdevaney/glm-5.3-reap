"""Tensor-level NVFP4 quantisation.

Every model-level path failed on this box, for reasons that are all the same reason - the model
is 157 GiB and nothing can hold it:

  * device_map="cpu"  -> does not fit 122 GiB of RAM
  * device_map="auto" -> cudaErrorIllegalAddress; Thor's memory is unified, so accelerate reads
                         ~122 GiB of "VRAM" and exhausts the pool it is measuring
  * accelerate DISK offload -> llm-compressor's oneshot asserts offloaded params live on `meta`,
                         which is only true for CPU offload

So quantise the way the rest of this pipeline works: one tensor at a time, straight from
safetensors, never building a model. Same shape as s04b surgery and s03 saliency.

Correctness is not assumed - `verify_against_library()` compares this against
compressed-tensors' own compressor on the same input and requires bit-identical output.
"""
from __future__ import annotations

import torch
from compressed_tensors.compressors.nvfp4.helpers import pack_fp4_to_uint8
from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy
from compressed_tensors.quantization.lifecycle.forward import quantize
from compressed_tensors.quantization.utils.helpers import calculate_qparams, generate_gparam

GROUP = 16

NVFP4_ARGS = QuantizationArgs(
    num_bits=4, type="float", symmetric=True,
    strategy=QuantizationStrategy.TENSOR_GROUP, group_size=GROUP,
    scale_dtype=torch.float8_e4m3fn,
)


def dequant_fp8_block(w: torch.Tensor, scale_inv: torch.Tensor, block: int = 128,
                      dtype=torch.float32) -> torch.Tensor:
    """FP8 E4M3 + per-128x128-block F32 scale -> dtype."""
    out, inn = w.shape
    s = scale_inv.to(torch.float32)
    s = s.repeat_interleave(block, dim=0).repeat_interleave(block, dim=1)[:out, :inn]
    return (w.to(torch.float32) * s).to(dtype)


def quantize_nvfp4(w: torch.Tensor) -> dict[str, torch.Tensor]:
    """Quantise a 2D weight to NVFP4, returning compressed-tensors' three parameters."""
    w = w.to(torch.float32)
    out, inn = w.shape
    if inn % GROUP:
        raise ValueError(f"in_features {inn} not divisible by group size {GROUP}")

    # Global scale maps the per-group maxima into the FP8 range that holds the local scales.
    gs = generate_gparam(w.min(), w.max())

    g = w.reshape(out, inn // GROUP, GROUP)
    scale, zp = calculate_qparams(g.min(dim=-1).values, g.max(dim=-1).values,
                                  NVFP4_ARGS, global_scale=gs)
    q = quantize(x=w, scale=scale, zero_point=zp, args=NVFP4_ARGS, global_scale=gs)
    return {"weight_packed": pack_fp4_to_uint8(q),
            "weight_scale": scale.to(torch.float8_e4m3fn),
            "weight_global_scale": gs.to(torch.float32)}


def verify_against_library(seed: int = 0, shape=(256, 512)) -> dict:
    """Prove this matches compressed-tensors' own compressor, bit for bit.

    Hand-rolling a checkpoint format is exactly the kind of thing that produces a file which
    loads and is silently wrong, so this is not optional.
    """
    from compressed_tensors.compressors.nvfp4.base import NVFP4PackedCompressor
    from compressed_tensors.quantization import QuantizationScheme

    torch.manual_seed(seed)
    w = torch.randn(*shape, dtype=torch.float32) * 0.05
    mine = quantize_nvfp4(w)

    gs = generate_gparam(w.min(), w.max())
    g = w.reshape(shape[0], shape[1] // GROUP, GROUP)
    scale, zp = calculate_qparams(g.min(dim=-1).values, g.max(dim=-1).values,
                                  NVFP4_ARGS, global_scale=gs)
    scheme = QuantizationScheme(targets=["Linear"], weights=NVFP4_ARGS)
    theirs = NVFP4PackedCompressor.compress(
        {"weight": w, "weight_scale": scale, "weight_global_scale": gs}, scheme)

    return {
        "packed_equal": bool(torch.equal(mine["weight_packed"], theirs["weight_packed"])),
        "scale_equal": bool(torch.equal(mine["weight_scale"].float(),
                                        theirs["weight_scale"].float())),
        "packed_dtype": str(mine["weight_packed"].dtype),
        "packed_shape": tuple(mine["weight_packed"].shape),
        "scale_dtype": str(mine["weight_scale"].dtype),
        "scale_shape": tuple(mine["weight_scale"].shape),
        "global_scale": float(mine["weight_global_scale"]),
    }


def roundtrip_error(w: torch.Tensor) -> float:
    """Relative Frobenius error of a quantise/dequantise round trip."""
    from compressed_tensors.compressors.nvfp4.helpers import unpack_fp4_from_uint8
    out = quantize_nvfp4(w)
    q = unpack_fp4_from_uint8(out["weight_packed"], *w.shape, dtype=torch.float32)
    s = out["weight_scale"].to(torch.float32) / out["weight_global_scale"].to(torch.float32)
    deq = (q.reshape(w.shape[0], -1, GROUP) * s.unsqueeze(-1)).reshape(w.shape)
    wf = w.to(torch.float32)
    return float((deq - wf).norm() / wf.norm().clamp(min=1e-12))
