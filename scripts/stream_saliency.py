"""Path B - custom layer-streaming REAP saliency.

Used when the model cannot be placed by transformers/accelerate at all (see
wiki/96-implementation.md). Peak residency is ONE decoder layer.

Feasible because the entire inter-layer state in Glm5NextTextModel.forward is a single tensor
plus topk_indices - mHC's four residual streams live in the hc_mult axis of that tensor and
are manipulated inside each decoder layer, so there is no cross-layer bookkeeping to replicate.

Chunk by samples, sweeping all layers per chunk. The hc_mult=4 expansion makes activations 4x
larger than intuition suggests, and on this box read is 7x faster than write (3.4 GB/s vs
487 MB/s), so re-reading weights is far cheaper than re-writing activations.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch

SAL_SUM: dict[str, torch.Tensor] = {}
SAL_CNT: dict[str, torch.Tensor] = {}
_CUR_LAYER = {"name": None}


def dequant_fp8_block(w: torch.Tensor, scale_inv: torch.Tensor,
                      block: int = 128, dtype=torch.bfloat16) -> torch.Tensor:
    """FP8 E4M3 + per-128x128-block F32 scale -> dtype."""
    out, inn = w.shape
    s = scale_inv.to(torch.float32)
    s = s.repeat_interleave(block, dim=0).repeat_interleave(block, dim=1)[:out, :inn]
    return (w.to(torch.float32) * s).to(dtype)


def patch_experts_for_saliency():
    """Replace Glm5NextTextExperts.forward with a copy that also accumulates REAP saliency.

    The arithmetic is the upstream forward verbatim; the only addition is that the expert
    output f_j is captured BEFORE it is scaled by the router gate, which is what REAP's
    S_j = mean(g_j * ||f_j||_2) is defined over.
    """
    import torch.nn.functional as F
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextExperts

    if getattr(Glm5NextTextExperts, "_reap_patched", False):
        return

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        lname = _CUR_LAYER["name"]
        if lname is not None and lname not in SAL_SUM:
            SAL_SUM[lname] = torch.zeros(self.num_experts, dtype=torch.float64)
            SAL_CNT[lname] = torch.zeros(self.num_experts, dtype=torch.float64)
        for expert_idx in hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(mask[expert_idx])
            cur = self._apply_gate(F.linear(hidden_states[token_idx],
                                            self.gate_up_proj[expert_idx]))
            f_j = F.linear(cur, self.down_proj[expert_idx])          # expert output, ungated
            g_j = top_k_weights[token_idx, top_k_pos]                 # router gate
            if lname is not None:
                with torch.no_grad():
                    s = (g_j.to(torch.float32) *
                         f_j.to(torch.float32).norm(dim=-1)).sum().double().cpu()
                    SAL_SUM[lname][expert_idx] += s
                    SAL_CNT[lname][expert_idx] += float(token_idx.numel())
            final.index_add_(0, token_idx, (f_j * g_j[:, None]).to(final.dtype))
        return final

    Glm5NextTextExperts.forward = forward
    Glm5NextTextExperts._reap_patched = True


class ShardReader:
    """Lazily mmap the source shards and hand back one layer's tensors at a time."""

    def __init__(self, src: Path):
        self.src = Path(src)
        idx = json.loads((self.src / "model.safetensors.index.json").read_text())
        self.map: dict[str, str] = idx["weight_map"]
        self._open: dict[str, object] = {}

    def _f(self, shard: str):
        from safetensors import safe_open
        if shard not in self._open:
            self._open[shard] = safe_open(str(self.src / shard), framework="pt", device="cpu")
        return self._open[shard]

    def names_for_layer(self, prefix: str) -> list[str]:
        return [k for k in self.map if k.startswith(prefix)]

    def get(self, name: str) -> torch.Tensor:
        return self._f(self.map[name]).get_tensor(name)

    def load_module(self, prefix: str, dtype=torch.bfloat16) -> dict[str, torch.Tensor]:
        """Return {relative_name: tensor}, dequantising any FP8+scale_inv pair."""
        out: dict[str, torch.Tensor] = {}
        names = self.names_for_layer(prefix)
        scales = {n for n in names if n.endswith("weight_scale_inv")}
        for n in names:
            if n in scales:
                continue
            rel = n[len(prefix):].lstrip(".")
            t = self.get(n)
            sname = n + "_scale_inv" if not n.endswith(".weight") else n[:-len("weight")] + "weight_scale_inv"
            if sname in scales:
                t = dequant_fp8_block(t, self.get(sname), dtype=dtype)
            elif t.dtype in (torch.float8_e4m3fn,):
                t = t.to(dtype)
            else:
                t = t.to(dtype) if t.is_floating_point() else t
            out[rel] = t
        return out

    def close(self):
        self._open.clear()


def set_current_layer(name: str | None):
    _CUR_LAYER["name"] = name


def reset_accumulators():
    SAL_SUM.clear()
    SAL_CNT.clear()


def dump(dirpath: Path, layer_name_fmt: str = "model.language_model.layers.{i}.mlp"):
    dirpath = Path(dirpath)
    dirpath.mkdir(parents=True, exist_ok=True)
    for lname in sorted(SAL_SUM):
        torch.save({"layer": lname, "sum_saliency": SAL_SUM[lname],
                    "count": SAL_CNT[lname], "num_experts": SAL_SUM[lname].numel()},
                   dirpath / f"{lname.replace('.', '__')}.pt")
    return len(SAL_SUM)
