"""Stage 8 - document the output format (directive section 6.10).

The point of this stage is the downstream custom-CUDA-kernel and inference-server work: that
project needs the per-tensor layout and precision map, not prose.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import ROOT, ARTIFACTS, log, kv_get, publish  # noqa: E402

STAGE = "s08_document"


def run() -> dict:
    import struct
    out_dir = Path(kv_get("nvfp4_path", str(ROOT / "output" / "glm-5.3-flash-reap50-nvfp4")))
    shards = sorted(out_dir.glob("*.safetensors"))
    if not shards:
        raise RuntimeError(f"no safetensors in {out_dir}")

    inv, dtypes, groups = {}, Counter(), Counter()
    for sh in shards:
        with sh.open("rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        hdr.pop("__metadata__", None)
        for k, v in hdr.items():
            inv[k] = {"dtype": v["dtype"], "shape": v["shape"], "shard": sh.name}
            dtypes[v["dtype"]] += 1
            if ".experts." in k:
                groups["routed_expert"] += 1
            elif "self_attn" in k:
                groups["attention"] += 1
            elif "visual" in k:
                groups["vision"] += 1
            elif "hc_" in k or "mapping_proj" in k:
                groups["mhc"] += 1
            elif "mlp.gate" in k:
                groups["router"] += 1
            else:
                groups["other"] += 1

    cfg = json.loads((out_dir / "config.json").read_text())
    doc = {
        "checkpoint": str(out_dir),
        "format": "compressed-tensors",
        "shards": len(shards),
        "tensors": len(inv),
        "tensor_dtypes": dict(dtypes),
        "tensors_by_group": dict(groups),
        "quantization_config": cfg.get("quantization_config"),
        "precision_map": {
            "routed_experts": "NVFP4 (E2M1 4-bit + FP8 E4M3 scale per 16 elems + F32 global)",
            "shared_experts": "NVFP4",
            "attention_mla_dsa": "FP8 E4M3 dynamic",
            "kda_linear_attn_state": "unquantised (recurrence compounds error along sequence)",
            "vision_tower": "unquantised BF16 (0.18% of mass, first-class capability)",
            "mhc_routers_lm_head_embed": "unquantised BF16",
        },
        "serving_notes_jetson_thor": {
            "fused_moe_backend": "cutlass - the Marlin FP4 MoE kernel faults in-kernel at >=256 experts",
            "attention_backend": "TRITON_MLA - FLASHINFER is invalid for MLA",
            "power": "nvpmodel -m 0 (MAXN) + jetson_clocks; EMC 2750->4266 MHz matters for a bandwidth-bound MoE",
        },
        "excluded": {
            "mtp_layer_45": "not instantiated by transformers Glm5NextForConditionalGeneration; "
                            "archived unmodified rather than inconsistently pruned",
        },
    }
    p = ARTIFACTS / "output_format.json"
    p.write_text(json.dumps(doc, indent=2))
    inv_p = ARTIFACTS / "output_tensor_inventory.json"
    inv_p.write_text(json.dumps(inv, indent=2))
    publish(p, "artifacts", "stage08/output_format.json", stage=STAGE)
    publish(inv_p, "artifacts", "stage08/output_tensor_inventory.json", stage=STAGE)
    log(f"documented {len(inv)} tensors across {len(shards)} shards", STAGE)
    return doc
