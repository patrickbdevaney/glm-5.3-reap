"""Stage 7 - NVFP4 quantisation, tensor by tensor.

No model is ever built. Every model-level route failed on this box for the same underlying
reason - 157 GiB does not fit anywhere:

  * device_map="cpu"        -> exceeds 122 GiB of RAM
  * device_map="auto"       -> cudaErrorIllegalAddress (unified memory: accelerate reads
                               ~122 GiB of "VRAM" and exhausts the pool it is measuring)
  * accelerate disk offload -> llm-compressor's oneshot asserts offloaded params are on `meta`,
                               which only holds for CPU offload

So this streams safetensors shard by shard, exactly like s04b surgery and s03 saliency.
`scripts/nvfp4_tensor.py` is verified bit-identical to compressed-tensors' own compressor.

Precision policy (wiki/60-quantization.md):
  routed + shared experts  -> NVFP4   ~97% of parameters; the only thing worth compressing
  everything else          -> BF16    3% of mass: attention, KDA state, vision tower, mHC,
                                      routers, lm_head, embeddings

Non-expert FP8 is dequantised to BF16 rather than left in block format, so the result carries
ONE quantisation format plus plain BF16 instead of two incompatible schemes in one file.
"""
from __future__ import annotations

import json
import re
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import ROOT, ARTIFACTS, log, metric, kv_get, kv_set, publish, free_gib  # noqa: E402

STAGE = "s07_quantize"
OUT = ROOT / "output" / "glm-5.3-flash-reap50-nvfp4"

EXPERT_RE = re.compile(r"\.mlp\.(experts\.\d+|shared_experts)\.(gate_proj|up_proj|down_proj)\.weight$")

# Untouched: quantising any of these saves almost nothing and risks real capability.
IGNORE = [
    "re:.*lm_head.*", "re:.*embed_tokens.*",
    "re:.*visual.*",            # dense ViT, 0.18% of mass, first-class capability
    "re:.*linear_attn.*",       # KDA recurrence: error compounds along the sequence
    "re:.*self_attn.*",
    "re:.*hc_.*", "re:.*mapping_proj.*",   # mHC, Sinkhorn-normalised
    "re:.*mlp\\.gate\\..*",     # routers: argmax-sensitive
    "re:.*norm.*",
]


def run() -> dict:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    import nvfp4_tensor as NV

    src = Path(kv_get("emit_path", str(ROOT / "output" / "glm-5.3-flash-reap50-fp8")))
    if not src.exists():
        raise RuntimeError(f"FP8 source not found at {src}")
    OUT.mkdir(parents=True, exist_ok=True)

    v = NV.verify_against_library()
    if not (v["packed_equal"] and v["scale_equal"]):
        raise RuntimeError(f"NVFP4 implementation does not match compressed-tensors: {v}")
    log(f"NVFP4 implementation verified bit-identical to compressed-tensors {v['packed_shape']}",
        STAGE)

    shards = sorted(src.glob("*.safetensors"))
    done = {p.name for p in OUT.glob("*.safetensors")}
    if done:
        log(f"resuming: {len(done)} shards already quantised", STAGE)

    weight_map: dict[str, str] = {}
    ip = OUT / "model.safetensors.index.json"
    if ip.exists():
        try:
            weight_map = json.loads(ip.read_text())["weight_map"]
        except Exception:
            weight_map = {}

    n_q = n_bf16 = 0
    t0 = time.time()
    for si, shard in enumerate(shards, 1):
        if shard.name in done:
            continue
        out_t: dict[str, torch.Tensor] = {}
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            keys = set(f.keys())
            for name in sorted(keys):
                if name.endswith("weight_scale_inv"):
                    continue                       # consumed with its weight
                t = f.get_tensor(name)
                sname = name[: -len("weight")] + "weight_scale_inv" if name.endswith("weight") else None
                if EXPERT_RE.search(name):
                    w = NV.dequant_fp8_block(t, f.get_tensor(sname)) if sname in keys \
                        else t.to(torch.float32)
                    q = NV.quantize_nvfp4(w)
                    base = name[: -len(".weight")]
                    out_t[f"{base}.weight_packed"] = q["weight_packed"]
                    out_t[f"{base}.weight_scale"] = q["weight_scale"]
                    out_t[f"{base}.weight_global_scale"] = q["weight_global_scale"]
                    n_q += 1
                    del w, q
                else:
                    # Non-expert: dequantise any block-FP8 to BF16 so the file carries one
                    # quantisation format plus plain BF16, not two incompatible schemes.
                    if sname in keys:
                        t = NV.dequant_fp8_block(t, f.get_tensor(sname), dtype=torch.bfloat16)
                    elif t.dtype == torch.float8_e4m3fn:
                        t = t.to(torch.bfloat16)
                    out_t[name] = t
                    n_bf16 += 1

        save_file(out_t, str(OUT / shard.name), metadata={"format": "pt"})
        for k in out_t:
            weight_map[k] = shard.name
        del out_t
        ip.write_text(json.dumps({"metadata": {"total_size": 0}, "weight_map": weight_map}))
        if si % 5 == 0 or si == 1:
            el = time.time() - t0
            log(f"shard {si}/{len(shards)}  quantised={n_q} bf16={n_bf16}  "
                f"elapsed {el/60:.1f} min  eta {(el/si)*(len(shards)-si)/60:.0f} min  "
                f"free {free_gib():.0f} GiB", STAGE)

    # config: compressed-tensors NVFP4 on the experts, everything else ignored
    cfg = json.loads((src / "config.json").read_text())
    cfg["quantization_config"] = {
        "quant_method": "compressed-tensors",
        "format": "nvfp4-pack-quantized",
        "quantization_status": "compressed",
        "ignore": IGNORE,
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": {"num_bits": 4, "type": "float", "symmetric": True,
                            "strategy": "tensor_group", "group_size": 16,
                            "scale_dtype": "float8_e4m3fn", "dynamic": False},
                "input_activations": None, "output_activations": None,
            }
        },
    }
    (OUT / "config.json").write_text(json.dumps(cfg, indent=2))
    for extra in ("tokenizer.json", "tokenizer_config.json", "generation_config.json",
                  "chat_template.jinja", "preprocessor_config.json", "LICENSE", "README.md"):
        sp = src / extra
        if sp.exists():
            shutil.copy2(sp, OUT / extra)

    total = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    gib = total / 2**30
    metric(STAGE, "nvfp4_bytes", total)
    metric(STAGE, "quantize_minutes", (time.time() - t0) / 60)
    res = {"path": str(OUT), "bytes": total, "gib": round(gib, 1),
           "tensors_nvfp4": n_q, "tensors_bf16": n_bf16,
           "minutes": round((time.time() - t0) / 60, 1), "fits_thor": gib < 117}
    log(f"NVFP4 checkpoint: {gib:.1f} GiB, {n_q} expert tensors quantised, "
        f"{n_bf16} kept", STAGE)
    if gib > 117:
        log(f"checkpoint is {gib:.1f} GiB, over the ~117 GiB Thor envelope", STAGE, "ERROR")
    p = ARTIFACTS / "s07_quantize.json"
    p.write_text(json.dumps(res, indent=2))
    publish(p, "artifacts", "stage07/s07_quantize.json", stage=STAGE)
    kv_set("nvfp4_path", str(OUT))
    return res
