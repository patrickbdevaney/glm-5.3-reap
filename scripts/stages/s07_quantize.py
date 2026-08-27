"""Stage 7 - NVFP4 quantisation to the Thor-resident checkpoint.

Per-component precision policy (wiki/60-quantization.md), following the GLM-5.2 precedent of
NVFP4 on MoE + FP8 on attention:

  routed + shared experts   -> NVFP4 (W4A4)   ~97% of parameters; the only thing worth compressing
  attention (MLA + DSA)     -> FP8            1.9% of mass, MLA latents are sensitive
  KDA linear-attn state     -> untouched      a recurrence compounds error along the sequence
  vision tower              -> untouched      0.18% of mass, first-class capability
  mHC / routers / lm_head   -> untouched      tiny, numerically delicate, argmax-sensitive

The whole non-expert model is ~3% of parameters, so everything worth protecting is affordable
to protect. That is why the 50% target has headroom at all.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import ROOT, ARTIFACTS, log, metric, kv_get, kv_set, publish, free_gib  # noqa: E402

STAGE = "s07_quantize"
OUT = ROOT / "output" / "glm-5.3-flash-reap50-nvfp4"
OFFLOAD = ROOT / "offload"

# Untouched: quantising any of these costs almost nothing in size and risks real capability.
IGNORE = [
    "re:.*lm_head.*",
    "re:.*embed_tokens.*",
    "re:.*visual.*",            # dense ViT, 0.18% of mass, first-class capability
    "re:.*linear_attn.*",       # KDA recurrence
    "re:.*\\.self_attn\\.(A_log|dt_bias|.*_conv1d|[fgb]_[ab]_proj).*",
    "re:.*hc_.*", "re:.*mapping_proj.*",   # mHC, Sinkhorn-normalised
    "re:.*mlp\\.gate\\..*",     # routers: argmax-sensitive
    "re:.*shared_experts.*gate\\..*",
]

N_CALIB = 256          # activation scales converge on a few hundred samples
MAX_LEN = 2048


def _calib():
    import torch
    from datasets import Dataset
    rows = []
    tdir = ROOT / "corpus" / "shards" / "text"
    for shard in sorted(tdir.glob("*.pt")):
        try:
            for it in torch.load(shard, weights_only=False):
                rows.append({"input_ids": it["input_ids"][:MAX_LEN].tolist()})
                if len(rows) >= N_CALIB:
                    return Dataset.from_list(rows)
        except Exception:
            continue
    return Dataset.from_list(rows) if rows else None


def run() -> dict:
    from transformers import AutoTokenizer
    from transformers.models.glm5_next import Glm5NextForConditionalGeneration
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from llmcompressor.modeling.moe.linearize import linearize_moe
    import glm5_next_support

    src = Path(kv_get("emit_path", str(ROOT / "output" / "glm-5.3-flash-reap50-fp8")))
    if not src.exists():
        raise RuntimeError(f"emit output not found at {src}")
    OUT.mkdir(parents=True, exist_ok=True)
    OFFLOAD.mkdir(parents=True, exist_ok=True)
    glm5_next_support.register()

    log(f"loading pruned checkpoint from {src} (free disk {free_gib():.0f} GiB)", STAGE)
    model = Glm5NextForConditionalGeneration.from_pretrained(
        src, device_map="cpu", dtype="auto")
    linearize_moe(model)
    tok = AutoTokenizer.from_pretrained(src)

    recipe = [QuantizationModifier(
        config_groups={
            "experts_nvfp4": {
                "targets": ["re:.*mlp\\.experts\\..*", "re:.*shared_experts\\.(gate|up|down)_proj"],
                "scheme": "NVFP4",
            },
            "attention_fp8": {
                "targets": ["re:.*self_attn\\.(q_a_proj|q_b_proj|kv_a_proj.*|kv_b_proj|o_proj)"],
                "scheme": "FP8_DYNAMIC",
            },
        },
        ignore=IGNORE,
    )]

    ds = _calib()
    log(f"NVFP4 oneshot: {len(ds) if ds else 0} calibration samples @ {MAX_LEN}", STAGE)
    t0 = time.time()
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        num_calibration_samples=len(ds) if ds else 0,
        max_seq_length=MAX_LEN,
        pipeline="sequential",
        sequential_offload_device="cpu",
        moe_calibrate_all_experts=True,   # quantisation DOES want every expert observed
        shuffle_calibration_samples=False,
        output_dir=str(OUT),
        save_compressed=True,
    )
    dt = time.time() - t0
    metric(STAGE, "quantize_minutes", dt / 60)

    if not any(OUT.glob("*.safetensors")):
        model.save_pretrained(str(OUT), safe_serialization=True)
    tok.save_pretrained(str(OUT))

    total = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    gib = total / 2**30
    metric(STAGE, "nvfp4_bytes", total)
    log(f"NVFP4 checkpoint: {gib:.1f} GiB in {dt/60:.1f} min", STAGE)
    if gib > 117:
        log(f"checkpoint is {gib:.1f} GiB, over the ~117 GiB Thor envelope", STAGE, "ERROR")

    res = {"path": str(OUT), "bytes": total, "gib": round(gib, 1),
           "minutes": round(dt / 60, 1), "fits_thor": gib < 117}
    out = ARTIFACTS / "s07_quantize.json"
    out.write_text(json.dumps(res, indent=2))
    publish(out, "artifacts", "stage07/s07_quantize.json", stage=STAGE)
    kv_set("nvfp4_path", str(OUT))
    publish(OUT, "nvfp4", ".", stage=STAGE)
    return res
