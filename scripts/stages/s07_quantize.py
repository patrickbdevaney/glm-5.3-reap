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
    from compressed_tensors.quantization import preset_name_to_scheme
    from llmcompressor.modeling.moe.linearize import linearize_moe
    import glm5_next_support

    if kv_get("skip_fp8_intermediate", False):
        nv = Path(kv_get("nvfp4_path", str(OUT)))
        total = sum(p.stat().st_size for p in nv.rglob("*") if p.is_file())
        gib = total / 2**30
        log(f"NVFP4 was written directly by stage 3 (R10 disk-pressure path): "
            f"{nv} ({gib:.1f} GiB). Nothing to re-quantise.", STAGE)
        return {"path": str(nv), "bytes": total, "gib": round(gib, 1),
                "fits_thor": gib < 117, "produced_by": "s03 (R10 path)"}
    src = Path(kv_get("emit_path", str(ROOT / "output" / "glm-5.3-flash-reap50-fp8")))
    if not src.exists():
        raise RuntimeError(f"emit output not found at {src}")
    OUT.mkdir(parents=True, exist_ok=True)
    OFFLOAD.mkdir(parents=True, exist_ok=True)
    glm5_next_support.register()

    # The pruned model is ~156 GiB against 116 GiB of RAM, so it still cannot sit in memory -
    # but surgery deletes the source as it goes, so by now there is ~300 GiB of free disk and
    # accelerate can offload. Decide from measured space rather than assuming.
    model_gib = sum(p.stat().st_size for p in src.glob("*.safetensors")) / 2**30
    have = free_gib()
    log(f"pruned checkpoint {model_gib:.0f} GiB, {have:.0f} GiB free", STAGE)
    # Only the part that does not fit in RAM lands on disk, so the requirement is
    # (model - usable RAM), not the whole model. offload_state_dict=True would write the ENTIRE
    # state dict out first and does need the full size - that is the difference between needing
    # ~173 GiB and ~60 GiB here.
    ram_gib = 0.0
    with open("/proc/meminfo") as fh:
        for line in fh:
            if line.startswith("MemAvailable"):
                ram_gib = int(line.split()[1]) / 1048576
                break
    need = max(model_gib - ram_gib * 0.8, 0) + 15
    if have < need:
        raise RuntimeError(
            f"need ~{need:.0f} GiB free to offload the part of the model that will not fit in "
            f"RAM ({model_gib:.0f} GiB model, {ram_gib:.0f} GiB available), have {have:.0f}.")
    log(f"loading with disk offload -> {OFFLOAD} (need ~{need:.0f} GiB, have {have:.0f})", STAGE)
    model = Glm5NextForConditionalGeneration.from_pretrained(
        src, device_map="auto", dtype="auto",
        offload_folder=str(OFFLOAD), offload_state_dict=False)
    linearize_moe(model)
    tok = AutoTokenizer.from_pretrained(src)

    # config_groups takes QuantizationScheme objects, NOT a {"scheme": "NVFP4"} dict -
    # pydantic rejects the latter with extra_forbidden. preset_name_to_scheme resolves the
    # preset and attaches the targets. Verified to yield: experts 4-bit float, tensor_group,
    # group_size=16 (NVFP4's 16-element blocks) with dynamic local activations; attention
    # 8-bit float, channel weights, token-dynamic activations.
    recipe = [QuantizationModifier(
        config_groups={
            "experts_nvfp4": preset_name_to_scheme(
                # NVFP4A16 (weight-only), not NVFP4 (W4A4). W4A4 needs calibration FORWARD
                # passes to fit activation scales, and this model's KDA layers cost ~13 GiB of
                # transient memory per 2048-token sequence - the wall that killed stage 3 six
                # times. Weight-only needs no forwards at all, so the whole class of failure
                # disappears. The cost is serving throughput, not accuracy: the weights are
                # identical NVFP4 either way. Activation scales can be fitted later, cheaply,
                # against the finished 91 GiB checkpoint rather than the 157 GiB one.
                "NVFP4A16",
                targets=["re:.*mlp\\.experts\\..*",
                         "re:.*shared_experts\\.(gate|up|down)_proj"]),
            "attention_fp8": preset_name_to_scheme(
                "FP8_DYNAMIC",
                targets=["re:.*self_attn\\.(q_a_proj|q_b_proj|kv_a_proj.*|kv_b_proj|o_proj)"]),
        },
        ignore=IGNORE,
    )]

    # No calibration data: NVFP4A16 weights are a deterministic per-block transform and
    # FP8_DYNAMIC computes activation scales at runtime. Neither needs a forward pass.
    ds = None
    log("NVFP4A16 oneshot: weight-only, no calibration forwards required", STAGE)
    t0 = time.time()
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        num_calibration_samples=0,
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
