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
# Versioned for the same reason as the FP8 emit: pass 2 must not overwrite pass 1's artifact.
OUT = ROOT / "output" / str(kv_get("nvfp4_name", "glm-5.3-flash-reap50-nvfp4"))
# Keep non-expert FP8 tensors in FP8 instead of upcasting them to BF16. Saves
# 1.45 GiB at bit-identical quality, but produces a MIXED-scheme checkpoint.
# Default off until such a checkpoint has been loaded successfully; see the branch
# in run() for the reasoning.
KEEP_FP8 = bool(kv_get("nvfp4_keep_fp8_passthrough", False))

EXPERT_RE = re.compile(r"\.mlp\.(experts\.\d+|shared_experts)\.(gate_proj|up_proj|down_proj)\.weight$")

# Attention projections, quantised only when `nvfp4_quantize_attention` is set.
#
# WHY THESE AND NOT ALL OF self_attn. Decode is dense-bound, not expert-bound: only 8 of 144
# experts are read per token while the entire attention stack is read every token, so attention
# is 76% of AR per-token traffic for 15% of the weights. Leaving it BF16 costs 11.12 GiB that
# decode re-reads on every single token. Quantising just these projections saves 8.00 GiB and
# takes the checkpoint from 98.2 to 90.2 GiB.
#
# PROTECTED, and deliberately so - together they are 0.40 GiB, so excluding them costs nothing
# measurable and removes the only places where 4-bit error would COMPOUND rather than average:
#   f_a/f_b/g_a/g_b   KDA forget and output gates. They feed exp()/sigmoid inside the delta rule,
#                     so their error propagates along the sequence recurrence.
#   b_proj            the KDA beta term - it scales the delta-rule update itself, and it is
#                     17 MiB. Earlier notes grouped it with q/k/v/o as "safe"; it is not, it is
#                     inside the recurrence, and it is far too small to be worth the argument.
#   *_conv1d          short causal convolutions, also inside the recurrence.
#   indexer.*         DSA's token selector. It decides WHICH tokens are attended, so it is
#                     argmax-sensitive exactly as the MoE router is, and gets the same treatment.
#   A_log, dt_bias    per-head scalars, and norms - not matmul weights at all.
# MODE, set by kv `nvfp4_attn_mode`:
#   "none"  attention stays BF16. The shipped default.
#   "post"  o_proj everywhere + the MLA projections. Everything quantised here sits OUTSIDE the
#           KDA recurrence: o_proj is applied to the attention output, and the 11 DSA layers are
#           ordinary softmax attention with no recurrent state at all.
#   "all"   the above plus KDA q/k/v.
#
# MEASURED 2026-08-29: "all" costs -0.00673 top-1 agreement (McNemar z = 14.5) for 8.00 GiB. That
# is MORE quality than the pass-2 mask and the healing fix together recovered, so it is not the
# lossless lever it was framed as. The suspicion "all" raises is specific: q/k/v on the 34
# linear-attention layers feed the delta-rule state, so 4-bit error there propagates along the
# SEQUENCE - which is exactly the argument already used to protect f_a/f_b/g_a/g_b, b_proj and
# the conv1ds, and it was not applied to q/k/v. "post" is the arm that tests that.
ATTN_SETS = {
    "none": None,
    "post": r"\.self_attn\.(o_proj|q_a_proj|q_b_proj|kv_a_proj_with_mqa|kv_b_proj)\.weight$",
    # MEASURED: "post" cost 82% of the damage for 43% of the bytes, so o_proj + the MLA
    # projections are the EXPENSIVE half and KDA q/k/v the cheap one - the opposite of what the
    # recurrence argument predicted. "kda" isolates the cheap half: 4.60 GiB at an implied
    # -0.00122 top-1, six times more byte-efficient than "post".
    "kda":  r"\.self_attn\.(q_proj|k_proj|v_proj)\.weight$",
    "all":  r"\.self_attn\.(q_proj|k_proj|v_proj|o_proj|q_a_proj|q_b_proj"
            r"|kv_a_proj_with_mqa|kv_b_proj)\.weight$",
}
ATTN_MODE = str(kv_get("nvfp4_attn_mode", "none"))
if bool(kv_get("nvfp4_quantize_attention", False)) and ATTN_MODE == "none":
    ATTN_MODE = "all"          # honour the older boolean flag
if ATTN_MODE not in ATTN_SETS:
    raise ValueError(f"nvfp4_attn_mode must be one of {sorted(ATTN_SETS)}, got {ATTN_MODE!r}")
ATTN_RE = re.compile(ATTN_SETS[ATTN_MODE]) if ATTN_SETS[ATTN_MODE] else None
QUANT_ATTN = ATTN_RE is not None

# Untouched: quantising any of these saves almost nothing and risks real capability.
IGNORE = [
    "re:.*lm_head.*", "re:.*embed_tokens.*",
    "re:.*visual.*",            # dense ViT, 0.18% of mass, first-class capability
    # Attention is SPLIT, not blanket-excluded.
    #
    # The old rule was `re:.*self_attn.*` with the reason "KDA recurrence: error compounds along
    # the sequence". That reason is right for the GATES and wrong for q/k/v/o/b - the gates are
    # what sits inside the recurrence, and they are 0.20 GiB. Excluding all of attention cost
    # 8.01 GiB of BF16 that DECODE READS ON EVERY TOKEN, and measurement says it dominates:
    # 76% of AR per-token traffic for 15% of the weights, because only 8 of 144 experts are read
    # per token while the whole attention stack is. Quantising it takes the AR roofline on Thor
    # from 13.5 to 23.5 tok/s and the checkpoint from 98.2 to 90.1 GiB.
    #
    # Protected, for 0.37 GiB total:
    #   f_a/f_b/g_a/g_b   KDA forget/output gates - feed exp()/sigmoid inside the delta rule,
    #                     so error compounds along the sequence rather than averaging out
    #   *_conv1d          short causal convolutions, also inside the recurrence
    #   indexer.*         DSA's token selector. It decides WHICH tokens are attended, so it is
    #                     argmax-sensitive in exactly the way the MoE router is, and gets the
    #                     same treatment.
    #   *norm*            per-channel scales; already covered by the norm rule below
    # Behind a FLAG, default OFF, because the recipe change and the healing correction must not
    # land in the same artifact. Pass 2 already shipped a mask change and a healing change
    # together and the resulting null hid two real effects of opposite sign; rebuilding the
    # published v2 from corrected weights AND a new attention recipe would repeat that exactly.
    # Set kv `nvfp4_quantize_attention` to run the split rule as its own measured arm.
    *({"none": ["re:.*self_attn.*", "re:.*indexer.*"],
       "post": [r"re:.*self_attn\.(q_proj|k_proj|v_proj|f_a_proj|f_b_proj|g_a_proj|g_b_proj"
                r"|b_proj)\..*",
                r"re:.*self_attn\..*_conv1d.*", "re:.*indexer.*"],
       "kda":  [r"re:.*self_attn\.(o_proj|q_a_proj|q_b_proj|kv_a_proj_with_mqa|kv_b_proj"
                r"|f_a_proj|f_b_proj|g_a_proj|g_b_proj|b_proj)\..*",
                r"re:.*self_attn\..*_conv1d.*", "re:.*indexer.*"],
       "all":  [r"re:.*self_attn\.(f_a_proj|f_b_proj|g_a_proj|g_b_proj|b_proj)\..*",
                r"re:.*self_attn\..*_conv1d.*", "re:.*indexer.*"]}[ATTN_MODE]),
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

    n_q = n_bf16 = n_fp8 = 0
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
                if EXPERT_RE.search(name) or (QUANT_ATTN and ATTN_RE.search(name)):
                    w = NV.dequant_fp8_block(t, f.get_tensor(sname)) if sname in keys \
                        else t.to(torch.float32)
                    q = NV.quantize_nvfp4(w)
                    base = name[: -len(".weight")]
                    out_t[f"{base}.weight_packed"] = q["weight_packed"]
                    out_t[f"{base}.weight_scale"] = q["weight_scale"]
                    out_t[f"{base}.weight_global_scale"] = q["weight_global_scale"]
                    n_q += 1
                    del w, q
                elif KEEP_FP8 and sname in keys:
                    # Pass the FP8 payload and its block scale through untouched.
                    #
                    # Upcasting FP8 to BF16 adds no information - it stores already-FP8 values in
                    # twice the bytes. Measured: 53 tensors, 1.45 GiB -> 2.91 GiB, so the file is
                    # 1.5% larger for exactly zero quality difference. Keeping them FP8 is
                    # bit-identical and smaller.
                    out_t[name] = t
                    out_t[sname] = f.get_tensor(sname)
                    n_fp8 += 1
                else:
                    # Default. Dequantise block-FP8 to BF16 so the file carries ONE quantisation
                    # format plus plain BF16 rather than two schemes in one checkpoint.
                    #
                    # This costs 1.45 GiB and buys loader simplicity. It stays the default until
                    # a mixed nvfp4-pack-quantized + fp8-block checkpoint has actually been LOADED
                    # - and nothing has yet run this architecture on this box, so that claim
                    # cannot be tested here. Shipping an untestable format change into the one
                    # artifact we cannot validate would trade 1.5% of disk for the whole model.
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
            log(f"shard {si}/{len(shards)}  quantised={n_q} bf16={n_bf16} fp8={n_fp8}  "
                f"elapsed {el/60:.1f} min  eta {(el/si)*(len(shards)-si)/60:.0f} min  "
                f"free {free_gib():.0f} GiB", STAGE)

    # config: compressed-tensors NVFP4 on the experts, everything else ignored
    cfg = json.loads((src / "config.json").read_text())
    # A mixed-scheme checkpoint must DECLARE both schemes, or a loader silently reads the
    # passed-through FP8 tensors as unquantised and produces garbage rather than an error.
    extra_group = {}
    if KEEP_FP8:
        extra_group["group_1"] = {
            "targets": ["Linear"],
            "weights": {"num_bits": 8, "type": "float", "symmetric": True,
                        "strategy": "block", "block_structure": [128, 128], "dynamic": False},
            "input_activations": None, "output_activations": None,
        }
    cfg["quantization_config"] = {
        "quant_method": "compressed-tensors",
        "format": "nvfp4-pack-quantized",
        "quantization_status": "compressed",
        "ignore": IGNORE,
        "config_groups": {
            **extra_group,
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
           "tensors_nvfp4": n_q, "tensors_bf16": n_bf16, "tensors_fp8_passthrough": n_fp8,
           "keep_fp8_passthrough": KEEP_FP8,
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
