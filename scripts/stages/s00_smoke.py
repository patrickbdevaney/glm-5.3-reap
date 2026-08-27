"""Stage 0 — structural smoke test for glm5_next x llm-compressor.

The cheapest possible early failure. Validates, without downloading a single weight of the
real model, that:
  1. the published config matches the parameter accounting this project is planned around
  2. llm-compressor can derive a LinearExperts2D class for Glm5NextTextExperts
  3. a *tiny* real glm5_next model can be linearized, introspected, and actually REAP-pruned
  4. the processor emits image tokens (the assertion that protects vision -- risk R3)
  5. what MoE forward throughput this box achieves, which every downstream estimate depends on

Finding recorded here rather than discovered later: transformers' Glm5NextForConditionalGeneration
does NOT instantiate the MTP block (layer 45). See MTP_NOTE below.
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import ARTIFACTS, MODEL_ID, log, metric, kv_set, publish  # noqa: E402

STAGE = "s00_smoke"

# Ground truth from research/glm53_tensors.json (all 62 shard headers, read 2026-08-27)
EXPECT = {
    "total_params": 321_342_220_638,
    "routed_expert_params": 311_672_586_240,
    "n_routed_experts": 288,
    "num_experts_per_tok": 8,
    "num_hidden_layers": 45,
    "moe_layers": 42,          # layers 3..44; layer 45 is MTP and is NOT built by transformers
    "hidden_size": 4096,
    "moe_intermediate_size": 2048,
    "vocab_size": 154880,
    "image_token_id": 154854,
    "video_token_id": 154855,
}

MTP_NOTE = (
    "transformers Glm5NextForConditionalGeneration does not instantiate the MTP block at "
    "layer index 45 (7.45B params, its own 288-expert MoE). The llm-compressor path therefore "
    "cannot see or prune it. Decision: exclude MTP from this run, archive its original tensors "
    "verbatim so the downstream speculative-decoding project can re-derive it. Prevents R7 "
    "(inconsistently-pruned MTP poisoning later spec-decode work)."
)


def _tiny_config():
    """A structurally faithful but tiny glm5_next: same mechanisms, ~1/1000th the size."""
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(MODEL_ID)
    t = cfg.text_config
    t.num_hidden_layers = 4
    t.layer_types = ["linear_attention", "linear_attention", "linear_attention",
                     "deepseek_sparse_attention"]
    t.linear_attn_config["kda_layers"] = [0, 1, 2]
    t.linear_attn_config["full_attn_layers"] = [3]
    t.indexer_types = ["full"] * 4
    t.mlp_layer_types = ["dense", "sparse", "sparse", "sparse"]
    t.first_k_dense_replace = 1
    t.n_routed_experts = t.num_local_experts = 16
    t.num_experts_per_tok = 4
    t.hidden_size = 256
    t.intermediate_size = 512
    t.moe_intermediate_size = 128
    t.num_attention_heads = 4
    t.num_key_value_heads = 4
    t.q_lora_rank = 64
    t.kv_lora_rank = 32
    t.qk_head_dim = t.qk_nope_head_dim = t.v_head_dim = 32
    t.index_n_heads = 2
    t.index_head_dim = 16
    t.index_topk = 16
    t.vocab_size = 512
    t.pad_token_id = 0
    t.eos_token_id = [1]
    t.num_nextn_predict_layers = 0
    t.linear_attn_config["num_heads"] = 4
    t.linear_attn_config["head_dim"] = 32
    cfg.vision_config.depth = 2
    cfg.vision_config.hidden_size = 64
    cfg.vision_config.intermediate_size = 128
    cfg.vision_config.num_heads = 2
    cfg.vision_config.out_hidden_size = 256
    cfg.vision_config.projection_intermediate_size = 128
    # token ids must fit the shrunken vocab or nn.Embedding rejects padding_idx
    cfg.pad_token_id = 0
    cfg.image_token_id = 2
    cfg.video_token_id = 3
    cfg.image_start_token_id, cfg.image_end_token_id = 4, 5
    cfg.video_start_token_id, cfg.video_end_token_id = 6, 7
    return cfg


def check_config() -> dict:
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(MODEL_ID)
    t = cfg.text_config
    got = {
        "n_routed_experts": t.n_routed_experts,
        "num_local_experts": t.num_local_experts,
        "num_experts_per_tok": t.num_experts_per_tok,
        "num_hidden_layers": t.num_hidden_layers,
        "hidden_size": t.hidden_size,
        "moe_intermediate_size": t.moe_intermediate_size,
        "vocab_size": t.vocab_size,
        "n_group": t.n_group,
        "topk_group": t.topk_group,
        "image_token_id": cfg.image_token_id,
        "video_token_id": cfg.video_token_id,
        "linear_attention_layers": sum(1 for x in t.layer_types if x == "linear_attention"),
        "sparse_attention_layers": sum(1 for x in t.layer_types if x == "deepseek_sparse_attention"),
        "num_nextn_predict_layers": t.num_nextn_predict_layers,
        "mhc": getattr(t, "mhc", None),
    }
    mismatches = []
    for k in ("n_routed_experts", "num_experts_per_tok", "num_hidden_layers", "hidden_size",
              "moe_intermediate_size", "vocab_size", "image_token_id", "video_token_id"):
        if got[k] != EXPECT[k]:
            mismatches.append(f"{k}: config={got[k]} expected={EXPECT[k]}")
    if mismatches:
        raise AssertionError("config disagrees with recorded accounting: " + "; ".join(mismatches))
    log(f"config validated: 288 experts/top-8, 34 KDA + 11 MLA+DSA, mHC={got['mhc']}", STAGE)
    return got


def check_class_derivation() -> dict:
    """The uncertainty that R4 was really about: can llm-compressor build a LinearExperts2D
    for an experts class it has never seen? glm5_next is NOT in ARCH_TO_IMPORT_PATHS."""
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextExperts
    from llmcompressor.modeling.moe.linear_experts import LinearExperts2D
    registered = LinearExperts2D.get_registration(Glm5NextTextExperts) is not None
    import glm5_next_support
    glm5_next_support.register()
    cls = LinearExperts2D.get_linear_experts_cls(Glm5NextTextExperts)
    info = {"pre_registered": registered, "derived_cls": cls.__name__,
            "has_gate": bool(cls.has_gate), "is_transposed": bool(cls.is_transposed)}
    log(f"LinearExperts2D derived for Glm5NextTextExperts: {info}", STAGE)
    return info


def tiny_end_to_end() -> dict:
    """The real test: linearize + introspect + actually REAP-prune a tiny glm5_next."""
    import torch
    from transformers.models.glm5_next import Glm5NextForConditionalGeneration
    from llmcompressor.modeling.moe.linearize import linearize_moe, get_non_linearized_moes
    from llmcompressor.modifiers.pruning.reap.utils import get_moe_attrs, prune_moe_layer
    import glm5_next_support
    glm5_next_support.register()

    cfg = _tiny_config()
    torch.manual_seed(0)
    model = Glm5NextForConditionalGeneration._from_config(cfg)
    model = model.eval()
    n_before = sum(p.numel() for p in model.parameters())

    before = len(get_non_linearized_moes(model))
    linearize_moe(model)
    after = len(get_non_linearized_moes(model))
    # linearize builds nn.Linear experts in the default dtype; re-cast so the whole
    # model is uniformly bf16 (mixed dtypes raise in F.linear).
    model = model.to(torch.bfloat16)
    if after != 0:
        raise AssertionError(f"linearize_moe left {after} fused MoE modules")

    attrs = get_moe_attrs(model, [])
    if attrs.num_experts != cfg.text_config.n_routed_experts:
        raise AssertionError(f"get_moe_attrs num_experts={attrs.num_experts}")

    # Structurally prune half the experts in every MoE layer, keeping a fixed arbitrary set.
    keep = list(range(attrs.num_experts // 2))
    for name in attrs.moe_layer_names:
        prune_moe_layer(model, name, keep, attrs)
    n_after = sum(p.numel() for p in model.parameters())

    # The pruned model must still run a forward pass.
    ids = torch.randint(0, cfg.text_config.vocab_size, (1, 16))
    with torch.no_grad():
        out = model(input_ids=ids)
    logits_ok = tuple(out.logits.shape) == (1, 16, cfg.text_config.vocab_size)

    res = {
        "fused_moe_before": before, "fused_moe_after": after,
        "detected_experts": attrs.num_experts, "detected_top_k": attrs.top_k,
        "detected_moe_layers": len(attrs.moe_layer_names),
        "router_attr": attrs.router_attr, "experts_attr": attrs.experts_attr,
        "n_group": attrs.n_group, "group_size": attrs.group_size,
        "params_before": n_before, "params_after": n_after,
        "param_reduction": round(1 - n_after / n_before, 4),
        "forward_after_prune_ok": logits_ok,
    }
    log(f"tiny end-to-end REAP prune OK: {res}", STAGE)
    if not logits_ok:
        raise AssertionError("pruned tiny model failed forward pass")
    return res


def check_processor() -> dict:
    """Highest-value assertion in the pipeline: images must survive into token ids."""
    from transformers import AutoProcessor
    from PIL import Image
    import numpy as np
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    img = Image.fromarray(np.random.randint(0, 255, (448, 448, 3), dtype=np.uint8))
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe."}]}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    batch = proc(text=[text], images=[img], return_tensors="pt")
    ids = batch["input_ids"][0].tolist()
    n_img = sum(1 for i in ids if i == EXPECT["image_token_id"])
    res = {"processor": type(proc).__name__, "seq_len": len(ids), "image_tokens": n_img,
           "keys": sorted(batch.keys())}
    log(f"processor check: {res}", STAGE)
    if n_img == 0:
        raise AssertionError(
            f"processor produced ZERO image tokens (id {EXPECT['image_token_id']}). "
            "Text-only calibration deletes vision experts with certainty (R3). Blocking.")
    return res


def bench_moe_layer() -> dict:
    """One real-shape MoE layer with random weights, to get a genuine tok/s number for this box.
    Extrapolated x42 it bounds the saliency pass."""
    import torch
    from transformers import AutoConfig
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextMoE
    if not torch.cuda.is_available():
        log("no CUDA; skipping throughput benchmark", STAGE, "WARN")
        return {"skipped": "no cuda"}

    cfg = AutoConfig.from_pretrained(MODEL_ID)
    torch.manual_seed(0)
    moe = Glm5NextTextMoE(cfg.text_config).to("cuda", torch.bfloat16).eval()
    gb = sum(p.numel() * p.element_size() for p in moe.parameters()) / 2**30
    log(f"built one real-shape MoE layer: {gb:.1f} GiB bf16 (288 experts x 2048)", STAGE)

    out = {"layer_gib_bf16": round(gb, 2)}
    for ntok in (512, 2048):
        x = torch.randn(1, ntok, cfg.text_config.hidden_size, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            for _ in range(2):
                moe(x)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            reps = 3
            for _ in range(reps):
                moe(x)
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / reps
        per_layer = ntok / dt
        full = per_layer / EXPECT["moe_layers"]
        out[f"tok_s_1layer_n{ntok}"] = round(per_layer, 1)
        out[f"est_tok_s_full_n{ntok}"] = round(full, 1)
        metric(STAGE, f"moe_layer_tok_s_n{ntok}", per_layer)
        metric(STAGE, f"est_full_model_tok_s_n{ntok}", full)
        log(f"  n={ntok}: {per_layer:,.0f} tok/s/layer -> ~{full:,.0f} tok/s full model", STAGE)
    del moe
    torch.cuda.empty_cache()
    return out


def run() -> dict:
    res: dict = {"model": MODEL_ID}
    res["config"] = check_config()
    res["class_derivation"] = check_class_derivation()
    res["tiny_e2e"] = tiny_end_to_end()
    res["processor"] = check_processor()
    res["throughput"] = bench_moe_layer()
    res["mtp_note"] = MTP_NOTE

    # Every downstream wall-clock estimate is revised from this number.
    est = res["throughput"].get("est_tok_s_full_n2048")
    if est:
        toks = 200_000_000
        kv_set("measured_full_model_tok_s", est)
        hrs = toks / est / 3600
        res["saliency_pass_estimate_hours_at_200M_tok"] = round(hrs, 1)
        log(f"PROJECTION: {est:,.0f} tok/s -> 200M-token saliency pass ~= {hrs:.1f} h", STAGE)

    out = ARTIFACTS / "s00_smoke.json"
    out.write_text(json.dumps(res, indent=2, default=str))
    publish(out, "artifacts", "stage00/s00_smoke.json", stage=STAGE)
    return res
