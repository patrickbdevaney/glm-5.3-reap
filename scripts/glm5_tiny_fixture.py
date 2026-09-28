"""Build a tiny random GLM-5.3-Flash and compare transformers logits against llama.cpp.

Converting the real 165B checkpoint costs 165 GiB and ~2 hours, and the failure modes in this
port are silent: a wrong Sinkhorn order, a transposed mHC mixing matmul, or KDA/MLA layer types
off by one all load cleanly and produce fluent-looking garbage. So the gate is a 6-layer model
with the same STRUCTURE - both attention types, dense and MoE FFN, a shared expert, an MTP
block, mHC everywhere - where a full forward pass costs a second.

The real tokenizer is copied in rather than synthesised: vocab_size only affects embedding size,
and a hand-rolled tokenizer is one more thing that can be wrong for reasons unrelated to the port.
"""
from __future__ import annotations

import json, shutil, sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "artifacts" / "glm5_tiny"
SRC  = ROOT / "output" / "glm-5.3-flash-reap50-fp8-pass2"

D, HEADS, KDA_HEAD = 128, 4, 32
LAYERS, NEXTN      = 5, 1
FULL_ATTN          = [3]


def build_config() -> dict:
    n_total = LAYERS
    return {
        "architectures": ["Glm5NextForCausalLM"],
        "model_type": "glm5_next_text",
        "hidden_size": D,
        "intermediate_size": 256,
        "num_hidden_layers": n_total,
        "num_attention_heads": HEADS,
        "num_key_value_heads": HEADS,
        "head_dim": 0,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-5,
        "vocab_size": 154880,
        "pad_token_id": 154820,
        "tie_word_embeddings": False,
        "max_position_embeddings": 4096,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "initializer_range": 0.02,
        "dtype": "float32",
        # MoE
        "n_routed_experts": 4, "num_local_experts": 4, "num_experts_per_tok": 2,
        "n_shared_experts": 1, "moe_intermediate_size": 64,
        "first_k_dense_replace": 3, "n_group": 1,
        "routed_scaling_factor": 2.5, "norm_topk_prob": True,
        "scoring_func": "sigmoid", "moe_router_dtype": "float32",
        "router_aux_loss_coef": 0.001, "output_router_logits": False,
        "swiglu_limit": 10.0,
        # MLA - NoPE, as in the real model
        "q_lora_rank": 64, "kv_lora_rank": 32,
        "qk_nope_head_dim": 32, "qk_rope_head_dim": 0, "qk_head_dim": 32,
        "v_head_dim": 32, "mla_use_nope": True,
        # DSA indexer
        "index_head_dim": 32, "index_n_heads": 2, "index_topk": 64,
        "index_kpool": 4, "index_kpool_compress": True,
        "index_kpool_always_select_tail": True,
        "index_share_for_mtp_iteration": True, "indexer_rope_interleave": True,
        # KDA
        "linear_attn_config": {
            "num_heads": HEADS, "head_dim": KDA_HEAD, "short_conv_kernel_size": 4,
            "gate_lower_bound": -5.0,
            "kda_layers": [i for i in range(n_total) if i not in FULL_ATTN],
            "full_attn_layers": FULL_ATTN,
        },
        # mHC
        "mhc": True, "hc_mult": 4, "hc_sinkhorn_iters": 20, "hc_eps": 1e-6,
        "num_nextn_predict_layers": NEXTN,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = build_config()
    (OUT / "config.json").write_text(json.dumps(cfg, indent=2))
    for f in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        if (SRC / f).exists():
            shutil.copy(SRC / f, OUT / f)

    from transformers import AutoConfig
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextModel

    torch.manual_seed(0)
    c = Glm5NextTextConfig(**{k: v for k, v in cfg.items() if k != "architectures"})
    # mlp_layer_types is derived by the config; make sure the MTP layer exists in the stack.
    model = Glm5NextTextModel(c)
    model = model.to(torch.float32).eval()
    # small random weights - default init is fine, but scale down so activations stay sane
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() > 1:
                p.mul_(0.5)

    lm_head = torch.nn.Linear(D, cfg["vocab_size"], bias=False)
    with torch.no_grad():
        lm_head.weight.mul_(0.5)

    # transformers 5.16 loads the RELEASE layout through conversion_mapping.py, so the fixture
    # must be written in that layout - it is the format the GGUF converter reads, and writing
    # the in-memory names instead would test nothing.
    mem = {f"model.{k}": v for k, v in model.state_dict().items()}
    sd: dict[str, torch.Tensor] = {}
    d_inner = HEADS * KDA_HEAD
    n_exp   = cfg["n_routed_experts"]
    ff      = cfg["moe_intermediate_size"]
    for k, v in mem.items():
        if ".self_attn.conv1d.weight" in k:                 # Concatenate(dim=0) of q,k,v
            q, kk, vv = torch.split(v, d_inner, dim=0)
            base = k.replace(".conv1d.weight", "")
            sd[f"{base}.q_conv1d.weight"] = q.contiguous()
            sd[f"{base}.k_conv1d.weight"] = kk.contiguous()
            sd[f"{base}.v_conv1d.weight"] = vv.contiguous()
        elif k.endswith("mlp.experts.gate_up_proj"):        # [E, 2*ff, D], gate then up
            base = k.rsplit(".experts.", 1)[0] + ".experts"
            for e in range(n_exp):
                g, u = torch.split(v[e], ff, dim=0)
                sd[f"{base}.{e}.gate_proj.weight"] = g.contiguous()
                sd[f"{base}.{e}.up_proj.weight"]   = u.contiguous()
        elif k.endswith("mlp.experts.down_proj"):           # [E, D, ff]
            base = k.rsplit(".experts.", 1)[0] + ".experts"
            for e in range(n_exp):
                sd[f"{base}.{e}.down_proj.weight"] = v[e].contiguous()
        else:
            k = (k.replace(".self_attn.forget_gate.", ".self_attn.")
                  .replace(".attn_hc.fn", ".hc_attn_fn").replace(".attn_hc.base", ".hc_attn_base")
                  .replace(".attn_hc.scale", ".hc_attn_scale")
                  .replace(".ffn_hc.fn", ".hc_ffn_fn").replace(".ffn_hc.base", ".hc_ffn_base")
                  .replace(".ffn_hc.scale", ".hc_ffn_scale"))
            sd[k] = v.contiguous()
    sd["lm_head.weight"] = lm_head.weight.data.clone()

    # The MTP block is a real layer in the release checkpoint and llama.cpp must load and then
    # ignore it. Synthesise it from an MLA+MoE layer so the loader path is actually exercised;
    # transformers drops it (_keys_to_ignore_on_load_unexpected matches layers.45), and here the
    # index is LAYERS, so it is dropped by shape of the config rather than by that literal.
    if NEXTN:
        src_l, dst_l = f"layers.{FULL_ATTN[0]}.", f"layers.{LAYERS}."
        for k in [k for k in sd if src_l in k]:
            # The real MTP block carries NO mHC tensors - it consumes the collapsed hidden
            # state, so there are no parallel streams left to mix. Copying them here made the
            # fixture MORE permissive than the real checkpoint and hid a loader bug that then
            # only surfaced after a 175 GB conversion.
            if "hc_attn" in k or "hc_ffn" in k:
                continue
            sd[k.replace(src_l, dst_l)] = sd[k].clone()
        for name, shape in (("eh_proj.weight", (D, 2*D)), ("enorm.weight", (D,)),
                            ("hnorm.weight", (D,)), ("shared_head.norm.weight", (D,))):
            sd[f"model.{dst_l}{name}"] = torch.randn(*shape) * 0.05

    from safetensors.torch import save_file
    save_file(sd, str(OUT / "model.safetensors"), metadata={"format": "pt"})
    print(f"wrote {OUT} ({sum(v.numel() for v in sd.values())/1e6:.1f}M params, {len(sd)} tensors)")

    ids = torch.arange(1, 17, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        h = model(input_ids=ids).last_hidden_state
        logits = lm_head(h)
    torch.save({"ids": ids, "logits": logits.float()}, OUT / "reference.pt")
    print(f"reference logits {tuple(logits.shape)}  "
          f"last-token top5 {logits[0, -1].topk(5).indices.tolist()}")


if __name__ == "__main__":
    main()
