# 10 — GLM-5.3-Flash: verified architecture and parameter accounting

Source of truth: `zai-org/GLM-5.3-Flash` `config.json` + all 62 safetensors headers,
read on 2026-08-27 via HTTP range requests (no download). Raw tensor inventory:
`research/glm53_tensors.json` (76,108 tensors).

## Headline correction to the project directive

> **The released checkpoint is FP8, not BF16.** `zai-org/GLM-5.3-Flash` ships a
> `quantization_config` with `quant_method: fp8`, `fmt: e4m3`, `activation_scheme: dynamic`,
> `weight_block_size: [128, 128]`. Every routed expert is F8_E4M3 with F32 `weight_scale_inv`
> block scales. `[EST]`

There is no public BF16 master. `unsloth/GLM-5.3-Flash` (BF16, 321,322,735,872 BF16 params
≈ 642 GB) is a **dequantised upcast** of the FP8 release — byte-for-byte the same information
in twice the space. `unsloth/GLM-5.3-Flash-FP8` has a dtype profile identical to `zai-org`. `[EST]`

**Consequence:** never download the 642 GB BF16 repo. FP8 is the native precision floor and
the correct working format. Operator confirmed this direction 2026-08-27.

## Measured totals

| dtype | params | tensors |
|---|---:|---:|
| F8_E4M3 | 314,396,639,232 | 37,338 |
| BF16 | 6,926,096,640 | 1,141 |
| F32 (block scales) | 19,484,766 | 37,629 |
| **total** | **321,342,220,638** | **76,108** |

On-disk: **328.3 GB / 305.8 GiB** across 62 shards. License MIT, ungated. `[EST]`

## Parameter accounting by function

| category | params | share | dtypes |
|---|---:|---:|---|
| **routed experts** | **311,672,586,240** | **96.99%** | F8_E4M3 + F32 scales |
| attention | 6,199,640,639 | 1.93% | BF16 4.99B + F8_E4M3 1.21B |
| shared expert | 1,082,196,480 | 0.34% | F8_E4M3 |
| dense MLP (layers 0–2) + vision MLP | 755,223,552 | 0.24% | F8_E4M3 0.45B + BF16 0.30B |
| lm_head | 634,388,480 | 0.20% | BF16 |
| embed_tokens | 634,388,480 | 0.20% | BF16 |
| vision tower (total incl. its MLPs) | 563,627,008 | 0.18% | BF16 |
| routers | 50,737,248 | 0.016% | BF16 |
| MTP `eh_proj` | 33,554,432 | 0.010% | BF16 |
| **mHC (all `hc_*`, `mapping_proj`)** | **17,695,935** | **0.006%** | BF16 + F32 |
| norms | 393,216 | ~0 | BF16 |

**The single most important number in this project: routed experts are 96.99% of the model.**
Expert pruning is the *only* lever that matters; every other component is rounding error by
mass. Correspondingly, everything else is cheap to protect. `[EST]`

**mHC is 17.7M parameters across 135 tensors — 0.006% of the model.** It is small enough to
*fully fine-tune* during healing rather than LoRA it. See [70-healing.md](70-healing.md). `[EST]`

## Architecture (from config.json)

- `architectures: ["Glm5NextForConditionalGeneration"]`, `model_type: glm5_next`,
  `transformers_version: 5.16.0`
- Text: `hidden_size 4096`, `num_hidden_layers 45`, `vocab_size 154880`,
  `tie_word_embeddings false`, `max_position_embeddings 1048576`
- **MoE:** `n_routed_experts 288`, `n_shared_experts 1`, `num_experts_per_tok 8`,
  `moe_intermediate_size 2048`, `scoring_func sigmoid`, `topk_method noaux_tc`,
  `n_group 1`, `topk_group 1`, `routed_scaling_factor 2.5`, `norm_topk_prob true`
- `first_k_dense_replace 3` → layers 0–2 dense (`intermediate_size 12288`), layers 3–44 sparse.
  `mlp_layer_types` = 3 dense / 42 sparse.
- **MTP:** `num_nextn_predict_layers 1` → a 46th block at index **45** with its own full
  288-expert MoE (7.43B params, 2.31% of model). Confirmed present in shard headers.
- `mhc: true`, `hc_mult 4`, `hc_sinkhorn_iters 20`, `hc_eps 1e-6`

### Hybrid attention — directive correction

The directive states this is "**not** MLA+DSA". That is **wrong**. `[EST]`

`layer_types` is **34 × `linear_attention` + 11 × `deepseek_sparse_attention`**, in a strict
**3:1 pattern** — full-attention at layers 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43.

- The 11 full layers **are MLA + DSA**: `q_lora_rank 1536`, `kv_lora_rank 512`,
  `qk_head_dim 256`, `qk_nope_head_dim 256`, `qk_rope_head_dim 0` (`mla_use_nope: true`),
  `v_head_dim 256`, 64 heads — plus a DSA indexer (`index_n_heads 32`, `index_head_dim 128`,
  `index_topk 2048`) with **IndexPool** compression (`index_kpool 4`, `index_kpool_compress`,
  `index_kpool_always_select_tail`).
- The 34 linear layers are **KDA (Kimi Delta Attention)** — the config literally names them
  `linear_attn_config.kda_layers`. `num_heads 64`, `head_dim 128`,
  `short_conv_kernel_size 4`, `gate_lower_bound -5.0`. Tensor names (`A_log`, `dt_bias`,
  `f_a_proj`/`f_b_proj`, `g_a_proj`/`g_b_proj`, `b_proj`, `k_conv1d`) confirm a gated-delta
  recurrent form.

**This is a very good discovery, not a bad one.** GLM-5.3-Flash's attention stack is
*Kimi Linear's* — KDA interleaved with full attention at exactly 3:1 — which means the
architecture has a direct, published REAP precedent. See [40-hybrid-fragility.md](40-hybrid-fragility.md).

### Vision

`vision_config.model_type: glm5_next_vision` — `depth 24`, `hidden_size 1024`,
`intermediate_size 4096`, `num_heads 16`, `patch_size 14`, `image_size 448`,
`spatial_merge_size 2`, `temporal_patch_size 2`, `out_hidden_size 4096`,
`projection_intermediate_size 10240`, `swiglu_limit 10.0`.

Special tokens: image 154854, video 154855, image_start/end 154830/154831,
video_start/end 154832/154833.

**The vision tower is a dense 563.6M-param ViT (0.18%) with no MoE in it.** REAP cannot
touch it directly. The vision risk is therefore *not* "pruning the vision tower" — it is
"pruning the language-model experts that vision tokens route to."
See [50-multimodal.md](50-multimodal.md). `[EST]`

### Tensor naming

Prefix is `model.language_model.*` for the text stack and `model.visual.*` for the tower.
Experts are **per-expert individual tensors**, not fused 3D stacks:

```
model.language_model.layers.{L}.mlp.experts.{E}.{gate,up,down}_proj.weight            F8_E4M3
model.language_model.layers.{L}.mlp.experts.{E}.{gate,up,down}_proj.weight_scale_inv  F32
```

`gate_proj`/`up_proj` are `[2048, 4096]` → scale `[16, 32]`; `down_proj` is `[4096, 2048]`
→ scale `[32, 16]`. Consistent with 128×128 blocks. `[EST]`

**This layout is ideal for pruning:** dropping an expert means dropping 6 whole tensors and
renumbering. No dequantisation, no requantisation, no reconstruction — the surgery is
lossless byte movement in FP8 space. `[EST]`
