"""Stage 3 - REAP saliency by layer streaming, then structural prune.

s01b determined by arithmetic that the model cannot be placed: 305.8 GiB of weights against
116 GiB of (unified) RAM and 162 GiB of free disk. So this does not load a model. It builds one
decoder layer at a time, runs it, accumulates saliency, and frees it. Peak residency is one
layer (~14.5 GiB dequantised) plus the activation batch.

That is only tractable because Glm5NextTextModel.forward carries its entire inter-layer state in
a single tensor plus topk_indices - mHC's four residual streams live in the hc_mult axis and are
manipulated inside each decoder layer, so there is no cross-layer bookkeeping to replicate.

    S_j = mean over tokens routed to expert j of  g_j * ||f_j||_2

with f_j captured BEFORE the router gate scales it, which is what REAP is defined over.

Chunk by samples, sweeping all layers per chunk. Re-reading weights (3.4 GB/s) is far cheaper
than re-writing activations (487 MB/s) on this box.
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import (ROOT, ARTIFACTS, MODEL_ID, log, metric, kv_get, kv_set,  # noqa: E402
                    publish, free_gib)

STAGE = "s03_saliency"
SRC = ROOT / "source" / "GLM-5.3-Flash"
CORPUS = ROOT / "corpus" / "shards"
SALIENCY = ROOT / "artifacts" / "saliency"

TARGET_SPARSITY = float(kv_get("chosen_ratio", 0.50) or 0.50)
N_CALIB = int(kv_get("n_calib_samples", 1024) or 1024)
MAX_LEN = int(kv_get("calib_max_len", 4096) or 4096)
BATCH = int(kv_get("calib_batch", 8) or 8)
MIN_TOKENS_PER_EXPERT = 2_000
DEV = "cuda"
DT = None  # set at runtime


def _load_calib():
    """Interleave text buckets in mixture proportion, then append image-text samples."""
    import torch
    import corpus_spec as SPEC
    per_bucket: dict[str, list] = {}
    tdir = CORPUS / "text"
    for shard in sorted(tdir.glob("*.pt")):
        try:
            per_bucket[shard.stem] = torch.load(shard, weights_only=False)
        except Exception as e:
            log(f"could not load {shard.name}: {e}", STAGE, "WARN")
    n_text = int(N_CALIB * (1 - SPEC.MIXTURE["multimodal"]))
    text_rows = []
    denom = 1 - SPEC.MIXTURE["multimodal"]
    for bucket, share in SPEC.MIXTURE.items():
        if bucket == "multimodal":
            continue
        items = per_bucket.get(bucket, [])
        want = int(n_text * share / denom)
        for it in items[:want]:
            text_rows.append(it["input_ids"][:MAX_LEN])
    mm_rows = []
    mdir = CORPUS / "multimodal"
    if mdir.exists():
        want_mm = max(N_CALIB - len(text_rows), 0)
        for f in sorted(mdir.glob("mm_*.pt")):
            if len(mm_rows) >= want_mm:
                break
            try:
                for rec in torch.load(f, weights_only=False):
                    if len(mm_rows) >= want_mm:
                        break
                    mm_rows.append(rec)
            except Exception:
                continue
    log(f"calibration: {len(text_rows)} text + {len(mm_rows)} image-text", STAGE)
    metric(STAGE, "calib_text_samples", len(text_rows))
    metric(STAGE, "calib_mm_samples", len(mm_rows))
    if not mm_rows:
        raise AssertionError(
            "no image-text calibration samples. Text-only calibration gives vision-serving "
            "experts an empty active set, so REAP deletes them with certainty at any prune "
            "ratio (R3). Blocking rather than silently dropping vision.")
    return text_rows, mm_rows


def _build_layer(cfg, i, reader, dtype):
    """Materialise decoder layer i from the source shards.

    The checkpoint stores experts as per-expert 2D tensors; the module wants fused 3D
    parameters, so gate/up are concatenated per expert and stacked.
    """
    import torch
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextDecoderLayer
    from accelerate import init_empty_weights
    from stream_saliency import dequant_fp8_block

    prefix = f"model.language_model.layers.{i}."
    with init_empty_weights():
        layer = Glm5NextTextDecoderLayer(cfg, i)
    layer = layer.to_empty(device="cpu")

    names = [k for k in reader.map if k.startswith(prefix)]
    scales = {n for n in names if n.endswith("weight_scale_inv")}

    def fetch(name):
        t = reader.get(name)
        s = name[: -len("weight")] + "weight_scale_inv"
        if s in scales:
            return dequant_fp8_block(t, reader.get(s), dtype=dtype)
        return t.to(dtype) if t.is_floating_point() else t

    sd = {}
    experts: dict[int, dict[str, "torch.Tensor"]] = {}
    for n in names:
        if n in scales:
            continue
        rel = n[len(prefix):]
        if rel.startswith("mlp.experts."):
            parts = rel.split(".")
            e = int(parts[2])
            experts.setdefault(e, {})[parts[3]] = fetch(n)
        else:
            sd[rel] = fetch(n)

    if experts:
        idx = sorted(experts)
        gu = torch.stack([torch.cat([experts[e]["gate_proj"], experts[e]["up_proj"]], dim=0)
                          for e in idx])
        dn = torch.stack([experts[e]["down_proj"] for e in idx])
        sd["mlp.experts.gate_up_proj"] = gu
        sd["mlp.experts.down_proj"] = dn
        experts.clear()

    missing, unexpected = layer.load_state_dict(sd, strict=False, assign=True)
    real_missing = [m for m in missing if "rotary" not in m and "inv_freq" not in m]
    if real_missing:
        log(f"layer {i}: {len(real_missing)} params missing from checkpoint "
            f"(e.g. {real_missing[:2]})", STAGE, "WARN")
    del sd
    return layer.to(DEV).eval()


def run() -> dict:
    import torch
    from transformers import AutoConfig
    from transformers.models.glm5_next.modeling_glm5_next import (
        Glm5NextForConditionalGeneration, create_recurrent_attention_mask)
    from accelerate import init_empty_weights
    import stream_saliency as SS

    global DT
    DT = torch.bfloat16
    SALIENCY.mkdir(parents=True, exist_ok=True)
    strategy = kv_get("load_strategy", "stream")
    if strategy != "stream":
        log(f"load strategy is '{strategy}', but this stage implements the streaming path only",
            STAGE, "WARN")

    cfg = AutoConfig.from_pretrained(SRC)
    tcfg = cfg.text_config
    reader = SS.ShardReader(SRC)
    SS.patch_experts_for_saliency()
    SS.reset_accumulators()

    text_rows, mm_rows = _load_calib()

    # The small, permanently-resident parts: embeddings and the vision tower (~2.5 GB bf16).
    log("materialising embeddings + vision tower (small, stays resident)", STAGE)
    with init_empty_weights():
        shell = Glm5NextForConditionalGeneration._from_config(cfg)
    shell = shell.to_empty(device="cpu")
    small = {}
    for name in reader.map:
        if (name.startswith("model.visual.") or "embed_tokens" in name
                or name.startswith("model.language_model.norm")
                or "hc_head" in name):
            if name.endswith("weight_scale_inv"):
                continue
            small[name] = reader.get(name).to(DT) if reader.get(name).is_floating_point() \
                else reader.get(name)
    shell.load_state_dict(small, strict=False, assign=True)
    del small
    embed = shell.model.language_model.embed_tokens.to(DEV)
    visual = shell.model.visual.to(DEV, DT).eval()
    mm_model = shell.model
    gc.collect()

    def embeds_for(batch_ids, mm_batch):
        if mm_batch is None:
            ids = torch.nn.utils.rnn.pad_sequence(
                [b.long() for b in batch_ids], batch_first=True,
                padding_value=tcfg.pad_token_id).to(DEV)
            return ids, embed(ids)
        rec = mm_batch
        ids = rec["input_ids"].long().unsqueeze(0).to(DEV)
        ie = embed(ids)
        pv = rec["pixel_values"].to(DEV, DT)
        thw = rec["image_grid_thw"].to(DEV)
        feats = mm_model.get_image_features(pv, thw).pooler_output
        feats = torch.cat(feats, dim=0).to(ie.device, ie.dtype)
        mask, _ = mm_model.get_placeholder_mask(ids, inputs_embeds=ie, image_features=feats)
        return ids, ie.masked_scatter(mask, feats)

    batches = []
    for i in range(0, len(text_rows), BATCH):
        batches.append((text_rows[i:i + BATCH], None))
    for rec in mm_rows:
        batches.append((None, rec))
    log(f"{len(batches)} batches ({len(text_rows)} text @ batch {BATCH} + "
        f"{len(mm_rows)} image-text singly) x {tcfg.num_hidden_layers} layers", STAGE)

    t0 = time.time()
    for bi, (tb, mb) in enumerate(batches):
        try:
            with torch.no_grad():
                ids, ie = embeds_for(tb, mb)
                am = create_recurrent_attention_mask(config=tcfg, inputs_embeds=ie,
                                                     attention_mask=None, past_key_values=None)
                if am is None:
                    am = torch.ones(ie.shape[0], ie.shape[1], dtype=torch.bool, device=DEV)
                am = am.bool()
                masks = {"deepseek_sparse_attention": am, "linear_attention": am}
                pos = torch.arange(ie.shape[1], device=DEV).unsqueeze(0)
                hs = ie.unsqueeze(2).expand(-1, -1, tcfg.hc_mult, -1).contiguous()
                topk = None
                for li in range(tcfg.num_hidden_layers):
                    layer = _build_layer(tcfg, li, reader, DT)
                    SS.set_current_layer(f"model.language_model.layers.{li}.mlp")
                    hs, topk = layer(hs, attention_mask=masks[tcfg.layer_types[li]],
                                     position_ids=pos, position_embeddings=None,
                                     input_ids=ids, past_key_values=None,
                                     use_cache=False, prev_topk_indices=topk)
                    SS.set_current_layer(None)
                    del layer
                    gc.collect()
                    torch.cuda.empty_cache()
                del hs, ie, ids
                gc.collect()
                torch.cuda.empty_cache()
        except Exception as e:
            log(f"batch {bi} failed ({type(e).__name__}: {str(e)[:180]}); continuing",
                STAGE, "WARN")
            gc.collect()
            torch.cuda.empty_cache()
            continue
        if (bi + 1) % 5 == 0 or bi == 0:
            el = time.time() - t0
            log(f"batch {bi+1}/{len(batches)}  elapsed {el/60:.1f} min  "
                f"eta {(el/(bi+1))*(len(batches)-bi-1)/60:.0f} min", STAGE)
            SS.dump(SALIENCY)      # checkpoint: a kill costs at most the current batch

    n = SS.dump(SALIENCY)
    dt = time.time() - t0
    metric(STAGE, "saliency_pass_minutes", dt / 60)
    log(f"saliency accumulated over {n} layers in {dt/60:.1f} min", STAGE)

    audit = _audit()
    kv_set("saliency_ready", True)
    res = {"layers": n, "minutes": round(dt / 60, 1), "batches": len(batches),
           "expert_audit": audit, "sparsity": TARGET_SPARSITY,
           "calib_samples": len(text_rows) + len(mm_rows), "max_len": MAX_LEN,
           "path": "stream"}
    p = ARTIFACTS / "s03_saliency.json"
    p.write_text(json.dumps(res, indent=2, default=str))
    publish(p, "artifacts", "stage03/s03_saliency.json", stage=STAGE)
    publish(SALIENCY, "saliency", "raw", stage=STAGE)
    return res


def _audit() -> dict:
    import torch
    files = sorted(SALIENCY.glob("*.pt"))
    if not files:
        raise RuntimeError("no saliency accumulated")
    mins, means, starved, tot = [], [], 0, 0
    for f in files:
        d = torch.load(f, weights_only=False)
        c = d["count"].double()
        mins.append(float(c.min()))
        means.append(float(c.mean()))
        starved += int((c < MIN_TOKENS_PER_EXPERT).sum())
        tot += c.numel()
    res = {"layers": len(files), "min_tokens_per_expert": min(mins),
           "mean_tokens_per_expert": round(sum(means) / len(means), 1),
           "experts_below_floor": starved, "expert_slots": tot,
           "floor": MIN_TOKENS_PER_EXPERT}
    for k, v in res.items():
        if isinstance(v, (int, float)):
            metric(STAGE, f"saliency_{k}", v)
    log(f"per-expert sampling audit: {res}", STAGE)
    if starved:
        log(f"{starved}/{tot} expert slots below the {MIN_TOKENS_PER_EXPERT}-token floor - "
            f"their saliency is noisy, so near-threshold deletions there are effectively "
            f"random (R1). Raise n_calib_samples to tighten.", STAGE, "WARN")
    return res
