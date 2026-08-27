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


def _reclaim_page_cache() -> None:
    """Drop page cache between layers.

    Each layer read faults in ~7 GB of mmap'd shard; across 45 layers that is 306 GB of page
    cache that is never re-read. Tegra under-reports it in MemAvailable and will not reclaim it
    fast enough to satisfy a driver allocation, which is how this stage died six times with no
    oom-kill line in dmesg. Cheap and safe: these are clean file-backed pages.
    """
    import subprocess
    try:
        subprocess.run(["sync"], timeout=30, check=False)
        subprocess.run(["sudo", "-n", "sh", "-c", "echo 1 > /proc/sys/vm/drop_caches"],
                       timeout=30, check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass                 # the memguard service is the backstop
SRC = ROOT / "source" / "GLM-5.3-Flash"
CORPUS = ROOT / "corpus" / "shards"
SALIENCY = ROOT / "artifacts" / "saliency"

TARGET_SPARSITY = float(kv_get("chosen_ratio", 0.50) or 0.50)
# Sized so that ALL activations fit in RAM at once, which lets each layer be loaded exactly
# once instead of once per batch - the difference between ~6 minutes and ~3 hours of weight
# re-reads. 512 x 2048 x hc_mult(4) x 4096 x 2B ~= 34 GB.
# REAP's saliency is a conditional mean, so tokens-per-expert governs, not corpus size:
# 512 x 2048 x 8/288 ~= 29k tokens per expert, far above the 2k sufficiency floor.
N_CALIB = int(kv_get("n_calib_samples", 256) or 256)
MAX_LEN = int(kv_get("calib_max_len", 2048) or 2048)
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


# transformers applies a checkpoint conversion mapping at load time; building layers by hand
# means applying it by hand. Source: transformers.conversion_mapping.get_checkpoint_conversion_mapping("glm5_next").
# Without these, mHC (attn_hc/ffn_hc) and the KDA forget gate silently fail to load - which
# would leave randomly-initialised weights in the two components most sensitive to pruning.
_RENAMES = [
    ("self_attn.f_a_proj.", "self_attn.forget_gate.f_a_proj."),
    ("self_attn.f_b_proj.", "self_attn.forget_gate.f_b_proj."),
    ("self_attn.dt_bias", "self_attn.forget_gate.dt_bias"),
    ("self_attn.A_log", "self_attn.forget_gate.A_log"),
    ("hc_attn_fn", "attn_hc.fn"),
    ("hc_attn_base", "attn_hc.base"),
    ("hc_attn_scale", "attn_hc.scale"),
    ("hc_ffn_fn", "ffn_hc.fn"),
    ("hc_ffn_base", "ffn_hc.base"),
    ("hc_ffn_scale", "ffn_hc.scale"),
]


def _rename(rel: str) -> str:
    for a, b in _RENAMES:
        if a in rel:
            return rel.replace(a, b)
    return rel


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
            return dequant_fp8_block(t, reader.get(s), dtype=dtype)   # already a new tensor
        # copy=True is required: .to(dtype) on a tensor already in that dtype returns SELF,
        # a view into the shard mmap. The mapping is released after each layer so its page
        # cache can be reclaimed, and a live view would then be a use-after-unmap.
        return t.to(dtype, copy=True) if t.is_floating_point() else t.clone()

    sd = {}
    experts: dict[int, dict[str, "torch.Tensor"]] = {}
    convs: dict[str, "torch.Tensor"] = {}
    for n in names:
        if n in scales:
            continue
        rel = n[len(prefix):]
        if rel.startswith("mlp.experts."):
            parts = rel.split(".")
            e = int(parts[2])
            experts.setdefault(e, {})[parts[3]] = fetch(n)
        elif rel.startswith("self_attn.") and "_conv1d." in rel:
            convs[rel.split(".")[1]] = fetch(n)      # q_conv1d / k_conv1d / v_conv1d
        else:
            sd[_rename(rel)] = fetch(n)

    if convs:
        order = [convs[k] for k in ("q_conv1d", "k_conv1d", "v_conv1d") if k in convs]
        if order:
            sd["self_attn.conv1d.weight"] = torch.cat(order, dim=0)

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
    #
    # CRITICAL: build the shell on meta and materialise ONLY these two submodules. Calling
    # shell.to_empty(device="cpu") on the whole model allocates all 321B parameters (~642 GB at
    # bf16) and takes the box from 116 GiB available to 1.5 GiB in seconds - that is what killed
    # this stage six times. Everything else stays on meta and costs nothing; the decoder layers
    # are streamed in one at a time later.
    log("materialising embeddings + vision tower ONLY (rest stays on meta)", STAGE)
    with init_empty_weights():
        shell = Glm5NextForConditionalGeneration._from_config(cfg)

    def _materialise(mod, prefix):
        mod.to_empty(device="cpu")
        sd = {}
        for name in reader.map:
            if not name.startswith(prefix) or name.endswith("weight_scale_inv"):
                continue
            t = reader.get(name)
            sd[name[len(prefix):]] = (t.to(DT, copy=True) if t.is_floating_point()
                                      else t.clone())
        missing, _ = mod.load_state_dict(sd, strict=False, assign=True)
        real = [m for m in missing if "inv_freq" not in m and "rotary" not in m]
        if real:
            log(f"{prefix}: {len(real)} params missing (e.g. {real[:2]})", STAGE, "WARN")
        del sd
        return mod

    _materialise(shell.model.visual, "model.visual.")
    _materialise(shell.model.language_model.embed_tokens, "model.language_model.embed_tokens.")
    embed = shell.model.language_model.embed_tokens.to(DEV)
    visual = shell.model.visual.to(DEV, DT).eval()
    mm_model = shell.model
    gc.collect()
    resident = (sum(p.numel() * p.element_size() for p in visual.parameters())
                + sum(p.numel() * p.element_size() for p in embed.parameters())) / 2**30
    log(f"resident small parts: {resident:.2f} GiB", STAGE)

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

    # ---- build every batch's layer-0 input once, and keep it in RAM -------------------
    log("preparing activations for all batches (kept in RAM so each layer loads once)", STAGE)
    states = []
    with torch.no_grad():
        for i in range(0, len(text_rows), BATCH):
            ids, ie = embeds_for(text_rows[i:i + BATCH], None)
            states.append({"ids": ids.cpu(), "hs": ie.unsqueeze(2)
                           .expand(-1, -1, tcfg.hc_mult, -1).contiguous().cpu(), "topk": None})
            del ie
        for rec in mm_rows:
            try:
                ids, ie = embeds_for(None, rec)
                states.append({"ids": ids.cpu(), "hs": ie.unsqueeze(2)
                               .expand(-1, -1, tcfg.hc_mult, -1).contiguous().cpu(),
                               "topk": None})
                del ie
            except Exception as e:
                log(f"image-text sample skipped ({type(e).__name__}: {str(e)[:120]})",
                    STAGE, "WARN")
    del visual, mm_model, shell
    gc.collect()
    torch.cuda.empty_cache()

    _reclaim_page_cache()
    act_gib = sum(st["hs"].numel() * st["hs"].element_size() for st in states) / 2**30
    log(f"{len(states)} batches prepared, {act_gib:.1f} GiB of activations on HOST "
        f"(reclaimable; device memory here is driver-pinned and is not)", STAGE)
    metric(STAGE, "activation_gib", act_gib)
    if not states:
        raise RuntimeError("no calibration batches could be prepared")

    # ---- sweep layers: each layer is built ONCE and every batch passes through it --------
    t0 = time.time()
    for li in range(tcfg.num_hidden_layers):
        layer = _build_layer(tcfg, li, reader, DT)
        SS.set_current_layer(f"model.language_model.layers.{li}.mlp")
        ltype = tcfg.layer_types[li]
        for st in states:
            try:
                with torch.no_grad():
                    hs = st["hs"].to(DEV, non_blocking=False)
                    ids = st["ids"].to(DEV)
                    am = torch.ones(ids.shape[0], ids.shape[1], dtype=torch.bool, device=DEV)
                    pos = torch.arange(ids.shape[1], device=DEV).unsqueeze(0)
                    topk = st["topk"].to(DEV) if st["topk"] is not None else None
                    out, topk = layer(hs, attention_mask=am, position_ids=pos,
                                      position_embeddings=None, input_ids=ids,
                                      past_key_values=None, use_cache=False,
                                      prev_topk_indices=topk)
                    # Back to host immediately. Device memory here is driver-pinned: the
                    # kernel cannot see, swap or reclaim it, so anything left resident is
                    # permanently unavailable until the process exits.
                    st["hs"] = out.cpu()
                    st["topk"] = topk.cpu() if topk is not None else None
                    del hs, out, ids, am, pos, topk
            except Exception as e:
                log(f"layer {li} batch failed ({type(e).__name__}: {str(e)[:160]})",
                    STAGE, "WARN")
        SS.set_current_layer(None)
        del layer
        reader.release()          # tear down shard mmaps so their pages become reclaimable
        gc.collect()
        torch.cuda.empty_cache()
        _reclaim_page_cache()
        el = time.time() - t0
        avail = 0.0
        try:
            with open("/proc/meminfo") as fh:
                for line in fh:
                    if line.startswith("MemAvailable"):
                        avail = int(line.split()[1]) / 1048576
                        break
        except OSError:
            pass
        log(f"layer {li+1}/{tcfg.num_hidden_layers} ({ltype})  elapsed {el/60:.1f} min  "
            f"eta {(el/(li+1))*(tcfg.num_hidden_layers-li-1)/60:.0f} min  "
            f"avail {avail:.0f} GiB", STAGE)
        SS.dump(SALIENCY)      # checkpoint per layer: a kill costs at most one layer

    n = SS.dump(SALIENCY)
    dt = time.time() - t0
    metric(STAGE, "saliency_pass_minutes", dt / 60)
    log(f"saliency accumulated over {n} layers in {dt/60:.1f} min", STAGE)

    audit = _audit()
    kv_set("saliency_ready", True)
    res = {"layers": n, "minutes": round(dt / 60, 1), "batches": len(states),
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
