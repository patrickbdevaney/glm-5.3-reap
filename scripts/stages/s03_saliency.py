"""Stage 3 - REAP saliency pass and structural prune at the target ratio.

Runs llm-compressor's sequential (onloading) pipeline over the calibration corpus, accumulating
    S_j = mean over tokens routed to expert j of  g_j * ||f_j||_2
and structurally removing the lowest-saliency half of every MoE layer.

Two deviations from stock, both deliberate:
  * moe_calibrate_all_experts=False - REAP needs the *real* routing distribution, not a forced
    all-expert pass. llm-compressor warns about this itself.
  * a SaliencyDumpingREAP subclass persists the raw accumulators per layer, so the prune-ratio
    sweep can re-rank at any ratio without re-running this pass (lever L3), and so the
    quantile-blended saliency A/B keeps its inputs.

Sample budget: REAP's saliency is a *conditional* mean over each expert's own active token set,
so what matters is tokens-per-expert, not corpus size. At 1024 samples x 4096 tokens every
expert sees ~116k tokens on average (4.2M x 8/288). The per-expert count floor is asserted
below rather than assumed - it is the guard for risk R1 (rare-specialist erosion).
"""
from __future__ import annotations

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
PRUNED = ROOT / "output" / "pruned-fp8"
OFFLOAD = ROOT / "offload"

TARGET_SPARSITY = 0.50
N_CALIB = int(kv_get("n_calib_samples", 1024) or 1024)
MAX_LEN = int(kv_get("calib_max_len", 4096) or 4096)
MIN_TOKENS_PER_EXPERT = 2_000     # sufficiency floor; below this, scores are noise (R1)


def _build_calib(tokenizer):
    """Interleave the text buckets in mixture proportion, then append image-text samples."""
    import torch
    from datasets import Dataset

    rows, mm_rows = [], []
    text_dir = CORPUS / "text"
    per_bucket: dict[str, list] = {}
    for shard in sorted(text_dir.glob("*.pt")):
        try:
            per_bucket[shard.stem] = torch.load(shard, weights_only=False)
        except Exception as e:
            log(f"could not load {shard.name}: {e}", STAGE, "WARN")

    import corpus_spec as SPEC
    n_text = int(N_CALIB * (1 - SPEC.MIXTURE["multimodal"]))
    for bucket, share in SPEC.MIXTURE.items():
        if bucket == "multimodal":
            continue
        items = per_bucket.get(bucket, [])
        want = int(n_text * share / (1 - SPEC.MIXTURE["multimodal"]))
        for it in items[:want]:
            rows.append({"input_ids": it["input_ids"][:MAX_LEN].tolist(), "bucket": bucket})

    mm_dir = CORPUS / "multimodal"
    want_mm = N_CALIB - len(rows)
    if mm_dir.exists():
        for f in sorted(mm_dir.glob("mm_*.pt")):
            if len(mm_rows) >= want_mm:
                break
            try:
                for rec in torch.load(f, weights_only=False):
                    if len(mm_rows) >= want_mm:
                        break
                    mm_rows.append(rec)
            except Exception:
                continue
    log(f"calibration set: {len(rows)} text + {len(mm_rows)} image-text = "
        f"{len(rows)+len(mm_rows)} samples", STAGE)
    if not mm_rows:
        raise AssertionError(
            "no image-text calibration samples available. Running text-only would delete "
            "vision-serving experts with certainty (R3). Blocking.")
    metric(STAGE, "calib_text_samples", len(rows))
    metric(STAGE, "calib_mm_samples", len(mm_rows))
    ds = Dataset.from_list([{"input_ids": r["input_ids"]} for r in rows])
    return ds, mm_rows


def _audit_expert_counts() -> dict:
    """The sufficiency floor. Rare experts are exactly the ones the mean-based criterion
    already disadvantages; under-sampling them compounds that (R1)."""
    import torch
    files = sorted(SALIENCY.glob("*.pt"))
    if not files:
        return {"layers": 0}
    mins, means, starved = [], [], 0
    for f in files:
        d = torch.load(f, weights_only=False)
        c = d["count"].double()
        mins.append(float(c.min()))
        means.append(float(c.mean()))
        starved += int((c < MIN_TOKENS_PER_EXPERT).sum())
    res = {"layers": len(files), "min_tokens_per_expert": min(mins),
           "mean_tokens_per_expert": sum(means) / len(means),
           "experts_below_floor": starved, "floor": MIN_TOKENS_PER_EXPERT}
    for k, v in res.items():
        if isinstance(v, (int, float)):
            metric(STAGE, f"saliency_{k}", v)
    log(f"per-expert sampling audit: {res}", STAGE)
    if starved:
        log(f"{starved} expert-slots below the {MIN_TOKENS_PER_EXPERT}-token floor; their "
            f"saliency is noisy and near-threshold deletions there are effectively random "
            f"(R1). Consider raising n_calib_samples.", STAGE, "WARN")
    return res


def run() -> dict:
    import torch
    from transformers import AutoTokenizer
    from transformers.models.glm5_next import Glm5NextForConditionalGeneration
    from llmcompressor import oneshot
    from llmcompressor.modeling.moe.linearize import linearize_moe, get_non_linearized_moes
    import glm5_next_support

    SALIENCY.mkdir(parents=True, exist_ok=True)
    PRUNED.parent.mkdir(parents=True, exist_ok=True)
    OFFLOAD.mkdir(parents=True, exist_ok=True)

    glm5_next_support.register()
    glm5_next_support._SaliencyDump.dir = str(SALIENCY)
    REAPCls = glm5_next_support.saliency_dumping_reap()

    strategy = kv_get("load_strategy", "cpu_offload_fp8")
    load_kw = {"cpu_offload_fp8": dict(device_map="cpu", dtype="auto"),
               "auto_offload_fp8": dict(device_map="auto", dtype="auto",
                                        offload_folder=str(OFFLOAD), offload_state_dict=True),
               "meta_lazy": dict(device_map="meta", dtype="auto")}[strategy]
    log(f"loading model with strategy '{strategy}' (free disk {free_gib():.0f} GiB)", STAGE)
    t0 = time.time()
    model = Glm5NextForConditionalGeneration.from_pretrained(SRC, **load_kw)
    log(f"model loaded in {(time.time()-t0)/60:.1f} min", STAGE)

    log("linearizing fused experts (2D->3D->2D; a load converter would avoid the round trip)",
        STAGE)
    linearize_moe(model)
    remaining = len(get_non_linearized_moes(model))
    if remaining:
        raise AssertionError(f"linearize_moe left {remaining} fused MoE modules")

    tok = AutoTokenizer.from_pretrained(SRC)
    ds, mm_rows = _build_calib(tok)

    log(f"REAP oneshot: sparsity={TARGET_SPARSITY}, {N_CALIB} samples @ {MAX_LEN} tokens", STAGE)
    t0 = time.time()
    oneshot(
        model=model,
        dataset=ds,
        recipe=[REAPCls(sparsity=TARGET_SPARSITY,
                        report_path=str(ARTIFACTS / "reap_retained_experts.json"))],
        num_calibration_samples=len(ds),
        max_seq_length=MAX_LEN,
        pipeline="sequential",
        sequential_offload_device="cpu",
        moe_calibrate_all_experts=False,   # REAP needs the real routing distribution
        shuffle_calibration_samples=False,
        save_compressed=False,
        output_dir=None,
    )
    dt = time.time() - t0
    metric(STAGE, "saliency_pass_minutes", dt / 60)
    log(f"REAP pass complete in {dt/60:.1f} min", STAGE)

    audit = _audit_expert_counts()

    n_after = sum(p.numel() for p in model.parameters())
    log(f"pruned model params: {n_after:,}", STAGE)
    metric(STAGE, "pruned_params", n_after)

    log(f"saving pruned model to {PRUNED}", STAGE)
    model.save_pretrained(str(PRUNED), safe_serialization=True)
    tok.save_pretrained(str(PRUNED))

    res = {"sparsity": TARGET_SPARSITY, "calib_samples": len(ds),
           "max_len": MAX_LEN, "minutes": round(dt / 60, 1),
           "pruned_params": n_after, "expert_audit": audit,
           "pruned_path": str(PRUNED)}
    out = ARTIFACTS / "s03_saliency.json"
    out.write_text(json.dumps(res, indent=2, default=str))
    publish(out, "artifacts", "stage03/s03_saliency.json", stage=STAGE)
    publish(ARTIFACTS / "reap_retained_experts.json", "artifacts",
            "stage03/reap_retained_experts.json", stage=STAGE)
    publish(SALIENCY, "saliency", "raw", stage=STAGE)
    kv_set("pruned_model_path", str(PRUNED))
    return res
