"""Stage 1b - can we materialise the model at all, and how?

328.3 GB of FP8 weights against 117 GiB of unified memory. This probes load strategies in
increasing order of desperation and records which one works, so the expensive saliency stage
does not discover the answer four hours in. Runs against the real staged checkpoint but only
touches weights lazily where the strategy allows.
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import ROOT, ARTIFACTS, log, metric, kv_set, publish, free_gib  # noqa: E402

STAGE = "s01b_loadcheck"
SRC = ROOT / "source" / "GLM-5.3-Flash"
OFFLOAD = ROOT / "offload"


def _strategies():
    """(name, kwargs) in preference order. Earlier = cheaper/faster if it works."""
    OFFLOAD.mkdir(parents=True, exist_ok=True)
    return [
        ("cpu_offload_fp8", dict(device_map="cpu", dtype="auto")),
        ("auto_offload_fp8", dict(device_map="auto", dtype="auto",
                                  offload_folder=str(OFFLOAD), offload_state_dict=True)),
        ("meta_lazy", dict(device_map="meta", dtype="auto")),
    ]


def run() -> dict:
    import torch
    from transformers import AutoConfig
    from transformers.models.glm5_next import Glm5NextForConditionalGeneration

    cfg = AutoConfig.from_pretrained(SRC)
    qc = getattr(cfg, "quantization_config", None)
    log(f"source quantization_config: {qc}", STAGE)

    results = {}
    winner = None
    for name, kw in _strategies():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log(f"trying load strategy '{name}': {kw}", STAGE)
        t0 = time.time()
        try:
            model = Glm5NextForConditionalGeneration.from_pretrained(SRC, **kw)
            dt = time.time() - t0
            dtypes = {}
            for n_, p in model.named_parameters():
                dtypes[str(p.dtype)] = dtypes.get(str(p.dtype), 0) + p.numel()
            info = {"ok": True, "seconds": round(dt, 1), "param_dtypes": dtypes,
                    "n_params": sum(dtypes.values())}
            log(f"  '{name}' OK in {dt/60:.1f} min; dtypes={dtypes}", STAGE)
            results[name] = info
            winner = name
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            break
        except Exception as e:
            results[name] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:400]}"}
            log(f"  '{name}' FAILED: {type(e).__name__}: {str(e)[:250]}", STAGE, "WARN")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if winner is None:
        raise RuntimeError(
            "no load strategy succeeded. The saliency stage cannot proceed via the "
            "llm-compressor path; a custom layer-streaming loader is required. "
            f"attempts: {json.dumps(results)[:1500]}")

    kv_set("load_strategy", winner)
    kv_set("load_strategy_kwargs", {k: str(v) for k, v in dict(_strategies())[winner].items()})
    metric(STAGE, "load_seconds", results[winner]["seconds"])
    out = ARTIFACTS / "s01b_loadcheck.json"
    out.write_text(json.dumps({"winner": winner, "results": results,
                               "free_gib_after": round(free_gib(), 1)}, indent=2))
    publish(out, "artifacts", "stage01b/loadcheck.json", stage=STAGE)
    log(f"load strategy selected: {winner}", STAGE)
    return {"winner": winner, "results": results}
