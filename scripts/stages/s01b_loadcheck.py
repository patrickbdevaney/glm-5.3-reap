"""Stage 1b - decide HOW the model can be placed, by arithmetic before experiment.

Learned the hard way on 2026-08-27. The first version of this stage simply tried
`device_map="cpu", dtype="auto"` on the theory that safetensors would stay mmap-backed and the
kernel would page it. It does not: transformers materialises into RAM. The probe reached
119 GB RSS with 5 GB available and had to be killed to protect the box - an OOM there would
have taken the orchestrator and the corpus build down with it.

So this stage now computes capacity first and never attempts a placement the arithmetic says
cannot fit. Guessing is fine; guessing with 122 GiB of RAM on the line is not.
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import ROOT, ARTIFACTS, log, metric, kv_set, publish, free_gib  # noqa: E402

STAGE = "s01b_load"
SRC = ROOT / "source" / "GLM-5.3-Flash"
OFFLOAD = ROOT / "offload"


def capacity_check() -> dict:
    model_bytes = sum(p.stat().st_size for p in SRC.glob("*.safetensors"))
    mem = {}
    with open("/proc/meminfo") as fh:
        for line in fh:
            if ":" in line:
                k, v = line.split(":", 1)
                mem[k] = int(v.split()[0]) * 1024
    ram_total = mem.get("MemTotal", 0)
    ram_avail = mem.get("MemAvailable", 0)
    disk_free = free_gib() * 2**30

    # Thor's memory is unified: RAM and VRAM are the same pool, so there is no second budget
    # to fall back on. Leave headroom for the orchestrator, the corpus build and the OS.
    fits_ram = model_bytes < ram_avail * 0.80
    fits_disk = model_bytes < disk_free * 0.90

    cap = {
        "model_gib": round(model_bytes / 2**30, 1),
        "ram_total_gib": round(ram_total / 2**30, 1),
        "ram_available_gib": round(ram_avail / 2**30, 1),
        "disk_free_gib": round(disk_free / 2**30, 1),
        "fits_in_ram": fits_ram,
        "fits_disk_offload": fits_disk,
        "verdict": "cpu_offload_fp8" if fits_ram else "auto_offload_fp8" if fits_disk else "stream",
    }
    for k in ("model_gib", "ram_available_gib", "disk_free_gib"):
        metric(STAGE, k, cap[k])
    return cap


def _try_load(name: str, kw: dict) -> dict:
    from transformers.models.glm5_next import Glm5NextForConditionalGeneration
    import torch
    log(f"attempting load strategy '{name}'", STAGE)
    t0 = time.time()
    try:
        model = Glm5NextForConditionalGeneration.from_pretrained(SRC, **kw)
        dt = time.time() - t0
        dtypes: dict[str, int] = {}
        for _, p in model.named_parameters():
            dtypes[str(p.dtype)] = dtypes.get(str(p.dtype), 0) + p.numel()
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log(f"  '{name}' OK in {dt/60:.1f} min", STAGE)
        return {"ok": True, "seconds": round(dt, 1), "param_dtypes": dtypes}
    except Exception as e:
        gc.collect()
        log(f"  '{name}' FAILED: {type(e).__name__}: {str(e)[:220]}", STAGE, "WARN")
        return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:400]}"}


def run() -> dict:
    OFFLOAD.mkdir(parents=True, exist_ok=True)
    cap = capacity_check()
    log(f"capacity: model {cap['model_gib']} GiB | RAM avail {cap['ram_available_gib']} GiB | "
        f"disk free {cap['disk_free_gib']} GiB -> verdict '{cap['verdict']}'", STAGE)

    results: dict = {"capacity": cap}
    winner = None

    if cap["verdict"] == "stream":
        log("neither RAM nor disk offload can hold 328 GB. Selecting the layer-streaming path "
            "(Path B) WITHOUT attempting a load that would OOM the box.", STAGE, "WARN")
        winner = "stream"
    else:
        strategies = [
            ("cpu_offload_fp8", dict(device_map="cpu", dtype="auto"), cap["fits_in_ram"]),
            ("auto_offload_fp8", dict(device_map="auto", dtype="auto",
                                      offload_folder=str(OFFLOAD),
                                      offload_state_dict=True), cap["fits_disk_offload"]),
        ]
        for name, kw, allowed in strategies:
            if not allowed:
                results[name] = {"ok": False, "error": "skipped: arithmetic says it cannot fit"}
                continue
            results[name] = _try_load(name, kw)
            if results[name]["ok"]:
                winner = name
                break
        if winner is None:
            log("all permitted strategies failed; falling back to layer streaming", STAGE, "WARN")
            winner = "stream"

    kv_set("load_strategy", winner)
    results["winner"] = winner
    out = ARTIFACTS / "s01b_loadcheck.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    publish(out, "artifacts", "stage01b/loadcheck.json", stage=STAGE)
    log(f"load strategy selected: {winner}", STAGE)
    return {"winner": winner, "capacity": cap}
