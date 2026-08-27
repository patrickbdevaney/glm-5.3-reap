"""Stage 1 - stage the FP8 source checkpoint locally.

328.3 GB / 305.8 GiB across 62 shards. ~52 min at the measured 105 MB/s link rate.

Local staging (rather than streaming from HF per pass) is deliberate: chunked calibration
makes 8-24 passes over the weights, and local NVMe read (3.4 GB/s) is ~32x the link. See
wiki/95-walltime.md L8.

snapshot_download resumes, so a killed run costs only the partial file.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import MODEL_ID, ROOT, log, metric, hf_token, free_gib  # noqa: E402

STAGE = "s01_source"
DEST = ROOT / "source" / "GLM-5.3-Flash"
EXPECTED_SHARDS = 62
EXPECTED_BYTES = 328_337_455_672


def _local_bytes() -> int:
    if not DEST.exists():
        return 0
    return sum(p.stat().st_size for p in DEST.rglob("*.safetensors"))


def run() -> dict:
    DEST.mkdir(parents=True, exist_ok=True)
    have = _local_bytes()
    if have >= EXPECTED_BYTES:
        log(f"source already staged ({have/1e9:.1f} GB); skipping download", STAGE)
    else:
        need_gib = (EXPECTED_BYTES - have) / 2**30 + 20
        if free_gib() < need_gib:
            raise RuntimeError(f"need {need_gib:.0f} GiB free, have {free_gib():.0f}")
        log(f"staging {MODEL_ID}: have {have/1e9:.1f} GB of {EXPECTED_BYTES/1e9:.1f} GB", STAGE)
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=str(DEST),
            token=hf_token(),
            max_workers=8,
            resume_download=True,
        )

    shards = sorted(DEST.glob("*.safetensors"))
    total = sum(p.stat().st_size for p in shards)
    if len(shards) != EXPECTED_SHARDS:
        raise AssertionError(f"expected {EXPECTED_SHARDS} shards, found {len(shards)}")
    if total != EXPECTED_BYTES:
        raise AssertionError(f"byte mismatch: {total} != {EXPECTED_BYTES}")

    # config must round-trip and still agree with the accounting stage 0 validated
    cfg = json.loads((DEST / "config.json").read_text())
    t = cfg["text_config"]
    assert t["n_routed_experts"] == 288 and t["num_experts_per_tok"] == 8, "config drift"
    assert cfg.get("quantization_config", {}).get("quant_method") == "fp8", "expected FP8 source"

    # snapshot_download leaves resume metadata and staged blobs under .cache inside the
    # local_dir. Once the shards verify byte-exact, that is ~40 GB of pure waste, and disk is
    # the binding constraint for the prune stage (R10).
    cache_dir = DEST / ".cache"
    if cache_dir.exists():
        import shutil
        freed = sum(p.stat().st_size for p in cache_dir.rglob("*") if p.is_file())
        shutil.rmtree(cache_dir, ignore_errors=True)
        log(f"reclaimed {freed/1e9:.1f} GB of download staging", STAGE)

    metric(STAGE, "source_bytes", total)
    metric(STAGE, "source_shards", len(shards))
    log(f"source staged and verified: {len(shards)} shards, {total/1e9:.1f} GB", STAGE)
    return {"path": str(DEST), "shards": len(shards), "bytes": total,
            "quant_method": cfg["quantization_config"]["quant_method"]}
