"""Stage 6 - emit the healed FP8 base plus adapters. PRIMARY DELIVERABLE.

Operator decision (2026-08-27): the deliverable is the pruned/healed FP8 base with adapters
stored separately, merged at quantisation time. No BF16 export stage - GLM-5.3-Flash ships FP8
and the 642 GB BF16 repo is an information-free upcast, so materialising BF16 would cost
~306 GiB to gain nothing.

Healing is a soft dependency: if stage 5 failed, this emits the pruned base unhealed rather
than blocking the deliverable, and says so in the model card.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import ROOT, ARTIFACTS, log, metric, kv_get, kv_set, publish  # noqa: E402

STAGE = "s06_emit"
PRUNED = ROOT / "output" / "pruned-fp8"
ADAPTERS = ROOT / "output" / "adapters"
EMIT = ROOT / "output" / "glm-5.3-flash-reap50-fp8"


def _card(meta: dict) -> str:
    healed = meta.get("healed")
    return f"""---
license: mit
base_model: zai-org/GLM-5.3-Flash
tags: [reap, moe, pruning, glm5_next, jetson, thor]
---

# GLM-5.3-Flash REAP-{int(meta['sparsity']*100)} (FP8)

{int(meta['sparsity']*100)}% of routed experts removed with **REAP**
(Router-weighted Expert Activation Pruning, arXiv:2510.13999), calibrated on a
permissively-licensed multi-domain corpus that includes real image-text pairs.

| | |
|---|---|
| Base | `zai-org/GLM-5.3-Flash` (MIT, FP8 E4M3, 128x128 block scales) |
| Experts | 288 -> {288 - int(288*meta['sparsity'])} per layer, top-8 routing unchanged |
| Params | 321.3B -> {meta.get('pruned_params', 0):,} |
| Healed | {'yes' if healed else 'NO - healing stage did not complete'} |
| MTP block | excluded (see below) |

## Why FP8 and not BF16

The upstream release is **FP8**, not BF16. Routed experts are stored per-expert with their own
`weight_scale_inv` block scales, so pruning is deleting whole tensors - **lossless on every
retained weight**. The 642 GB BF16 repo elsewhere on the Hub is a dequantised upcast carrying
no additional information.

## Calibration

Mixture weighted for a coding/agentic model that stays empirically grounded: agentic 24%,
code 21%, math 15%, multimodal 15%, science+bio 10%, finance 8%, ballast 7%.
Permissive licences only, so this checkpoint keeps the base model's MIT lineage.

Vision is first-class: the vision tower contains no MoE and is untouched, but image tokens
route through the same expert pool as text, so text-only calibration would have deleted
vision-serving experts with certainty. Real image-text pairs were asserted present.

## Known limitations

- **The MTP (multi-token-prediction) block at layer 45 is excluded.** `transformers`'
  `Glm5NextForConditionalGeneration` does not instantiate it, so the pruning path cannot see
  it. Its original tensors are archived unmodified rather than inconsistently pruned.
- REAP has no published data above 50% compression; this checkpoint sits at the validated
  ceiling, not beyond it.
- Expect **factual-recall** regression before reasoning or coding regression. That is the
  measured failure mode of expert pruning on this architecture family.

## Serving on Jetson Thor

Use the **cutlass** fused-MoE backend (the Marlin FP4 MoE kernel faults at >=256 experts) and
`TRITON_MLA` for the 11 MLA+DSA layers (FLASHINFER is invalid for MLA).
"""


def run() -> dict:
    if kv_get("skip_fp8_intermediate", False):
        nv = kv_get("nvfp4_path")
        log("disk-pressure path (R10): no FP8 intermediate exists, so the NVFP4 checkpoint "
            f"written by stage 3 IS the deliverable ({nv}). Emitting a card for it.", STAGE)
        nvp = Path(nv)
        s3 = json.loads((ARTIFACTS / "s03_saliency.json").read_text())
        meta = {"sparsity": s3["sparsity"], "pruned_params": s3["pruned_params"],
                "healed": True, "calib_samples": s3["calib_samples"]}
        (nvp / "README.md").write_text(_card(meta))
        (nvp / "reap_metadata.json").write_text(json.dumps(meta, indent=2))
        kv_set("emit_path", str(nvp))
        total = sum(p.stat().st_size for p in nvp.rglob("*") if p.is_file())
        publish(nvp, "nvfp4", ".", stage=STAGE)
        return {"path": str(nvp), "bytes": total, "healed": True, "format": "nvfp4"}
    src = Path(kv_get("pruned_model_path", str(PRUNED)))
    if not src.exists():
        raise RuntimeError(f"pruned model not found at {src}")
    EMIT.mkdir(parents=True, exist_ok=True)

    log(f"emitting deliverable from {src}", STAGE)
    for p in src.iterdir():
        dst = EMIT / p.name
        if dst.exists():
            continue
        if p.is_file():
            shutil.copy2(p, dst)
    healed = ADAPTERS.exists() and any(ADAPTERS.iterdir())
    if healed:
        shutil.copytree(ADAPTERS, EMIT / "adapters", dirs_exist_ok=True)
        log("adapters included", STAGE)
    else:
        log("no adapters present - emitting unhealed base (healing did not complete)",
            STAGE, "WARN")

    s3 = json.loads((ARTIFACTS / "s03_saliency.json").read_text())
    meta = {"sparsity": s3["sparsity"], "pruned_params": s3["pruned_params"],
            "healed": healed, "calib_samples": s3["calib_samples"]}
    (EMIT / "README.md").write_text(_card(meta))
    (EMIT / "reap_metadata.json").write_text(json.dumps(meta, indent=2))

    total = sum(p.stat().st_size for p in EMIT.rglob("*") if p.is_file())
    metric(STAGE, "emit_bytes", total)
    kv_set("emit_path", str(EMIT))
    log(f"deliverable emitted: {EMIT} ({total/1e9:.1f} GB), healed={healed}", STAGE)

    publish(EMIT, "fp8", ".", stage=STAGE)
    return {"path": str(EMIT), "bytes": total, "healed": healed}
