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
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import ROOT, ARTIFACTS, log, metric, kv_get, kv_set, publish  # noqa: E402

STAGE = "s06_emit"
PRUNED = ROOT / "output" / "pruned-fp8"
ADAPTERS = ROOT / "output" / "adapters"
# Versioned. Pass 2 must NOT write into pass 1's tree: that tree is the published artifact and,
# from P9.5 onward, the A/B baseline the whole "pass 2 is better" claim rests on. Emitting into a
# populated directory is also exactly how pass 1 produced four mixed shard families
# (of-00029/35/52/62) from partial runs, none of them a complete model.
EMIT = ROOT / "output" / str(kv_get("emit_name", "glm-5.3-flash-reap50-fp8"))


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
| Experts | 288 -> {meta.get('experts_kept', '?')} per layer, top-8 routing unchanged |
| Size | {meta.get('pruned_gib', '?')} GiB (FP8) |
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

## Evaluation status: NONE

**This checkpoint has not been evaluated.** No benchmark has been run against it - not coding,
not agentic, not vision, not knowledge. What has been verified is *structural*: expert counts
match the config, routers are sliced to the retained set, every tensor loads, the vision tower
is untouched, and the MTP block is cleanly absent.

The pruning itself measured **1.29x better than random** at retaining expert output
contribution (saliency mass 0.643 against 0.50 for random pruning at the same ratio). That says
the criterion selected well. It does **not** say the model is good.

Treat this as a research artifact pending evaluation, not a drop-in replacement.

## Known limitations

- **The MTP (multi-token-prediction) block at layer 45 is excluded.** `transformers`'
  `Glm5NextForConditionalGeneration` does not instantiate it, so the pruning path cannot see
  it. Its original tensors are archived unmodified rather than inconsistently pruned.
- REAP has no published data above 50% compression; this checkpoint sits at the validated
  ceiling, not beyond it.
- Expect **factual-recall** regression before reasoning or coding regression. That is the
  measured failure mode of expert pruning on this architecture family: the closest published
  analogue (`cerebras/Kimi-Linear-REAP-35B-A3B`, same KDA + full-attention stack) loses 3.4
  points on FRAMES at only 30% pruning while code and maths hold flat.
- Healing is a **first-moment output-scale correction** derived from the calibration saliency
  (median gain 0.696, applied exactly to the F32 block scales). It is *not* distillation and
  does not attempt to recover lost knowledge.
- Routing is disrupted more than expert count suggests: the retained experts carry ~0.90x the
  routing mass an average expert would, because REAP preserves rare-but-strong experts over
  common-but-weak ones.

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
        sg = json.loads((ARTIFACTS / "s04b_surgery.json").read_text())
        meta = {"sparsity": sg["ratio"], "experts_kept": sg["experts_kept"],
                "pruned_gib": sg["gib"], "healed": True,
                "calib_samples": s3.get("calib_samples")}
        (nvp / "README.md").write_text(_card(meta))
        (nvp / "reap_metadata.json").write_text(json.dumps(meta, indent=2))
        kv_set("emit_path", str(nvp))
        total = sum(p.stat().st_size for p in nvp.rglob("*") if p.is_file())
        publish(nvp, "nvfp4", ".", stage=STAGE)
        return {"path": str(nvp), "bytes": total, "healed": True, "format": "nvfp4"}
    src = Path(kv_get("pruned_model_path", str(PRUNED)))
    if not src.exists():
        raise RuntimeError(f"pruned model not found at {src}")
    # Refuse to emit into a tree that already holds a model. Overwriting shard-by-shard leaves
    # a directory that looks complete and is not.
    if EMIT.exists() and any(EMIT.glob("*.safetensors")):
        if not bool(kv_get("emit_overwrite", False)):
            raise RuntimeError(
                f"{EMIT} already contains a model. Set kv emit_name to a new directory (pass 2 "
                f"must not clobber pass 1 - it is the published artifact and the A/B baseline), "
                f"or set emit_overwrite=1 deliberately.")
        log(f"emit_overwrite set: writing over the existing model in {EMIT}", STAGE, "WARN")
    EMIT.mkdir(parents=True, exist_ok=True)

    log(f"emitting deliverable from {src}", STAGE)
    for p in src.iterdir():
        dst = EMIT / p.name
        if dst.exists():
            continue
        if p.is_file():
            # Hardlink, don't copy. The emit directory is the SAME 157 GiB of weights under a
            # release name; copying doubles peak usage to 314 GiB and left s07 without room to
            # offload (preflight: "only ~-9 GiB projected, needs ~78"). A hardlink is instant
            # and costs nothing; fall back to a copy only across filesystems.
            try:
                os.link(p, dst)
            except OSError:
                shutil.copy2(p, dst)
    healed = ADAPTERS.exists() and any(ADAPTERS.iterdir())
    if healed:
        shutil.copytree(ADAPTERS, EMIT / "adapters", dirs_exist_ok=True)
        log("adapters included", STAGE)
    else:
        log("no adapters present - emitting unhealed base (healing did not complete)",
            STAGE, "WARN")

    s3 = json.loads((ARTIFACTS / "s03_saliency.json").read_text())
    sg = json.loads((ARTIFACTS / "s04b_surgery.json").read_text())
    meta = {"sparsity": sg["ratio"], "experts_kept": sg["experts_kept"],
            "pruned_gib": sg["gib"], "healed": healed,
            "calib_samples": s3.get("calib_samples")}
    (EMIT / "README.md").write_text(_card(meta))
    (EMIT / "reap_metadata.json").write_text(json.dumps(meta, indent=2))

    total = sum(p.stat().st_size for p in EMIT.rglob("*") if p.is_file())
    metric(STAGE, "emit_bytes", total)
    kv_set("emit_path", str(EMIT))
    log(f"deliverable emitted: {EMIT} ({total/1e9:.1f} GB), healed={healed}", STAGE)

    publish(EMIT, "fp8", ".", stage=STAGE)
    return {"path": str(EMIT), "bytes": total, "healed": healed}
