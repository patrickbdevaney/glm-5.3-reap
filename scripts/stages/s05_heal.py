"""Stage 5 - first-moment recovery correction. NON-CRITICAL by design.

Honest scope statement, because this is much narrower than the healing originally planned.

Full layer-local distillation against the unpruned teacher is the right repair and is what
wiki/70-healing.md recommends. It is not tractable here: the pruned model is ~165B parameters
against a 117 GiB envelope, so even holding the student for a backward pass requires heavy
offload, and the teacher is another 328 GB. Gradient-based healing on this box is a multi-day
job with real failure modes, and it is explicitly allowed to be skipped.

What IS both principled and cheap is correcting the *first moment* of the MoE output, and it
needs no teacher and no forward pass at all - only the saliency accumulators already cached.

The argument:
  REAP keeps the HIGHEST-saliency experts. `norm_topk_prob=True` renormalises the gates over
  the surviving support, so lost gate mass is already compensated. What is NOT compensated is
  that the retained experts have systematically LARGER ||f_j|| than the deleted ones - so the
  pruned layer's expected output magnitude is biased HIGH, by exactly the ratio between the
  all-expert and retained-expert saliency means.

      E[g||f||] over all experts        sum_j c_j m_j / sum_j c_j
      ---------------------------  =  --------------------------------
      E[g||f||] over retained         sum_{j in keep} c_j m_j / sum_{keep} c_j

  Applying that ratio as a gain on each retained expert's `down_proj` restores the layer's
  expected output scale in weight space - persistent, no runtime change, no config edit.

This matters most for mHC (risk R5): its Sinkhorn-normalised connection matrices were fitted
against the ORIGINAL expert-output distribution and conserve feature means, so a first-order
scale shift in what feeds the residual streams is precisely the perturbation they are
sensitive to.

This is a first-moment correction, NOT distillation. It does not recover rare-knowledge
erosion (R1). It is labelled as such in the model card.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import ROOT, ARTIFACTS, log, metric, kv_get, kv_set, publish  # noqa: E402

STAGE = "s05_heal"
SALIENCY = ROOT / "artifacts" / "saliency"
PRUNED = ROOT / "output" / "pruned-fp8"
ADAPTERS = ROOT / "output" / "adapters"

GAIN_FLOOR, GAIN_CEIL = 0.5, 1.0   # a gain outside this is a bug, not a correction


def run() -> dict:
    import torch
    if kv_get("skip_fp8_intermediate", False):
        log("stage 3 took the disk-pressure path and applied the first-moment correction "
            "in-memory before quantising; nothing to rewrite here", STAGE)
        return {"skipped": "applied in-memory during s03 (R10 path)"}
    retained_p = ARTIFACTS / "reap_retained_experts.json"
    if not retained_p.exists():
        raise RuntimeError("reap_retained_experts.json missing; stage 3 must run first")
    retained = json.loads(retained_p.read_text())

    gains, skipped = {}, []
    for f in sorted(SALIENCY.glob("*.pt")):
        d = torch.load(f, weights_only=False)
        layer = d["layer"]
        keep = retained.get(layer)
        if not keep:
            skipped.append(layer)
            continue
        c = d["count"].double()
        s = d["sum_saliency"].double()
        m = torch.where(c > 0, s / c.clamp(min=1), torch.zeros_like(s))
        idx = torch.tensor(keep, dtype=torch.long)
        tot_c = c.sum()
        keep_c = c[idx].sum()
        if tot_c <= 0 or keep_c <= 0:
            skipped.append(layer)
            continue
        e_all = float((c * m).sum() / tot_c)
        e_keep = float((c[idx] * m[idx]).sum() / keep_c)
        if e_keep <= 0:
            skipped.append(layer)
            continue
        g = e_all / e_keep
        gains[layer] = g
        metric(STAGE, "first_moment_gain", g, tag=layer)

    if not gains:
        raise RuntimeError("no per-layer gains computed; saliency accumulators unusable")

    vals = list(gains.values())
    stats = {"layers": len(gains), "skipped": len(skipped),
             "gain_min": round(min(vals), 4), "gain_max": round(max(vals), 4),
             "gain_mean": round(sum(vals) / len(vals), 4)}
    log(f"first-moment gains: {stats}", STAGE)

    out_of_range = [k for k, v in gains.items() if not (GAIN_FLOOR <= v <= GAIN_CEIL)]
    if out_of_range:
        log(f"{len(out_of_range)} layers have a gain outside [{GAIN_FLOOR},{GAIN_CEIL}] "
            f"- refusing to apply, this indicates a saliency bug not a correction",
            STAGE, "ERROR")
        raise RuntimeError(f"gain out of range for {len(out_of_range)} layers "
                           f"(e.g. {out_of_range[:3]})")

    # Apply in weight space, directly on the saved shards, so the correction is persistent
    # and needs no runtime support.
    from safetensors.torch import load_file, save_file
    src = Path(kv_get("pruned_model_path", str(PRUNED)))
    applied, touched_files = 0, 0
    for shard in sorted(src.glob("*.safetensors")):
        tensors = load_file(str(shard))
        changed = False
        for name in list(tensors):
            if ".mlp.experts." not in name or not name.endswith("down_proj.weight"):
                continue
            layer_key = name.split(".mlp.experts.")[0] + ".mlp"
            g = gains.get(layer_key)
            if g is None:
                continue
            t = tensors[name]
            orig_dtype = t.dtype
            tensors[name] = (t.to(torch.float32) * g).to(orig_dtype)
            applied += 1
            changed = True
        if changed:
            save_file(tensors, str(shard), metadata={"format": "pt"})
            touched_files += 1
            log(f"applied gains in {shard.name}", STAGE)

    ADAPTERS.mkdir(parents=True, exist_ok=True)
    (ADAPTERS / "first_moment_gains.json").write_text(json.dumps(
        {"method": "first-moment MoE output-scale correction (no teacher, no forward pass)",
         "gains": gains, "stats": stats,
         "experts_scaled": applied, "shards_rewritten": touched_files,
         "not_done": "layer-local distillation and LoRA SFT - not tractable for 165B on a "
                     "117 GiB box; see wiki/70-healing.md"}, indent=2))
    kv_set("healed", True)
    res = {**stats, "experts_scaled": applied, "shards_rewritten": touched_files}
    p = ARTIFACTS / "s05_heal.json"
    p.write_text(json.dumps(res, indent=2))
    publish(p, "artifacts", "stage05/s05_heal.json", stage=STAGE)
    log(f"first-moment correction applied to {applied} expert tensors "
        f"across {touched_files} shards", STAGE)
    return res
