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

# The gain is E[g*||f||] over ALL experts divided by the same over the RETAINED ones, so it
# is <= 1 by construction (REAP keeps the most salient). A gain ABOVE 1 really would be a bug.
# The lower bound is not symmetric: under non-uniform allocation a layer whose saliency is
# concentrated gets pruned hardest, and its retained experts are then far above the layer
# average - layer 44 keeps 86 of 288 experts and lands at 0.441. That is a large correction,
# not a wrong one. A 0.5 floor rejected it and would have failed the stage after surgery.
GAIN_FLOOR, GAIN_CEIL = 0.25, 1.0
# A per-expert coefficient is NOT bounded above by 1 the way the layer scalar is: the layer
# mean can be right while an individual expert that lost most of its top-8 competitors needs
# scaling UP. The fitter clamps to [0.25, 1.60]; this is the same band, re-checked on load.
PEREXPERT_FLOOR, PEREXPERT_CEIL = 0.25, 1.60
LEDGER = ROOT / "state" / "heal_done.json"


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

    # Per-layer gains measured by scripts/heal_refit.py, which replays post-prune routing from
    # the cached router scores rather than deriving the correction from pre-prune means.
    measured: dict[str, float] = {}
    rf = ARTIFACTS / "heal_refit.json"
    if rf.exists():
        try:
            d = json.loads(rf.read_text())
            # Refuse gains measured against a DIFFERENT keep-set. Pass 2 re-ranks the experts,
            # so a heal_refit.json left over from pass 1 describes a mask that no longer exists.
            # Silently applying it would repeat, in a new costume, exactly the failure P5 found:
            # a confidently-wrong correction derived from stale bookkeeping.
            import hashlib
            want = hashlib.sha256(retained_p.read_bytes()).hexdigest()[:16]
            got = d.get("keep_set_sha")
            if got and got != want:
                raise RuntimeError(
                    f"heal_refit.json was measured against keep-set {got} but this run applies "
                    f"{want}. Re-run scripts/heal_refit.py AFTER s04_sweep has written the "
                    f"current keep-set; healing gains are only valid for the mask they were "
                    f"derived from.")
            if not got:
                log("heal_refit.json predates keep-set stamping; cannot prove it matches this "
                    "mask - re-run heal_refit.py to be safe", STAGE, "WARN")
            measured = {r["layer"]: r["measured_gain"] for r in d.get("per_layer", [])
                        if r.get("measured_gain")}
            log(f"using MEASURED healing gains for {len(measured)} layers "
                f"(median {d.get('measured_median')}); first-moment retained only as fallback",
                STAGE)
        except Exception as e:
            log(f"heal_refit.json unreadable ({type(e).__name__}); falling back to first-moment",
                STAGE, "WARN")
    # Tier 1.1 - PER-EXPERT gains. `measured` above is one scalar per layer, which is the
    # degenerate case of the least-squares problem heal_perexpert.py actually solves: choose a
    # coefficient per RETAINED expert minimising ||y_unpruned - y_pruned||^2 under the replayed
    # post-prune routing. It ships a layer's vector only if it beat the best possible scalar on
    # HELD-OUT tokens, so an empty dict here means the scalar genuinely could not be improved on,
    # not that the fit failed.
    perexp: dict[str, list[float]] = {}
    pe = ARTIFACTS / "heal_perexpert.json"
    if pe.exists():
        try:
            d = json.loads(pe.read_text())
            import hashlib
            want = hashlib.sha256(retained_p.read_bytes()).hexdigest()[:16]
            got = d.get("keep_set_sha")
            if got != want:
                raise RuntimeError(
                    f"heal_perexpert.json was fitted against keep-set {got} but this run applies "
                    f"{want}; per-expert coefficients are only valid for the mask that produced "
                    f"the routing they were fitted from.")
            for r in d.get("per_layer", []):
                if r.get("chosen", "").startswith("per_expert") and r.get("c"):
                    ln, c = r["layer"], r["c"]
                    kept = retained.get(ln) or []
                    # The checkpoint indexes experts by POSITION in the sorted keep-list (see
                    # s04b_surgery's `pos` map), which is exactly the order `kept_ids` was
                    # written in. Refuse the layer if that correspondence does not hold rather
                    # than scale the wrong experts.
                    if r.get("kept_ids") != sorted(kept) or len(c) != len(kept):
                        log(f"{ln}: per-expert vector does not line up with the keep-set "
                            f"({len(c)} coeffs vs {len(kept)} experts) - using the scalar",
                            STAGE, "WARN")
                        continue
                    if not all(PEREXPERT_FLOOR <= x <= PEREXPERT_CEIL for x in c):
                        log(f"{ln}: per-expert coefficient out of "
                            f"[{PEREXPERT_FLOOR},{PEREXPERT_CEIL}] - using the scalar",
                            STAGE, "WARN")
                        continue
                    perexp[ln] = c
            if perexp:
                mg = (d.get("median_gain_vs_scalar") or {}).get("per_expert_diag")
                log(f"using PER-EXPERT healing for {len(perexp)}/{d.get('layers')} layers "
                    f"(median held-out residual reduction vs the optimal scalar: "
                    f"{mg:.1%})" if mg is not None else
                    f"using PER-EXPERT healing for {len(perexp)} layers", STAGE)
            else:
                log("heal_perexpert.json present but no layer beat its scalar out of sample; "
                    "applying per-layer scalars", STAGE)
        except Exception as e:
            log(f"heal_perexpert.json rejected ({type(e).__name__}: {e}); falling back to "
                f"per-layer scalars", STAGE, "WARN")
            perexp = {}

    if not measured:
        log("NO measured gains available - applying the first-moment derivation, which is known "
            "to over-correct by ~30% on this architecture because it ignores norm_topk_prob "
            "renormalisation. Run scripts/heal_refit.py first.", STAGE, "WARN")

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
        # MEASURED overrides DERIVED. See wiki/70-healing.md.
        #
        # The first-moment ratio above ignores that `norm_topk_prob` renormalises the surviving
        # top-8, so a surviving expert's gate GROWS after pruning and the router hands most of the
        # lost mass back by itself. Measured 2026-08-28: gate mass is 2.5000 before and 2.5000
        # after, and the real output inflation is 1.10x, not the 1.39x this ratio implies. Pass 1
        # therefore shrank every retained expert by 0.696 where 0.911 was correct - a 30.8% error,
        # baked into the published checkpoint.
        m = measured.get(layer)
        if m:
            g = m
        gains[layer] = g
        metric(STAGE, "first_moment_gain", g, tag=layer)

    if not gains:
        raise RuntimeError("no per-layer gains computed; saliency accumulators unusable")

    vals = sorted(gains.values())
    stats = {"layers": len(gains), "skipped": len(skipped),
             "gain_min": round(vals[0], 4), "gain_max": round(vals[-1], 4),
             "gain_median": round(vals[len(vals) // 2], 4),
             "gain_mean": round(sum(vals) / len(vals), 4)}
    log(f"first-moment gains: {stats}", STAGE)
    low = sorted(((v, k) for k, v in gains.items()))[:3]
    log("largest corrections (most-pruned layers): "
        + ", ".join(f"{k.split('.')[-2]}:{v:.3f}" for v, k in low), STAGE)

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

    # This stage MUTATES the checkpoint in place, so it must be idempotent. Without a ledger a
    # retry after a partial pass would apply the gain a SECOND time to already-scaled shards -
    # silently, since a doubly-scaled expert is still a valid tensor. Track completed shards.
    done: set[str] = set()
    if LEDGER.exists():
        try:
            done = set(json.loads(LEDGER.read_text()))
        except Exception:
            done = set()
    if done:
        log(f"resuming: {len(done)} shards already healed, skipping them", STAGE)

    applied, touched_files, perexpert_applied = 0, 0, 0
    for shard in sorted(src.glob("*.safetensors")):
        if shard.name in done:
            continue
        tensors = load_file(str(shard))
        changed = False
        for name in list(tensors):
            if ".mlp.experts." not in name or not name.endswith("down_proj.weight"):
                continue
            head, tail = name.split(".mlp.experts.", 1)
            layer_key = head + ".mlp"
            g = gains.get(layer_key)
            if g is None:
                continue
            pv = perexp.get(layer_key)
            if pv is not None:
                # Local expert index in the PRUNED checkpoint == position in the sorted
                # keep-list, which is the order `pv` is stored in.
                try:
                    g = pv[int(tail.split(".", 1)[0])]
                except (ValueError, IndexError):
                    pass
                else:
                    perexpert_applied += 1
            # Scale the BLOCK SCALE, not the FP8 values. The dequantised weight is
            # w_fp8 * weight_scale_inv, so scaling the F32 scale is mathematically identical
            # and exact, whereas round-tripping E4M3 values through float32 and back would
            # requantise every weight and lose precision on a correction of only a few percent.
            sname = name[: -len("weight")] + "weight_scale_inv"
            if sname in tensors:
                tensors[sname] = (tensors[sname].to(torch.float32) * g)
            else:
                t = tensors[name]
                tensors[name] = (t.to(torch.float32) * g).to(t.dtype)
            applied += 1
            changed = True
        if changed:
            # Write to a sibling temp then rename: an interrupted save_file over the live path
            # leaves a truncated shard and loses those weights outright.
            tmp = shard.with_suffix(".safetensors.tmp")
            save_file(tensors, str(tmp), metadata={"format": "pt"})
            tmp.replace(shard)
            touched_files += 1
        done.add(shard.name)
        LEDGER.write_text(json.dumps(sorted(done)))
        del tensors

    ADAPTERS.mkdir(parents=True, exist_ok=True)
    (ADAPTERS / "first_moment_gains.json").write_text(json.dumps(
        {"method": ("per-expert least-squares output matching (closed form, no teacher)"
                    if perexp else
                    "first-moment MoE output-scale correction (no teacher, no forward pass)"),
         "gains": gains, "stats": stats,
         "per_expert_layers": sorted(perexp), "per_expert_tensors": perexpert_applied,
         "per_expert_coefficients": perexp,
         "experts_scaled": applied, "shards_rewritten": touched_files,
         "not_done": "layer-local distillation and LoRA SFT - not tractable for 165B on a "
                     "117 GiB box; see wiki/70-healing.md"}, indent=2))
    kv_set("healed", True)
    res = {**stats, "experts_scaled": applied, "shards_rewritten": touched_files,
           "per_expert_layers": len(perexp), "per_expert_tensors": perexpert_applied}
    p = ARTIFACTS / "s05_heal.json"
    p.write_text(json.dumps(res, indent=2))
    publish(p, "artifacts", "stage05/s05_heal.json", stage=STAGE)
    log(f"correction applied to {applied} expert tensors across {touched_files} shards "
        f"({perexpert_applied} with a per-expert coefficient, "
        f"{applied - perexpert_applied} with the layer scalar)", STAGE)
    return res
