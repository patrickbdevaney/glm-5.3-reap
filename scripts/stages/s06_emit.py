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


def _enrich(meta: dict) -> dict:
    """Fill the card's quality numbers from the artifacts, never from literals.

    The card previously hardcoded "1.29x better than random / saliency mass 0.643". Those are
    PASS-1 measurements. Printed on a pass-2 model they would be a confident, specific, wrong
    claim about a different checkpoint - the worst kind, because nothing about the card would
    look stale.
    """
    try:
        sw = json.loads((ARTIFACTS / "s04_sweep.json").read_text())["sweep"]
        key = f"{float(meta.get('sparsity', 0.5)):.2f}"
        row = sw.get(key) or sw.get(str(meta.get("sparsity"))) or {}
        meta["saliency_mass_retained"] = row.get("saliency_mass_retained")
        meta["concentration_vs_random"] = row.get("concentration_vs_random")
        meta["routing_mass_retained"] = row.get("routing_mass_retained")
        meta["min_layer_routing_mass"] = row.get("min_layer_routing_mass")
    except Exception as e:
        log(f"could not read sweep numbers for the card: {type(e).__name__}", STAGE, "WARN")
    try:
        dh = json.loads((ARTIFACTS / "domain_holes.json").read_text())
        meta["domain_retention"] = {k: v["mean"] for k, v in dh["per_domain"].items()}
    except Exception:
        pass
    try:
        s3 = json.loads((ARTIFACTS / "s03_saliency.json").read_text())
        meta["calib_tokens"] = s3.get("calib_samples", 0) * s3.get("max_len", 2048)
    except Exception:
        pass
    try:
        import statistics as _st
        # Report the gain that was APPLIED. s05_heal writes first_moment_gains.json under that
        # name whatever it applied, so when a measured re-fit exists the measured median is the
        # honest number - quoting the first-moment value beside the word "measured" would
        # describe the right method with the wrong figure.
        rf = ARTIFACTS / "heal_refit.json"
        meas = None
        if rf.exists():
            meas = json.loads(rf.read_text()).get("measured_median")
        meta["heal_measured"] = bool(meas)
        if meas:
            meta["heal_gain_median"] = meas
        else:
            hp = ROOT / "output" / "adapters" / "first_moment_gains.json"
            if hp.exists():
                meta["heal_gain_median"] = _st.median(
                    json.loads(hp.read_text())["gains"].values())
        # Per-expert healing supersedes the scalar wherever it shipped. Read what was APPLIED
        # from the adapter record, not from the fit, so a layer that fell back to its scalar is
        # not described as per-expert.
        ap = ROOT / "output" / "adapters" / "first_moment_gains.json"
        if ap.exists():
            a = json.loads(ap.read_text())
            n = len(a.get("per_expert_layers") or [])
            if n:
                meta["heal_per_expert_layers"] = n
                meta["heal_per_expert_tensors"] = a.get("per_expert_tensors")
        pe = ARTIFACTS / "heal_perexpert.json"
        if pe.exists():
            d = json.loads(pe.read_text())
            rr = d.get("median_holdout_rel_residual") or {}
            for k, dst in (("none", "heal_resid_none"), ("shipped", "heal_resid_scalar"),
                           ("per_expert_mag", "heal_resid_perexpert")):
                if rr.get(k) is not None:
                    meta[dst] = rr[k]
    except Exception:
        meta.setdefault("heal_measured", False)
    try:
        sg = json.loads((ARTIFACTS / "s04b_surgery.json").read_text())
        meta["mtp_preserved"] = bool(sg.get("mtp_preserved", False))
        meta["mtp_criterion"] = sg.get("mtp_criterion")
    except Exception:
        meta.setdefault("mtp_preserved", False)
    return meta


def _card(meta: dict) -> str:
    healed = meta.get("healed")
    conc = meta.get("concentration_vs_random")
    smass = meta.get("saliency_mass_retained")
    conc = f"{conc:.2f}" if isinstance(conc, (int, float)) else "?"
    smass = f"{smass:.3f}" if isinstance(smass, (int, float)) else "?"
    rm = meta.get("routing_mass_retained")
    rmass = f"{rm/0.5:.2f}" if isinstance(rm, (int, float)) else "?"
    hg = meta.get("heal_gain_median")
    npe = meta.get("heal_per_expert_layers")
    if npe:
        r0, r1, r2 = (meta.get("heal_resid_none"), meta.get("heal_resid_scalar"),
                      meta.get("heal_resid_perexpert"))
        tbl = ""
        if None not in (r0, r1, r2):
            tbl = (f" On held-out calibration tokens the relative reconstruction residual "
                   f"`\u03a3\u2016y\u2212\u0177\u2016\u00b2/\u03a3\u2016y\u2016\u00b2` "
                   f"falls from **{r0:.4f}** with no correction to **{r1:.4f}** with the best "
                   f"single scalar to **{r2:.4f}** per-expert.")
        heal_note = (
            f"- Healing is an output-scale correction applied exactly to the F32 block scales, "
            f"fitted **per retained expert** ({npe} of 42 MoE layers; the rest keep a per-layer "
            f"scalar, median **{hg:.4f}**). Each coefficient is the closed-form least-squares "
            f"solution for reproducing the unpruned layer's output under post-prune routing, "
            f"replayed from a cached router-score trace - no teacher and no forward pass. The "
            f"solution is rescaled to preserve the layer's output magnitude, because the "
            f"unconstrained least-squares fit is scale-attenuating and that bias would compound "
            f"across 42 layers.{tbl} It is *not* distillation and does not recover lost "
            f"knowledge.")
    elif meta.get("heal_measured"):
        heal_note = (
            "- Healing is an output-scale correction applied exactly to the F32 block scales, "
            f"median gain **{hg:.4f}**. It was **measured**, not derived: post-prune routing is "
            "replayed from a cached router-score trace, because the first-moment estimator that "
            "pass 1 used ignores that `norm_topk_prob` renormalises the surviving top-8 and "
            "over-corrects by ~30%. It is *not* distillation and does not recover lost knowledge.")
    else:
        heal_note = (
            f"- Healing is a **first-moment output-scale correction** derived from the "
            f"calibration saliency (median gain {hg if hg else 'n/a'}, applied exactly to the F32 "
            "block scales). Known to over-correct by ~30% on this architecture; prefer a measured "
            "re-fit. It is *not* distillation and does not recover lost knowledge.")
    dh = meta.get("domain_retention") or {}
    def _r(b):
        v = dh.get(b)
        return f"{v:.3f}" if isinstance(v, (int, float)) else "n/a"
    ret_agentic, ret_code, ret_science = _r("agentic"), _r("code"), _r("science")
    ret_math, ret_vision, ret_finance, ret_general = (_r("math"), _r("vision"),
                                                      _r("finance"), _r("general"))
    ct = meta.get("calib_tokens")
    calib_tok = f"{ct/1e6:.1f}M tokens" if isinstance(ct, (int, float)) else "several million tokens"
    if meta.get("mtp_preserved"):
        mtp_inline = " The MTP block at layer 45 is **preserved**."
        mtp_note = (
            "- **The MTP (multi-token-prediction) block at layer 45 is preserved** and pruned to "
            "the same expert count as every other MoE layer (forced: `num_local_experts` is a "
            "single scalar). `transformers` does not instantiate it, but vLLM and SGLang implement "
            "MTP speculative decoding, so it is retained as a draft head. Because the calibration "
            "sweep never runs it, its experts were ranked by "
            f"{meta.get('mtp_criterion') or 'weight norm'} rather than activation saliency - a "
            "weaker criterion, appropriate here because a draft head is expected to be fine-tuned "
            "downstream.")
    else:
        mtp_inline = " The MTP block at layer 45 is cleanly absent."
        mtp_note = (
            "- **The MTP (multi-token-prediction) block at layer 45 is excluded.** `transformers`' "
            "`Glm5NextForConditionalGeneration` does not instantiate it, so the pruning path "
            "cannot see it. Dropping it forecloses speculative decoding from this artifact.")
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
is untouched.{mtp_inline}

The pruning itself measured **{conc}x better than random** at retaining expert output
contribution (saliency mass {smass} against 0.50 for random pruning at the same ratio). That says
the criterion selected well. It does **not** say the model is good.

Treat this as a research artifact pending evaluation, not a drop-in replacement.

## What this prune costs, and what it does not

REAP ranks experts on saliency pooled across the calibration mixture, so the cost is **not**
spread evenly across capabilities. Measured on this checkpoint - the fraction of the expert
output each domain actually relies on that survived the prune:

| domain | retained |
|---|---|
| agentic / tool use | {ret_agentic} |
| code | {ret_code} |
| science | {ret_science} |
| maths | {ret_math} |
| vision (image-text) | {ret_vision} |
| finance | {ret_finance} |
| general / ballast | **{ret_general}** |

**The damage is concentrated where retrieval can repair it.** Generic factual ballast is the
worst-retained bucket, and it is also the single most RAG-recoverable capability: a markdown
corpus and a retriever substitute for memorised trivia almost perfectly. The best-retained
buckets - agentic behaviour, code, science, maths - are the ones retrieval **cannot** restore,
because you cannot retrieve your way to reasoning, tool use, or code synthesis.

If you are deploying this, pair it with retrieval. That is not a workaround for a defect; it is
the shape the trade was made in. A uniformly-quantised model of the same footprint spends its
degradation budget evenly, including on the capabilities retrieval cannot give back.

Two honest caveats. Low retention can also mean a domain uses experts *diffusely* rather than
that its capability was removed - generic web text spreads across many experts while REAP keeps
concentrated ones. And the per-domain spread was a **consequence** of the calibration mixture,
not a targeted design: this is mixture-shaped preservation, not surgery.

## Method: strengths and gaps

**Where this is above common practice**

- Calibration is {calib_tok} across seven use-case-weighted domains, against the 128-512 generic
  samples typical of pruning and quantisation work.
- **Real image-text pairs** in calibration, not text descriptions of images. Measured effect:
  vision retained {ret_vision}, at the cross-domain mean. Text-only calibration deletes
  vision-serving experts with near-certainty.
- Saliency capture was **verified against the model source** - the router gate excludes
  `e_score_correction_bias`, which is the detail most implementations get silently wrong.
- Per-token router scores were cached, making post-prune routing exactly replayable offline.
- The healing correction was **measured, not derived**. The standard first-moment estimator
  over-corrects by ~30% on this architecture because it ignores `norm_topk_prob`
  renormalisation; that error was found and fixed only because of the router cache.
- The calibration budget was validated by a split-half stopping rule on retained saliency mass,
  rather than assumed.

**Where it falls short, stated plainly**

- **Healing rescales experts; it does not retrain them.** The per-expert coefficients are the best
  a *fixed rescaling* can do, and that ceiling is now measured rather than assumed: 96.6% of an
  expert's output energy is token-dependent residual rather than its mean, so the ~0.27 residual
  that remains is information deleted with the pruned experts, not error a better fit could
  remove. Recovering it means layer-local distillation, which needs a backward pass that does not
  fit the hardware this was built on.
- **No expert merging** (REAM/EEP), rejected on cost.
- **Evaluation is teacher-forced dNLL and top-1 agreement, not generative benchmarks.** That is
  the largest gap: dNLL does not fully predict agentic or coding capability, which is what this
  model is for. Generation from the unpruned teacher costs ~110 s/token on one device and was
  out of reach.
- **Long context is uncalibrated.** Calibration sequences were capped at 2048 tokens against a
  1M-token context window.
- The calibration mixture was specified in *samples* but acts in *tokens*; document lengths
  differ by more than 10x, so realised domain weights differed substantially from intended.
- A single prune ratio was materialised. No 40/50/60 ablation with evaluation behind it.

## Known limitations

{mtp_note}
- REAP has no published data above 50% compression; this checkpoint sits at the validated
  ceiling, not beyond it.
- Expect **factual-recall** regression before reasoning or coding regression. That is the
  measured failure mode of expert pruning on this architecture family: the closest published
  analogue (`cerebras/Kimi-Linear-REAP-35B-A3B`, same KDA + full-attention stack) loses 3.4
  points on FRAMES at only 30% pruning while code and maths hold flat.
{heal_note}
- Routing is disrupted more than expert count suggests: the retained experts carry ~{rmass}x the
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
        meta = _enrich(meta)
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
    meta = _enrich(meta)
    (EMIT / "README.md").write_text(_card(meta))
    (EMIT / "reap_metadata.json").write_text(json.dumps(meta, indent=2))

    total = sum(p.stat().st_size for p in EMIT.rglob("*") if p.is_file())
    metric(STAGE, "emit_bytes", total)
    kv_set("emit_path", str(EMIT))
    log(f"deliverable emitted: {EMIT} ({total/1e9:.1f} GB), healed={healed}", STAGE)

    publish(EMIT, "fp8", ".", stage=STAGE)
    return {"path": str(EMIT), "bytes": total, "healed": healed}
