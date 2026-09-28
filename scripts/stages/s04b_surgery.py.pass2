"""Stage 4b - structural surgery at the tensor level.

The streaming path never materialises a model, so the prune is done directly on safetensors:
copy through the surviving expert tensors (renumbered), shrink the routers, drop the rest.

Nothing is dequantised. Experts are stored per-expert with their own weight_scale_inv block
scales, so removing an expert is deleting six tensors - the retained weights come through
bit-identical. There is no prune-time quantisation error at all.

Disk is the binding constraint (R10): 162 GiB free against a ~156 GiB output. So each source
shard is DELETED once its survivors have been written, which makes free space grow monotonically
through the pass. That is safe because the source is public and re-downloadable in ~23 min, and
the stage is resumable - a completed-shard ledger means a crash re-runs only what is unfinished.

The MTP block (layer 45) is PRESERVED and pruned to the same expert count as every other MoE
layer (forced: `num_local_experts` is a single scalar). It is a draft head - vLLM and SGLang
implement MTP speculative decoding even though `transformers` does not - so dropping it, as pass
1 did, forecloses spec-decode from the artifact entirely. Because the streaming sweep never runs
it, its experts are ranked by weight norm rather than activation saliency; see `_mtp_keep_set`.
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import ROOT, ARTIFACTS, log, metric, kv_get, kv_set, publish, free_gib  # noqa: E402

STAGE = "s04b_surgery"
SRC = ROOT / "source" / "GLM-5.3-Flash"
OUT = ROOT / "output" / "pruned-fp8"
SALIENCY = ROOT / "artifacts" / "saliency"
LEDGER = ROOT / "state" / "surgery_done.json"
MTP_LAYER = 45


MIN_KEEP_FRAC, MAX_KEEP_FRAC = 0.30, 0.75


def _layer_curves():
    """Per layer: expert order (best first) and the cumulative saliency-mass fraction."""
    import torch
    out = {}
    for f in sorted(SALIENCY.glob("*.pt")):
        d = torch.load(f, weights_only=False)
        c = d["count"].double()
        m = torch.where(c > 0, d["sum_saliency"].double() / c.clamp(min=1),
                        torch.zeros_like(c))
        # An expert with zero routed tokens has an UNDEFINED mean, not a low one - rank it
        # last explicitly rather than letting a 0.0 outrank a genuinely weak observed expert.
        rank_key = torch.where(c > 0, m, torch.full_like(m, float("-inf")))
        order = torch.argsort(rank_key, descending=True)
        contrib = (m * c)[order]
        total = contrib.sum().clamp(min=1e-12)
        out[d["layer"]] = (order.tolist(), (contrib.cumsum(0) / total).tolist())
    return out


def _mtp_keep_set(n_keep: int, n_orig: int) -> list[int] | None:
    """Rank the MTP block's experts by a WEIGHT-ONLY criterion, and say so.

    Layer 45 is not part of the main forward - `transformers` never instantiates it - so the
    streaming sweep produces no activation saliency for it. Pruning it therefore has to fall
    back to weights alone: score_e = log||W_gate||_F + log||W_up||_F + log||W_down||_F, which
    bounds the expert's operator magnitude.

    That is a genuinely weaker criterion than REAP, and it is acceptable HERE and ONLY here,
    because the MTP head's intended use is as a draft head that gets fine-tuned against the
    pruned target afterwards - training repairs a mediocre prune. The same argument must never
    be used for the main stack, where nothing downstream repairs anything.

    Pruning it at all is forced, not chosen: `num_local_experts` is a single scalar, so layer 45
    must carry exactly as many experts as every other MoE layer or the config cannot describe
    the checkpoint.
    """
    import torch
    from safetensors import safe_open

    cache = ARTIFACTS / "mtp_keep.json"
    if cache.exists():
        try:
            return json.loads(cache.read_text())["keep"]
        except Exception:
            pass

    ip = SRC / "model.safetensors.index.json"
    if not ip.exists():
        return None
    wm = json.loads(ip.read_text())["weight_map"]
    pref = f"model.language_model.layers.{MTP_LAYER}.mlp.experts."
    names = [k for k in wm if k.startswith(pref) and k.endswith(".weight")]
    if not names:
        return None

    handles: dict[str, object] = {}

    def _get(n):
        sh = wm[n]
        if sh not in handles:
            fp = SRC / sh
            if not fp.exists():
                raise FileNotFoundError(fp)
            handles[sh] = safe_open(str(fp), framework="pt", device="cpu")
        return handles[sh].get_tensor(n)

    score = torch.zeros(n_orig, dtype=torch.float64)
    seen = torch.zeros(n_orig, dtype=torch.bool)
    try:
        for n in names:
            try:
                e = int(n[len(pref):].split(".")[0])
            except ValueError:
                continue
            if e >= n_orig:
                continue
            w = _get(n)
            sn = n[: -len("weight")] + "weight_scale_inv"
            if sn in wm:
                sc = _get(sn).to(torch.float32)
                out, inn = w.shape
                sc = sc.repeat_interleave(128, 0).repeat_interleave(128, 1)[:out, :inn]
                fro = (w.to(torch.float32) * sc).norm()
            else:
                fro = w.to(torch.float32).norm()
            score[e] += float(torch.log(fro.clamp_min(1e-30)))
            seen[e] = True
    except FileNotFoundError:
        # Source shards already consumed by a previous run and no cache: cannot rank.
        return None
    finally:
        handles.clear()

    if not bool(seen.all()):
        log(f"MTP keep-set: only {int(seen.sum())}/{n_orig} experts scored", STAGE, "WARN")
    keep = sorted(int(i) for i in torch.argsort(score, descending=True)[:n_keep])
    cache.write_text(json.dumps({"keep": keep, "criterion": "log-Frobenius weight norm",
                                 "layer": MTP_LAYER, "note": "no activation saliency exists "
                                 "for the MTP block; intended to be fine-tuned downstream"}))
    return keep


def compute_retained(ratio: float, uniform: bool = True) -> dict[str, list[int]]:
    """Allocate the expert budget across layers by EQUALISING retained saliency mass.

    Uniform pruning is badly suboptimal on this model. Measured over all 42 MoE layers at 50%:
    retained saliency mass ranges from 0.491 (layer 35 - literally no better than random
    pruning, x0.98) to 0.861 (layer 7, x1.72), with 12 layers below 0.60. Spending the same
    50% everywhere over-prunes the layers that cannot afford it and under-prunes the ones that
    can.

    So instead: pick the largest fraction f such that giving every layer just enough experts to
    retain f of its own saliency mass still fits the global budget. Layers that concentrate
    their mass in few experts give experts up; layers that spread it keep more. This is the
    non-uniform-allocation idea from EvoESAP/DiEP, but computed analytically from the cached
    accumulators rather than searched - no extra forward passes, no evolutionary loop.

    Bounded to [30%, 75%] kept per layer so no layer is pushed far outside REAP's validated
    territory, and top_k reachability is guaranteed.

    DEFAULT IS UNIFORM, and that is not a preference - it is forced by the architecture.
    `Glm5NextTextExperts.__init__` and `Glm5NextTextTopkRouter.__init__` both read a single
    scalar `config.num_local_experts` and apply it to EVERY layer. There is no per-layer expert
    count in glm5_next, and vLLM and the GGUF converters read the same field. A non-uniform
    checkpoint is therefore unloadable by anything that does not carry a patched model
    definition, which makes it worthless as a deliverable however good its saliency numbers
    look. Measured on the real model: non-uniform lifted the worst layer from 0.491 to 0.649
    retained saliency mass and cleared all 12 layers that sat below 0.60 - a real gain, and
    unusable. Keep the code; it becomes deployable the day the config grows a per-layer field.
    """
    curves = _layer_curves()
    if not curves:
        raise RuntimeError("no saliency available")
    n_exp = len(next(iter(curves.values()))[0])
    n_layers = len(curves)
    budget = int(round(n_exp * (1 - ratio))) * n_layers
    lo_k = max(int(n_exp * MIN_KEEP_FRAC), 8)
    hi_k = int(n_exp * MAX_KEEP_FRAC)

    if uniform:
        per = {L: int(round(n_exp * (1 - ratio))) for L in curves}
    else:
        def keeps_for(f):
            out = {}
            for L, (_, cum) in curves.items():
                k = next((i + 1 for i, v in enumerate(cum) if v >= f), n_exp)
                out[L] = min(max(k, lo_k), hi_k)
            return out

        lo, hi = 0.0, 1.0
        per = keeps_for(0.5)
        for _ in range(60):                       # bisect on the equalised fraction
            mid = (lo + hi) / 2
            cand = keeps_for(mid)
            if sum(cand.values()) > budget:
                hi = mid
            else:
                lo = mid
                per = cand
        log(f"non-uniform allocation: equalised retained saliency fraction f={lo:.4f}; "
            f"experts kept per layer min={min(per.values())} max={max(per.values())} "
            f"total={sum(per.values())} vs budget {budget}", STAGE)

    retained = {}
    for L, (order, _) in curves.items():
        retained[L] = sorted(order[:per[L]])
    return retained


def _original_expert_count() -> int:
    import torch
    for f in sorted(SALIENCY.glob("*.pt")):
        return int(torch.load(f, weights_only=False)["num_experts"])
    raise RuntimeError("no saliency files; cannot determine original expert count")


def _load_ledger() -> set[str]:
    if LEDGER.exists():
        try:
            return set(json.loads(LEDGER.read_text()))
        except Exception:
            return set()
    return set()


def _save_ledger(done: set[str]) -> None:
    LEDGER.write_text(json.dumps(sorted(done)))


def run() -> dict:
    import re
    import torch
    from safetensors.torch import save_file
    from safetensors import safe_open

    ratio = float(kv_get("chosen_ratio", 0.50) or 0.50)
    retained = compute_retained(ratio)
    if not retained:
        raise RuntimeError("no saliency available; stage 3 must run first")
    n_keep = len(next(iter(retained.values())))
    n_orig = _original_expert_count()
    # Preserve the MTP block (R14). Dropping it forecloses speculative decoding: an MTP block is
    # a draft head, and vLLM/SGLang implement MTP even though `transformers` does not, so
    # "transformers cannot instantiate it" was a fact about our validation harness rather than
    # about the weight's value. Injecting its keep-set here means the existing expert-renumbering
    # and router-slicing paths handle it with no special-casing downstream.
    mtp_lname = f"model.language_model.layers.{MTP_LAYER}.mlp"
    mtp_keep = _mtp_keep_set(n_keep, n_orig)
    if mtp_keep:
        retained[mtp_lname] = mtp_keep
        log(f"MTP block (layer {MTP_LAYER}) PRESERVED: {len(mtp_keep)}/{n_orig} experts kept by "
            f"weight norm (no activation saliency exists for it)", STAGE)
    else:
        log(f"MTP block (layer {MTP_LAYER}) could not be ranked - it will be dropped, which "
            f"forecloses speculative decoding from this artifact", STAGE, "WARN")
    (ARTIFACTS / "reap_retained_experts.json").write_text(json.dumps(retained))
    log(f"ratio {ratio:.0%}: keeping {n_keep} of {n_orig} experts in {len(retained)} MoE layers",
        STAGE)

    OUT.mkdir(parents=True, exist_ok=True)
    # The output file existing IS the resume record now that names are 1:1 - a separate
    # ledger could disagree with what is actually on disk, and did.
    shards = sorted(SRC.glob("*.safetensors"))
    done = {p.name for p in OUT.glob("*.safetensors")}
    if done:
        log(f"resuming: {len(done)} output shards already present", STAGE)

    pos = {lname: {e: i for i, e in enumerate(keep)} for lname, keep in retained.items()}
    exp_re = re.compile(r"^(?P<pre>.*\.layers\.(?P<layer>\d+)\.mlp)\.experts\.(?P<e>\d+)\.(?P<rest>.+)$")

    weight_map: dict[str, str] = {}
    if (OUT / "model.safetensors.index.json").exists():
        try:
            weight_map = json.loads((OUT / "model.safetensors.index.json").read_text())["weight_map"]
        except Exception:
            weight_map = {}

    kept_t = dropped_t = routers_sliced = 0
    t0 = time.time()
    for si, shard in enumerate(shards):
        if shard.name in done:
            continue
        out_t: dict[str, torch.Tensor] = {}
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            for name in f.keys():
                m = exp_re.match(name)
                layer_m = re.search(r"\.layers\.(\d+)\.", name)
                if layer_m and int(layer_m.group(1)) == MTP_LAYER and not mtp_keep:
                    dropped_t += 1
                    continue          # only if it could not be ranked; see _mtp_keep_set
                if m:
                    lname, e = m.group("pre"), int(m.group("e"))
                    keep_map = pos.get(lname)
                    if keep_map is None or e not in keep_map:
                        dropped_t += 1
                        continue
                    new = f"{lname}.experts.{keep_map[e]}.{m.group('rest')}"
                    out_t[new] = f.get_tensor(name)
                    kept_t += 1
                    continue
                t = f.get_tensor(name)
                # Routers emit logits over the expert set; they must shrink with it.
                if name.endswith("mlp.gate.weight") or name.endswith("e_score_correction_bias"):
                    base = name.split(".mlp.")[0] + ".mlp"
                    keep = retained.get(base)
                    # Compare against the ORIGINAL expert count read from the saliency, not a
                    # hardcoded 288. A router left emitting logits over the full expert set
                    # while only half the experts exist does not crash - it silently produces
                    # garbage routing, which is the worst way for this to fail.
                    if keep is not None and t.shape[0] == n_orig:
                        t = t[torch.tensor(keep, dtype=torch.long)].contiguous()
                        routers_sliced += 1
                out_t[name] = t
                kept_t += 1

        # Name the output after its SOURCE shard, 1:1. Deriving the name from the count of
        # REMAINING shards meant every resumed run used a different name-space - a single
        # output dir ended up holding shards labelled of-00029, of-00035, of-00052 and
        # of-00062 from four partial runs, none of them a complete model.
        out_name = shard.name
        n_written = len(out_t)
        if out_t:
            save_file(out_t, str(OUT / out_name), metadata={"format": "pt"})
            for k in out_t:
                weight_map[k] = out_name
        del out_t

        # Deleting the source is irreversible within this run (re-download is ~23 min), so
        # verify the output is on disk and actually readable before dropping the input. A
        # truncated or unreadable output plus a deleted source would lose the shard silently.
        if n_written:
            outp = OUT / out_name
            if not outp.exists() or outp.stat().st_size == 0:
                raise RuntimeError(f"output {out_name} missing or empty; refusing to delete "
                                   f"source {shard.name}")
            try:
                with safe_open(str(outp), framework="pt", device="cpu") as vf:
                    got = len(vf.keys())
            except Exception as e:
                raise RuntimeError(f"output {out_name} unreadable ({type(e).__name__}); "
                                   f"refusing to delete source {shard.name}") from e
            if got != n_written:
                raise RuntimeError(f"output {out_name} has {got} tensors, expected "
                                   f"{n_written}; refusing to delete source {shard.name}")

        shard.unlink()                            # free space as we go (R10)
        done.add(shard.name)
        (OUT / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": 0}, "weight_map": weight_map}))
        if (si + 1) % 5 == 0 or si == 0:
            el = time.time() - t0
            log(f"shard {si+1}/{len(shards)}  free {free_gib():.0f} GiB  "
                f"elapsed {el/60:.1f} min  eta {(el/(si+1))*(len(shards)-si-1)/60:.0f} min",
                STAGE)

    # config + tokenizer
    cfg = json.loads((SRC / "config.json").read_text()) if (SRC / "config.json").exists() \
        else json.loads((OUT / "config.json").read_text())
    cfg["text_config"]["n_routed_experts"] = n_keep
    cfg["text_config"]["num_local_experts"] = n_keep
    cfg["text_config"]["num_nextn_predict_layers"] = 1 if mtp_keep else 0
    cfg["reap"] = {"ratio": ratio, "experts_kept": n_keep, "experts_original": n_orig,
                   "mtp_preserved": bool(mtp_keep),
                   "mtp_criterion": "weight-norm (no activation saliency)" if mtp_keep else None}
    (OUT / "config.json").write_text(json.dumps(cfg, indent=2))
    for extra in ("tokenizer.json", "tokenizer_config.json", "generation_config.json",
                  "chat_template.jinja", "preprocessor_config.json", "LICENSE"):
        sp = SRC / extra
        if sp.exists() and not (OUT / extra).exists():
            shutil.copy2(sp, OUT / extra)

    total = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    metric(STAGE, "pruned_bytes", total)
    metric(STAGE, "tensors_kept", kept_t)
    metric(STAGE, "tensors_dropped", dropped_t)
    # Verify routers by INSPECTING THE OUTPUT, not by counting what this run happened to
    # touch. A per-run counter is wrong on any resume: it fired at "sliced 8, expected 84"
    # on a run that had legitimately done the other 76 earlier.
    from safetensors import safe_open as _so
    seen_routers, wrong = 0, []
    for outp in sorted(OUT.glob("*.safetensors")):
        with _so(str(outp), framework="pt", device="cpu") as f:
            for k in f.keys():
                if k.endswith("mlp.gate.weight") or k.endswith("e_score_correction_bias"):
                    seen_routers += 1
                    base = k.split(".mlp.")[0] + ".mlp"
                    want = len(retained.get(base, []))
                    got = f.get_slice(k).get_shape()[0]
                    if want and got != want:
                        wrong.append((k, got, want))
    expected_routers = 2 * len(retained)
    if wrong:
        raise RuntimeError(f"{len(wrong)} router tensors have the wrong width, e.g. {wrong[:2]}")
    if seen_routers != expected_routers:
        raise RuntimeError(f"found {seen_routers} router tensors in the output, expected "
                           f"{expected_routers} (2 per MoE layer). The output is incomplete.")
    log(f"verified {seen_routers} router tensors, all sized to their layer's expert count",
        STAGE)
    res = {"ratio": ratio, "experts_kept": n_keep, "routers_sliced": routers_sliced,
           "experts_original": n_orig, "tensors_kept": kept_t,
           "tensors_dropped": dropped_t, "bytes": total,
           "gib": round(total / 2**30, 1), "path": str(OUT),
           "free_gib_after": round(free_gib(), 1)}
    kv_set("pruned_model_path", str(OUT))
    p = ARTIFACTS / "s04b_surgery.json"
    p.write_text(json.dumps(res, indent=2))
    publish(p, "artifacts", "stage04b/s04b_surgery.json", stage=STAGE)
    log(f"surgery complete: {res['gib']} GiB, {kept_t} tensors kept / {dropped_t} dropped, "
        f"{res['free_gib_after']} GiB free", STAGE)
    return res
