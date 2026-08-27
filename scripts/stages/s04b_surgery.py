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

The MTP block (layer 45) is excluded: transformers does not instantiate it, so it cannot be
pruned consistently. Its tensors are simply not copied; they remain unmodified upstream.
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


def compute_retained(ratio: float) -> dict[str, list[int]]:
    import torch
    retained = {}
    for f in sorted(SALIENCY.glob("*.pt")):
        d = torch.load(f, weights_only=False)
        c = d["count"].double()
        m = torch.where(c > 0, d["sum_saliency"].double() / c.clamp(min=1),
                        torch.zeros_like(c))
        n = m.numel()
        keep = int(round(n * (1 - ratio)))
        # Never rank an unobserved expert above an observed one: an expert with zero routed
        # tokens has an undefined mean, not a low one.
        m = torch.where(c > 0, m, torch.full_like(m, float("-inf")))
        idx = sorted(torch.argsort(m, descending=True)[:keep].tolist())
        retained[d["layer"]] = idx
    return retained


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
    (ARTIFACTS / "reap_retained_experts.json").write_text(json.dumps(retained))
    n_keep = len(next(iter(retained.values())))
    log(f"ratio {ratio:.0%}: keeping {n_keep} of 288 experts in {len(retained)} MoE layers",
        STAGE)

    OUT.mkdir(parents=True, exist_ok=True)
    done = _load_ledger()
    shards = sorted(SRC.glob("*.safetensors"))
    if not shards and done:
        log("all source shards already consumed (resuming a completed pass)", STAGE)

    pos = {lname: {e: i for i, e in enumerate(keep)} for lname, keep in retained.items()}
    exp_re = re.compile(r"^(?P<pre>.*\.layers\.(?P<layer>\d+)\.mlp)\.experts\.(?P<e>\d+)\.(?P<rest>.+)$")

    weight_map: dict[str, str] = {}
    if (OUT / "model.safetensors.index.json").exists():
        try:
            weight_map = json.loads((OUT / "model.safetensors.index.json").read_text())["weight_map"]
        except Exception:
            weight_map = {}

    kept_t = dropped_t = 0
    t0 = time.time()
    for si, shard in enumerate(shards):
        if shard.name in done:
            continue
        out_t: dict[str, torch.Tensor] = {}
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            for name in f.keys():
                m = exp_re.match(name)
                layer_m = re.search(r"\.layers\.(\d+)\.", name)
                if layer_m and int(layer_m.group(1)) == MTP_LAYER:
                    dropped_t += 1
                    continue                      # MTP excluded, see module docstring
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
                    lname = name.rsplit(".gate.", 1)[0] if ".gate." in name else None
                    base = name.split(".mlp.")[0] + ".mlp"
                    keep = retained.get(base)
                    if keep is not None and t.shape[0] == 288:
                        t = t[torch.tensor(keep, dtype=torch.long)].contiguous()
                out_t[name] = t
                kept_t += 1

        out_name = f"model-{si+1:05d}-of-{len(shards):05d}.safetensors"
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
        _save_ledger(done)
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
    cfg["text_config"]["num_nextn_predict_layers"] = 0      # MTP excluded
    cfg["reap"] = {"ratio": ratio, "experts_kept": n_keep, "experts_original": 288,
                   "mtp_excluded": True}
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
    res = {"ratio": ratio, "experts_kept": n_keep, "tensors_kept": kept_t,
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
