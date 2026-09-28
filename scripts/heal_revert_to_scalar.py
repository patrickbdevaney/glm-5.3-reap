"""Undo per-expert healing on the shipped checkpoint, restoring the per-layer scalar.

The ablation (`artifacts/eval/ablation_scalar_healing.json`) measured the per-expert correction
END-TO-END against the scalar it replaced, on the same held-out tokens, by overriding the
coefficients at load. The scalar won on every metric and in every sufficiently-sampled domain:

    top-1 agreement  0.83693 -> 0.84238   (+0.00545, 7.3 sigma at sigma=0.00075)
    dNLL             0.19396 -> 0.17601
    top-k KL         0.69388 -> 0.65248

So the per-expert vectors must come OFF the checkpoint on disk. Healing is a multiply on
`down_proj`, hence so is its inverse:

    multiplier_j = scalar_gain_layer / c_j

which leaves exactly the correction s05_heal would have applied had `heal_perexpert.json` never
existed. Same operation the ablation applied in memory - identical arithmetic, now persistent.

Why revert rather than annotate: a card that says "per-expert healing, which we measured to be
worse" describes a checkpoint nobody should download. The correction is invertible in weight
space at zero quality cost, so there is no reason to ship the inferior one.

Idempotent under its own {target, keep_set_sha} fingerprint - the same discipline s05_heal
learned the hard way when a pass-1 ledger silently made pass-2 healing a no-op. Verified
afterwards against `artifacts/preheal_probe.json`: post-revert scale / pre-heal scale must equal
the LAYER SCALAR, not c_j.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

LEDGER = ROOT / "state" / "heal_revert_done.json"
ADAPTERS = ROOT / "output" / "adapters" / "first_moment_gains.json"
TOL = 0.02


def main():
    from common import kv_get, log

    src = Path(kv_get("pruned_model_path") or "")
    if not (src / "model.safetensors.index.json").exists():
        src = ROOT / "output" / str(kv_get("emit_name") or "")
    if not (src / "model.safetensors.index.json").exists():
        raise SystemExit(f"no checkpoint at {src}")

    pe = json.loads((ROOT / "artifacts" / "heal_perexpert.json").read_text())
    scal = {r["layer"]: r.get("measured_gain")
            for r in json.loads((ROOT / "artifacts" / "heal_refit.json").read_text())["per_layer"]}

    # multiplier[layer_key] = [scalar/c_j for each retained expert, in keep-list order]
    mult: dict[str, list[float]] = {}
    for r in pe["per_layer"]:
        if not r.get("chosen", "").startswith("per_expert") or not r.get("c"):
            continue
        g = scal.get(r["layer"])
        if not g:
            continue
        mult[r["layer"]] = [g / c for c in r["c"]]
    if not mult:
        raise SystemExit("no per-expert layers on this checkpoint - nothing to revert")
    lo = min(min(v) for v in mult.values())
    hi = max(max(v) for v in mult.values())
    n = sum(len(v) for v in mult.values())
    log(f"reverting {len(mult)} layers / {n} experts to the per-layer scalar; "
        f"multiplier range {lo:.4f}..{hi:.4f}", "heal_revert")

    retained_p = ROOT / "artifacts" / "reap_retained_experts.json"
    fingerprint = {"target": str(src.resolve()), "op": "perexpert->scalar",
                   "keep_set_sha": hashlib.sha256(retained_p.read_bytes()).hexdigest()[:16]}
    done: set[str] = set()
    if LEDGER.exists():
        led = json.loads(LEDGER.read_text())
        if led.get("fingerprint") == fingerprint:
            done = set(led.get("shards") or [])
    shard_names = {f.name for f in src.glob("*.safetensors")}
    done &= shard_names
    if done:
        log(f"resuming: {len(done)}/{len(shard_names)} shards already reverted", "heal_revert")

    applied = files = 0
    for shard in sorted(src.glob("*.safetensors")):
        if shard.name in done:
            continue
        tensors = load_file(str(shard))
        changed = False
        for name in list(tensors):
            if ".mlp.experts." not in name or not name.endswith("down_proj.weight"):
                continue
            head, tail = name.split(".mlp.experts.", 1)
            mv = mult.get(head + ".mlp")
            if mv is None:
                continue
            try:
                m = mv[int(tail.split(".", 1)[0])]
            except (ValueError, IndexError):
                continue
            sname = name[: -len("weight")] + "weight_scale_inv"
            if sname in tensors:
                tensors[sname] = tensors[sname].to(torch.float32) * m
            else:
                t = tensors[name]
                tensors[name] = (t.to(torch.float32) * m).to(t.dtype)
            applied += 1
            changed = True
        if changed:
            tmp = shard.with_suffix(".safetensors.tmp")
            save_file(tensors, str(tmp), metadata={"format": "pt"})
            tmp.replace(shard)
            files += 1
        done.add(shard.name)
        LEDGER.write_text(json.dumps({"fingerprint": fingerprint, "shards": sorted(done)}))
        del tensors
        if len(done) % 10 == 0:
            log(f"  {len(done)}/{len(shard_names)} shards, {applied} experts reverted",
                "heal_revert")

    if applied == 0 and len(done) < len(shard_names):
        raise SystemExit("reverted 0 tensors but not every shard is recorded done")

    # DECISIVE CHECK. The probe holds the block scale as it was BEFORE any healing, so after the
    # revert the ratio must be the layer scalar. If per-expert coefficients were still present
    # the ratio would be c_j, and for these probes c_j and the scalar differ by 6-27%.
    wm = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    cvec = {r["layer"]: r["c"] for r in pe["per_layer"] if r.get("c")}
    bad = 0
    print("\npost-revert block scale / pre-heal block scale (want the LAYER SCALAR):")
    for key, _sh, before in json.loads((ROOT / "artifacts" / "preheal_probe.json").read_text()):
        ln = key.split(".mlp.experts.")[0] + ".mlp"
        e = int(key.split(".mlp.experts.")[1].split(".")[0])
        g = scal.get(ln)
        if g is None or key not in wm:
            continue
        with safe_open(str(src / wm[key]), framework="pt") as f:
            after = float(f.get_tensor(key).float().mean())
        got = after / before
        ok = abs(got - g) / g < TOL
        bad += not ok
        c = cvec.get(ln, [None] * (e + 1))[e]
        print(f"  {'ok ' if ok else 'BAD'} L{ln.split('.')[-2]:>3}/e{e:<3} ratio {got:.6f}  "
              f"scalar {g:.6f}  (per-expert was {c:.6f})" if c is not None else
              f"  {'ok ' if ok else 'BAD'} L{ln.split('.')[-2]:>3}/e{e:<3} ratio {got:.6f}  "
              f"scalar {g:.6f}")
    if bad:
        raise SystemExit(f"FAIL: {bad} probe(s) do not carry the layer scalar")

    a = json.loads(ADAPTERS.read_text())
    a["reverted_per_expert"] = {
        "when": "2026-08-29",
        "why": "ablation measured per-expert healing WORSE than the per-layer scalar end-to-end "
               "(top-1 agreement 0.83693 -> 0.84238, +0.00545 at sigma=0.00075); "
               "see artifacts/eval/ablation_scalar_healing.json",
        "coefficients": a.pop("per_expert_coefficients", {}),
        "tensors_reverted": applied, "shards_rewritten": files,
    }
    a["method"] = ("per-layer first-moment MoE output-scale correction, gains measured by "
                   "replaying post-prune routing (no teacher, no forward pass)")
    a["per_expert_layers"] = []
    a["per_expert_tensors"] = 0
    ADAPTERS.write_text(json.dumps(a, indent=2))
    print(f"\nPASS: {applied} expert tensors across {files} shards now carry the layer scalar")


if __name__ == "__main__":
    main()
