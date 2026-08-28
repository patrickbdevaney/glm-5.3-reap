"""Re-heal the published pass-1 FP8 with the MEASURED healing gain.

Pass 1 applied a first-moment gain (median 0.6964) to every retained expert's output. P5 measured
the correct value by replaying post-prune routing: median 0.9111. The shipped checkpoint therefore
under-scales its entire MoE pathway by ~0.76 relative to attention, the shared experts and the
residual stream, in all 42 layers.

This corrects it in place by multiplying each layer's block scales by

    factor[layer] = measured_gain[layer] / shipped_gain[layer]

Three properties make in-place safe here:

  * **Exact.** s05 scaled the F32 `weight_scale_inv`, never the FP8 values. The dequantised weight
    is `w_fp8 * weight_scale_inv`, so this is a scalar multiply on a float32 tensor - no
    requantisation, no rounding, nothing to lose.
  * **Reversible.** A scalar multiply inverts exactly. The factors are recorded, so the
    as-published state can be restored by dividing. In-place is not destructive in any
    information sense.
  * **Idempotent.** A per-shard ledger, because a doubly-scaled expert is still a perfectly valid
    tensor - a partial retry would corrupt the model silently, which is exactly how this class of
    bug hides.

Only `down_proj` scales are touched, mirroring s05 exactly: scaling the expert's output
projection scales its contribution, and touching gate/up as well would apply the correction
multiple times per expert.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parent.parent
SHIPPED = ROOT / "output" / "adapters" / "first_moment_gains.json"
REFIT = ROOT / "artifacts" / "heal_refit.json"
LEDGER = ROOT / "state" / "reheal_done.json"
RECORD = ROOT / "artifacts" / "reheal_factors.json"

FACTOR_LO, FACTOR_HI = 0.80, 2.00      # measured/shipped; outside this is a bug, not a correction


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "output" / "glm-5.3-flash-reap50-fp8"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    model = Path(a.model)

    shipped = json.loads(SHIPPED.read_text())["gains"]
    refit = json.loads(REFIT.read_text())
    measured = {r["layer"]: r["measured_gain"] for r in refit["per_layer"] if r.get("measured_gain")}

    factors, missing = {}, []
    for layer, g_old in shipped.items():
        g_new = measured.get(layer)
        if not g_new:
            missing.append(layer)
            continue
        factors[layer] = g_new / g_old
    if missing:
        raise SystemExit(f"no measured gain for {len(missing)} layers (e.g. {missing[:3]}) - "
                         f"refusing to half-correct the model")
    vals = sorted(factors.values())
    lo, hi = vals[0], vals[-1]
    print(f"layers                : {len(factors)}")
    print(f"shipped gain   median : {st.median(shipped.values()):.4f}")
    print(f"measured gain  median : {st.median(measured.values()):.4f}")
    print(f"correction factor     : min {lo:.4f}  median {st.median(vals):.4f}  max {hi:.4f}")
    if not (FACTOR_LO <= lo and hi <= FACTOR_HI):
        raise SystemExit(f"factor outside [{FACTOR_LO}, {FACTOR_HI}] - this indicates a bug in "
                         f"the gain bookkeeping, not a correction. Refusing.")
    RECORD.write_text(json.dumps({"factors": factors, "shipped": shipped,
                                  "measured": measured,
                                  "note": "multiply down_proj.weight_scale_inv by factor; "
                                          "divide to restore the as-published state"}, indent=2))
    if a.dry_run:
        print("\ndry run - nothing written")
        return

    done = set()
    if LEDGER.exists():
        try:
            done = set(json.loads(LEDGER.read_text()))
        except Exception:
            done = set()
    if done:
        print(f"resuming: {len(done)} shards already corrected")

    shards = sorted(model.glob("*.safetensors"))
    scaled_t = 0
    sample = None
    for i, shard in enumerate(shards, 1):
        if shard.name in done:
            continue
        tensors = load_file(str(shard))
        changed = False
        for name in list(tensors):
            if ".mlp.experts." not in name or not name.endswith("down_proj.weight_scale_inv"):
                continue
            layer_key = name.split(".mlp.experts.")[0] + ".mlp"
            f = factors.get(layer_key)
            if f is None:
                continue
            if sample is None:
                sample = (name, float(tensors[name].flatten()[0]), f)
            tensors[name] = tensors[name].to(torch.float32) * f
            scaled_t += 1
            changed = True
        if changed:
            tmp = shard.with_suffix(".safetensors.tmp")
            save_file(tensors, str(tmp), metadata={"format": "pt"})
            os.replace(tmp, shard)      # atomic: a kill leaves either old or new, never a mix
        done.add(shard.name)
        LEDGER.parent.mkdir(parents=True, exist_ok=True)
        LEDGER.write_text(json.dumps(sorted(done)))
        del tensors
        if i % 10 == 0 or i == len(shards):
            print(f"  shard {i}/{len(shards)}  scales corrected: {scaled_t}", flush=True)

    if sample:
        n, before, f = sample
        after = float(load_file(str(model / json.loads(
            (model / "model.safetensors.index.json").read_text())["weight_map"][n]))[n].flatten()[0])
        print(f"\nverification on {n.split('.mlp.')[0].split('.')[-1]}:")
        print(f"  before {before:.6g}  x{f:.4f}  ->  expected {before*f:.6g}  actual {after:.6g}")
        print(f"  match: {abs(after - before*f) < abs(before*f)*1e-5}")
    print(f"\ncorrected {scaled_t} block-scale tensors across {len(shards)} shards")
    print(f"factors recorded in {RECORD} (divide by them to restore the published state)")


if __name__ == "__main__":
    main()
