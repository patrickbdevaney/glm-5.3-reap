"""Ablation: does PER-EXPERT healing beat the per-layer scalar end-to-end, or only on the residual?

Pass 2 changed two things at once against pass 1 - the mask and the healing method - and measured
no aggregate movement in top-1 agreement. That leaves both changes unattributed. This isolates the
healing half by re-evaluating the SHIPPED pass-2 checkpoint with its per-expert coefficients
replaced by the per-layer scalar that would otherwise have been applied.

Healing is a multiply on `down_proj`, so the swap is a multiply:

    multiplier_j = scalar_gain_layer / c_j

applied when the layer is materialised (`s03_saliency.HEAL_OVERRIDE`). Nothing on disk is touched.
That matters twice over: there is not enough free disk for a second 161 GiB copy, and mutating the
published artifact in place to measure it would risk shipping whichever variant an interruption
left behind - which is exactly the failure class this project has already hit once.

Read against `artifacts/eval/eval_<pass2>.json`:
  * per-expert BETTER  -> the technique earns its place, and the mask is what went sideways
  * indistinguishable  -> per-expert healing is harmless but not load-bearing; the honest card
                          claim stays "reduces reconstruction residual", never "more accurate"
  * per-expert WORSE   -> the magnitude-preserving rescale is not the right objective after all
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / "artifacts" / "eval" / "ablation_scalar_healing.json"


def main():
    from common import kv_get, log
    import stages.s03_saliency as S3
    import stages.s09_eval as EV

    ckpt = ROOT / "output" / str(kv_get("emit_name") or "")
    if not (ckpt / "model.safetensors.index.json").exists():
        raise SystemExit(f"no checkpoint at {ckpt}")

    pe = json.loads((ROOT / "artifacts" / "heal_perexpert.json").read_text())
    scal = {r["layer"]: r.get("measured_gain")
            for r in json.loads((ROOT / "artifacts" / "heal_refit.json").read_text())["per_layer"]}

    override, n = {}, 0
    for r in pe["per_layer"]:
        if not r.get("chosen", "").startswith("per_expert") or not r.get("c"):
            continue
        g = scal.get(r["layer"])
        if not g:
            continue
        li = int(r["layer"].split("layers.")[1].split(".")[0])
        override[li] = [g / c for c in r["c"]]
        n += len(r["c"])
    if not override:
        raise SystemExit("no per-expert layers to ablate")
    S3.HEAL_OVERRIDE = override
    print(f"replacing per-expert healing with the layer scalar in {len(override)} layers "
          f"({n} experts); multiplier range "
          f"{min(min(v) for v in override.values()):.4f}..{max(max(v) for v in override.values()):.4f}")

    T = torch.load(ROOT / "artifacts" / "eval" / "teacher.pt", weights_only=False)
    rows, mm = EV.load_heldout()
    S = EV.score_checkpoint(ckpt, rows, mm, "ablation-scalar-heal")
    res = EV.compare(T, S)
    res["ablation"] = "per-layer scalar healing substituted for per-expert, applied at load"
    res["layers_overridden"] = len(override)
    OUT.write_text(json.dumps(res, indent=2))

    ref = ROOT / "artifacts" / "eval" / f"eval_{ckpt.name}.json"
    print(f"\n{'metric':<20}{'per-expert':>13}{'scalar':>13}{'delta':>13}")
    if ref.exists():
        a = json.loads(ref.read_text())
        for k in ("top1_agreement", "dNLL_mean", "topk_KL", "student_nll"):
            if a.get(k) is not None and res.get(k) is not None:
                print(f"{k:<20}{a[k]:>13.5f}{res[k]:>13.5f}{res[k]-a[k]:>+13.5f}")
        print("\nper domain (top-1 agreement):")
        for d in sorted(res["by_domain"], key=lambda x: -res["by_domain"][x]["tokens"]):
            q = res["by_domain"][d]
            pa = a["by_domain"].get(d)
            if not pa or not q.get("sufficient"):
                continue
            print(f"  {d:<9} per-expert {pa['top1_agreement']:.4f}  scalar "
                  f"{q['top1_agreement']:.4f}  delta {q['top1_agreement']-pa['top1_agreement']:+.4f}")
    else:
        print(json.dumps({k: res[k] for k in ("top1_agreement", "dNLL_mean")}, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
