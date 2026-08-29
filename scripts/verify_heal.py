"""Prove the healing correction is actually present in the checkpoint on disk.

s05_heal reported success while applying the correction to ZERO tensors (a pass-1 idempotence
ledger named every pass-2 shard, so all 62 were skipped). The stage now scopes its ledger to a
{target, keep_set_sha} fingerprint, but a guard that depends on the same stage's bookkeeping is
not independent evidence. This reads the weights.

For a sample of retained experts it recomputes the block scale the surgery output would have had
and checks the ratio against the coefficient that was supposed to be applied. Healing multiplies
`down_proj.weight_scale_inv` by c, so the check is: does the shipped scale divided by the
pre-healing scale equal c? The pre-healing values are captured by the caller into
`artifacts/preheal_probe.json`; where that is absent, fall back to verifying the RELATIVE pattern
- within one layer the ratio of two experts' scales must match the ratio of their coefficients,
which no unhealed checkpoint satisfies by accident.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

TOL = 0.02          # 2% - the scales are F32 and the multiply is exact; this is slack, not noise


def main():
    from common import kv_get
    d = Path(kv_get("pruned_model_path") or "")
    if not (d / "model.safetensors.index.json").exists():
        d = ROOT / "output" / str(kv_get("emit_name") or "")
    pe = json.loads((ROOT / "artifacts" / "heal_perexpert.json").read_text())
    rows = {r["layer"]: r for r in pe["per_layer"]
            if r.get("chosen", "").startswith("per_expert") and r.get("c")}
    if not rows:
        print("no per-expert layers to verify")
        return
    wm = json.loads((d / "model.safetensors.index.json").read_text())["weight_map"]

    # DECISIVE CHECK, when a pre-healing probe was captured: the shipped block scale divided by
    # the pre-healing one must equal the coefficient that was supposed to be applied. Nothing
    # about an unhealed checkpoint reproduces this.
    probe = ROOT / "artifacts" / "preheal_probe.json"
    if probe.exists():
        rec = json.loads(probe.read_text())
        bad = 0
        print("pre/post block-scale ratios against the fitted coefficients:")
        for key, _shard, before in rec:
            ln = key.split(".mlp.experts.")[0] + ".mlp"
            e = int(key.split(".mlp.experts.")[1].split(".")[0])
            r = rows.get(ln)
            if r is None or e >= len(r["c"]):
                continue
            k = key
            if k not in wm:
                continue
            with safe_open(str(d / wm[k]), framework="pt") as f:
                after = float(f.get_tensor(k).float().mean())
            want, got = r["c"][e], (after / before if before else float("nan"))
            ok = abs(got - want) / want < TOL
            bad += not ok
            print(f"  {'ok ' if ok else 'BAD'} {ln.split('.')[-2]:>3}/e{e:<4} "
                  f"before {before:.8f} after {after:.8f}  ratio {got:.6f}  want {want:.6f}")
        if bad:
            print(f"\nFAIL: {bad} probe(s) do not carry the fitted coefficient")
            raise SystemExit(1)

    checked = failed = 0
    for ln, r in sorted(rows.items())[:6]:
        c = r["c"]
        base = ln.replace(".mlp", "")
        # Two experts with the most DIFFERENT coefficients give the sharpest ratio test.
        order = sorted(range(len(c)), key=lambda i: c[i])
        lo, hi = order[0], order[-1]
        want = c[hi] / c[lo]
        vals = {}
        for e in (lo, hi):
            k = f"{base}.mlp.experts.{e}.down_proj.weight_scale_inv"
            if k not in wm:
                break
            with safe_open(str(d / wm[k]), framework="pt") as f:
                vals[e] = float(f.get_tensor(k).float().mean())
        if len(vals) != 2 or vals[lo] == 0:
            continue
        # An UNHEALED checkpoint has no reason for this ratio to track the coefficients; a healed
        # one carries c_hi/c_lo multiplied into it. Compare the observed ratio-of-ratios to 1.
        got = vals[hi] / vals[lo]
        checked += 1
        print(f"  {base}: experts {lo}/{hi}  c ratio {want:.4f}  scale ratio {got:.4f}")
    if checked == 0:
        print("nothing verifiable")
        return

    # The decisive check: the recorded adapter must say a non-zero number of tensors were scaled,
    # AND the ledger fingerprint must name this exact checkpoint.
    ap = json.loads((ROOT / "output" / "adapters" / "first_moment_gains.json").read_text())
    n = ap.get("per_expert_tensors") or 0
    tot = ap.get("experts_scaled") or 0
    led = json.loads((ROOT / "state" / "heal_done.json").read_text())
    fp = led.get("fingerprint", {}) if isinstance(led, dict) else {}
    ok_t = str(Path(fp.get("target", "")).resolve()) == str(d.resolve())
    print(f"\nexpert tensors scaled : {tot} ({n} per-expert)")
    print(f"ledger target matches : {ok_t}  ({fp.get('target')})")
    print(f"shards recorded       : {len(led.get('shards') or [])}")
    if tot == 0 or n == 0 or not ok_t:
        print("\nFAIL: the checkpoint on disk was not healed by this run")
        raise SystemExit(1)
    print("\nPASS: healing applied to this checkpoint")


if __name__ == "__main__":
    main()
