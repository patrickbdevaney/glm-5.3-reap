"""Paired (McNemar) test of scalar vs per-expert healing, on the identical token set.

The headline +0.00545 was quoted against an UNPAIRED binomial s.e. of 0.00075. That is the
conservative bound and it was already decisive, but both arms score the same 241,516 tokens
against the same cached teacher, so the paired test is the correct one and is strictly sharper:
tokens where both arms agree with the teacher, or where both disagree, carry no information about
which arm is better and should not be in the denominator.

    b = tokens the PER-EXPERT arm gets right and the SCALAR arm does not
    c = tokens the SCALAR arm gets right and the PER-EXPERT arm does not
    chi2 = (|b - c| - 1)^2 / (b + c)          (Edwards continuity correction)

Run after the corrected eval has written its capture; `revert_chain.sh` preserves the per-expert
capture as `student_pass2_fp8_perexpert_healed.pt` first, because s09_eval writes the corrected
one to the same path.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
E = ROOT / "artifacts" / "eval"


def main():
    import stages.s09_eval as EV

    T = torch.load(E / "teacher.pt", weights_only=False)
    A = torch.load(E / "student_pass2_fp8_perexpert_healed.pt", weights_only=False)   # per-expert
    B = torch.load(E / "student_glm-5.3-flash-reap50-fp8-pass2.pt", weights_only=False)  # scalar

    n = min(T["nll"].numel(), A["nll"].numel(), B["nll"].numel())
    keep = torch.ones(n, dtype=torch.bool)
    gold, skip, _sl, _nt = EV.gold_tokens()
    g = gold[:n]
    for sid in skip:
        keep &= g != sid

    t = T["argmax"][:n][keep]
    a = (A["argmax"][:n][keep] == t)
    b_ = (B["argmax"][:n][keep] == t)

    b = int((a & ~b_).sum())      # per-expert right, scalar wrong
    c = int((b_ & ~a).sum())      # scalar right, per-expert wrong
    both = int((a & b_).sum())
    neither = int((~a & ~b_).sum())
    N = int(keep.sum())

    chi2 = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) else 0.0
    z = math.sqrt(chi2)
    # Paired s.e. of the difference in proportions is sqrt(b+c)/N.
    se = math.sqrt(b + c) / N
    delta = (c - b) / N

    res = {
        "n_scored": N, "both_agree": both, "neither_agrees": neither,
        "per_expert_only": b, "scalar_only": c, "discordant": b + c,
        "delta_top1_scalar_minus_perexpert": delta,
        "paired_se": se, "mcnemar_chi2": chi2, "z": z,
        "unpaired_se_for_comparison": math.sqrt(0.84 * 0.16 / N),
    }
    (E / "healing_paired_test.json").write_text(json.dumps(res, indent=2))

    print(f"scored tokens            : {N}")
    print(f"both arms agree w/teacher: {both}   neither: {neither}")
    print(f"per-expert only right    : {b}")
    print(f"scalar only right        : {c}")
    print(f"discordant pairs         : {b + c}  ({(b+c)/N:.2%} of tokens carry the signal)")
    print(f"delta (scalar - perexp)  : {delta:+.5f}")
    print(f"paired s.e.              : {se:.5f}   (unpaired was {res['unpaired_se_for_comparison']:.5f})")
    print(f"McNemar chi2             : {chi2:.1f}   z = {z:.1f}")
    print(f"\nwrote {E / 'healing_paired_test.json'}")


if __name__ == "__main__":
    main()
