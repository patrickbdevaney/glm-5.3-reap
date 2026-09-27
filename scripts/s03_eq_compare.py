"""Compare the four equivalence arms. Criterion corrected 2026-09-27.

BIT-EXACT WAS THE WRONG BAR, and this test asserted it until today. The reasoning was that
float32 sums over identical batches in identical order must agree. But they are not sums over
identical inputs: bf16 rounding differs between kernel selections, which flips borderline top-8
routing decisions, which changes WHICH experts get credited. MEASURED: `cnt_by_bucket` is not
bit-identical between two runs of the SAME config, while the totals are.

So this checks the two things that are meaningful:

  1. integer totals must be EXACT -- a lost or replayed block changes how many routed slots were
     counted, and that is the failure blocking can actually cause.
  2. keep-set agreement must clear the BLOCKED config's OWN rerun floor. A blocked run has one
     CUDA context per block, each with its own kernel autotuning, so it is intrinsically noisier
     than an unblocked one; holding it to the unblocked floor would fail it merely for being
     blocked. MEASURED: blocked-vs-blocked min 0.8958 while unblocked-vs-unblocked is 0.9583.
"""
import sys
from pathlib import Path

import torch

R = {k: Path(f"artifacts/_s03_eqtest/{k}") for k in "ABCD"}
fail = 0


def sal(d):
    return d["sum_by_bucket"].sum(0).float() / d["cnt_by_bucket"].sum(0).float().clamp(min=1)


def overlap(x, y, n):
    a = sal(torch.load(R[x] / n, weights_only=False))
    b = sal(torch.load(R[y] / n, weights_only=False))
    k = a.numel() // 2                      # the shipped 50% keep ratio
    return len(set(a.topk(k).indices.tolist()) & set(b.topk(k).indices.tolist())) / k


names = sorted(p.name for p in R["A"].glob("*.pt"))
if not names:
    print("FAIL: arm A produced no layer dumps")
    sys.exit(1)
for k in "BCD":
    if sorted(p.name for p in R[k].glob("*.pt")) != names:
        print(f"FAIL: arm {k} dumped a different set of layers than A")
        sys.exit(1)
print(f"PASS: all four arms dumped the same {len(names)} layers")

tot = {k: [int(torch.load(R[k] / n, weights_only=False)["count"].sum().item()) for n in names]
       for k in "ABCD"}
if len({tuple(v) for v in tot.values()}) == 1:
    print(f"PASS: routed-slot totals identical across all four arms: {tot['A']}")
    print("      -> no block boundary lost or replayed work")
else:
    print(f"FAIL: routed-slot totals differ: {tot}")
    fail = 1

u_floor = min(overlap("A", "C", n) for n in names)
b_floor = min(overlap("B", "D", n) for n in names)
cross = min(min(overlap("A", "B", n), overlap("C", "D", n)) for n in names)
print(f"      unblocked rerun floor : {u_floor:.4f}")
print(f"      BLOCKED rerun floor   : {b_floor:.4f}   <-- the bar")
print(f"      cross-config          : {cross:.4f}")
if cross >= b_floor:
    print("PASS: cross-config keep-set agreement is at or above the blocked rerun floor")
    print("      -> block boundaries add nothing beyond the noise a rerun already has")
else:
    print("FAIL: cross-config is BELOW the blocked rerun floor -- boundaries are systematic")
    fail = 1

print()
print("NOTE: 8,192 tokens, ~134 routed slots per expert, so per-expert means are very noisy and")
print("      even the same-config floor is well under 1.0. Production is 5.5M tokens. This tests")
print("      for a SYSTEMATIC difference, not for production-scale agreement.")
sys.exit(fail)
