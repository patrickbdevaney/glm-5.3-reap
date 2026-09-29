"""Repair layer 3's accumulator, which counted chunk 8 three times instead of once.

THE DAMAGE (2026-09-29). Every MoE layer's `count` was uniform at 15,633,048 (chunks 0-7)
except layer 3, at 21,564,088 = 1.379x. Layer 3 was swept three times for chunk 8 during the
retry storm that preceded the durable-ledger fix: each re-sweep called load_accumulators(),
added its contribution, and dumped, so the running total absorbed chunk 8 three times.

WHY THE SNAPSHOT CANNOT FIX IT. artifacts/saliency_snapshots/chunk_007 is a dump_light: 5 of the
17 fields (layer, sum_saliency, count, sum_by_bucket, cnt_by_bucket). s04_sweep reads all 17 --
f_sum, out_sum, gate_*, norm_*, sq_by_bucket, hist. Restoring from it would silently drop twelve.

THE REPAIR. With `full` the value after the run and `c8` chunk 8's true layer-3 contribution:

    full      = base(chunks 0-7) + 3*c8 + c9
    corrected = base              +   c8 + c9  =  full - 2*c8

c8 is recomputed here in isolation: prepare chunk 8, sweep layers 0-2 to position the states at
layer 3, then sweep layer 3 alone against ZEROED accumulators. That yields a pure chunk-8 delta
for every field, so the subtraction is exact in all 17 rather than just the 5 the snapshot kept.

Residual error is floating-point only: the recomputed c8 can differ from the original in the last
bits of a GPU reduction. That is many orders below a whole extra chunk of weight.

Run AFTER s03 completes and BEFORE s04_sweep builds the mask.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SAL = ROOT / "artifacts" / "saliency"
WORK = ROOT / "artifacts" / "_layer3_repair"
LAYER = 3
CHUNK = 8
NAME = "model__language_model__layers__3__mlp.pt"


def _expected_uniform() -> float:
    """The count every other MoE layer agrees on, as a sanity anchor."""
    import re, glob
    vals = []
    for f in glob.glob(str(SAL / "*.pt")):
        i = int(re.search(r"layers__(\d+)__", f).group(1))
        if i != LAYER:
            vals.append(torch.load(f, weights_only=False)["count"].float().sum().item())
    vals.sort()
    return vals[len(vals) // 2]


def main() -> int:
    if not (SAL / NAME).exists():
        print(f"missing {SAL/NAME}"); return 1
    full = torch.load(SAL / NAME, weights_only=False)
    median = _expected_uniform()
    print(f"layer {LAYER} count      : {full['count'].float().sum().item():,.0f}")
    print(f"median other MoE layer  : {median:,.0f}")
    if full["count"].float().sum().item() <= median * 1.02:
        print("layer 3 is already in line with its peers -- nothing to repair.")
        return 0

    WORK.mkdir(parents=True, exist_ok=True)
    st, sal, ledger = WORK / "st", WORK / "sal", WORK / "ledger.json"
    for d in (st, sal):
        shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
    ledger.write_text(json.dumps({"done": []}))

    # Zeroed accumulators: copy the real dumps and zero every tensor, so the sweep's output IS
    # the contribution rather than a running total. Structure must match what the code expects,
    # which is why we zero real files instead of synthesising them.
    for f in SAL.glob("*.pt"):
        d = torch.load(f, weights_only=False)
        for k, v in d.items():
            if torch.is_tensor(v):
                d[k] = torch.zeros_like(v)
        torch.save(d, sal / f.name)
    print(f"zeroed {len(list(sal.glob('*.pt')))} accumulators in {sal}")

    env = dict(os.environ, S03_ROLE="worker", S03_CHUNK=str(CHUNK),
               S03_STATES_DIR=str(st), S03_SALIENCY_DIR=str(sal),
               S03_BLOCK_LEDGER=str(ledger))
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    worker = str(ROOT / "scripts" / "s03_worker_main.py")

    def run(phase=None, lo=0, hi=0, label=""):
        e = dict(env, S03_LO=str(lo), S03_HI=str(hi))
        if phase:
            e["S03_PHASE"] = phase
        print(f"  -> {label}", flush=True)
        rc = subprocess.run([sys.executable, worker], env=e, cwd=str(ROOT)).returncode
        if rc != 0:
            raise RuntimeError(f"{label} failed rc={rc}")

    run(phase="prepare", label=f"prepare chunk {CHUNK}")
    for li in range(LAYER):
        run(lo=li, hi=li + 1, label=f"position: sweep layer {li}")
    run(lo=LAYER, hi=LAYER + 1, label=f"measure: sweep layer {LAYER} against zeros")

    c8 = torch.load(sal / NAME, weights_only=False)
    print(f"\nchunk {CHUNK} layer {LAYER} contribution: count={c8['count'].float().sum().item():,.0f}")

    out = {}
    for k, v in full.items():
        if torch.is_tensor(v) and k in c8 and torch.is_tensor(c8[k]) and v.shape == c8[k].shape:
            out[k] = v - 2 * c8[k]
        else:
            out[k] = v
    newc = out["count"].float().sum().item()
    print(f"corrected count         : {newc:,.0f}  (target ~{median:,.0f} x chunks-swept)")
    if newc < 0 or newc > full["count"].float().sum().item():
        print("REFUSING: corrected count is not between zero and the original."); return 1

    shutil.copy2(SAL / NAME, SAL / (NAME + ".corrupt"))
    torch.save(out, SAL / NAME)
    print(f"\nwrote repaired {NAME}; original kept as {NAME}.corrupt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
