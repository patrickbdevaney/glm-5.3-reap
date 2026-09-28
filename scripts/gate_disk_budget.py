"""Will this run fit on the disk, at its PEAK, not at its start?

Written 2026-09-28 after s03_saliency filled a 936 GB disk to 100% and died with
`OSError: [Errno 28] No space left on device`, taking sqlite down with it ("database or disk is
full"). The cause was a per-chunk scratch filename -- `chunk_{CI:03d}.pt` -- so ten chunks meant
ten ~17 GB files instead of one reused file. The comment beside it said the file was
"OVERWRITTEN rather than deleted"; the filename made that false.

The memory lookahead (mem_lookahead.py) exists because memory ceilings were being discovered by
hitting them. Disk is the same class of problem and had no equivalent, which is how a 132 GB
scratch directory grew unnoticed next to a 308 GB source tree.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GIB = 2 ** 30


def free_gib(p=ROOT) -> float:
    import shutil
    return shutil.disk_usage(p).free / GIB


def dir_gib(p: Path) -> float:
    if not p.exists():
        return 0.0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / GIB


# Every scratch path a run can grow, with the size it is ALLOWED to reach.
# A path whose actual size exceeds its budget is a leak, not a big file.
BUDGETS = {
    "artifacts/s03_states": (20.0, "one chunk's hidden states, reused across that chunk's blocks"),
    "artifacts/teacher_routing": (2.0, "top-8 ids, int16: 0.66 GiB at 1M tokens"),
    "artifacts/saliency": (10.0, "per-layer accumulators, 45 layers"),
    "artifacts/router_cache": (30.0, "per-chunk router score cache"),
}


def main() -> int:
    fail = 0
    print(f"free on {ROOT.anchor}: {free_gib():.1f} GiB\n")
    print(f"{'path':34s} {'actual':>9} {'budget':>9}  note")
    for rel, (budget, note) in BUDGETS.items():
        actual = dir_gib(ROOT / rel)
        flag = "" if actual <= budget else "   <-- OVER BUDGET"
        if actual > budget:
            fail = 1
        print(f"{rel:34s} {actual:8.1f}G {budget:8.1f}G  {note}{flag}")
    print()
    if free_gib() < 40:
        print(f"FAIL: only {free_gib():.1f} GiB free; a run needs headroom for one states file "
              f"(~17 GiB) plus artifacts")
        fail = 1
    else:
        print(f"PASS: {free_gib():.1f} GiB free")
    if fail:
        print("\nA path over budget means something is accumulating that should be reused.")
    return fail


if __name__ == "__main__":
    sys.exit(main())
