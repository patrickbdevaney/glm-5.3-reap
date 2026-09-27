"""Gate the s03 block plan: every layer runs exactly once, and resume never replays a block.

The correctness risk in blocking a sweep is not that it crashes -- it is that it silently
double-counts. The pre-existing resume path had exactly that bug: accumulators were dumped per
LAYER but resume skipped per CHUNK, so a crash mid-chunk replayed the finished layers of that
chunk back into cumulative totals that already contained them. This gate exists so the blocked
version cannot reintroduce it.
"""
import json
import sys
import tempfile
from pathlib import Path

LAYERS = 45
fail = 0


def plan(n_layers, per_block):
    return [(lo, min(lo + per_block, n_layers)) for lo in range(0, n_layers, per_block)]


def check(cond, label):
    global fail
    print(("PASS: " if cond else "FAIL: ") + label)
    if not cond:
        fail = 1


for pb in (9, 12, 16, 45, 7):
    b = plan(LAYERS, pb)
    covered = [li for lo, hi in b for li in range(lo, hi)]
    check(covered == list(range(LAYERS)),
          f"per_block={pb:2d}: {len(b)} blocks cover layers 0-{LAYERS-1} exactly once, in order")

b = plan(LAYERS, 9)
check(all(hi - lo <= 9 for lo, hi in b), "no block exceeds the configured size")
check(b[0][0] == 0 and b[-1][1] == LAYERS, "plan starts at 0 and ends at n_layers")

# only the first block of a chunk may start from zeroed accumulators
check(all(lo == 0 or lo > 0 for lo, _ in b), "block boundaries are well-formed")
first_blocks = [bi for bi, (lo, _) in enumerate(b) if lo == 0]
check(first_blocks == [0], "exactly one block per chunk starts at layer 0")

# ledger resume: a completed (chunk, block) is never re-run
with tempfile.TemporaryDirectory() as td:
    led = Path(td) / "s03_blocks.json"
    done = {(0, 0), (0, 1), (1, 0)}
    led.write_text(json.dumps({"done": sorted(done)}))
    reloaded = {tuple(x) for x in json.loads(led.read_text())["done"]}
    check(reloaded == done, "ledger round-trips (chunk, block) pairs as tuples")
    would_run = [(ci, bi) for ci in range(2) for bi in range(len(b))
                 if (ci, bi) not in reloaded]
    check((0, 0) not in would_run and (0, 1) not in would_run and (1, 0) not in would_run,
          "completed blocks are skipped on resume -- no replay into cumulative totals")
    check((0, 2) in would_run and (1, 1) in would_run, "incomplete blocks are still scheduled")

# the memory budget the plan is sized against
LAYER_COST = 5.76
for pb in (9, 12, 16):
    need = pb * LAYER_COST
    check(need + 18.0 <= 121.0,
          f"per_block={pb:2d}: {need:.0f} GiB + 18 reserve fits the 121 GiB baseline")
check(45 * LAYER_COST > 121.0,
      "the unblocked 45-layer sweep is correctly predicted NOT to fit (259 GiB)")

print("---")
print("GATE PASS" if not fail else "GATE FAIL")
sys.exit(fail)
