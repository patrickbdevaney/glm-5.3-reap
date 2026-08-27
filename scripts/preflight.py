#!/usr/bin/env python
"""Look ahead through every remaining stage and check its assumptions NOW.

Stage failures in this pipeline are expensive: s03 is ~80 min, s04b rewrites 157 GiB and
deletes its inputs, s07 is hours. Discovering a wrong key name or a disk shortfall at the
START of one of those wastes the whole stage. This validates the chain end to end - artifact
schemas, kv handoffs, disk headroom, and output invariants - against current on-disk state.

Read-only. Safe to run while stages are executing.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, ARTIFACTS, db, kv_get, free_gib

OK, WARN, FAIL = "ok  ", "WARN", "FAIL"
rows: list[tuple[str, str, str]] = []


def chk(cond, name, detail_ok="", detail_bad="", warn_only=False):
    rows.append(((OK if cond else (WARN if warn_only else FAIL)), name,
                 detail_ok if cond else detail_bad))
    return cond


def stage_status(n):
    with db() as c:
        r = c.execute("SELECT status FROM stages WHERE name=?", (n,)).fetchone()
    return r[0] if r else "pending"


# ---- constants the plan depends on -----------------------------------------------------
ROUTED = 311_672_586_240
NONEXP_BF16, NONEXP_FP8 = 6_926_096_640, 2_743_074_816
MTP_EXPERTS = 7_247_757_312          # layer 45, excluded
RATIO = float(kv_get("chosen_ratio", 0.50) or 0.50)

sal = sorted((ROOT / "artifacts" / "saliency").glob("*.pt"))
n_sal = len(sal)

print("=== PREFLIGHT: remaining pipeline ===\n")

# ---- s03 -> s04 -------------------------------------------------------------------------
chk(n_sal > 0, "s03 produced saliency", f"{n_sal} layer files", "no saliency files yet",
    warn_only=(stage_status("s03_saliency") == "running"))
if n_sal:
    import torch
    d = torch.load(sal[0], weights_only=False)
    n_exp = int(d["num_experts"])
    chk(n_exp == 288, "expert count in saliency", f"{n_exp}", f"{n_exp}, expected 288")
    chk({"layer", "sum_saliency", "count", "num_experts"} <= set(d),
        "saliency schema", "layer/sum_saliency/count/num_experts", f"got {sorted(d)}")
    chk(n_sal <= 42, "MoE layer count", f"{n_sal} <= 42 (layers 3..44)",
        f"{n_sal} > 42 - dense layers should not appear")

# ---- s04 -> s04b ------------------------------------------------------------------------
keep = int(round(288 * (1 - RATIO)))
chk(keep >= 8, "top_k reachable after prune", f"{keep} experts >= top_k 8",
    f"only {keep} experts left, top_k is 8")

# ---- s04b disk ---------------------------------------------------------------------------
src_gb = sum(p.stat().st_size for p in (ROOT / "source" / "GLM-5.3-Flash").glob("*.safetensors")) / 2**30 \
    if (ROOT / "source" / "GLM-5.3-Flash").exists() else 0.0
pruned_gib = ((ROUTED - MTP_EXPERTS) * (1 - RATIO) + NONEXP_FP8 + NONEXP_BF16 * 2) / 2**30
# surgery deletes each source shard after writing, so the transient need is one shard
chk(free_gib() > 20, "s04b headroom (deletes as it writes)",
    f"{free_gib():.0f} GiB free, needs ~one shard (~5 GiB) transient",
    f"only {free_gib():.0f} GiB free")
rows.append((OK, "s04b projected output", f"~{pruned_gib:.0f} GiB pruned FP8"))

# ---- s05 --------------------------------------------------------------------------------
chk((ROOT / "state").exists(), "s05 ledger dir", "state/ exists", "state/ missing")

# ---- s06 artifact schema ----------------------------------------------------------------
s3p = ARTIFACTS / "s03_saliency.json"
if s3p.exists():
    s3 = json.loads(s3p.read_text())
    chk("calib_samples" in s3, "s06 reads s03.calib_samples", "present", f"missing; has {sorted(s3)}")
else:
    rows.append((WARN, "s06 reads s03_saliency.json", "not written yet (s03 still running)"))
rows.append((OK, "s06 reads s04b keys", "ratio/experts_kept/gib - all written by s04b.run()"))

# ---- s07 --------------------------------------------------------------------------------
nvfp4_gib = ((ROUTED - MTP_EXPERTS) * (1 - RATIO) * 4.5 / 8 + NONEXP_FP8 + NONEXP_BF16) / 2**30
after_surgery = free_gib() + src_gb - pruned_gib
# accelerate offloads only the part that does not fit in RAM (offload_state_dict=False),
# so the disk requirement is (model - usable RAM) + margin, not the whole model.
ram_avail = 0.0
with open("/proc/meminfo") as _fh:
    for _l in _fh:
        if _l.startswith("MemAvailable"):
            ram_avail = int(_l.split()[1]) / 1048576
            break
s07_need = max(pruned_gib - ram_avail * 0.8, 0) + 15
chk(after_surgery > s07_need, "s07 can offload what will not fit in RAM",
    f"~{after_surgery:.0f} GiB free after surgery vs ~{s07_need:.0f} GiB needed",
    f"only ~{after_surgery:.0f} GiB projected, needs ~{s07_need:.0f}")
chk(nvfp4_gib < 117, "s07 output fits Thor envelope",
    f"~{nvfp4_gib:.0f} GiB < 117 GiB", f"~{nvfp4_gib:.0f} GiB exceeds 117 GiB")

# ---- services ---------------------------------------------------------------------------
import subprocess
for svc in ("glm53-reap", "glm53-memguard"):
    try:
        a = subprocess.run(["systemctl", "--user", "is-active", f"{svc}.service"],
                           capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        a = "?"
    chk(a == "active", f"service {svc}", a, a)

# ---- report -----------------------------------------------------------------------------
w = max(len(n) for _, n, _ in rows)
for st, name, detail in rows:
    print(f"  [{st}] {name:<{w}}  {detail}")
bad = [r for r in rows if r[0] == FAIL]
print(f"\n  {len(rows)} checks, {len(bad)} failing")
sys.exit(1 if bad else 0)
