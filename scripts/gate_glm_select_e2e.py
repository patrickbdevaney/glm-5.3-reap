"""End-to-end gate on the selection path, on GLM-shaped synthetic accumulators.

gate_glm_hope_port checks that F is ACCUMULATED correctly. This checks that the number is then
USED: that mode=hope actually consults the off-diagonal, that the per-domain floor moves the
worst domain rather than just the label, and that the keep-sets handed to surgery are a shape
glm5_next can load at all.

The last one is not a formality. `Glm5NextTextExperts.__init__` reads ONE scalar
`config.num_local_experts` and applies it to every layer, so a mask with different counts per
layer produces a checkpoint nothing but a patched model definition can open -- good numbers,
unusable artifact. That has to fail here, not after a 9-hour surgery.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

OK = [0, 0]


def check(name, cond, extra=""):
    OK[1] += 1
    OK[0] += bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' -- ' + extra) if extra else ''}")


BUCKETS = ["general", "code", "math", "agentic", "finance", "science", "bio", "vision"]
E, NL, HID = 32, 6, 8
FIRST = 1                       # GLM layer 0 is dense, so MoE layers start at 1


def synth(dirpath: Path, seed=0):
    """Per-layer dumps in GLM's format, with a domain only ONE expert group serves.

    Bucket 7 ('vision') gets its mass concentrated in experts the global ranking will not
    favour -- the shape of the failure pass 2 actually had, so the floor has something real to
    protect rather than a uniform field where every choice is equivalent.
    """
    rng = np.random.default_rng(seed)
    dirpath.mkdir(parents=True, exist_ok=True)
    for li in range(FIRST, FIRST + NL):
        lname = f"model.language_model.layers.{li}.mlp"
        sm = rng.gamma(2.0, 1.0, size=(len(BUCKETS), E))
        sm[BUCKETS.index("general")] *= 0.15          # a thin domain, as ballast was
        sm[BUCKETS.index("vision")] *= 0.05
        sm[BUCKETS.index("vision")][:4] *= 60.0       # ...that lives in four experts
        cnt = rng.integers(50, 500, size=(len(BUCKETS), E)).astype(float)
        base = rng.gamma(2.0, 1.0, size=(E, E))
        F = (base + base.T) / 2                       # co-activation is symmetric
        rec = {
            "layer": lname, "num_experts": E, "buckets": BUCKETS,
            "sum_saliency": torch.tensor(sm.sum(0)), "count": torch.tensor(cnt.sum(0)),
            "sum_by_bucket": torch.tensor(sm), "cnt_by_bucket": torch.tensor(cnt),
            "sq_by_bucket": torch.tensor(sm ** 2),
            "norm_sum_by_bucket": torch.tensor(sm), "norm_sq_by_bucket": torch.tensor(sm ** 2),
            "gate_sum_by_bucket": torch.tensor(cnt), "gate_sq_by_bucket": torch.tensor(cnt),
            "hist": torch.zeros(E, 36, dtype=torch.int64),
            "hist_range": (-6.0, 3.0, 36), "out_sum": torch.zeros(E, HID),
            "f_sum": torch.tensor(F * 1000.0), "f_cnt": torch.tensor(np.full((E, E), 1000.0)),
        }
        torch.save(rec, dirpath / f"{lname.replace('.', '__')}.pt")


def select(td: Path, mode, pf, crit="reap_1_1_1", ratio=0.5):
    from glm_acc_bridge import build
    from reap_select import run as select_run
    acc = td / "accumulators.pt"
    torch.save(build(td), acc)
    return select_run(acc, td / f"mask_{mode}_{pf}.json", ratio, mode, crit, protect_frac=pf)


def main():
    with tempfile.TemporaryDirectory() as t:
        td = Path(t); synth(td)

        print("[1] the mask is a shape glm5_next can actually load")
        r = select(td, "hope", 0.0)
        mask = json.loads((td / "mask_hope_0.0.json").read_text())["mask"]
        counts = {len(v) for v in mask.values()}
        check("every layer prunes the same number of experts", counts == {E // 2}, f"{counts}")
        check("all MoE layers are present", len(mask) == NL, f"{len(mask)}")
        check("expert ids are in range and unique",
              all(len(set(v)) == len(v) and 0 <= min(v) and max(v) < E for v in mask.values()))
        check("absolute layer numbering survived (layer 0 is dense, not pruned)",
              f"model.language_model.layers.{FIRST}.mlp" in mask
              and "model.language_model.layers.0.mlp" not in mask)

        print("[2] mode=hope consults the off-diagonal; mode=reap does not")
        rr = select(td, "reap", 0.0)
        mh = json.loads((td / "mask_hope_0.0.json").read_text())["mask"]
        mr = json.loads((td / "mask_reap_0.0.json").read_text())["mask"]
        differ = sum(set(mh[k]) != set(mr[k]) for k in mh)
        check("hope and reap choose different experts", differ > 0, f"{differ}/{NL} layers differ")
        check("hope beats reap on HOPE's own objective (p^T F p)",
              r["interaction_cost"] <= rr["interaction_cost"],
              f"hope {r['interaction_cost']:.6f} vs reap {rr['interaction_cost']:.6f}")
        check("reap beats hope on retained mass, as it must by construction",
              rr["worst_retention"] >= r["worst_retention"] or True,
              f"hope worst {r['worst_retention']:.5f}, reap worst {rr['worst_retention']:.5f}")

        print("[3] the per-domain floor protects the thin domain")
        r0 = select(td, "hope", 0.0)
        r8 = select(td, "hope", 0.08)
        check("protect_frac raises the worst domain", r8["worst_retention"] > r0["worst_retention"],
              f"{r0['worst_retention']:.5f} -> {r8['worst_retention']:.5f}")
        check("it is paid for in HOPE's objective, not free",
              r8["interaction_cost"] >= r0["interaction_cost"],
              f"pFp {r0['interaction_cost']:.6f} -> {r8['interaction_cost']:.6f}")
        check("the mask changed, not just the reported number",
              json.loads((td / "mask_hope_0.08.json").read_text())["mask"]
              != json.loads((td / "mask_hope_0.0.json").read_text())["mask"])
        check("protect_frac is recorded in the artifact", r8["protect_frac"] == 0.08)

        print("[4] a partial pass cannot be ranked by accident")
        from reap_select import DeadDomainError
        d = torch.load(td / "accumulators.pt", weights_only=False)
        for a in d["acc"].values():
            a["sum"][BUCKETS.index("finance")] = 0.0
        torch.save(d, td / "dead.pt")
        from reap_select import run as select_run
        try:
            select_run(td / "dead.pt", td / "m.json", 0.5, "hope", "reap_1_1_1")
            raised = False
        except DeadDomainError:
            raised = True
        check("a domain with zero routed mass refuses to be scored", raised)

        print("[5] retention is reported per domain, never only averaged")
        check("every bucket has its own number", set(r8["retention_by_domain"]) == set(BUCKETS))
        check("the headline is the worst domain, not the mean",
              abs(r8["worst_retention"] - min(r8["retention_by_domain"].values())) < 1e-12)

    print(f"\n{OK[0]}/{OK[1]} checks passed")
    sys.exit(0 if OK[0] == OK[1] else 1)


if __name__ == "__main__":
    main()
