"""Turn artifacts/exp_seqlen/comparison.json into a wiki results page.

Deliberately mechanical. It emits the measured table and the verdict rule stated BEFORE the run
(an S arm at or above the control floor is indistinguishable from resampling noise), and nothing
else. Interpretation is written by hand afterwards, underneath, so that the numbers in the repo
are never entangled with a reading of them that was composed at the same time.
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMP = ROOT / "artifacts" / "exp_seqlen" / "comparison.json"
OUT = ROOT / "wiki" / "93-seqlen-ladder-results.md"


def main() -> None:
    if not COMP.exists():
        print(f"no comparison.json at {COMP}; nothing to publish", file=sys.stderr)
        sys.exit(1)
    rows = json.loads(COMP.read_text())
    if not rows:
        print("comparison.json is empty", file=sys.stderr)
        sys.exit(1)

    ctrl = [r for r in rows if r["arm"].startswith("control")]
    if not ctrl:
        print("no control arm in results -- refusing to publish a table with no noise floor",
              file=sys.stderr)
        sys.exit(1)
    floor = min(r["keep_overlap"] for r in ctrl)
    floor_rho = min(r["spearman"] for r in ctrl)

    arms = []
    for r in rows:
        if r["arm"] not in arms:
            arms.append(r["arm"])

    L = []
    L.append("# 93 — Sequence-length ladder: measured results")
    L.append("")
    L.append(f"*Run {date.today().isoformat()} by `scripts/exp_seqlen_saliency.py`, "
             f"published by `scripts/publish_exp_results.py`. Raw: "
             f"`artifacts/exp_seqlen/comparison.json`.*")
    L.append("")
    L.append("Question, method and the **pre-registered** verdict rule are in "
             "[91-capability-coverage.md](91-capability-coverage.md) §2 and "
             "[92-long-context-program.md](92-long-context-program.md) §2. In short: the same "
             "tokens are run at several S and compared against the S=2048 reference by Spearman "
             "rho over per-expert saliency and by top-144 keep-set overlap — the quantity that "
             "actually decides the mask. A control arm reruns S=2048 on **disjoint** tokens to "
             "establish the noise floor, because \"the arms differ\" is not a result without one.")
    L.append("")
    L.append("## Noise floor (control: same S, disjoint tokens)")
    L.append("")
    L.append(f"- worst keep-overlap: **{floor:.4f}**")
    L.append(f"- worst Spearman rho: **{floor_rho:.4f}**")
    L.append("")
    L.append("**Verdict rule, fixed before the run:** an S arm at or above that keep-overlap is "
             "indistinguishable from resampling noise — S does not move the keep-set and "
             "`MAX_LEN=2048` is vindicated. An arm below it is a real effect.")
    L.append("")
    L.append("## Per-layer")
    L.append("")
    L.append("| arm | layer | attention type | Spearman rho | keep-overlap @144 |")
    L.append("|---|---:|---|---:|---:|")
    for r in rows:
        L.append(f"| {r['arm']} | {r['layer']} | {r['type']} | "
                 f"{r['spearman']:.4f} | {r['keep_overlap']:.4f} |")
    L.append("")
    L.append("## Verdict per arm")
    L.append("")
    L.append("| arm | worst keep-overlap | vs floor | verdict |")
    L.append("|---|---:|---|---|")
    for a in arms:
        if a.startswith("control"):
            continue
        sub = [r for r in rows if r["arm"] == a]
        worst = min(r["keep_overlap"] for r in sub)
        v = "WITHIN noise floor" if worst >= floor else "**BELOW floor — REAL EFFECT**"
        L.append(f"| {a} | {worst:.4f} | {worst - floor:+.4f} | {v} |")
    L.append("")
    L.append("## By attention type")
    L.append("")
    L.append("The hypothesis was that KDA (`linear_attention`) layers, which carry recurrent "
             "state over the whole context, should be more S-sensitive than "
             "`deepseek_sparse_attention` layers, which are capped at `index_topk=2048`.")
    L.append("")
    L.append("| arm | attention type | mean keep-overlap | worst |")
    L.append("|---|---|---:|---:|")
    for a in arms:
        for t in sorted({r["type"] for r in rows}):
            sub = [r for r in rows if r["arm"] == a and r["type"] == t]
            if not sub:
                continue
            m = sum(r["keep_overlap"] for r in sub) / len(sub)
            L.append(f"| {a} | {t} | {m:.4f} | {min(r['keep_overlap'] for r in sub):.4f} |")
    L.append("")
    OUT.write_text("\n".join(L) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
