"""Fold measured evaluation results into an already-emitted model card.

`s06_emit` runs BEFORE `s09_eval` - it has to, since the eval scores the emitted checkpoint - so
the card it writes says "this checkpoint has not been evaluated". By upload time that is false,
and shipping it would understate the artifact in the one place a reader looks first.

This rewrites that section in place from `artifacts/eval/eval_<name>.json`. It is idempotent: the
section is delimited, so re-running replaces rather than accumulates. If no eval exists for the
named output the card is left exactly as it was - an unevaluated card is honest, a card claiming
numbers it does not have is not.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "artifacts" / "eval"

BEGIN = "<!-- EVAL:BEGIN -->"
END = "<!-- EVAL:END -->"
PLACEHOLDER = "**This checkpoint has not been evaluated.**"


def section(ev: dict, name: str) -> str:
    by = ev.get("by_domain", {})
    rows = []
    for b, r in sorted(by.items(), key=lambda x: -x[1]["tokens"]):
        if r.get("sufficient", True):
            rows.append(f"| {b} | {r['tokens']:,} | **{r['top1_agreement']:.3f}** | "
                        f"{r['dNLL_mean']:+.3f} |")
        else:
            rows.append(f"| {b} | {r['tokens']:,} | *n too small* | *n too small* |")
    tbl = "\n".join(rows)
    sup = ev.get("superseded") or {}
    note = ""
    if isinstance(sup.get("top1_agreement"), float):
        note = (f"\n\nAn earlier run of this evaluation reported top-1 agreement "
                f"{sup['top1_agreement']:.3f}. That figure scored image-placeholder positions, "
                f"which the training objective masks out; it is superseded and the corrected "
                f"number is above. See `wiki/97-evaluation.md`.")
    return f"""{BEGIN}
## Measured against the unpruned teacher

Teacher-forced paired evaluation on **{ev['tokens']:,} held-out tokens** the calibration never
saw, scored against unpruned `zai-org/GLM-5.3-Flash` on identical inputs.

| | |
|---|---|
| **Top-1 agreement** | **{ev['top1_agreement']:.3f}** |
| ΔNLL (student − teacher), mean | {ev['dNLL_mean']:+.4f} |
| ΔNLL, median | {ev.get('dNLL_p50', float('nan')):+.4f} |
| Top-k KL (teacher ‖ student) | {ev.get('topk_KL', float('nan')):.4f} |
| Teacher NLL / student NLL | {ev['teacher_nll']:.3f} / {ev['student_nll']:.3f} |

| domain | tokens | top-1 agreement | ΔNLL |
|---|---|---|---|
{tbl}

`1 − flip_rate` against the unpruned model is the same quantity aggressively-quantised releases
quote as "retains X% of top-1 accuracy", so this number is comparable to a heavily-quantised GGUF
of the same base — provided both are measured against that same unpruned reference.

**Image-placeholder positions are excluded**, as the training objective excludes them. After that
exclusion the vision bucket is too small to support a conclusion: vision is **unmeasured here**,
which is not the same as measured-and-fine. Its per-domain saliency retention (0.682, mid-pack of
seven domains) says the mask did not strip vision-serving experts; whether the survivors suffice
is untested.

**This is teacher-forced agreement, not capability.** It measures how far the student moved from
the teacher on ground-truth prefixes. It does not measure whether the model is *smart* — only
generative benchmarks do, and none has been run.{note}
{END}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="output/<name> directory to update")
    a = ap.parse_args()
    card = ROOT / "output" / a.name / "README.md"
    ep = EVAL / f"eval_{a.name}.json"
    if not card.exists():
        raise SystemExit(f"no card at {card}")
    if not ep.exists():
        print(f"no evaluation for {a.name}; leaving the card unevaluated (which is honest)")
        return
    ev = json.loads(ep.read_text())
    body = section(ev, a.name)
    txt = card.read_text()
    if BEGIN in txt and END in txt:
        pre, rest = txt.split(BEGIN, 1)
        txt = pre + body + rest.split(END, 1)[1]
    elif PLACEHOLDER in txt:
        # Replace the whole "not evaluated" paragraph up to the next heading.
        i = txt.index(PLACEHOLDER)
        j = txt.index("\n## ", i)
        txt = txt[:i] + body + "\n\n" + txt[j + 1:]
    else:
        txt = txt.rstrip() + "\n\n" + body + "\n"
    card.write_text(txt)
    print(f"updated {card} with top-1 agreement {ev['top1_agreement']:.4f} "
          f"over {ev['tokens']:,} tokens")


if __name__ == "__main__":
    main()
