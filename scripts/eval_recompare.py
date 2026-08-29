"""Recompute s09_eval's metrics from the CACHED captures, without re-running the scoring passes.

WHY THIS EXISTS
---------------
`s09_eval` originally masked scored positions with `gold != pad_token_id` alone. For an image-text
record that keeps every image-placeholder position in the loss - and those are exactly the
positions whose *embedding was replaced by an image feature* before layer 0. The model is asked to
predict the literal placeholder token id from a hidden state that is a picture, at positions the
training loss masks out. It is not a hard question; it is a meaningless one.

MEASURED 2026-08-28 on the pass-1 baseline capture:

                              n        teacher NLL   student NLL     dNLL      flip
    vision, placeholder    96 597          16.94         15.19      -1.752     0.732
    vision, real text         532          10.45          8.49      -1.958     0.547
    everything else       240 984           0.997         1.200     +0.203     0.162

Those 96,597 positions were 99.5% of the vision bucket and 29% of every scored token, and they
dragged the headline figures with them:

    dNLL_mean       -0.359  ->  +0.198
    top1_agreement   0.674  ->   0.837
    teacher_nll      5.567  ->   1.018
    topk_KL          4.108  ->   0.695

A pruned student appearing to BEAT its teacher is the tell. That cannot happen, and the sign was
the only reason to look.

The fix lives in `s09_eval.compare`, which now rebuilds the gold-token array (deterministically,
from the same `load_heldout`) and drops those positions. Crucially the *capture* mask is
unchanged, so the cached teacher - 93 minutes of compute whose source `s04b_surgery` deletes -
stays positionally alignable with every student capture. Narrowing the mask at capture time would
have made a fresh student capture (241,516 tokens) compare against the cached teacher (338,113)
by `[:n]` truncation: the same silent-misalignment failure in a new costume.

This script is a thin re-runner over the cached captures so an already-completed evaluation can be
corrected in place. It deliberately calls `s09_eval.compare` rather than reimplementing it; two
copies of this arithmetic is exactly how the corrected number and the shipped number drift apart.

A SECOND, honest finding it exposes: after excluding placeholders the vision bucket is 532 tokens
across 28 records, because the held-out image-text records average ~3,450 placeholder positions
against ~19 real text tokens. That is far too few to conclude anything. R3 - vision capability -
is UNMEASURED by this evaluation, which is a different statement from measured-and-fine, and is
reported as such.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

EVAL = ROOT / "artifacts" / "eval"
OUT = ROOT / "artifacts" / "s09_eval.json"
RAW = ROOT / "artifacts" / "s09_eval.raw.json"


def main():
    import stages.s09_eval as EV
    tf = EVAL / "teacher.pt"
    students = sorted(EVAL.glob("student_*.pt"))
    if not tf.exists() or not students:
        raise SystemExit("no cached captures to recompare")
    T = torch.load(tf, weights_only=False)

    if OUT.exists() and not RAW.exists():
        RAW.write_text(OUT.read_text())
    prev = json.loads(RAW.read_text()) if RAW.exists() else {}

    for sp in students:
        name = sp.stem.replace("student_", "")
        S = torch.load(sp, weights_only=False)
        res = EV.compare(T, S)
        res["student"] = name
        res["teacher_from_cache"] = True
        res["correction"] = ("image/video placeholder positions excluded from the loss, as the "
                             "training objective does; see scripts/eval_recompare.py")
        res["superseded"] = {k: prev.get(k) for k in
                             ("tokens", "dNLL_mean", "top1_agreement", "teacher_nll",
                              "student_nll", "topk_KL")}
        OUT.write_text(json.dumps(res, indent=2))
        (EVAL / f"eval_{name}.json").write_text(json.dumps(res, indent=2))

        print(f"student: {name}")
        print(f"  scored {res['tokens']} of {res['tokens_captured']} captured "
              f"({res['excluded_placeholder_tokens']} placeholder positions excluded)")
        for k, fmt in (("dNLL_mean", "+.4f"), ("top1_agreement", ".4f"),
                       ("teacher_nll", ".4f"), ("topk_KL", ".4f")):
            o = prev.get(k)
            o = format(o, fmt) if isinstance(o, (int, float)) else "n/a"
            print(f"  {k:<15} {o:>9}  ->  {format(res[k], fmt)}")
        print("  by domain:")
        for b, r in sorted(res["by_domain"].items(), key=lambda x: -x[1]["tokens"]):
            warn = "" if r["sufficient"] else "   <-- too few tokens to conclude anything"
            print(f"    {b:<9} n={r['tokens']:>6}  dNLL {r['dNLL_mean']:+.4f}  "
                  f"top1 {r['top1_agreement']:.4f}{warn}")
        print(f"  wrote {OUT.name} and eval/eval_{name}.json "
              f"(uncorrected run preserved at {RAW.name})")


if __name__ == "__main__":
    main()
