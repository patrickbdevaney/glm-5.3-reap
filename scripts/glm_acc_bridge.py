"""Fold GLM's per-layer saliency dumps into the single accumulator file reap_select reads.

Two pipelines that were built months apart have to meet somewhere. MiMo's selector is already
model-agnostic -- it touches neither the model nor the corpus, only statistics -- so the meeting
point is the statistics file, and this is the only piece of the port that is GLM-specific.

The formats differ in three ways and nothing else:

    GLM (stream_saliency.dump)          MiMo (reap_select.load_acc)
    one .pt per layer                   one .pt for the run
    "sum_by_bucket", "cnt_by_bucket"    "sum", "cnt"
    f_sum/f_cnt per layer, [E, E]       f_sum/f_cnt for the run, [n_layers, E, E]

WHY NOT JUST RENAME THE KEYS IN stream_saliency. Because the GLM keys are what s04_sweep,
domain_holes.py, heal_refit.py and every pass-1/pass-2 artifact reader already expect, and pass
3 has to stay comparable to pass 2 on exactly those readers. A translation that lives in one
file is cheaper than a rename that reaches into eight.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reap_select import layer_index  # noqa: E402

# GLM dump key -> the name reap_select's criterion table indexes by.
KEYMAP = {
    "sum": "sum_by_bucket", "sq": "sq_by_bucket", "cnt": "cnt_by_bucket",
    "nrm": "norm_sum_by_bucket", "nsq": "norm_sq_by_bucket",
    "gat": "gate_sum_by_bucket", "gsq": "gate_sq_by_bucket",
}


class MissingF(RuntimeError):
    pass


def build(dirpath: Path, require_f: bool = True) -> dict:
    dirpath = Path(dirpath)
    files = sorted(dirpath.glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"no per-layer dumps in {dirpath}")

    acc, f_by_layer, buckets, n_exp = {}, {}, None, None
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        lname = d.get("layer")
        if lname is None or "sum_by_bucket" not in d:
            continue                      # a pass-1 dump; no per-bucket tensors to carry over
        if buckets is None:
            buckets = list(d["buckets"])
        elif list(d["buckets"]) != buckets:
            raise RuntimeError(f"{f.name}: bucket list differs from the rest of the run; these "
                               f"dumps are from two different passes and must not be merged")
        acc[lname] = {k: d[src].double() for k, src in KEYMAP.items()}
        e = acc[lname]["sum"].shape[1]
        if n_exp is None:
            n_exp = e
        elif e != n_exp:
            raise RuntimeError(f"{lname}: {e} experts, but another layer has {n_exp}")
        if "f_sum" in d:
            f_by_layer[lname] = (d["f_sum"].double(), d["f_cnt"].double())

    if not acc:
        raise RuntimeError(f"{dirpath} holds only pass-1 dumps: no per-bucket statistics")

    missing = [l for l in acc if l not in f_by_layer]
    if missing and require_f:
        raise MissingF(
            f"{len(missing)} of {len(acc)} layers have no f_sum -- these dumps were written by a "
            f"pass that did not accumulate HOPE's off-diagonal ({missing[:3]}...). It cannot be "
            f"reconstructed from what is here; ranking them under mode=hope would silently "
            f"solve against an all-zero F, which is REAP with extra steps and a different name. "
            f"Re-run the saliency pass, or select with --mode reap and say so.")

    # f_sum is addressed by ABSOLUTE layer number, the same way reap_select indexes it, so it is
    # sized by the highest layer present -- not by the layer COUNT. GLM's layer 0 is dense, so
    # those differ, and sizing by count would put layer 42 out of bounds.
    n_layers = max(layer_index(l) for l in acc) + 1
    f_sum = torch.zeros(n_layers, n_exp, n_exp, dtype=torch.float64)
    f_cnt = torch.zeros(n_layers, n_exp, n_exp, dtype=torch.float64)
    for lname, (fs, fc) in f_by_layer.items():
        li = layer_index(lname)
        f_sum[li], f_cnt[li] = fs, fc

    return {"acc": acc, "f_sum": f_sum, "f_cnt": f_cnt, "buckets": buckets}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", default="artifacts/saliency")
    ap.add_argument("--out", default="artifacts/saliency/accumulators.pt")
    ap.add_argument("--allow-missing-f", action="store_true",
                    help="write the file even with no HOPE matrix; mode=hope will be meaningless")
    a = ap.parse_args()
    d = build(Path(a.dumps), require_f=not a.allow_missing_f)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(d, out)
    live = int((d["f_cnt"].sum(dim=(1, 2)) > 0).sum())
    print(f"wrote {out}: {len(d['acc'])} layers, {d['f_sum'].shape[1]} experts, "
          f"buckets={d['buckets']}, layers with F={live}")


if __name__ == "__main__":
    main()
