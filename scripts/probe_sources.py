"""Probe every corpus source: does it resolve, and does its text_fn actually extract text?

29 datasets with inferred field names is the largest untested surface in the corpus stage.
Cheaper to find a wrong key here than six hours into a build.
"""
import itertools, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
import corpus_spec as SPEC
from datasets import load_dataset
from common import hf_token

tok = hf_token()


def probe(hf_id, config, split, fn, label):
    try:
        ds = load_dataset(hf_id, config, split=split, streaming=True, token=tok)
        rows = list(itertools.islice(iter(ds), 3))
        if not rows:
            return f"EMPTY   {label}"
        outs = []
        for r in rows:
            try:
                outs.append(fn(r))
            except Exception:
                outs.append(None)
        ok = sum(1 for o in outs if o and len(o) >= 200)
        if ok:
            return f"OK({ok}/3) {label:54s} len={len(outs[0] or '')}"
        return f"ZERO    {label:54s} cols={list(rows[0].keys())[:9]}"
    except Exception as e:
        return f"FAIL    {label:54s} {type(e).__name__}: {str(e)[:110]}"


for bucket, srcs in SPEC.SOURCES.items():
    print(f"--- {bucket} ---", flush=True)
    for hf_id, config, split, w, fn in srcs:
        print("  " + probe(hf_id, config, split, fn, f"{hf_id}({config or '-'})/{split}"), flush=True)

print("--- multimodal ---", flush=True)
for hf_id, config, split, w in SPEC.MM_SOURCES:
    label = f"{hf_id}({config or '-'})/{split}"
    try:
        ds = load_dataset(hf_id, config, split=split, streaming=True, token=tok)
        rows = list(itertools.islice(iter(ds), 2))
        if not rows:
            print(f"  EMPTY   {label}", flush=True); continue
        cols = list(rows[0].keys())[:9]
        has_img = any(k in rows[0] for k in ("image", "images"))
        print(f"  {'OK  ' if has_img else 'NOIMG'}   {label:54s} cols={cols}", flush=True)
    except Exception as e:
        print(f"  FAIL    {label:54s} {type(e).__name__}: {str(e)[:110]}", flush=True)
