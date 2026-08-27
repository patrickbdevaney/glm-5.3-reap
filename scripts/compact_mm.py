#!/usr/bin/env python
"""Recompress multimodal shards to bf16 pixel_values.

The processor emits float32, which cost ~30 MB per sample and made the shard directory 55 GB.
The forward pass casts to bf16 regardless, so this loses nothing and halves the footprint -
and disk is the binding constraint for surgery (R10).
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import torch

d = Path(sys.argv[1] if len(sys.argv) > 1 else "corpus/shards/multimodal")
files = sorted(d.glob("mm_*.pt"))
before = sum(f.stat().st_size for f in files)
n = 0
for f in files:
    recs = torch.load(f, weights_only=False)
    changed = False
    for r in recs:
        pv = r.get("pixel_values")
        if pv is not None and pv.dtype == torch.float32:
            r["pixel_values"] = pv.to(torch.bfloat16)
            changed = True
    if changed:
        torch.save(recs, f)
        n += 1
    del recs
after = sum(f.stat().st_size for f in files)
print(f"recompressed {n}/{len(files)} shards: {before/2**30:.1f} -> {after/2**30:.1f} GiB "
      f"(freed {(before-after)/2**30:.1f} GiB)")
