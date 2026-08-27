#!/usr/bin/env python
"""One-screen pipeline status. Safe to run any time, including mid-stage."""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import db, ROOT, free_gib


def sh(cmd):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        return r.stdout.strip().replace("\n", " ")
    except Exception:
        return "?"


print(f"=== GLM-5.3 REAP pipeline @ {sh('date -Is')} ===")
print(f"service : {sh('systemctl --user show glm53-reap.service -p ActiveState --value')}"
      f"/{sh('systemctl --user show glm53-reap.service -p SubState --value')}")
print(f"disk    : {free_gib():.0f} GiB free")

src = ROOT / "source" / "GLM-5.3-Flash"
if src.exists():
    b = sum(p.stat().st_size for p in src.rglob("*.safetensors"))
    print(f"source  : {b/1e9:.1f} / 328.3 GB ({100*b/328_337_455_672:.1f}%)")
for d, label in [(ROOT / "corpus", "corpus"), (ROOT / "output", "output"),
                 (ROOT / "artifacts" / "saliency", "saliency")]:
    if d.exists():
        print(f"{label:8s}: {sh(f'du -sh {d}').split()[0] if sh(f'du -sh {d}') else '-'}")

print("\nstages:")
with db() as c:
    for n, st, a, err in c.execute("SELECT name,status,attempts,error FROM stages ORDER BY name"):
        e = f"  <- {str(err)[:70]}" if err and st in ("failed", "retry", "blocked") else ""
        print(f"  {n:16s} {st:14s} attempts={a}{e}")

print("\nrecent events:")
with db() as c:
    rows = list(c.execute("SELECT ts,stage,level,msg FROM events ORDER BY id DESC LIMIT 14"))
for ts, stg, lv, msg in reversed(rows):
    print(f"  {ts[11:19]} {lv:5s} {str(stg or '-'):16s} {msg[:104]}")
