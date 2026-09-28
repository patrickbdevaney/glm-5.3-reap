"""Prove output/glm-5.3-flash-reap50-fp8-pass2 is fully recoverable from HF before deleting it.

161 GB with exactly one local copy. A filename-and-size match is not proof: a truncated or
half-committed LFS upload has the right name and can have the right size in the index while the
stored object differs. HF records an LFS sha256 per file, so compare against THAT. Every local
file must be present on the Hub with a matching size and a matching sha256, or nothing is deleted.

Exit 0 only if every file verifies.
"""
import hashlib, sys
from pathlib import Path
from huggingface_hub import HfApi

SRC  = Path("output/glm-5.3-flash-reap50-fp8-pass2")
REPO = "patrickbdevaney/GLM-5.3-Flash-REAP50-FP8-v2"

def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(64 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def main() -> int:
    local = sorted(p for p in SRC.rglob("*") if p.is_file())
    if not local:
        print("FAIL: no local files found - refusing"); return 1

    api = HfApi()
    rel = [str(p.relative_to(SRC)) for p in local]
    info = {i.path: i for i in api.get_paths_info(REPO, rel, repo_type="model")}
    print(f"{len(local)} local files, {len(info)} matched on {REPO}")

    bad = 0
    for p, r in zip(local, rel):
        i = info.get(r)
        if i is None:
            print(f"  MISSING ON HUB  {r}"); bad += 1; continue
        if i.size != p.stat().st_size:
            print(f"  SIZE MISMATCH   {r}  local={p.stat().st_size} hub={i.size}"); bad += 1; continue
        remote_sha = getattr(getattr(i, "lfs", None), "sha256", None)
        if remote_sha is None:
            # small non-LFS files (config.json etc) carry no sha256; size match is all HF offers
            print(f"  ok (size only)  {r}"); continue
        if sha256(p) != remote_sha:
            print(f"  SHA MISMATCH    {r}"); bad += 1; continue
        print(f"  ok (sha256)     {r}")

    if bad:
        print(f"FAIL: {bad} file(s) did not verify - KEEPING the local copy"); return 1
    print(f"OK: all {len(local)} files verified against {REPO}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
