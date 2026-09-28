#!/usr/bin/env bash
# Detached: restore the promoted DSpark head, then re-establish the archive that lost it.
#
# Uses tools/build_trained_head.py -- the CANONICAL rebuilder -- not a hand-rolled tensor
# substitution. mtp_trained.safetensors stores all 72 tensors as BF16, but the engine reads
# 8 of them per shard as FP8 e4m3 + a separate .scale, and the published file carries no
# scales at all. Substituting the bf16 tensors directly "produces a plausible load and garbage
# weights" (tools/build_trained_head.py docstring). The rebuilder re-quantises each tensor back
# to its ORIGINAL format and rewrites the scales.
#
# head_card.json records sha256 + byte length for all three shards, so the rebuild is verified
# against the promotion record rather than merely assumed correct.
set -u
R="$HOME/deepseek-v4-flash-0731-cuda"
LOG="$HOME/glm-5.3-reap/logs/head_restore.log"
HEAD="$HOME/models/dspark-head-s3recap-p25-b0.1"
STORE="$HOME/model-backups/heads"
OUT="$HOME/models/promoted-head-shards"
say() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

say "waiting for NIAH to release memory"
while pgrep -x glm5-server >/dev/null 2>&1; do sleep 60; done
say "free $(free -g | awk '/^Mem:/{print $7}') GiB"

cd "$HOME/glm-5.3-reap"
say "build_trained_head.py --selftest"
.venv/bin/python "$R/tools/build_trained_head.py" --selftest >> "$LOG" 2>&1 \
  || { say "SELFTEST FAILED -- refusing to rebuild"; exit 1; }

say "rebuilding promoted head shards"
.venv/bin/python "$R/tools/build_trained_head.py" \
  --base "$HOME/models/DeepSeek-V4-Flash-0731-REAP" \
  --trained "$HEAD/mtp_trained.safetensors" --out "$OUT" >> "$LOG" 2>&1 \
  || { say "REBUILD FAILED"; exit 1; }

say "verifying against head_card.json"
.venv/bin/python - "$HEAD" "$OUT" >> "$LOG" 2>&1 <<'PY'
import hashlib, json, os, sys
head, out = sys.argv[1], sys.argv[2]
card = json.load(open(os.path.join(head, "head_card.json")))
want = {f["file"]: (f["bytes"], f["sha256"]) for f in card["files"]
        if f["file"].startswith("model-") and f["file"].endswith(".safetensors")}
ok = True
for fn, (nb, sha) in sorted(want.items()):
    p = os.path.join(out, fn)
    if not os.path.exists(p):
        print(f"{fn}: ABSENT"); ok = False; continue
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(16 << 20), b""): h.update(c)
    g, b = h.hexdigest(), os.path.getsize(p)
    good = g == sha and b == nb
    ok &= good
    print(f"{fn}: bytes {'OK' if b==nb else f'{b}!={nb}'} sha256 {'MATCH' if g==sha else 'DIFFER'}")
sys.exit(0 if ok else 3)
PY
rc=$?
if [ $rc -ne 0 ]; then
  say "VERIFY FAILED rc=$rc -- NOT installing. Staged shards left in $OUT for inspection."
  exit 1
fi
say "verified against the promotion record"

# install into live_ckpt
LIVE="$HOME/models/ckpt-head-s3recap-p25-b0.1"
for n in 46 47 48; do
  f="model-000$n-of-00048.safetensors"
  ln -sfn "$OUT/$f" "$LIVE/$f" && say "  $LIVE/$f -> promoted"
done

# re-establish the archive whose loss started this
mkdir -p "$STORE/s3recap-p25-b0.1"
cp -n "$HEAD/mtp_trained.safetensors" "$STORE/s3recap-p25-b0.1/" 2>/dev/null
cp -n "$HEAD/head_card.json" "$STORE/s3recap-p25-b0.1/" 2>/dev/null
say "archived source-of-truth -> $STORE/s3recap-p25-b0.1/ (1 GB, regenerates shards deterministically)"
say "DONE -- next: scripts/baseline_tau.sh (expect tau ~3.8413, ~28.38 tok/s at block 5)"
