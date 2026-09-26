#!/usr/bin/env bash
# Pass 3, up to the decision point and NOT past it.
#
# Deliberately stops after saliency. s04b_surgery deletes source shards as it writes survivors,
# so it is the one stage that cannot be taken back: once it runs, re-deciding the mask means
# re-downloading 328 GB. protect_frac is supposed to be read off THIS run's accumulators, which
# do not exist until s03 finishes -- so the chain hands back at exactly the point where that
# measurement becomes possible, and a human decides before anything irreversible happens.
set -u
cd /home/patrickd/glm-5.3-reap
PY=./.venv/bin/python
LOG=logs/pass3_chain.log
ST=logs/PASS3_STATUS.txt
say(){ echo "[$(date -Is)] pass3: $*" >> "$LOG"; }
status(){ printf 'pass 3 -- %s\n%s\n' "$(date -Is)" "$1" > "$ST"; }

say "chain start"

# ---- wait for the source ---------------------------------------------------------------------
status "waiting for s01_source"
while :; do
  s=$($PY - <<'EOF'
import sys; sys.path.insert(0,'scripts')
from common import db
with db() as c:
    r=c.execute("select status from stages where name='s01_source'").fetchone()
print(r[0] if r else "missing")
EOF
)
  [ "$s" = "done" ] && break
  if [ "$s" = "failed" ]; then say "s01_source FAILED; stopping"; status "s01_source failed"; exit 1; fi
  sleep 120
done
say "source staged"

# ---- corpus ------------------------------------------------------------------------------------
# Re-run even though pass 2 built one: corpus_spec's pass-3 quotas are different (ballast 860 ->
# ~2095 samples), and s02 tops buckets up rather than rebuilding, so this is additive.
status "s02_corpus (pass-3 mixture: ballast 4.8% -> 15% of routed tokens)"
say "s02_corpus"
$PY scripts/run_stage.py s02_corpus stages.s02_corpus >> "$LOG" 2>&1
rc=$?; say "s02_corpus rc=$rc"
if [ $rc -ne 0 ]; then status "s02_corpus failed (rc=$rc)"; exit 1; fi

# ---- saliency ----------------------------------------------------------------------------------
status "s03_saliency (HOPE: F's off-diagonal accumulates on this forward or not at all)"
say "s03_saliency under the GPU lock"
./scripts/gpulock.sh s03-pass3 $PY scripts/run_stage.py s03_saliency stages.s03_saliency >> "$LOG" 2>&1
rc=$?; say "s03_saliency rc=$rc"
if [ $rc -ne 0 ]; then status "s03_saliency failed (rc=$rc)"; exit 1; fi

# ---- hand back ----------------------------------------------------------------------------------
nf=$(ls artifacts/saliency/*.pt 2>/dev/null | wc -l)
say "saliency complete: $nf layer dumps"
status "SALIENCY DONE ($nf layers). STOPPED BEFORE SURGERY ON PURPOSE.
Next, in order:
  1. sweep protect_frac on artifacts/saliency and write conf/protect_frac_override.txt
  2. review artifacts/masks/mask.json -- worst domain must beat pass 2's ballast 0.487
  3. only then run s04b_surgery, which DELETES source shards as it goes"
say "chain complete; surgery NOT started"
