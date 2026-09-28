#!/usr/bin/env bash
# Rebuild the v2 NVFP4 artifact from the healing-corrected FP8 base.
#
# Gated on the corrected FP8 eval reproducing the ablation. If the revert did not land the number
# the ablation predicted, something is wrong with the weights and nothing downstream of them
# should be built, let alone published.
#
# RECIPE IS UNCHANGED. `nvfp4_quantize_attention` stays off, so this build differs from the
# published v2 in exactly one variable - the healing correction. Rebuilding it with the new
# attention split at the same time would recreate the confound that caused this whole detour.
set -u
cd /home/patrickd/glm-5.3-reap
PY=./.venv/bin/python
LOG=logs/revert_chain.log
OLD=output/glm-5.3-flash-reap50-nvfp4-pass2
say(){ echo "[$(date -Is)] nvfp4: $*" >> $LOG; }

while kill -0 "$1" 2>/dev/null; do sleep 60; done

$PY - >> $LOG 2>&1 <<'EOF'
import json, sys
d = json.load(open('artifacts/eval/eval_glm-5.3-flash-reap50-fp8-pass2.json'))
ok = abs(d['top1_agreement'] - 0.84238) < 2e-3
print(f"gate: corrected FP8 top1={d['top1_agreement']:.5f} vs ablation 0.84238 -> "
      f"{'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
EOF
if [ $? -ne 0 ]; then say "GATE FAILED - not rebuilding NVFP4"; exit 1; fi
say "gate passed"

# The superseded build has to go before the new one can be written: 98.2 GiB needed, and it is
# the only 98 GiB on the disk. Verified byte-for-byte present on the Hub first - every weight
# file, sizes matching - so this is reclaiming a cache, not destroying the only copy.
$PY - >> $LOG 2>&1 <<'EOF'
from huggingface_hub import HfApi
import pathlib, sys
loc = pathlib.Path('output/glm-5.3-flash-reap50-nvfp4-pass2')
remote = {s.rfilename: s.size for s in HfApi().model_info(
    "patrickbdevaney/GLM-5.3-Flash-REAP50-NVFP4-v2", files_metadata=True).siblings}
bad = [str(p.relative_to(loc)) for p in loc.rglob('*') if p.is_file()
       and str(p.relative_to(loc)) != 'README.md'
       and (str(p.relative_to(loc)) not in remote
            or remote[str(p.relative_to(loc))] != p.stat().st_size)]
print("hub backup check:", "complete" if not bad else f"INCOMPLETE {bad[:5]}")
sys.exit(0 if not bad else 1)
EOF
if [ $? -ne 0 ]; then say "hub backup incomplete - refusing to delete the local NVFP4"; exit 1; fi

if pgrep -f 's07_quantiz[e]' > /dev/null; then
  say "a quantiser is running - refusing to remove $OLD out from under a live writer"; exit 1
fi
say "removing superseded $OLD ($(du -sh $OLD 2>/dev/null | cut -f1))"
rm -rf "$OLD"
say "free: $(df -h /home/patrickd | tail -1 | awk '{print $4}')"

$PY - >> $LOG 2>&1 <<'EOF'
import sys; sys.path.insert(0,'scripts')
from common import kv_set
kv_set("nvfp4_name", "glm-5.3-flash-reap50-nvfp4-pass2")
kv_set("emit_path", "/home/patrickd/glm-5.3-reap/output/glm-5.3-flash-reap50-fp8-pass2")
kv_set("nvfp4_quantize_attention", False)
EOF
say "quantising from the corrected FP8 base (experts only, attention untouched)"
./scripts/gpulock.sh s07-nvfp4-corrected $PY scripts/run_stage.py s07_quantize stages.s07_quantize >> $LOG 2>&1
say "quantise rc=$?"

$PY - >> $LOG 2>&1 <<'EOF'
import sys; sys.path.insert(0,'scripts')
from common import kv_set
kv_set("eval_student", "/home/patrickd/glm-5.3-reap/output/glm-5.3-flash-reap50-nvfp4-pass2")
EOF
say "eval: corrected NVFP4"
./scripts/gpulock.sh eval-nvfp4-corrected $PY scripts/run_stage.py s09_eval stages.s09_eval >> $LOG 2>&1
say "eval rc=$?  chain complete"
