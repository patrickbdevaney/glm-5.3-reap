#!/usr/bin/env bash
# Publish the MTP-enablement artifacts to the GGUF repo.
#
# The weights themselves are unchanged and are NOT re-uploaded: blk.45 already ships inside all
# five quants, so what was missing was never the weights, only the means to run them. Four files:
#
#   glm5-next-llama.cpp.patch   15 commits now, was 12 - verified to reproduce the fork HEAD
#                               byte for byte when git am'd onto upstream 761797ff
#   glm5-next-llama.cpp.bundle  regenerated to match; git bundle verify says complete history
#   make_mtp_draft.py           the card now tells people to run this, so it has to be here
#   README.md                   the MTP section no longer claims llama.cpp cannot run the block,
#                               and reports the measured slowdown rather than implying a speedup
set -uo pipefail
cd /home/patrickd/glm-5.3-reap
LOG=logs/ship_mtp.log
say() { echo "[$(date -Is)] ship_mtp: $*" >> "$LOG"; }

say "starting"
./.venv/bin/python - >> "$LOG" 2>&1 <<'PY'
import sys
from huggingface_hub import HfApi
REPO = "patrickbdevaney/GLM-5.3-Flash-REAP50-GGUF"
api = HfApi()
uploads = [
    ("gguf_README.md",                          "README.md"),
    ("gguf/glm5-next-llama.cpp.patch",          "glm5-next-llama.cpp.patch"),
    ("scripts/make_mtp_draft.py",               "make_mtp_draft.py"),
    ("artifacts/glm5-next-llama.cpp.bundle",    "glm5-next-llama.cpp.bundle"),
]
for local, remote in uploads:
    print(f"uploading {local} -> {remote}", flush=True)
    api.upload_file(path_or_fileobj=local, path_in_repo=remote, repo_id=REPO,
                    commit_message="MTP block is runnable: recurrent-state rollback + draft repack")
    print(f"  done {remote}", flush=True)

have = set(api.list_repo_files(REPO))
missing = [r for _, r in uploads if r not in have]
print("missing after upload:", missing if missing else "none")
sys.exit(1 if missing else 0)
PY
say "rc=$?"
say "done"
