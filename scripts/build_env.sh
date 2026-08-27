#!/usr/bin/env bash
# Build the project venv. Torch comes from the official cu130 aarch64 index (the exact wheel
# already proven on this box: sm_110 in arch_list, reports "NVIDIA Thor").
# llmcompressor comes from git main, NOT pypi: the 0.13.0 release pins transformers<=5.14.1,
# but glm5_next only exists in transformers>=5.16.1. main's dev build asks for
# transformers>=5.15.0, which admits 5.16.1.
set -euo pipefail
cd /home/patrickd/glm-5.3-reap
export UV_HTTP_TIMEOUT=600
LOG=logs/build_env.log
exec >>"$LOG" 2>&1
echo "=== build_env start $(date -Is) ==="
uv venv --python 3.12 .venv
export VIRTUAL_ENV=/home/patrickd/glm-5.3-reap/.venv
uv pip install --index-url https://download.pytorch.org/whl/cu130 "torch==2.13.0+cu130"
uv pip install "transformers==5.16.1" "accelerate" "datasets" "safetensors" "peft" \
               "pillow" "huggingface_hub[cli]" "sentencepiece" "protobuf" "tqdm" "numpy" "scipy"
# --no-deps so pip cannot drag transformers back down to satisfy a stale pin
uv pip install --no-deps "git+https://github.com/vllm-project/llm-compressor.git@main"
uv pip install --index-url https://download.pytorch.org/whl/cu130 "torchvision==0.28.0+cu130"
uv pip install "compressed-tensors>=0.18.0" "pydantic" "loguru" "pynvml" "tdigest"
echo "=== versions ==="
.venv/bin/python - <<'PY'
import importlib.metadata as m
for p in ['torch','transformers','llmcompressor','compressed-tensors','accelerate','datasets','peft','huggingface_hub']:
    try: print(f'{p:20s} {m.version(p)}')
    except Exception as e: print(f'{p:20s} MISSING')
import torch
print('cuda:', torch.cuda.is_available(), torch.cuda.get_arch_list() if torch.cuda.is_available() else '')
PY
echo "=== build_env done $(date -Is) rc=$? ==="
