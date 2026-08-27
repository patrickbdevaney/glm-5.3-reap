# 20 — Jetson AGX Thor: measured capability and storage envelope

All numbers `MEASURED` on 2026-08-27 unless marked otherwise.

## Platform

| Property | Measured value |
|---|---|
| Board | NVIDIA Jetson AGX Thor Developer Kit (`/proc/device-tree/model`) |
| L4T | R38 rev 4.0, GCID 43443517, kernel `6.8.12-tegra` (oot), openrm |
| Arch | aarch64, 14-core Arm Neoverse-V3AE |
| CUDA | 13.0.48 (`nvcc`), toolkit at `/usr/local/cuda-13.0` |
| Compute capability | **sm_110a** (Blackwell) |
| RAM | 122 GiB total, 115 GiB free, 117 GiB available (`free -g`) |
| Swap | 31 GiB |
| NVMe | WD PC SN5000S SDEPNSJ-1T00, 1.02 TB, `/dev/nvme0n1p1` → `/` |

Memory is unified LPDDR5X with full UVM coherence; the practical GPU-visible envelope from
prior projects on this box is **~117 GiB**, not the nominal 128 GB.

## Measured throughput

| Path | Measured | Method |
|---|---|---|
| NVMe sequential **read** | **3.4 GB/s** | `dd bs=1M count=4096 iflag=direct` |
| NVMe sequential **write** | **487 MB/s** | `dd bs=1M count=4096 oflag=direct` |
| HuggingFace download, 1 stream | **103 MB/s** | 2 GiB ranged `curl` |
| HuggingFace download, 8 streams | **107 MB/s** | 8 × 256 MiB parallel ranged `curl` |

**Two consequences that shape the whole plan:**

1. **The network link is saturated at ~105 MB/s and parallelism does not help.** One full
   streaming pass over the 328.3 GB source costs **~52 minutes**. That is cheap. Re-reading
   the source from HF is therefore a *legitimate alternative to storing it*.
2. **NVMe write (487 MB/s) is ~7× slower than read (3.4 GB/s) and ~4.6× faster than the
   network.** Writing is the local bottleneck; reading never is. Sequential-onload passes
   that re-read local weights are essentially free; passes that rewrite them are not.

## Storage envelope

```
/dev/nvme0n1p1   936G total   592G used   298G available   67% full
```

**298 GiB free vs. a 305.8 GiB source. The source does not fit as-is.** `[EST]`

See [90-open-risks.md](90-open-risks.md) for the resolution (streaming) and the reclaim
inventory presented to the operator.

## Large on-disk items (reclaim inventory — NOTHING deleted, operator decides)

| Path | Size | What it is | Risk to delete |
|---|---:|---|---|
| `models/DeepSeek-V4-Flash-0731-REAP` | 101 G | **The §7 incumbent eval baseline** | **Do not touch** |
| docker images (5) | 102 G | vllm-dflash-thor:dllm 68G, ghcr jetson-vllm 50G, vllm-openai gemma 32G, phoenix, searxng (shared layers → 102 G real) | High — rebuilds are long |
| `s5-capture` | 80 G | dspark draft-head trace captures (s3recap 31 G, agentic-p25 14 G, …) | Medium — regenerable but expensive |
| `laguna-s1-cuda-server/models` | 70 G | Laguna S1 server weights | Medium |
| `model-backups` | 62 G | `heads` 50 G, dspark-mtp-base 6.6 G, releases 3.5 G | Medium |
| `thor-vllm-cache` | 31 G | torch_compile_cache 12 G, vllm-dflash 9.5 G, … | **Low — regenerable cache** |
| `Qwen3.6-35B-A3B-NVFP4` | 24 G | Prior NVFP4 checkpoint | Medium |
| `models/gemma-4-26B-A4B-it-NVFP4` | 16 G | Prior NVFP4 checkpoint | Medium |
| docker dangling layers | 21 G | Unreferenced image layers | **None — `docker image prune`** |

Zero-risk reclaim available immediately: **~52 GiB** (21 G dangling docker + 31 G
regenerable vLLM cache). That alone is enough margin for the streaming plan.

## Dedication

The DSv4 flywheel Claude Code loop is **confirmed stopped**: crontab still carries the hourly
`flywheel_cron.sh` entry, but a `FLYWHEEL_STOP` sentinel is present and every tick since
2026-08-26 logs `FLYWHEEL_STOP present; exiting` with no work done. `dsv4-oomsentry.service`
is still running — that is a passive OOM backstop, harmless, and arguably useful to keep. `[EST]`
