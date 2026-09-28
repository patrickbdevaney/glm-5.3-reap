"""Stage 3 - REAP saliency by layer streaming, then structural prune.

s01b determined by arithmetic that the model cannot be placed: 305.8 GiB of weights against
116 GiB of (unified) RAM and 162 GiB of free disk. So this does not load a model. It builds one
decoder layer at a time, runs it, accumulates saliency, and frees it. Peak residency is one
layer (~14.5 GiB dequantised) plus the activation batch.

That is only tractable because Glm5NextTextModel.forward carries its entire inter-layer state in
a single tensor plus topk_indices - mHC's four residual streams live in the hc_mult axis and are
manipulated inside each decoder layer, so there is no cross-layer bookkeeping to replicate.

    S_j = mean over tokens routed to expert j of  g_j * ||f_j||_2

with f_j captured BEFORE the router gate scales it, which is what REAP is defined over.

Chunk by samples, sweeping all layers per chunk. Re-reading weights (3.4 GB/s) is far cheaper
than re-writing activations (487 MB/s) on this box.
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import memfence as MF  # noqa: E402
from common import (ROOT, ARTIFACTS, MODEL_ID, log, metric, kv_get, kv_set,  # noqa: E402
                    publish, free_gib)

STAGE = "s03_saliency"


def _reclaim_page_cache() -> None:
    """Drop page cache between layers.

    Each layer read faults in ~7 GB of mmap'd shard; across 45 layers that is 306 GB of page
    cache that is never re-read. Tegra under-reports it in MemAvailable and will not reclaim it
    fast enough to satisfy a driver allocation, which is how this stage died six times with no
    oom-kill line in dmesg. Cheap and safe: these are clean file-backed pages.
    """
    import subprocess
    try:
        subprocess.run(["sync"], timeout=30, check=False)
        subprocess.run(["sudo", "-n", "sh", "-c", "echo 1 > /proc/sys/vm/drop_caches"],
                       timeout=30, check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass                 # the memguard service is the backstop
SRC = ROOT / "source" / "GLM-5.3-Flash"
CORPUS = ROOT / "corpus" / "shards"
# Overridable so an equivalence test can write somewhere isolated instead of the real dumps.
SALIENCY = Path(os.environ.get("S03_SALIENCY_DIR") or (ROOT / "artifacts" / "saliency"))
# Router-score cache, written per chunk. Grows with TOKENS rather than with experts, so it is
# flushed and cleared each chunk instead of accumulating across the run.
ROUTER_CACHE_DIR = Path(os.environ.get("S03_ROUTER_CACHE_DIR")
                        or (ROOT / "artifacts" / "router_cache"))
# Per-chunk cumulative snapshots of the ranking statistics; see SS.dump_light.
SNAPSHOT_DIR = Path(os.environ.get("S03_SNAPSHOT_DIR")
                    or (ROOT / "artifacts" / "saliency_snapshots"))

TARGET_SPARSITY = float(kv_get("chosen_ratio", 0.50) or 0.50)
# Sized so that ALL activations fit in RAM at once, which lets each layer be loaded exactly
# once instead of once per batch - the difference between ~6 minutes and ~3 hours of weight
# re-reads. 512 x 2048 x hc_mult(4) x 4096 x 2B ~= 34 GB.
# REAP's saliency is a conditional mean, so tokens-per-expert governs, not corpus size:
# 512 x 2048 x 8/288 ~= 29k tokens per expert, far above the 2k sufficiency floor.
def _tunable(env_name: str, kv_name: str, default: int) -> int:
    """Env overrides the kv store, for isolated test runs that must not mutate pipeline state."""
    v = os.environ.get(env_name)
    return int(v) if v else int(kv_get(kv_name, default) or default)


MAX_LEN = _tunable("S03_MAX_LEN", "calib_max_len", 2048)
# PASS 2. Pass 1 ran 256 samples = 0.52M tokens: 1.1% of the 48.3M-token corpus we built, which
# left 501/12096 expert slots decided on under 2000 tokens and ~25.5 experts per layer sitting
# within +-5% of the 50% cut. The budget is now expressed in TOKENS and swept in chunks, so it
# is bounded by wall-clock rather than by RAM.
CALIB_TOKENS = _tunable("S03_CALIB_TOKENS", "calib_tokens", 5_500_000)
CHUNK_TOKENS = _tunable("S03_CHUNK_TOKENS", "calib_chunk_tokens", 500_000)
N_CALIB = int(kv_get("n_calib_samples", 0) or 0) or -(-CALIB_TOKENS // MAX_LEN)
# MEASURED 2026-08-27: the KDA (linear_attention) forward costs ~13 GiB of transient memory
# per 2048-token sequence and scales linearly with batch, so B=8 needed ~104 GiB and took the
# box to 0.2 GiB available. That is why every run died at layer 5 - the first layer that is
# BOTH linear_attention and MoE. B=2 puts the transient at ~26 GiB.
BATCH = int(kv_get("calib_batch", 2) or 2)
MIN_TOKENS_PER_EXPERT = 2_000
DEV = "cuda"
DT = None  # set at runtime


def _load_calib():
    """Interleave text buckets in mixture proportion, then append image-text samples."""
    import torch
    import corpus_spec as SPEC
    per_bucket: dict[str, list] = {}
    tdir = CORPUS / "text"
    for shard in sorted(tdir.glob("*.pt")):
        try:
            per_bucket[shard.stem] = torch.load(shard, weights_only=False)
        except Exception as e:
            log(f"could not load {shard.name}: {e}", STAGE, "WARN")
    n_text = int(N_CALIB * (1 - SPEC.MIXTURE["multimodal"]))
    text_rows = []
    denom = 1 - SPEC.MIXTURE["multimodal"]
    for bucket, share in SPEC.MIXTURE.items():
        if bucket == "multimodal":
            continue
        items = per_bucket.get(bucket, [])
        want = int(n_text * share / denom)
        for it in items[:want]:
            # Carry the bucket. Pass 1 discarded it, which made the calibration mixture
            # impossible to re-weight after the fact - any change meant another full pass.
            text_rows.append((it["input_ids"][:MAX_LEN], bucket))
    mm_rows = []
    mdir = CORPUS / "multimodal"
    if mdir.exists():
        want_mm = max(N_CALIB - len(text_rows), 0)
        for f in sorted(mdir.glob("mm_*.pt")):
            if len(mm_rows) >= want_mm:
                break
            try:
                for rec in torch.load(f, weights_only=False):
                    if len(mm_rows) >= want_mm:
                        break
                    mm_rows.append(rec)
            except Exception:
                continue
    # Stratify the text rows so EVERY chunk carries the full mixture.
    #
    # The rows are built bucket-by-bucket, so slicing them into chunks in that order gives chunk 0
    # entirely to `agentic`, chunk 1 to agentic+code, and so on. Two things break: an interrupted
    # run leaves a domain-skewed accumulator, and - worse - the split-half gate would compare
    # early chunks against late ones, i.e. one set of domains against a different set, and report
    # spuriously low keep-set overlap. We would then read a mixture artefact as "the token budget
    # was insufficient" and buy hours of calibration we did not need.
    #
    # Interleave by within-bucket rank so each bucket spreads evenly over the whole list and the
    # proportions hold in every slice. Deterministic - no RNG - so a resumed run rebuilds the
    # identical order.
    by_b: dict[str, list] = {}
    for row in text_rows:
        by_b.setdefault(row[1], []).append(row)
    # Sort on an explicit key. Sorting (fraction, row) tuples lets equal fractions fall through
    # to comparing the rows themselves, and a row holds an input_ids TENSOR - so ties raise
    # "size of tensor a (130) must match tensor b (2048)". Ties are common here because the first
    # item of every bucket has fraction 0.5/n.
    keyed = [((i + 0.5) / max(1, len(rows)), b, i, r)
             for b, rows in by_b.items() for i, r in enumerate(rows)]
    keyed.sort(key=lambda t: (t[0], t[1], t[2]))
    text_rows = [t[3] for t in keyed]
    log(f"calibration: {len(text_rows)} text + {len(mm_rows)} image-text; stratified across "
        f"{len(by_b)} buckets so every chunk carries the mixture", STAGE)
    metric(STAGE, "calib_text_samples", len(text_rows))
    metric(STAGE, "calib_mm_samples", len(mm_rows))
    if not mm_rows:
        raise AssertionError(
            "no image-text calibration samples. Text-only calibration gives vision-serving "
            "experts an empty active set, so REAP deletes them with certainty at any prune "
            "ratio (R3). Blocking rather than silently dropping vision.")
    return text_rows, mm_rows


# transformers applies a checkpoint conversion mapping at load time; building layers by hand
# means applying it by hand. Source: transformers.conversion_mapping.get_checkpoint_conversion_mapping("glm5_next").
# Without these, mHC (attn_hc/ffn_hc) and the KDA forget gate silently fail to load - which
# would leave randomly-initialised weights in the two components most sensitive to pruning.
_RENAMES = [
    ("self_attn.f_a_proj.", "self_attn.forget_gate.f_a_proj."),
    ("self_attn.f_b_proj.", "self_attn.forget_gate.f_b_proj."),
    ("self_attn.dt_bias", "self_attn.forget_gate.dt_bias"),
    ("self_attn.A_log", "self_attn.forget_gate.A_log"),
    ("hc_attn_fn", "attn_hc.fn"),
    ("hc_attn_base", "attn_hc.base"),
    ("hc_attn_scale", "attn_hc.scale"),
    ("hc_ffn_fn", "ffn_hc.fn"),
    ("hc_ffn_base", "ffn_hc.base"),
    ("hc_ffn_scale", "ffn_hc.scale"),
]


def _rename(rel: str) -> str:
    for a, b in _RENAMES:
        if a in rel:
            return rel.replace(a, b)
    return rel


# layer index -> per-expert multiplier on down_proj, applied when the layer is materialised.
# Empty in every normal run; populated only by scripts/heal_ablation.py.
HEAL_OVERRIDE: dict[int, list] = {}


def _build_layer(cfg, i, reader, dtype):
    """Materialise decoder layer i from the source shards.

    The checkpoint stores experts as per-expert 2D tensors; the module wants fused 3D
    parameters, so gate/up are concatenated per expert and stacked.
    """
    import torch
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextDecoderLayer
    from accelerate import init_empty_weights
    from stream_saliency import dequant_fp8_block

    prefix = f"model.language_model.layers.{i}."
    with init_empty_weights():
        layer = Glm5NextTextDecoderLayer(cfg, i)
    layer = layer.to_empty(device="cpu")

    names = [k for k in reader.map if k.startswith(prefix)]
    scales = {n for n in names if n.endswith("weight_scale_inv")}
    packed = {n[: -len("_packed")] for n in names if n.endswith("weight_packed")}

    # Iterate LOGICAL weight names, not raw tensor names.
    #
    # FP8 stores `w.weight` alongside `w.weight_scale_inv`, so the bare name is present and the
    # old code could simply skip the companion. NVFP4 stores THREE companions - `weight_packed`,
    # `weight_scale`, `weight_global_scale` - and NO bare `weight`. Iterating raw names therefore
    # handed the module `...gate_proj.weight_packed`, which is not a parameter it has: the tensor
    # was reported "missing from checkpoint" and the module silently kept the uninitialised f32
    # buffer from `to_empty`. The failure surfaced two layers later as
    # `mat1 and mat2 to have the same dtype: BFloat16 != float`, which names neither the tensor
    # nor the format. Collapse every companion onto its logical `...weight` first; `fetch` already
    # knows how to reconstitute either format from that name.
    def _logical(n: str) -> str:
        for suf in ("weight_packed", "weight_scale_inv", "weight_global_scale", "weight_scale"):
            if n.endswith(suf):
                return n[: -len(suf)] + "weight"
        return n

    seen: set[str] = set()
    logical: list[str] = []
    for n in names:
        b = _logical(n)
        if b not in seen:
            seen.add(b)
            logical.append(b)

    def fetch(name):
        # NVFP4 path. The size-matched competitor to an aggressively-quantised GGUF is our
        # NVFP4 checkpoint, not the FP8 master - so the evaluation has to be able to read it,
        # or it measures a 157 GiB artifact and reports it against 93-109 GB rivals.
        if name in packed:
            from compressed_tensors.compressors.nvfp4.helpers import unpack_fp4_from_uint8
            pk = reader.get(name + "_packed")
            sc = reader.get(name + "_scale").to(torch.float32)
            gs = reader.get(name + "_global_scale").to(torch.float32)
            out_f, in_f = pk.shape[0], pk.shape[1] * 2
            q = unpack_fp4_from_uint8(pk, out_f, in_f, dtype=torch.float32)
            # scale is per group of 16, stored as FP8 E4M3 and scaled by the per-tensor global
            g = q.reshape(out_f, in_f // 16, 16)
            w = g * (sc.reshape(out_f, in_f // 16, 1) / gs)
            return w.reshape(out_f, in_f).to(dtype)
        t = reader.get(name)
        s = name[: -len("weight")] + "weight_scale_inv"
        if s in scales:
            return dequant_fp8_block(t, reader.get(s), dtype=dtype)   # already a new tensor
        # copy=True is required: .to(dtype) on a tensor already in that dtype returns SELF,
        # a view into the shard mmap. The mapping is released after each layer so its page
        # cache can be reclaimed, and a live view would then be a use-after-unmap.
        return t.to(dtype, copy=True) if t.is_floating_point() else t.clone()

    sd = {}
    convs: dict[str, "torch.Tensor"] = {}
    expert_names: dict[int, dict[str, str]] = {}
    for n in logical:
        if n in scales:
            continue
        rel = n[len(prefix):]
        if rel.startswith("mlp.experts."):
            parts = rel.split(".")
            expert_names.setdefault(int(parts[2]), {})[parts[3]] = n
        elif rel.startswith("self_attn.") and "_conv1d." in rel:
            convs[rel.split(".")[1]] = fetch(n)      # q_conv1d / k_conv1d / v_conv1d
        else:
            sd[_rename(rel)] = fetch(n)

    if convs:
        order = [convs[k] for k in ("q_conv1d", "k_conv1d", "v_conv1d") if k in convs]
        if order:
            sd["self_attn.conv1d.weight"] = torch.cat(order, dim=0)
        convs.clear()

    if expert_names:
        # Fill preallocated 3D tensors expert by expert, freeing each source as it is consumed.
        # Building a list of 288 dequantised experts and then torch.stack()-ing it holds THREE
        # copies at once (dict + list + stack output) - about 44 GB for one MoE layer, which is
        # what spiked the box to 800 MB available at layer 5.
        idx = sorted(expert_names)
        inter = cfg.moe_intermediate_size
        hidden = cfg.hidden_size
        gu = torch.empty(len(idx), 2 * inter, hidden, dtype=dtype)
        dn = torch.empty(len(idx), hidden, inter, dtype=dtype)
        for j, e in enumerate(idx):
            names_e = expert_names[e]
            g = fetch(names_e["gate_proj"])
            gu[j, :inter].copy_(g)
            del g
            u = fetch(names_e["up_proj"])
            gu[j, inter:].copy_(u)
            del u
            d = fetch(names_e["down_proj"])
            dn[j].copy_(d)
            del d
        # Optional per-expert rescale applied AT LOAD, for healing ablations. Healing is a
        # multiply on down_proj, so an alternative healing can be evaluated by multiplying here
        # instead of rewriting 161 GiB of checkpoint - which matters because there is not enough
        # disk for a second copy, and mutating the shipped artifact in place to measure it would
        # risk publishing whichever variant an interruption left behind.
        mult = HEAL_OVERRIDE.get(i)
        if mult is not None:
            if len(mult) != dn.shape[0]:
                raise RuntimeError(f"heal override for layer {i} has {len(mult)} coefficients "
                                   f"but the layer has {dn.shape[0]} experts")
            dn.mul_(torch.tensor(mult, dtype=dn.dtype).view(-1, 1, 1))
        sd["mlp.experts.gate_up_proj"] = gu
        sd["mlp.experts.down_proj"] = dn
        expert_names.clear()

    missing, unexpected = layer.load_state_dict(sd, strict=False, assign=True)
    real_missing = [m for m in missing if "rotary" not in m and "inv_freq" not in m]
    if real_missing:
        log(f"layer {i}: {len(real_missing)} params missing from checkpoint "
            f"(e.g. {real_missing[:2]})", STAGE, "WARN")
    del sd
    return layer.to(DEV).eval()


# ---------------------------------------------------------------------------------------------
# Block execution. MEASURED 2026-09-27 (scripts/probe_layer_mem.py): each layer's forward
# consumes ~5.76 GiB of host memory that is NEVER returned while the process lives --
# empty_cache() gave back 0.03 GiB against 13.54 GiB consumed. 45 layers is ~259 GiB on a
# 122 GiB box, so this stage could not complete a single chunk: the pass-3 ledger was empty
# after five attempts because it died around layer 17-28 of chunk 0 every time. It was looping,
# not progressing.
#
# Only two things recover that memory, and BOTH are required:
#     process exit (CUDA context teardown) ..... 88.71 -> 93.66 GiB
#     then `echo 3 > drop_caches` .............. 93.66 -> 121.23 GiB  (full baseline)
#
# So the stage now runs as an ORCHESTRATOR that spawns one worker process per (chunk, layer
# block). The orchestrator never touches CUDA, so it never leaks; each worker exits and its
# memory comes back.
S03_ROLE = os.environ.get("S03_ROLE", "orchestrator")
# Cap the sweep for tests only. The equivalence gate needs an UNBLOCKED reference arm, and an
# unblocked 45-layer sweep is precisely the thing that cannot run here (259 GiB). Capping to a
# handful of layers makes both arms feasible so the comparison is possible at all. Unset in
# production, where it must never truncate the sweep.
S03_MAX_LAYERS = int(os.environ.get("S03_MAX_LAYERS", "0")) or None
# MEASURED 2026-09-28 by sampling /proc/meminfo every 2 s through a real 2-layer block:
#
#   t=  0s  117.9 GiB available
#   t= 24s   65.8   <- states loaded + layer 0 swept
#   t=145s   66.3   <- plateau: the memory NEVER comes back
#   t=157s   12.8   <- layer 1 swept, another ~53 GiB gone
#   t=171s    1.9   <- memfence refused layer 2
#
# 5.76 GiB is the WEIGHT size of a layer. The true host footprint of sweeping one is ~53 GiB --
# about 9x larger -- because on this integrated board torch.cuda.empty_cache() returns
# essentially nothing (this file, line ~334: "gave back 0.03 GiB against 13.54 GiB consumed").
# Device memory comes from system RAM via nvmap, is charged to no cgroup, and is reclaimed ONLY
# by process exit.
#
# Sizing against 5.76 is why 9, then 5, then even 2 layers per block all died: two layers alone
# exceed the box. One layer per process is the only configuration that fits, and the per-block
# process boundary is what returns the memory.
LAYER_COST_GIB = 53.0          # MEASURED host footprint, NOT the weight size
LAYER_WEIGHT_GIB = 5.76        # weight size only -- kept for logging, never for budgeting
# The FIXED cost a worker pays before it touches a single layer. The old budget counted only
# `layers x LAYER_COST_GIB` and therefore under-stated a 9-layer block by ~20 GiB:
#   reloaded states  17.5 GiB  ("reloaded 166 states (17.5 GiB)")
#   resident parts    2.2 GiB  (embeddings + vision tower, "resident small parts: 2.23 GiB")
STATES_GIB = 17.5
RESIDENT_GIB = 2.23
FIXED_GIB = STATES_GIB + RESIDENT_GIB
# Transients: forward activations, allocator fragmentation, the accumulator write. Empirical --
# 9 layers projected 71.5 GiB against a 72.0 GiB cap and was SIGKILLed, so the true overshoot is
# small but nonzero and there is no swap (MemorySwapMax=0) to absorb it.
MARGIN_GIB = 8.0
# Derived from the ceiling that actually binds, NOT hardcoded. An explicit override still wins so
# the equivalence gate can force an unblocked reference arm, but it is preflighted like any other.
LAYERS_PER_BLOCK = int(os.environ.get("S03_LAYERS_PER_BLOCK", "0")) or None
# Both overridable, so an equivalence test cannot pollute the real ledger or state files.
STATES_DIR = Path(os.environ.get("S03_STATES_DIR") or (ROOT / "artifacts" / "s03_states"))
BLOCK_LEDGER = Path(os.environ.get("S03_BLOCK_LEDGER") or (ROOT / "state" / "s03_blocks.json"))


def _effective_gib() -> float:
    """The ceiling that actually binds: min(host MemAvailable, cgroup headroom).

    MEASURED 2026-09-28: three consecutive crash-loop kills of chunk 8 block 0, each logging
    `114.8 GiB available` from /proc/meminfo while the cgroup enforced MemoryMax=72G. The guard
    was reading a ceiling that does not kill.
    """
    sys.path.insert(0, str(ROOT / "scripts"))
    from memceiling import effective_gib
    return effective_gib()


def _disk_free_gib() -> float:
    import shutil
    return shutil.disk_usage(ROOT).free / 2**30


def _require_disk(need_gib: float, label: str) -> None:
    """Refuse to start work that cannot land.

    MEASURED 2026-09-28: a per-chunk scratch filename grew artifacts/s03_states to 132 GB and
    filled a 936 GB disk to 100%. The run died with `OSError: [Errno 28] No space left on device`
    and took sqlite with it ("database or disk is full"), which is a far worse failure than
    stopping -- a half-written 17 GB tensor file and a half-written ledger are both garbage.
    Memory has had a pre-flight since the first wedge; disk had none.
    """
    free = _disk_free_gib()
    if free < need_gib:
        raise RuntimeError(
            f"{label}: {free:.1f} GiB free, need {need_gib:.1f} GiB. Refusing to start work that "
            f"cannot be written. Check artifacts/s03_states and run scripts/gate_disk_budget.py.")


def _avail_gib() -> float:
    with open("/proc/meminfo") as fh:
        for line in fh:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1048576
    return 0.0


def _full_reclaim() -> float:
    """process exit is only half of it; `echo 3` releases the nvmap pool. MEASURED: +27.5 GiB."""
    import subprocess
    import time as _t
    try:
        subprocess.run(["sync"], timeout=60, check=False)
        subprocess.run(["sudo", "-n", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"],
                       timeout=60, check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass
    _t.sleep(3)
    return _avail_gib()



# The block ledger records LAYER RANGES, not block indices.
#
# A block index is meaningless without the block size that produced it. When the size became
# derived from the observed memory ceiling (2026-09-28), index 4 stopped meaning "layers 36-44"
# and started meaning "layers 28-34" -- so a resumed run would have re-swept layers 35-44 for the
# eight finished chunks and added them to the accumulators a SECOND time. Silent double-counting
# of saliency is worse than a crash: it survives, and it biases which experts get pruned.
#
# Ranges are geometry-independent. Coverage is what matters, not indices.
LEGACY_BLOCK_SIZE = 9


def _migrate_ledger(done: set, n_layers: int):
    """Return (ledger, swept) where swept[chunk] is the set of layers already accumulated."""
    out, swept = set(), {}
    for e in done:
        if len(e) == 2 and e[1] == "prep":
            out.add(e)
        elif len(e) == 2 and isinstance(e[1], int):           # legacy (chunk, block_index)
            ci, bi = e
            lo = bi * LEGACY_BLOCK_SIZE
            hi = min(lo + LEGACY_BLOCK_SIZE, n_layers)
            out.add((ci, lo, hi))
            swept.setdefault(ci, set()).update(range(lo, hi))
        elif len(e) == 3:                                     # already (chunk, lo, hi)
            ci, lo, hi = e
            out.add((ci, lo, hi))
            swept.setdefault(ci, set()).update(range(lo, hi))
    return out, swept


def _block_state(swept: dict, ci: int, lo: int, hi: int) -> str:
    """"done" | "todo" | "partial" -- partial means the geometry changed mid-chunk."""
    have = swept.get(ci, set())
    want = set(range(lo, hi))
    if want <= have:
        return "done"
    if want & have:
        return "partial"
    return "todo"


def _oom_kills() -> int:
    sys.path.insert(0, str(ROOT / "scripts"))
    from memceiling import oom_kills
    return oom_kills()


def _check_worker_rc(rc: int, what: str, oom_before: int) -> None:
    """A worker that returns -9 was SIGKILLed -- almost always the cgroup OOM killer.

    MEASURED 2026-09-28: chunk 8 block 0 died three times at ~10 min with no traceback, no error
    line and no log entry of any kind, because SIGKILL leaves none. The orchestrator saw a
    non-zero rc and restarted, which reproduced it exactly. Naming the signal turns a silent
    crash loop into a one-line diagnosis.
    """
    if rc == 0:
        return
    sys.path.insert(0, str(ROOT / "scripts"))
    from memceiling import describe, oom_kills, peak_gib
    if rc == -9 or oom_kills() > oom_before:
        raise MemoryError(
            f"{what}: worker was OOM-KILLED (rc={rc}, cgroup peak {peak_gib() or 0:.1f} GiB, "
            f"{describe()}). Not a code fault -- the block does not fit. Reduce "
            f"S03_LAYERS_PER_BLOCK or raise the unit's MemoryMax.")
    raise RuntimeError(f"{what}: worker failed rc={rc}")

def _orchestrate(n_layers: int, n_chunks: int, n_dense: int = 0) -> dict:
    """Spawn one worker per (chunk, block); never import CUDA in this process."""
    import subprocess

    sys.path.insert(0, str(ROOT / "scripts"))
    from memceiling import describe, effective_gib, fit_units, host_ceiling_gib, require

    # HOST, not effective_gib(): this stage's dominant cost is device memory, which comes from
    # system RAM via nvmap and is charged to no cgroup. min(host, cgroup) would wrongly refuse --
    # the cgroup bounds only the ~20 GiB CPU-side working set, which fits it comfortably.
    from memceiling import host_ceiling_gib
    ceiling = host_ceiling_gib()
    pinned = None
    if BLOCK_LEDGER.exists():
        try:
            pinned = json.loads(BLOCK_LEDGER.read_text()).get("block_size")
        except Exception:
            pinned = None
    if LAYERS_PER_BLOCK is not None:
        per_block = LAYERS_PER_BLOCK          # explicit override (equivalence gate)
        why = f"S03_LAYERS_PER_BLOCK={per_block} (explicit override)"
    elif pinned:
        per_block = int(pinned)
        why = f"pinned by an earlier run in this ledger (ceiling now {ceiling:.1f} GiB)"
        need_pinned = FIXED_GIB + per_block * LAYER_COST_GIB
        if need_pinned + MARGIN_GIB > ceiling:
            raise MemoryError(
                f"pinned block size {per_block} needs {need_pinned:.1f} GiB but only "
                f"{describe()}. The ledger's geometry no longer fits. Free memory, or re-sweep "
                f"the unfinished chunks after clearing their ledger entries.")
    else:
        per_block = fit_units(ceiling, FIXED_GIB, LAYER_COST_GIB, MARGIN_GIB, n_layers)
        why = (f"derived: ({ceiling:.1f} ceiling - {FIXED_GIB:.1f} fixed - {MARGIN_GIB:.1f} "
               f"margin) / {LAYER_COST_GIB} per layer")
        if per_block < 1:
            raise MemoryError(
                f"cannot fit even ONE layer: {describe()}, fixed cost {FIXED_GIB:.1f} GiB. "
                f"Raise the unit's MemoryMax or reduce the chunk size.")
    log(f"block size {per_block} layers -- {why}", STAGE)

    blocks = [(lo, min(lo + per_block, n_layers))
              for lo in range(0, n_layers, per_block)]
    # Size the budget on the LARGEST ACTUAL block, not on the configured maximum. A model with
    # 4 layers and S03_LAYERS_PER_BLOCK=64 runs one 4-layer block costing ~23 GiB, but the
    # configured size predicted 369 GiB and refused to start -- found by
    # gate_s03_equivalence.sh, whose arm A deliberately uses an oversized block to mean
    # "unblocked".
    # The PEAK a worker reaches, every term included. The old form was
    # `max_layers * LAYER_COST_GIB`, which omitted the 17.5 GiB reloaded states and the 2.2 GiB
    # resident parts -- it predicted 51.8 GiB for a block that actually needed 71.5 GiB, and the
    # 72 GiB cgroup killed it three times in a row with no error line.
    need = FIXED_GIB + max(hi - lo for lo, hi in blocks) * LAYER_COST_GIB
    _have = host_ceiling_gib()
    if need + MARGIN_GIB > _have:
        raise MemoryError(
            f"s03 block sweep: needs {need:.1f} GiB + {MARGIN_GIB:.1f} margin, host has "
            f"{_have:.1f} GiB. The CPU-side share (~{FIXED_GIB:.1f} GiB) is what the cgroup "
            f"sees; the rest is nvmap and is charged to no cgroup.")
    done = set()
    if BLOCK_LEDGER.exists():
        try:
            done = {tuple(x) for x in json.loads(BLOCK_LEDGER.read_text()).get("done", [])}
        except Exception:
            done = set()
    done, swept = _migrate_ledger(done, n_layers)
    log(f"orchestrating {n_chunks} chunks x {len(blocks)} blocks of <={per_block} layers "
        f"(~{need:.0f} GiB per worker, measured {LAYER_COST_GIB} GiB/layer)", STAGE)

    for ci in range(n_chunks):
        # PREPARE is its own process. MEASURED 2026-09-27: chunk 9's block 0 was started by the
        # orchestrator with 113.7 GiB available and reached layer 1 with 6 GiB -- ~107 GiB
        # consumed by _prepare before a single layer ran. _prepare embeds every batch on the GPU
        # in a loop (166 batches for that chunk) and those allocations never come back, the same
        # leak as the layer sweep but in the one step that had neither a fence nor a process
        # boundary. Giving it its own process means its memory is returned before the sweep
        # starts, instead of being subtracted from the sweep's budget.
        # Don't prepare a chunk whose blocks are all finished -- on resume that would rebuild
        # 17 GiB of states for nothing. The existing 40-block ledger predates this phase.
        outstanding = [bi for bi, (lo, hi) in enumerate(blocks)
                       if _block_state(swept, ci, lo, hi) != "done"]
        if outstanding and (ci, "prep") not in done:
            # 17 GiB for the states file plus headroom for the accumulators and the ledger.
            _require_disk(25.0, f"chunk {ci} prepare")
            avail = _full_reclaim()
            log(f"chunk {ci} PREPARE starting, {avail:.1f} GiB available", STAGE)
            env = dict(os.environ, S03_ROLE="worker", S03_PHASE="prepare", S03_CHUNK=str(ci),
                       S03_LO="0", S03_HI="0")
            env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
            _oom0 = _oom_kills()
            rc = subprocess.run([sys.executable,
                                 str(ROOT / "scripts" / "s03_worker_main.py")],
                                env=env, cwd=str(ROOT)).returncode
            _check_worker_rc(rc, f"chunk {ci} prepare", _oom0)
            done.add((ci, "prep"))
            BLOCK_LEDGER.parent.mkdir(parents=True, exist_ok=True)
            BLOCK_LEDGER.write_text(json.dumps({"block_size": per_block,
                                               "done": sorted(map(list, done), key=str)}))
            log(f"chunk {ci} PREPARE done", STAGE)

        for bi, (lo, hi) in enumerate(blocks):
            state = _block_state(swept, ci, lo, hi)
            if state == "done":
                continue
            if state == "partial":
                # Some of these layers are already in the accumulators. Re-running the block would
                # count them twice. Refuse loudly instead of silently corrupting the saliency.
                raise RuntimeError(
                    f"chunk {ci} block {bi} (layers {lo}-{hi-1}) partially swept under a previous "
                    f"block geometry. Re-running would double-count. Delete this chunk's entries "
                    f"from {BLOCK_LEDGER} and re-sweep the whole chunk.")
            _full_reclaim()
            # Refuse rather than start a worker that cannot finish. A run that begins below
            # baseline is the run that wedges the box.
            #
            # `need` now carries the fixed 19.7 GiB, and the ceiling is min(host, cgroup), so the
            # old `+ 18.0` fudge is gone: the reserve it stood in for is a counted term.
            avail = host_ceiling_gib()
            if avail < need + MARGIN_GIB:
                log(f"ABORT before chunk {ci} block {bi}: need {need:.1f} + {MARGIN_GIB:.1f} "
                    f"margin, have {describe()}. Refusing -- this would be OOM-killed.",
                    STAGE, "ERROR")
                raise RuntimeError(f"insufficient memory: {describe()}")
            log(f"chunk {ci} block {bi} (layers {lo}-{hi-1}) starting, need {need:.1f} GiB, "
                f"{describe()}", STAGE)
            env = dict(os.environ, S03_ROLE="worker", S03_CHUNK=str(ci),
                       S03_LO=str(lo), S03_HI=str(hi))
            # expandable_segments is wrong on Tegra (unsloth-zoo#1235) and this device reports
            # is_integrated=1; make sure an inherited value cannot reach the worker.
            env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
            # Fingerprint the dumps so a worker that exits 0 without doing anything cannot be
            # mistaken for one that succeeded -- exactly what happened when workers were spawned
            # through run_stage.py and silently lost the stage flock.
            before = {p.name: (p.stat().st_mtime_ns, p.stat().st_size)
                      for p in SALIENCY.glob("*.pt")}
            _oom0 = _oom_kills()
            rc = subprocess.run([sys.executable,
                                 str(ROOT / "scripts" / "s03_worker_main.py")],
                                env=env, cwd=str(ROOT)).returncode
            _check_worker_rc(rc, f"chunk {ci} block {bi} (layers {lo}-{hi-1})", _oom0)
            after = {p.name: (p.stat().st_mtime_ns, p.stat().st_size)
                     for p in SALIENCY.glob("*.pt")}
            # Only a block that actually CONTAINS a MoE layer is expected to write a dump. The
            # first `first_k_dense_replace` layers are dense and have no router or experts, so a
            # block wholly inside them legitimately changes nothing -- the first version of this
            # check called that a silent no-op and failed a correct run.
            has_moe = hi > n_dense
            if has_moe and after == before:
                raise RuntimeError(
                    f"chunk {ci} block {bi} (layers {lo}-{hi-1}) contains MoE layers but the "
                    f"worker exited 0 without changing any layer dump. A zero exit from a worker "
                    f"that never ran is not success.")
            done.add((ci, lo, hi))
            swept.setdefault(ci, set()).update(range(lo, hi))
            BLOCK_LEDGER.parent.mkdir(parents=True, exist_ok=True)
            BLOCK_LEDGER.write_text(json.dumps({"block_size": per_block,
                                               "done": sorted(map(list, done), key=str)}))
            log(f"chunk {ci} block {bi} done", STAGE)
    _full_reclaim()

    # Completion is the ORCHESTRATOR's job now. Workers return early, so without this nothing
    # would ever run the audit or set saliency_ready and the pipeline would wait forever on a
    # stage that had actually finished. _audit() only torch.loads the dumps on CPU, so the
    # parent still never initialises CUDA.
    audit = _audit()
    n = len(sorted(SALIENCY.glob("*.pt")))
    kv_set("saliency_ready", True)
    res = {"layers": n, "chunks": n_chunks, "blocks": len(blocks),
           "layers_per_block": LAYERS_PER_BLOCK, "expert_audit": audit,
           "sparsity": TARGET_SPARSITY, "max_len": MAX_LEN, "path": "stream-blocked"}
    (ARTIFACTS / "s03_saliency.json").write_text(json.dumps(res, indent=2, default=str))
    log(f"saliency COMPLETE: {n} layer dumps over {n_chunks} chunks x {len(blocks)} blocks",
        STAGE)
    return res


def run() -> dict:
    import torch
    from transformers import AutoConfig
    from transformers.models.glm5_next.modeling_glm5_next import (
        Glm5NextForConditionalGeneration, create_recurrent_attention_mask)
    from accelerate import init_empty_weights
    import stream_saliency as SS

    global DT
    DT = torch.bfloat16
    SALIENCY.mkdir(parents=True, exist_ok=True)

    # ---- orchestrator branch: plan the blocks and spawn workers, touching no CUDA ----------
    # Importing torch does not initialise CUDA; only a device op does. Nothing below this branch
    # runs in the parent, so the parent cannot accumulate the 5.76 GiB/layer that never returns.
    if S03_ROLE == "orchestrator":
        _cfg = AutoConfig.from_pretrained(SRC)
        _t = getattr(_cfg, "text_config", _cfg)
        _text_rows, _ = _load_calib()
        _per_chunk = max(1, CHUNK_TOKENS // MAX_LEN)
        _n_chunks = max(1, -(-len(_text_rows) // _per_chunk))
        del _text_rows
        gc.collect()
        _nl = min(_t.num_hidden_layers, S03_MAX_LAYERS or _t.num_hidden_layers)
        if _nl != _t.num_hidden_layers:
            log(f"S03_MAX_LAYERS={_nl} -- TEST MODE, sweeping only the first {_nl} layers",
                STAGE, "WARN")
        return _orchestrate(_nl, _n_chunks, getattr(_t, "first_k_dense_replace", 0))
    strategy = kv_get("load_strategy", "stream")
    if strategy != "stream":
        log(f"load strategy is '{strategy}', but this stage implements the streaming path only",
            STAGE, "WARN")

    cfg = AutoConfig.from_pretrained(SRC)
    tcfg = cfg.text_config
    reader = SS.ShardReader(SRC)
    SS.patch_experts_for_saliency()
    # Cache the router's per-token scores. This is what turns one saliency pass into an offline
    # laboratory: post-prune routing (selection AND the renormalised gate) becomes exactly
    # replayable for any candidate keep-set, which is what P5's healing re-fit and P9's
    # router-aware re-scoring need and what pass 1 could not do at any price.
    SS.patch_router_for_cache()
    SS.reset_accumulators()

    text_rows, mm_rows = _load_calib()

    # The small, permanently-resident parts: embeddings and the vision tower (~2.5 GB bf16).
    #
    # CRITICAL: build the shell on meta and materialise ONLY these two submodules. Calling
    # shell.to_empty(device="cpu") on the whole model allocates all 321B parameters (~642 GB at
    # bf16) and takes the box from 116 GiB available to 1.5 GiB in seconds - that is what killed
    # this stage six times. Everything else stays on meta and costs nothing; the decoder layers
    # are streamed in one at a time later.
    log("materialising embeddings + vision tower ONLY (rest stays on meta)", STAGE)
    with init_empty_weights():
        shell = Glm5NextForConditionalGeneration._from_config(cfg)

    def _materialise(mod, prefix):
        mod.to_empty(device="cpu")
        sd = {}
        for name in reader.map:
            if not name.startswith(prefix) or name.endswith("weight_scale_inv"):
                continue
            t = reader.get(name)
            sd[name[len(prefix):]] = (t.to(DT, copy=True) if t.is_floating_point()
                                      else t.clone())
        missing, _ = mod.load_state_dict(sd, strict=False, assign=True)
        real = [m for m in missing if "inv_freq" not in m and "rotary" not in m]
        if real:
            log(f"{prefix}: {len(real)} params missing (e.g. {real[:2]})", STAGE, "WARN")
        del sd
        return mod

    _materialise(shell.model.visual, "model.visual.")
    _materialise(shell.model.language_model.embed_tokens, "model.language_model.embed_tokens.")
    embed = shell.model.language_model.embed_tokens.to(DEV)
    visual = shell.model.visual.to(DEV, DT).eval()
    mm_model = shell.model
    gc.collect()
    resident = (sum(p.numel() * p.element_size() for p in visual.parameters())
                + sum(p.numel() * p.element_size() for p in embed.parameters())) / 2**30
    log(f"resident small parts: {resident:.2f} GiB", STAGE)

    def embeds_for(batch_ids, mm_batch):
        if mm_batch is None:
            # Pad every text batch to the SAME length. Variable shapes make the caching
            # allocator reserve a fresh ~13 GiB arena per distinct length; one shape lets it
            # reuse a single block for the whole sweep.
            ids = torch.full((len(batch_ids), MAX_LEN), tcfg.pad_token_id, dtype=torch.long)
            for r, b in enumerate(batch_ids):
                n = min(len(b), MAX_LEN)
                ids[r, :n] = b[:n].long()
            ids = ids.to(DEV)
            return ids, embed(ids)
        rec = mm_batch
        ids = rec["input_ids"].long().unsqueeze(0).to(DEV)
        ie = embed(ids)
        pv = rec["pixel_values"].to(DEV, DT)
        thw = rec["image_grid_thw"].to(DEV)
        feats = mm_model.get_image_features(pv, thw).pooler_output
        feats = torch.cat(feats, dim=0).to(ie.device, ie.dtype)
        mask, _ = mm_model.get_placeholder_mask(ids, inputs_embeds=ie, image_features=feats)
        return ids, ie.masked_scatter(mask, feats)

    # ---- chunked, resumable sweep ----------------------------------------------------
    # Pass 1 kept every batch's activations in RAM for the whole layer sweep, which is what
    # capped it at 0.52M tokens. Chunking trades one extra read of the weights per chunk
    # (306 GiB at 3.4 GB/s, ~90 s) for an effectively unbounded token budget. On this box read
    # is 7x faster than write, so re-reading weights beats spilling activations every time.
    def _prepare(chunk_text, chunk_mm):
        states = []
        with torch.no_grad():
            # Batches must be homogeneous in bucket - the accumulators attribute a whole batch
            # to one domain. Grouping first is free; it only changes the packing order.
            by_bucket: dict[str, list] = {}
            for ids_row, bkt in chunk_text:
                by_bucket.setdefault(bkt, []).append(ids_row)
            for bkt, rows in by_bucket.items():
                for i0 in range(0, len(rows), BATCH):
                    ids, ie = embeds_for(rows[i0:i0 + BATCH], None)
                    # Padding routes like any other token. Pass 1 accumulated it as though it
                    # were text, most heavily for the shortest documents.
                    valid = (ids != tcfg.pad_token_id).reshape(-1)
                    states.append({"ids": ids.cpu(), "bucket": bkt, "valid": valid.cpu(),
                                   "hs": ie.unsqueeze(2).expand(-1, -1, tcfg.hc_mult, -1)
                                   .contiguous().cpu(), "topk": None})
                    del ie
            for rec in chunk_mm:
                try:
                    ids, ie = embeds_for(None, rec)
                    states.append({"ids": ids.cpu(), "bucket": "vision",
                                   "valid": torch.ones(ids.numel(), dtype=torch.bool),
                                   "hs": ie.unsqueeze(2).expand(-1, -1, tcfg.hc_mult, -1)
                                   .contiguous().cpu(), "topk": None})
                    del ie
                except Exception as e:
                    log(f"image-text sample skipped ({type(e).__name__}: {str(e)[:120]})",
                        STAGE, "WARN")
        return states

    def _sweep(states, tag, lo=0, hi=None):
        t0 = time.time()
        hi = tcfg.num_hidden_layers if hi is None else hi
        for li in range(lo, hi):
            # The orchestrator checks before a BLOCK; nothing checked within one. Chunk 9 ran off
            # that cliff. An exception unwinds Python and frees tensors; a wedged box does not.
            MF.require(LAYER_COST_GIB, f"chunk {CI} layer {li}")
            layer = _build_layer(tcfg, li, reader, DT)
            SS.set_current_layer(f"model.language_model.layers.{li}.mlp")
            ltype = tcfg.layer_types[li]
            bi_seen = 0
            for st in states:
                try:
                    with torch.no_grad():
                        hs = st["hs"].to(DEV, non_blocking=False)
                        ids = st["ids"].to(DEV)
                        SS.set_bucket(st["bucket"])
                        SS.set_valid_mask(st["valid"].to(DEV))
                        am = torch.ones(ids.shape[0], ids.shape[1], dtype=torch.bool, device=DEV)
                        pos = torch.arange(ids.shape[1], device=DEV).unsqueeze(0)
                        topk = st["topk"].to(DEV) if st["topk"] is not None else None
                        out, topk = layer(hs, attention_mask=am, position_ids=pos,
                                          position_embeddings=None, input_ids=ids,
                                          past_key_values=None, use_cache=False,
                                          prev_topk_indices=topk)
                        # Back to host immediately. Device memory here is driver-pinned: the
                        # kernel cannot see, swap or reclaim it, so anything left resident is
                        # permanently unavailable until the process exits.
                        st["hs"] = out.cpu()
                        st["topk"] = topk.cpu() if topk is not None else None
                        del hs, out, ids, am, pos, topk
                except Exception as e:
                    log(f"layer {li} batch failed ({type(e).__name__}: {str(e)[:160]})",
                        STAGE, "WARN")
                    gc.collect()
                    torch.cuda.empty_cache()
                finally:
                    SS.set_valid_mask(None)
                bi_seen += 1
                if bi_seen % 25 == 0:
                    torch.cuda.empty_cache()
            SS.set_current_layer(None)
            del layer
            reader.release()      # tear down shard mmaps so their pages become reclaimable
            gc.collect()
            torch.cuda.empty_cache()
            _reclaim_page_cache()
            el = time.time() - t0
            avail = 0.0
            try:
                with open("/proc/meminfo") as fh:
                    for line in fh:
                        if line.startswith("MemAvailable"):
                            avail = int(line.split()[1]) / 1048576
                            break
            except OSError:
                pass
            n_here = li - lo + 1
            log(f"{tag} layer {li+1}/{tcfg.num_hidden_layers} (block {lo}-{hi-1}, "
                f"{n_here}/{hi-lo}) ({ltype})  elapsed {el/60:.1f} min  "
                f"eta {(el/n_here)*(hi-li-1)/60:.0f} min  avail {avail:.0f} GiB", STAGE)
            SS.dump(SALIENCY)  # checkpoint per layer: a kill costs at most one layer

    # Chunk by token count, not sample count, so a chunk's activation footprint is predictable
    # regardless of how the mixture happens to be packed.
    per_chunk = max(1, CHUNK_TOKENS // MAX_LEN)
    text_chunks = [text_rows[k:k + per_chunk] for k in range(0, len(text_rows), per_chunk)]
    if not text_chunks:
        raise RuntimeError("no calibration text rows")
    mm_per = max(1, -(-len(mm_rows) // len(text_chunks)))
    mm_chunks = [mm_rows[k:k + mm_per] for k in range(0, len(mm_rows), mm_per)]
    mm_chunks += [[]] * (len(text_chunks) - len(mm_chunks))

    ledger = ROOT / "state" / "s03_chunks.json"
    done = set()
    if ledger.exists():
        try:
            done = set(json.loads(ledger.read_text()).get("done", []))
        except Exception:
            done = set()
    if done:
        # Resume: the accumulators live in the per-layer dumps, so a killed run reloads them
        # rather than starting the token budget over.
        loaded = SS.load_accumulators(SALIENCY, DEV)
        log(f"resuming after chunk(s) {sorted(done)}; reloaded {loaded} layer accumulators",
            STAGE, )
    log(f"{len(text_chunks)} chunks x ~{per_chunk} samples ({CHUNK_TOKENS/1e6:.1f}M tokens each), "
        f"{CALIB_TOKENS/1e6:.1f}M tokens total", STAGE)

    t0 = time.time()

    # ---- worker: exactly one (chunk, block) ------------------------------------------------
    # Hidden states are the only thing that must survive a block boundary, so they go to disk.
    # They are OVERWRITTEN rather than deleted: nothing in this pipeline removes an artifact.
    CI = int(os.environ["S03_CHUNK"])
    LO = int(os.environ["S03_LO"])
    HI = int(os.environ["S03_HI"])
    STATES_DIR.mkdir(parents=True, exist_ok=True)
    # ONE file, reused. It was `chunk_{CI:03d}.pt`, which is per-chunk, so every chunk wrote a
    # NEW ~17 GB file and they accumulated: 8 chunks = 132 GB, which filled a 936 GB disk to 100%
    # and killed the run with `OSError: [Errno 28] No space left on device`. The comment next to
    # it claimed the file was "OVERWRITTEN rather than deleted" -- the filename made that false.
    #
    # States are scratch that lives only between the blocks of ONE chunk: the next chunk's
    # prepare overwrites them and nothing ever reads a previous chunk's. A single reused path is
    # both correct and bounded at ~17 GB.
    # STATES_DIR is overridable so a measurement or gate run cannot advance the real states.
    # 2026-09-28: a 2-layer probe redirected S03_SALIENCY_DIR but NOT this path; had it completed
    # it would have written states advanced by two layers while the ledger recorded nothing, and
    # the next real run would have re-swept layers 0-1 from an already-advanced tensor. It raised
    # on the memfence before the save, so nothing was lost -- luck, not design.
    spath = STATES_DIR / "states.pt"

    if os.environ.get("S03_PHASE") == "prepare":
        ct, cm = text_chunks[CI], mm_chunks[CI]
        states = _prepare(ct, cm)
        if not states:
            raise RuntimeError(f"chunk {CI}: no batches prepared")
        gib = sum(st["hs"].numel() * st["hs"].element_size() for st in states) / 2**30
        torch.save(states, spath)
        log(f"chunk {CI} prepared: {len(states)} batches, {gib:.1f} GiB states -> {spath.name}; "
            f"exiting so the embedding pass's memory is returned", STAGE)
        return {"chunk": CI, "phase": "prepare", "batches": len(states)}

    if not spath.exists():
        raise RuntimeError(f"chunk {CI} block {LO}-{HI-1} needs hidden states, but {spath} is "
                           f"missing -- the prepare phase must run first")
    states = torch.load(spath, map_location="cpu", weights_only=False)
    log(f"chunk {CI} block {LO}-{HI-1}: reloaded {len(states)} states "
        f"({spath.stat().st_size/2**30:.1f} GiB)", STAGE)
    act_gib = sum(st["hs"].numel() * st["hs"].element_size() for st in states) / 2**30
    log(f"chunk {CI+1}/{len(text_chunks)} block {LO}-{HI-1}: {len(states)} batches, "
        f"{act_gib:.1f} GiB activations on HOST", STAGE)

    # ALWAYS continue from whatever has already been accumulated. load_accumulators() restores
    # the per-layer dumps onto the device and is a no-op on an empty directory, so the very first
    # worker loads nothing and every later one resumes the running totals.
    #
    # This was `if LO > 0` and that was WRONG, caught by gate_s03_equivalence.sh. Every chunk's
    # block 0 starts at LO=0, so it skipped the load, began from zero, and its dump OVERWROTE
    # every previous chunk's contribution for the layers in that block. With the production block
    # of 9, layers 3-8 would have retained only the LAST chunk while layers 9+ accumulated all
    # ten -- a per-layer data imbalance of 10x, silent, in the numbers the mask is chosen from.
    #
    # Safety of loading unconditionally rests entirely on the (chunk, block) ledger: a block is
    # never rescheduled, so a restored total is never added to twice.
    SS.load_accumulators(SALIENCY, DEV)

    _sweep(states, f"chunk {CI+1}/{len(text_chunks)}", LO, HI)
    SS.dump(SALIENCY)

    _n_layers = min(tcfg.num_hidden_layers, S03_MAX_LAYERS or tcfg.num_hidden_layers)
    if HI < _n_layers:
        torch.save(states, spath)
        log(f"chunk {CI} block ends at layer {HI-1}; states saved for the next worker", STAGE)
        return {"chunk": CI, "block": [LO, HI], "more": True}

    nrc = SS.dump_router_cache(ROUTER_CACHE_DIR / f"chunk_{CI:03d}.pt")
    SS.dump_light(SNAPSHOT_DIR / f"chunk_{CI:03d}")
    log(f"chunk {CI+1}/{len(text_chunks)} COMPLETE ({nrc} cached router rows)", STAGE)
    return {"chunk": CI, "block": [LO, HI], "more": False, "router_rows": nrc}

    # ---- unreachable: the single-process path, kept for reference --------------------------
    for ci, (ct, cm) in enumerate(zip(text_chunks, mm_chunks)):
        if ci in done:
            continue
        states = _prepare(ct, cm)
        if not states:
            log(f"chunk {ci}: no batches prepared, skipping", STAGE, "WARN")
            continue
        act_gib = sum(st["hs"].numel() * st["hs"].element_size() for st in states) / 2**30
        log(f"chunk {ci+1}/{len(text_chunks)}: {len(states)} batches, {act_gib:.1f} GiB "
            f"activations on HOST (reclaimable; device memory here is driver-pinned)", STAGE)
        _sweep(states, f"chunk {ci+1}/{len(text_chunks)}")
        nrc = SS.dump_router_cache(ROUTER_CACHE_DIR / f"chunk_{ci:03d}.pt")
        SS.dump(SALIENCY)
        # Cumulative snapshot per chunk: the split-half gate (P6) cannot be computed from a
        # single running total, and the sweep is far too expensive to re-run to get one.
        SS.dump_light(SNAPSHOT_DIR / f"chunk_{ci:03d}")
        done.add(ci)
        ledger.parent.mkdir(parents=True, exist_ok=True)
        ledger.write_text(json.dumps({"done": sorted(done), "chunks": len(text_chunks)}))
        del states
        gc.collect()
        torch.cuda.empty_cache()
        _reclaim_page_cache()
        log(f"chunk {ci+1}/{len(text_chunks)} done ({nrc} cached router rows); "
            f"elapsed {(time.time()-t0)/60:.1f} min", STAGE)

    n = SS.dump(SALIENCY)
    dt = time.time() - t0
    metric(STAGE, "saliency_pass_minutes", dt / 60)
    log(f"saliency accumulated over {n} layers in {dt/60:.1f} min", STAGE)

    audit = _audit()
    kv_set("saliency_ready", True)
    res = {"layers": n, "minutes": round(dt / 60, 1), "chunks": len(text_chunks),
           "expert_audit": audit, "sparsity": TARGET_SPARSITY,
           "calib_samples": len(text_rows) + len(mm_rows), "max_len": MAX_LEN,
           "path": "stream"}
    p = ARTIFACTS / "s03_saliency.json"
    p.write_text(json.dumps(res, indent=2, default=str))
    publish(p, "artifacts", "stage03/s03_saliency.json", stage=STAGE)
    publish(SALIENCY, "saliency", "raw", stage=STAGE)
    return res


def _audit() -> dict:
    import torch
    files = sorted(SALIENCY.glob("*.pt"))
    if not files:
        raise RuntimeError("no saliency accumulated")
    mins, means, starved, tot = [], [], 0, 0
    for f in files:
        d = torch.load(f, weights_only=False)
        c = d["count"].double()
        mins.append(float(c.min()))
        means.append(float(c.mean()))
        starved += int((c < MIN_TOKENS_PER_EXPERT).sum())
        tot += c.numel()
    res = {"layers": len(files), "min_tokens_per_expert": min(mins),
           "mean_tokens_per_expert": round(sum(means) / len(means), 1),
           "experts_below_floor": starved, "expert_slots": tot,
           "floor": MIN_TOKENS_PER_EXPERT}
    for k, v in res.items():
        if isinstance(v, (int, float)):
            metric(STAGE, f"saliency_{k}", v)
    log(f"per-expert sampling audit: {res}", STAGE)
    if starved:
        log(f"{starved}/{tot} expert slots below the {MIN_TOKENS_PER_EXPERT}-token floor - "
            f"their saliency is noisy, so near-threshold deletions there are effectively "
            f"random (R1). Raise n_calib_samples to tighten.", STAGE, "WARN")
    return res
