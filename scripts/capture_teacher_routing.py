"""Capture the UNPRUNED teacher's routing decisions at long context, before surgery destroys it.

WHY THIS EXISTS AND WHY IT HAS A DEADLINE. `s04b_surgery` deletes source shards as it writes
survivors. Once it runs, the unpruned GLM teacher cannot be recovered without re-downloading
328 GB. This is the only long-context test that becomes impossible afterwards, so it runs first.

WHAT IT BUYS. The question is whether a mask calibrated at 2,048 tokens is still the right mask
at position 100,000 -- i.e. whether the fraction of routed slots landing on a pruned expert rises
with token POSITION. That needs the teacher's top-8 choices per token per layer, and nothing
else. 42 MoE layers x 1M tokens x top-8 as int16 is 0.66 GiB, so the teacher's complete routing
behaviour at full context fits in under a gigabyte.

Once captured, affected-rate-versus-position is computable OFFLINE against any candidate mask --
including masks that do not exist yet -- with no GPU and no teacher. That is the same trick that
dissolved the router-KD sequencing constraint (wiki/98): the teacher is needed as a source of
decisions, not as a live model.

MEMORY. This sweeps layers exactly like s03_saliency, so it carries the same ~5.76 GiB/layer that
is never returned in-process (wiki/96). It therefore uses the same block-per-process structure:
one worker per layer block, `drop_caches 3` between, and a verified return to baseline before the
next block starts.

    python scripts/capture_teacher_routing.py                 # orchestrator
    S03C_ROLE=worker S03C_LO=0 S03C_HI=9 python ...           # one block (spawned)
"""
from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, "scripts")
sys.path.insert(0, "scripts/stages")

from common import ROOT, log  # noqa: E402
import memfence as MF  # noqa: E402

STAGE = "s03c_routing"
SRC = ROOT / "source" / "GLM-5.3-Flash"
OUT = ROOT / "artifacts" / "teacher_routing"
STATES = OUT / "_states.pt"
LEDGER = ROOT / "state" / "s03c_blocks.json"
DEV = "cuda"
DT = torch.bfloat16

ROLE = os.environ.get("S03C_ROLE", "orchestrator")
# MEASURED 2026-09-29 on s03_saliency, which sweeps layers identically. 5.76 GiB is the WEIGHT
# size of a layer; the true HOST footprint of sweeping one is ~53 GiB, because on this integrated
# board torch.cuda.empty_cache() returns essentially nothing and device memory (nvmap) comes from
# system RAM and is reclaimed only by drop_caches=3. Sizing against 5.76 is why s03 died at 9, at
# 5 and even at 2 layers per block. One layer per process is the only configuration that fits.
LAYERS_PER_BLOCK = int(os.environ.get("S03C_LAYERS_PER_BLOCK", "1"))
LAYER_COST_GIB = 53.0          # MEASURED host footprint
LAYER_WEIGHT_GIB = 5.76        # weight size only -- never for budgeting
FIXED_GIB = 19.73              # reloaded states (17.5) + resident embeddings/vision (2.23)
# Reclaim as soon as headroom falls below this, regardless of batch count. s03 measured a 40 GiB
# collapse inside a single 25-batch interval, which a fixed schedule cannot see coming.
RECLAIM_FLOOR_GIB = float(os.environ.get("S03C_RECLAIM_FLOOR", "30.0"))
HARD_FLOOR_GIB = float(os.environ.get("S03C_HARD_FLOOR", "6.0"))
# Real documents only. The corpus caps at 16,384 (corpus_spec.MAX_TOKENS), and concatenating
# shorter samples to reach a longer sequence would manufacture long-range structure that is not
# there -- the routing at a fake boundary is not the routing we are asking about.
SEQ = int(os.environ.get("S03C_SEQ", "16384"))
N_ROWS = int(os.environ.get("S03C_ROWS", "16"))


def _avail_gib() -> float:
    with open("/proc/meminfo") as fh:
        for line in fh:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1048576
    return 0.0


def _reclaim() -> float:
    subprocess.run(["sync"], timeout=60, check=False)
    subprocess.run(["sudo", "-n", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"],
                   timeout=60, check=False, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)
    time.sleep(3)
    return _avail_gib()


def _rows():
    """Longest real samples available, untruncated."""
    import glob
    rows = []
    for f in sorted(glob.glob(str(ROOT / "corpus/shards/text/*.pt"))):
        d = torch.load(f, map_location="cpu", weights_only=False)
        items = d if isinstance(d, list) else d.get("items", d)
        for it in items:
            ids = it.get("input_ids")
            if ids is not None and len(ids) >= SEQ:
                rows.append(torch.as_tensor(ids[:SEQ], dtype=torch.long))
                if len(rows) >= N_ROWS:
                    return rows
    return rows


def orchestrate(n_layers: int) -> int:
    blocks = [(lo, min(lo + LAYERS_PER_BLOCK, n_layers))
              for lo in range(0, n_layers, LAYERS_PER_BLOCK)]
    # FIXED_GIB is the part the old form omitted: it counted only layers x cost and under-stated
    # a block by ~20 GiB.
    need = FIXED_GIB + max(hi - lo for lo, hi in blocks) * LAYER_COST_GIB
    done = set()
    if LEDGER.exists():
        try:
            done = {tuple(x) for x in json.loads(LEDGER.read_text()).get("done", [])}
        except Exception:
            done = set()
    log(f"{len(blocks)} blocks of <={LAYERS_PER_BLOCK} layers (~{need:.0f} GiB per worker); "
        f"{N_ROWS} rows x {SEQ} tokens of REAL documents", STAGE)
    for bi, (lo, hi) in enumerate(blocks):
        if (bi,) in done or bi in {d[0] for d in done if len(d) == 1}:
            continue
        avail = _reclaim()
        if avail < need + 8.0:
            log(f"ABORT before block {bi}: {avail:.1f} GiB available, need {need:.0f} + 8 "
                f"reserve. Memory did not return after the last worker.", STAGE, "ERROR")
            return 5
        log(f"block {bi} (layers {lo}-{hi-1}) starting, {avail:.1f} GiB available", STAGE)
        env = dict(os.environ, S03C_ROLE="worker", S03C_LO=str(lo), S03C_HI=str(hi))
        env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        before = {p.name: p.stat().st_mtime_ns for p in OUT.glob("layer_*.pt")}
        rc = subprocess.run([sys.executable, str(ROOT / "scripts" / "capture_teacher_routing.py")],
                            env=env, cwd=str(ROOT)).returncode
        if rc != 0:
            log(f"block {bi} worker rc={rc}", STAGE, "ERROR")
            return rc
        after = {p.name: p.stat().st_mtime_ns for p in OUT.glob("layer_*.pt")}
        if hi > 3 and after == before:
            log(f"block {bi}: worker exited 0 but wrote nothing. A zero exit from a worker that "
                f"never ran is not success.", STAGE, "ERROR")
            return 6
        done.add((bi,))
        LEDGER.parent.mkdir(parents=True, exist_ok=True)
        LEDGER.write_text(json.dumps({"done": sorted(list(d) for d in done)}))
        log(f"block {bi} done", STAGE)
    _reclaim()
    n = len(list(OUT.glob("layer_*.pt")))
    (OUT / "manifest.json").write_text(json.dumps(
        {"seq": SEQ, "rows": N_ROWS, "layers_captured": n,
         "source": "unpruned GLM-5.3-Flash, pre-surgery", "topk": 8}, indent=2))
    log(f"COMPLETE: routing captured for {n} MoE layers at {SEQ} tokens", STAGE)
    return 0


def worker(lo: int, hi: int) -> int:
    import stream_saliency as SS
    from s03_saliency import _build_layer
    from transformers import AutoConfig
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextForConditionalGeneration
    from accelerate import init_empty_weights

    cfg = AutoConfig.from_pretrained(SRC)
    tcfg = getattr(cfg, "text_config", cfg)
    reader = SS.ShardReader(SRC)
    OUT.mkdir(parents=True, exist_ok=True)

    if lo == 0:
        rows = _rows()
        if len(rows) < N_ROWS:
            log(f"only {len(rows)} rows of >={SEQ} tokens available of {N_ROWS}", STAGE, "ERROR")
            return 1
        with init_empty_weights():
            shell = Glm5NextForConditionalGeneration._from_config(cfg)
        emb = shell.model.language_model.embed_tokens
        emb.to_empty(device="cpu")
        pfx = "model.language_model.embed_tokens."
        emb.load_state_dict({n[len(pfx):]: reader.get(n).to(DT, copy=True)
                             for n in reader.map if n.startswith(pfx)},
                            strict=False, assign=True)
        emb = emb.to(DEV)
        del shell
        gc.collect()
        states = []
        with torch.no_grad():
            for r in rows:
                ids = r.unsqueeze(0).to(DEV)
                ie = emb(ids)
                states.append({"ids": ids.cpu(),
                               "hs": ie.unsqueeze(2).expand(-1, -1, tcfg.hc_mult, -1)
                                     .contiguous().cpu(), "topk": None})
                del ie
        del emb
        gc.collect()
        torch.cuda.empty_cache()
    else:
        states = torch.load(STATES, map_location="cpu", weights_only=False)

    n_dense = getattr(tcfg, "first_k_dense_replace", 0)
    for li in range(lo, hi):
        MF.require(LAYER_COST_GIB, f"routing capture layer {li}")
        layer = _build_layer(tcfg, li, reader, DT)
        per_row = []
        _seen = 0
        with torch.no_grad():
            for st in states:
                hs = st["hs"].to(DEV)
                ids = st["ids"].to(DEV)
                am = torch.ones(ids.shape[0], ids.shape[1], dtype=torch.bool, device=DEV)
                pos = torch.arange(ids.shape[1], device=DEV).unsqueeze(0)
                tk = st["topk"].to(DEV) if st["topk"] is not None else None
                out, tk = layer(hs, attention_mask=am, position_ids=pos,
                                position_embeddings=None, input_ids=ids,
                                past_key_values=None, use_cache=False, prev_topk_indices=tk)
                st["hs"] = out.cpu()
                st["topk"] = tk.cpu() if tk is not None else None
                if li >= n_dense and tk is not None:
                    # [tokens, top_k] expert ids, int16 -- position is the row index, which is
                    # the whole point: this is what lets affected-rate be plotted against it.
                    per_row.append(tk.reshape(-1, tk.shape[-1]).to(torch.int16).cpu())
                del hs, out, ids, am, pos, tk
                _seen += 1
                # Watch the floor every batch, not a batch counter. MEASURED on s03 layer 4:
                # a stable ~52 GiB plateau then 40 GiB gone in 26 s, ending 550 MB above
                # memguard's kill floor -- entirely inside one fixed reclaim interval.
                if _seen % 25 == 0 or _avail_gib() < RECLAIM_FLOOR_GIB:
                    torch.cuda.empty_cache()
                    _reclaim()
                    if _avail_gib() < HARD_FLOOR_GIB:
                        raise RuntimeError(
                            f"routing capture layer {li} batch {_seen}: {_avail_gib():.1f} GiB "
                            f"left after a full reclaim. Stopping rather than being SIGKILLed.")
        if per_row:
            torch.save({"layer": li, "seq": SEQ, "rows": len(per_row),
                        "topk_ids": torch.stack(per_row)}, OUT / f"layer_{li:03d}.pt")
        del layer, per_row
        reader.release()
        gc.collect()
        torch.cuda.empty_cache()
        log(f"layer {li} captured ({li-lo+1}/{hi-lo} in block)", STAGE)

    if hi < tcfg.num_hidden_layers:
        torch.save(states, STATES)
    return 0


if __name__ == "__main__":
    try:
        if ROLE == "worker":
            sys.exit(worker(int(os.environ["S03C_LO"]), int(os.environ["S03C_HI"])))
        from transformers import AutoConfig
        _c = AutoConfig.from_pretrained(SRC)
        sys.exit(orchestrate(getattr(_c, "text_config", _c).num_hidden_layers))
    except MF.MemFence as e:
        print(f"MEMFENCE ABORT: {e}", file=sys.stderr)
        sys.exit(5)
