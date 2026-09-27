"""Does calibration sequence length change WHICH experts look salient on GLM-5.3-Flash?

The question matters because `s03_saliency` runs at MAX_LEN=2048 while the corpus is built to
16,384. MEASURED 2026-09-26: that truncation discards 73.5% of all collected calibration tokens
(35.5M of 48.3M), and 38.2% of samples are cut. If saliency is S-sensitive, the mask is
calibrated for short context and the 1M-context capability is being pruned blind.

MiMo asked the same question and predicted near-S-invariance, because 39 of its 48 layers are
SWA with a window of 128 -- structurally, S cannot change what those layers see past position
128. **That argument inverts here.** From config.json, GLM is a hybrid: `linear_attn_config.
kda_layers` covers ~3 of every 4 layers, the rest are `deepseek_sparse_attention`. KDA is linear
attention carrying a recurrent state that accumulates over the WHOLE context, with no window
capping it. So the prediction, with a sign:

    GLM saliency should be S-SENSITIVE, where MiMo's was predicted S-invariant.

CLAUDE.md §6 wants a comparison whose sign is already known. The S arms alone cannot supply one:
any two runs differ somewhat, and without a noise floor "they differ" is not a result. So this
runs a CONTROL arm -- S=2048 again, on a DISJOINT token sample. Same-S-different-tokens is known
to be the most similar pair that can exist here. If the S=16384 arm is no further from the
reference than the control is, S does not matter and MAX_LEN=2048 is vindicated.

Reports, per MoE layer, against the S=2048 reference:
  - Spearman rho over per-expert saliency
  - top-144 keep-set overlap (288 -> 144 is the shipped 50% ratio; this is the quantity that
    actually decides the mask, and rho can look fine while the keep-set moves)
split by layer_type, because the whole hypothesis is that KDA and DSA layers behave differently.
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, "scripts")
sys.path.insert(0, "scripts/stages")

from common import ROOT, log  # noqa: E402

STAGE = "exp_seqlen"
SRC = ROOT / "source" / "GLM-5.3-Flash"
OUT = ROOT / "artifacts" / "exp_seqlen"
DEV = "cuda"
DT = torch.bfloat16

SEQS = [2048, 8192, 16384]
TOTAL_TOKENS = 16384 * 8      # same token budget in every arm
L_MAX = int(sys.argv[1]) if len(sys.argv) > 1 else 12
KEEP = 144                    # 288 experts -> 50%


def _rows(pool: torch.Tensor, S: int, total: int) -> list[torch.Tensor]:
    """Same token stream, cut into rows of length S. Identical tokens across arms."""
    n = total // S
    return [pool[i * S:(i + 1) * S] for i in range(n)]


def _spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    ra = a.argsort().argsort().float()
    rb = b.argsort().argsort().float()
    ra = ra - ra.mean(); rb = rb - rb.mean()
    d = (ra.norm() * rb.norm())
    return float((ra @ rb) / d) if d > 0 else float("nan")


def _saliency(rec: dict) -> torch.Tensor:
    """REAP (1,1,1): mean gate-weighted output norm per expert, summed over buckets."""
    num = rec["sum_by_bucket"].sum(0).float()
    cnt = rec["cnt_by_bucket"].sum(0).float().clamp(min=1.0)
    return num / cnt


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    import stream_saliency as SS
    from s03_saliency import _build_layer
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(SRC)
    tcfg = getattr(cfg, "text_config", cfg)
    tcfg._attn_implementation = "eager"
    reader = SS.ShardReader(SRC)

    # ---- token pool: real long samples, two DISJOINT halves ------------------------------
    # A synthetic repeat would have no long-range structure and would answer the wrong question.
    pool_a, pool_b = [], []
    need = TOTAL_TOKENS
    import glob
    for f in sorted(glob.glob(str(ROOT / "corpus/shards/text/*.pt"))):
        d = torch.load(f, map_location="cpu", weights_only=False)
        items = d if isinstance(d, list) else d.get("items", d)
        for it in items:
            ids = it.get("input_ids")
            if ids is None or len(ids) < 16384:
                continue                      # only untruncated samples carry the long structure
            (pool_a if sum(len(x) for x in pool_a) < need else pool_b).append(
                torch.as_tensor(ids, dtype=torch.long))
            if sum(len(x) for x in pool_b) >= need:
                break
        if sum(len(x) for x in pool_b) >= need:
            break
    A = torch.cat(pool_a)[:need]
    B = torch.cat(pool_b)[:need]
    if A.numel() < need or B.numel() < need:
        log(f"only {A.numel()}/{B.numel()} tokens available of {need}; "
            f"corpus lacks enough >=16384-token samples", STAGE, "ERROR")
        sys.exit(1)
    log(f"token pools: A={A.numel()} B={B.numel()} (disjoint, real, >=16384-token samples)", STAGE)

    arms = [(f"S{S}", S, A) for S in SEQS] + [("control_S2048_disjoint", 2048, B)]

    # ---- embedding table, the only large part that stays resident -------------------------
    # Glm5NextConfig is not registered for AutoModelForCausalLM; s03 builds the concrete class
    # directly and so must this. Only embed_tokens is materialised -- everything else stays on
    # meta and costs nothing, and the decoder layers are streamed one at a time below.
    from accelerate import init_empty_weights
    from transformers.models.glm5_next.modeling_glm5_next import (
        Glm5NextForConditionalGeneration)
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

    SS.patch_experts_for_saliency()
    results: dict[str, dict] = {}

    for tag, S, pool in arms:
        adir = OUT / tag
        if (adir / "_done").exists():
            log(f"{tag}: already done, skipping", STAGE)
            continue
        SS.reset_accumulators()
        rows = _rows(pool, S, TOTAL_TOKENS)
        log(f"{tag}: S={S}, {len(rows)} rows x {S} tok = {len(rows)*S} tokens", STAGE)

        # Embed every row once; the sweep then walks layers over the held states, exactly as
        # s03 does. One bucket for the whole arm -- this experiment compares rankings across
        # arms, so per-domain attribution is not what is being measured.
        states = []
        with torch.no_grad():
            for r in rows:
                ids = r.unsqueeze(0).to(DEV)
                ie = emb(ids)
                states.append({"ids": ids.cpu(),
                               "valid": torch.ones(ids.numel(), dtype=torch.bool),
                               "hs": ie.unsqueeze(2).expand(-1, -1, tcfg.hc_mult, -1)
                                     .contiguous().cpu(),
                               "topk": None})
                del ie
        torch.cuda.empty_cache()

        t0 = time.time()
        for li in range(min(L_MAX, tcfg.num_hidden_layers)):
            layer = _build_layer(tcfg, li, reader, DT)
            SS.set_current_layer(f"model.language_model.layers.{li}.mlp")
            SS.set_bucket("general")
            for st in states:
                with torch.no_grad():
                    hs = st["hs"].to(DEV)
                    ids = st["ids"].to(DEV)
                    SS.set_valid_mask(st["valid"].to(DEV))
                    am = torch.ones(ids.shape[0], ids.shape[1], dtype=torch.bool, device=DEV)
                    pos = torch.arange(ids.shape[1], device=DEV).unsqueeze(0)
                    tk = st["topk"].to(DEV) if st["topk"] is not None else None
                    out, tk = layer(hs, attention_mask=am, position_ids=pos,
                                    position_embeddings=None, input_ids=ids,
                                    past_key_values=None, use_cache=False,
                                    prev_topk_indices=tk)
                    st["hs"] = out.cpu()
                    st["topk"] = tk.cpu() if tk is not None else None
                    del hs, out, ids, am, pos, tk
                SS.set_valid_mask(None)
            SS.set_current_layer(None)
            del layer
            reader.release()
            gc.collect()
            torch.cuda.empty_cache()
            log(f"{tag} layer {li+1}/{L_MAX} ({tcfg.layer_types[li]}) "
                f"elapsed {(time.time()-t0)/60:.1f} min", STAGE)
        SS.dump(adir)
        (adir / "_done").write_text("ok\n")
        del states
        gc.collect()
        torch.cuda.empty_cache()

    # ---- compare ---------------------------------------------------------------------------
    def load(tag):
        out = {}
        for f in sorted((OUT / tag).glob("*.pt")):
            rec = torch.load(f, map_location="cpu", weights_only=False)
            out[rec["layer"]] = _saliency(rec)
        return out

    ref = load("S2048")
    rows_out = []
    for tag, S, _ in arms:
        if tag == "S2048":
            continue
        cur = load(tag)
        for lname in sorted(ref, key=lambda n: int(n.split(".")[-2])):
            if lname not in cur:
                continue
            li = int(lname.split(".")[-2])
            a, b = ref[lname], cur[lname]
            ka = set(a.topk(KEEP).indices.tolist())
            kb = set(b.topk(KEEP).indices.tolist())
            rows_out.append({"arm": tag, "layer": li, "type": tcfg.layer_types[li],
                             "spearman": _spearman(a, b),
                             "keep_overlap": len(ka & kb) / KEEP})

    (OUT / "comparison.json").write_text(json.dumps(rows_out, indent=2))

    ctrl = [r for r in rows_out if r["arm"].startswith("control")]
    floor = min((r["keep_overlap"] for r in ctrl), default=float("nan"))
    print()
    print(f"{'arm':26s} {'layer':>5s} {'type':>26s} {'rho':>7s} {'keep@144':>9s}")
    for r in rows_out:
        print(f"{r['arm']:26s} {r['layer']:5d} {r['type']:>26s} "
              f"{r['spearman']:7.4f} {r['keep_overlap']:9.3f}")
    print()
    print(f"CONTROL FLOOR (same S=2048, disjoint tokens): keep-overlap >= {floor:.3f}")
    print("An S arm at or above that floor is indistinguishable from resampling noise:")
    print("S does not move the keep-set, and MAX_LEN=2048 is vindicated.")
    print("An S arm BELOW it is a real effect, and pass 3 should raise calib_max_len.")
    for tag, _, _ in arms:
        if tag == "S2048" or tag.startswith("control"):
            continue
        arm = [r for r in rows_out if r["arm"] == tag]
        if not arm:
            continue
        worst = min(r["keep_overlap"] for r in arm)
        verdict = "WITHIN noise floor" if worst >= floor else "BELOW floor -- REAL EFFECT"
        print(f"  {tag:14s} worst keep-overlap {worst:.3f}  -> {verdict}")


if __name__ == "__main__":
    main()
