"""P12/P13 - paired teacher-vs-student evaluation. The thing pass 1 never had.

Pass 1 shipped a 50%-pruned 321B model with **zero** evaluation of any kind. Every quality claim
about it - including "REAP keeps the important experts" - was an argument, not a measurement. This
stage produces the first numbers.

WHAT IT MEASURES, and why each one
----------------------------------
On a held-out split the calibration never saw, teacher-forced (no generation - see below):

  dNLL per domain   the direct cost of pruning, in nats/token, split by the domains that were
                    weighted differently in calibration. A single global number would hide a
                    catastrophic loss in `code` behind a large `agentic` sample.
  flip rate         fraction of positions where argmax changes. Interpretable in a way NLL is
                    not: it is roughly "how often would the model have started a different token".
  top-k KL          distributional distance, k-truncated. Full KL over 154,880 logits is not
                    storable; k=32 captures the mass that matters for both sampling and drafting.
  tap drift         relative L2 change of the hidden states at layers [5,14,24,33,42] - the exact
                    layers DFlash 2 taps. Predicts how much drafter re-training the prune costs,
                    which is otherwise only discoverable by doing it.

WHY TEACHER-FORCED, not generative
----------------------------------
Generation from the unpruned teacher is impossible here: 328 GB / 3.0 GB/s is ~110 s per decoded
token. Teacher-forced scoring needs one forward pass per sequence, which the streaming harness
already does. Generative benchmarks (AIME/HumanEval/BFCL) are listed in CLOUD_COUNTERFACTUAL.md
as hardware-bound, and are honestly out of reach on one Thor - so this measures what CAN be
measured rather than pretending otherwise.

DESIGN
------
Teacher and student are scored in SEPARATE passes, not lockstep. Lockstep would need one MoE layer
from each resident simultaneously (~20 GiB dequantised) for no benefit: the comparison is
per-token and both passes see identical inputs in identical order, so the join is exact either
way. Separate passes halve peak memory, which on this box is the binding constraint.

Per token we keep 4 B (gold NLL) + 4 B (argmax) + k*(2+2) B, ~140 B/token at k=32 - so a 200k-token
eval is ~28 MB. Taps are kept for a subsample only, at 40 KB/token.
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import ROOT, ARTIFACTS, log, metric, kv_get, publish  # noqa: E402

STAGE = "s09_eval"
CORPUS = ROOT / "corpus" / "shards"
OUT = ARTIFACTS / "eval"
TAP_LAYERS = [5, 14, 24, 33, 42]        # DFlash 2 target_layer_ids
TOPK = 32
N_EVAL = int(kv_get("eval_samples", 192) or 192)
MAX_LEN = int(kv_get("calib_max_len", 2048) or 2048)
BATCH = int(kv_get("eval_batch", 2) or 2)
TAP_SUBSAMPLE = 0.02
DEV = "cuda"


def load_heldout():
    """Rows the calibration never saw.

    `_load_calib` takes `items[:want]` per bucket; held-out is the slice immediately after. Pass 1
    defined a HELDOUT_FRACTION and then never used it, so its calibration and its (absent)
    evaluation were drawn from the same pool - which would have made any number it produced
    meaningless anyway.
    """
    import torch
    import corpus_spec as SPEC
    from stages.s03_saliency import N_CALIB

    n_text = int(N_CALIB * (1 - SPEC.MIXTURE["multimodal"]))
    denom = 1 - SPEC.MIXTURE["multimodal"]
    rows = []
    for bucket, share in SPEC.MIXTURE.items():
        if bucket == "multimodal":
            continue
        f = CORPUS / "text" / f"{bucket}.pt"
        if not f.exists():
            continue
        items = torch.load(f, weights_only=False)
        used = int(n_text * share / denom)
        want = max(1, int(N_EVAL * share / denom))
        for it in items[used:used + want]:
            rows.append((it["input_ids"][:MAX_LEN], bucket))
    log(f"held-out: {len(rows)} rows the calibration never saw", STAGE)
    return rows


def score_checkpoint(ckpt: Path, rows, tag: str) -> dict:
    """Stream one checkpoint over the held-out rows and record per-token statistics."""
    import torch
    from transformers import AutoConfig
    from accelerate import init_empty_weights
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextForConditionalGeneration
    import stream_saliency as SS
    from stages.s03_saliency import _build_layer

    cfg = AutoConfig.from_pretrained(ckpt)
    tcfg = cfg.text_config
    reader = SS.ShardReader(ckpt)

    with init_empty_weights():
        shell = Glm5NextForConditionalGeneration(cfg)
    lm = shell.model.language_model
    for mod, prefix in ((lm.embed_tokens, "model.language_model.embed_tokens"),
                        (lm.norm, "model.language_model.norm"),
                        (lm.hc_head, "model.language_model.hc_head")):
        sd = {k[len(prefix):].lstrip("."): reader.get(k)
              for k in reader.map if k.startswith(prefix)}
        mod.to_empty(device="cpu")
        if sd:
            mod.load_state_dict(sd, strict=False, assign=True)
        mod.to(DEV).eval()
    head_w = reader.get("lm_head.weight").to(DEV, torch.bfloat16)
    embed, norm, hc_head = lm.embed_tokens, lm.norm, lm.hc_head

    # Build all batch states up front; the held-out set is small by construction.
    states = []
    with torch.no_grad():
        for i in range(0, len(rows), BATCH):
            chunk = rows[i:i + BATCH]
            ids = torch.full((len(chunk), MAX_LEN), tcfg.pad_token_id, dtype=torch.long)
            for r, (b, _) in enumerate(chunk):
                n = min(len(b), MAX_LEN)
                ids[r, :n] = b[:n].long()
            ids = ids.to(DEV)
            ie = embed(ids)
            states.append({"ids": ids.cpu(), "buckets": [c[1] for c in chunk],
                           "hs": ie.unsqueeze(2).expand(-1, -1, tcfg.hc_mult, -1)
                           .contiguous().cpu(), "topk": None, "taps": {}})
            del ie
    del shell
    gc.collect(); torch.cuda.empty_cache()

    t0 = time.time()
    for li in range(tcfg.num_hidden_layers):
        layer = _build_layer(tcfg, li, reader, torch.bfloat16)
        for st in states:
            with torch.no_grad():
                hs = st["hs"].to(DEV)
                ids = st["ids"].to(DEV)
                am = torch.ones(ids.shape[0], ids.shape[1], dtype=torch.bool, device=DEV)
                pos = torch.arange(ids.shape[1], device=DEV).unsqueeze(0)
                topk = st["topk"].to(DEV) if st["topk"] is not None else None
                out, topk = layer(hs, attention_mask=am, position_ids=pos,
                                  position_embeddings=None, input_ids=ids,
                                  past_key_values=None, use_cache=False,
                                  prev_topk_indices=topk)
                if li in TAP_LAYERS:
                    # Collapse hc_mult the same way the model's head does, so the stored tap is
                    # the [B,S,H] feature a drafter would actually consume.
                    st["taps"][li] = hc_head(out).to(torch.float16).cpu()
                st["hs"] = out.cpu()
                st["topk"] = topk.cpu() if topk is not None else None
                del hs, out, ids, am, pos, topk
        del layer
        reader.release(); gc.collect(); torch.cuda.empty_cache()
        log(f"[{tag}] layer {li+1}/{tcfg.num_hidden_layers}  "
            f"elapsed {(time.time()-t0)/60:.1f} min", STAGE)

    # Final projection and per-token statistics.
    nll, amax, tk_i, tk_p, bkt, taps = [], [], [], [], [], {li: [] for li in TAP_LAYERS}
    with torch.no_grad():
        for st in states:
            h = norm(hc_head(st["hs"].to(DEV)))
            ids = st["ids"].to(DEV)
            logits = torch.nn.functional.linear(h, head_w).float()
            lp = torch.log_softmax(logits, dim=-1)
            # Teacher forcing: position t predicts token t+1.
            gold = ids[:, 1:]
            lp_s = lp[:, :-1]
            valid = gold != tcfg.pad_token_id
            nll.append((-lp_s.gather(-1, gold.unsqueeze(-1)).squeeze(-1))[valid].cpu())
            amax.append(lp_s.argmax(-1)[valid].to(torch.int32).cpu())
            v, i = lp_s.topk(TOPK, dim=-1)
            tk_i.append(i[valid].to(torch.int32).cpu())
            tk_p.append(v[valid].to(torch.float16).cpu())
            for r, b in enumerate(st["buckets"]):
                bkt += [b] * int(valid[r].sum())
            for li in TAP_LAYERS:
                t = st["taps"][li][:, :-1][valid.cpu()]
                n = max(1, int(t.shape[0] * TAP_SUBSAMPLE))
                taps[li].append(t[torch.arange(0, t.shape[0], max(1, t.shape[0] // n))[:n]])
            del h, ids, logits, lp, lp_s
    res = {"nll": torch.cat(nll), "argmax": torch.cat(amax),
           "topk_idx": torch.cat(tk_i), "topk_logp": torch.cat(tk_p),
           "buckets": bkt, "taps": {li: torch.cat(v) for li, v in taps.items()}}
    log(f"[{tag}] scored {res['nll'].numel()} tokens in {(time.time()-t0)/60:.1f} min", STAGE)
    return res


def compare(T: dict, S: dict) -> dict:
    import torch
    n = min(T["nll"].numel(), S["nll"].numel())
    dn = (S["nll"][:n] - T["nll"][:n]).double()
    flip = (S["argmax"][:n] != T["argmax"][:n]).double()
    out = {"tokens": int(n),
           "dNLL_mean": float(dn.mean()), "dNLL_p50": float(dn.median()),
           "dNLL_p95": float(dn.quantile(0.95)),
           "teacher_nll": float(T["nll"][:n].double().mean()),
           "student_nll": float(S["nll"][:n].double().mean()),
           "flip_rate": float(flip.mean()),
           # Reported explicitly because it is the metric published quantization work uses.
           # "Retains X% of top-1 accuracy" against the unpruned model is exactly 1 - flip_rate
           # measured against the teacher, so this number is directly comparable to e.g. an
           # aggressively-quantized GGUF of the SAME base model - provided both are measured
           # against that same unpruned reference, which ours is.
           "top1_agreement": float(1.0 - flip.mean())}
    # k-truncated KL(teacher || student) on the teacher's top-k support.
    #
    # Chunked deliberately. Materialising a dense [n, 154880] scatter to look up the student's
    # logprob at the teacher's indices is 124 GB at n=200k - it would OOM the box after the
    # multi-hour scoring passes had already been paid for, which is the most expensive possible
    # place to discover an allocation bug. Both top-k lists are sorted-by-index-searchable per
    # chunk instead, so peak memory is O(chunk * k).
    ti, tp = T["topk_idx"][:n].long(), T["topk_logp"][:n].float()
    si, sp = S["topk_idx"][:n].long(), S["topk_logp"][:n].float()
    FLOOR = -30.0
    tot, seen = 0.0, 0
    CH = 4096
    for a0 in range(0, n, CH):
        a1 = min(n, a0 + CH)
        ti_c, tp_c, si_c, sp_c = ti[a0:a1], tp[a0:a1], si[a0:a1], sp[a0:a1]
        # match[i,j,k] : teacher index j equals student index k
        eq = ti_c.unsqueeze(2) == si_c.unsqueeze(1)          # [c, K, K]
        hit = eq.any(-1)
        pos = eq.float().argmax(-1)
        s_on_t = torch.where(hit, sp_c.gather(1, pos), torch.full_like(tp_c, FLOOR))
        pk = tp_c.exp()
        tot += float((pk * (tp_c - s_on_t)).sum(-1).sum())
        seen += a1 - a0
    out["topk_KL"] = tot / max(1, seen)
    by = {}
    for b in sorted(set(T["buckets"][:n])):
        m = torch.tensor([x == b for x in T["buckets"][:n]])
        if int(m.sum()) == 0:
            continue
        by[b] = {"tokens": int(m.sum()), "dNLL_mean": float(dn[m].mean()),
                 "flip_rate": float(flip[m].mean())}
    out["by_domain"] = by
    drift = {}
    for li in TAP_LAYERS:
        a, b = T["taps"][li].float(), S["taps"][li].float()
        m = min(a.shape[0], b.shape[0])
        a, b = a[:m], b[:m]
        drift[li] = float(((b - a).norm(dim=-1) / a.norm(dim=-1).clamp_min(1e-6)).mean())
    out["tap_drift"] = drift          # DFlash 2 reads exactly these layers
    return out


def run() -> dict:
    import torch
    teacher = Path(kv_get("eval_teacher", str(ROOT / "source" / "GLM-5.3-Flash")))
    student = Path(kv_get("eval_student", str(ROOT / "output" / "glm-5.3-flash-reap50-fp8")))
    if not (student / "model.safetensors.index.json").exists():
        raise RuntimeError(f"student checkpoint not usable: {student}")
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_heldout()

    # THE TEACHER CAPTURE MUST OUTLIVE THE TEACHER.
    #
    # s04b_surgery deletes each source shard right after writing its survivors - that is the R10
    # mechanism that keeps the prune inside the disk envelope - and the source IS the teacher. So
    # after any materialisation there is no teacher left to score against, and a teacher-vs-student
    # number becomes impossible without a 3-hour re-download.
    #
    # The capture is per-token on a FIXED held-out set, so it is reusable across every student:
    # score the teacher once, persist it WITH taps, and compare each subsequent student to the
    # saved copy. Missing this would have cost pass 2 its evaluation entirely - the sequencing is
    # the whole point, not an optimisation.
    cache = OUT / "teacher.pt"
    if cache.exists() and bool(kv_get("eval_reuse_teacher", True)):
        T = torch.load(cache, weights_only=False)
        log(f"reusing cached teacher capture ({T['nll'].numel()} tokens) - the source may be "
            f"gone, and it does not need to be here", STAGE)
    else:
        if not (teacher / "model.safetensors.index.json").exists():
            raise RuntimeError(
                f"no teacher at {teacher} and no cached capture at {cache}. Surgery consumes the "
                f"source, so the teacher must be scored BEFORE materialising a student.")
        T = score_checkpoint(teacher, rows, "teacher")
        torch.save(T, cache)          # WITH taps: pass-2 tap drift needs the teacher side too
        log(f"teacher capture saved to {cache} - safe to materialise now", STAGE)

    S = score_checkpoint(student, rows, "student")
    torch.save(S, OUT / f"student_{student.name}.pt")
    res = compare(T, S)
    res["teacher_from_cache"] = bool(cache.exists())
    res["student"] = student.name

    (ARTIFACTS / "s09_eval.json").write_text(json.dumps(res, indent=2))
    for k, v in res.items():
        if isinstance(v, (int, float)):
            metric(STAGE, f"eval_{k}", v)
    log(f"dNLL {res['dNLL_mean']:+.4f} nats/token | top-1 agreement "
        f"{100*res['top1_agreement']:.2f}% (flip {100*res['flip_rate']:.2f}%) | "
        f"topk-KL {res['topk_KL']:.4f}", STAGE)
    for b, d in res["by_domain"].items():
        log(f"  {b:10s} dNLL {d['dNLL_mean']:+.4f}  flip {100*d['flip_rate']:.2f}%  "
            f"({d['tokens']} tok)", STAGE)
    log(f"  DFlash tap drift: {_fmt_drift(res['tap_drift'])}", STAGE)
    publish(ARTIFACTS / "s09_eval.json", "artifacts", "stage09/s09_eval.json", stage=STAGE)
    return res


def _fmt_drift(d):
    return ", ".join(f"L{k}: {v:.4f}" for k, v in d.items())
