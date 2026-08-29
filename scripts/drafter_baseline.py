"""Baseline: stock DFlash 2's acceptance length against OUR REAP target, measured offline.

`DRAFTER_PLAN.md` step 7.2: "Measure stock DFlash 2's acceptance length against the REAP target
BEFORE training anything. Without it there is no way to tell whether fine-tuning helped or whether
REAP simply did not hurt the drafter much."

WHY OFFLINE, AND WHAT THAT COSTS
--------------------------------
`dflash_generate` needs the target doing autoregressive decode. That is plan step D2, and it does
not exist: there is no vLLM/SGLang on this box, and the NVFP4 artifact needs an FP4 matmul kernel
that is the entire point of the not-yet-written CUDA server. Dequantising to bf16 for a resident
forward is ~320 GB against 122 GiB. So real-decode acceptance is blocked on the same thing that
blocks the generative benchmarks.

What IS available is the drafting algorithm itself, run against captured taps. Per block:

    target_hidden  = taps[:, :t]                       (context the drafter attends over)
    block_ids      = [tok_t, MASK, MASK, ... ]         (block_size 8 -> 7 drafted)
    draft_hidden   = draft(target_hidden, embed(block_ids), position_ids)[:, 1-B:, :]
    draft_tokens   = draft.propose(draft_hidden, tok_t, lm_head, temperature=0)

and under GREEDY verification a drafted token is accepted exactly when it equals the target's own
argmax at that position. So acceptance length = 1 + (leading matches), capped at the block.

**The honest caveat.** Real serving acceptance is measured on SELF-GENERATED context; this is
measured on ground-truth context, teacher-forced. Those differ, and the absolute number here will
NOT reproduce the 5.78 GSM8K figure on the upstream card - different target, different text,
different conditioning. What it IS: a fixed instrument for the comparison the plan actually needs,
stock-vs-retrained on the same target and the same data. Both arms pay the same bias.

Reported per domain as well, because REAP's damage is domain-structured (ballast 0.572 top-1
against code 0.921) and a drafter's acceptance should track that.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "vendor"))

CAP = ROOT / "artifacts" / "drafter"
OUT = ROOT / "artifacts" / "drafter" / "baseline.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", default=str(ROOT / "source" / "DFlash2-stock"))
    ap.add_argument("--target-ckpt", default=None)
    ap.add_argument("--max-ctx", type=int, default=2048, help="drafter sliding window")
    ap.add_argument("--label", default="stock")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    from common import kv_get, log
    from transformers import AutoConfig
    from dflash.model import DFlash2DraftModel, _raw_input_embeddings, _draft_value
    import stream_saliency as SS

    seqs = sorted(CAP.glob("seq_*.pt"))
    if not seqs:
        raise SystemExit(f"no captures in {CAP}; run scripts/drafter_capture.py first")

    dev = "cuda"
    draft = DFlash2DraftModel.from_pretrained(a.draft, dtype=torch.bfloat16).to(dev).eval()
    B = draft.block_size
    mask_id = int(draft.mask_token_id)
    scale = float(_draft_value(draft.config, "input_embedding_scale", 1.0))

    tgt = Path(a.target_ckpt) if a.target_ckpt else ROOT / "output" / str(kv_get("emit_name") or "")
    reader = SS.ShardReader(tgt)
    emb_w = reader.get("model.language_model.embed_tokens.weight").to(dev, torch.bfloat16)
    head_w = reader.get("lm_head.weight").to(dev, torch.bfloat16)

    class _Shim(torch.nn.Module):
        """`propose` wants an output head module and `_raw_input_embeddings` an embedding."""
        def __init__(self):
            super().__init__()
            self.lm_head = torch.nn.Linear(head_w.shape[1], head_w.shape[0], bias=False)
            self.lm_head.weight = torch.nn.Parameter(head_w, requires_grad=False)
            self.emb = torch.nn.Embedding(emb_w.shape[0], emb_w.shape[1])
            self.emb.weight = torch.nn.Parameter(emb_w, requires_grad=False)

        def get_input_embeddings(self):
            return self.emb

    shim = _Shim().to(dev).eval()
    log(f"drafter {Path(a.draft).name}: block_size {B}, mask id {mask_id}, "
        f"taps {draft.target_layer_ids}", "drafter_baseline")

    per_dom: dict[str, list[int]] = {}
    all_acc: list[int] = []
    with torch.no_grad():
        for sp in seqs:
            d = torch.load(sp, weights_only=False)
            ids = d["ids"].to(dev)                 # [S]
            greedy = d["greedy"].to(dev).long()    # [S] target argmax at each position
            taps = d["taps"].to(dev, torch.bfloat16)   # [S, 20480]
            S = ids.shape[0]
            bucket = d.get("bucket", "unknown")
            accs = []
            # Step through the sequence one block at a time, as decoding would.
            for t in range(B, S - B, B):
                ctx0 = max(0, t - a.max_ctx)
                th = taps[ctx0:t].unsqueeze(0)                      # [1, ctx, 20480]
                ctx = th.shape[1]
                blk = torch.full((1, B), mask_id, dtype=torch.long, device=dev)
                blk[0, 0] = ids[t]
                pos = torch.arange(ctx0, t + B, device=dev).unsqueeze(0)
                noise = _raw_input_embeddings(shim, blk, scale)
                dh = draft(target_hidden=th, noise_embedding=noise, position_ids=pos,
                           past_key_values=None, use_cache=False)[:, 1 - B:, :]
                dtok, _, _ = draft.propose(dh, blk[:, 0], shim.lm_head, 0.0)
                # Greedy verification: accept the leading run that matches the target's argmax.
                # greedy[t+j-1] is the target's prediction FOR position t+j.
                want = greedy[t: t + B - 1]
                got = dtok[0]
                m = (got == want)
                k = int((~m).float().argmax()) if not bool(m.all()) else int(m.numel())
                accs.append(1 + k)
            if accs:
                per_dom.setdefault(bucket, []).extend(accs)
                all_acc.extend(accs)
            del ids, greedy, taps
            log(f"{sp.name} [{bucket}] blocks {len(accs)} mean accept "
                f"{sum(accs)/max(1,len(accs)):.3f}", "drafter_baseline")

    def stat(xs):
        n = len(xs)
        mean = sum(xs) / max(1, n)
        full = sum(1 for x in xs if x >= B) / max(1, n)
        return {"blocks": n, "acceptance_length": mean,
                "full_block_rate": full,
                "speedup_vs_AR": mean}      # tokens per verification step == AR speedup ceiling
    res = {"label": a.label, "draft": Path(a.draft).name, "target": tgt.name,
           "block_size": B, "overall": stat(all_acc),
           "by_domain": {k: stat(v) for k, v in sorted(per_dom.items())},
           "method": "offline greedy-verification acceptance on teacher-forced context; NOT "
                     "comparable to served acceptance on self-generated context"}
    Path(a.out).write_text(json.dumps(res, indent=2))
    o = res["overall"]
    print(f"\n{a.label}: acceptance length {o['acceptance_length']:.3f} / {B} "
          f"({o['blocks']} blocks, full-block {o['full_block_rate']:.1%})")
    for k, v in res["by_domain"].items():
        print(f"  {k:<9} {v['acceptance_length']:.3f}  ({v['blocks']} blocks)")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
