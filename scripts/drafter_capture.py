"""Capture the five DFlash-2 taps densely, plus the target's greedy continuation.

WHY A DEDICATED CAPTURE
-----------------------
`s09_eval` already taps layers [5, 14, 24, 33, 42] - the exact `target_layer_ids` in DFlash 2's
config - but it subsamples them to 2% and then valid-masks and concatenates across sequences,
which destroys the contiguity a block drafter needs. Drafting is inherently positional: the model
proposes tokens t+1..t+7 from the context up to t. So the taps must be dense and per-sequence.

WHAT THE DRAFTER ACTUALLY CONSUMES
----------------------------------
`extract_context_feature` selects `hidden_states[layer_id + 1]`, i.e. the OUTPUT of each tapped
layer, and concatenates: 5 x 4096 = 20480, which is exactly the input width of `fc.weight`
[4096, 20480] read from the checkpoint. Our streaming loop already collapses `hc_mult` with the
model's own `hc_head` before storing a tap, so what is stored is the [B, S, H] feature the drafter
would receive from a real forward - that was the stated reason for collapsing it there.

Also stored: the target's greedy argmax at every position, teacher-forced on the ground-truth
prefix. That is the verification target for the offline acceptance measurement - under greedy
verification a drafted token is accepted exactly when it equals the target's argmax.

48 KB/token in bf16 (5 taps + nothing else; the final hidden is not stored because we store the
argmax it produces, which is 4 bytes rather than 8 KB).
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

TAPS = [5, 14, 24, 33, 42]
OUT = ROOT / "artifacts" / "drafter"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", type=int, default=48)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--ckpt", default=None, help="defaults to the emitted pass-2 FP8")
    a = ap.parse_args()

    from common import kv_get, log
    from transformers import AutoConfig
    from accelerate import init_empty_weights
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextForConditionalGeneration
    import stream_saliency as SS
    from stages.s03_saliency import _build_layer
    import stages.s09_eval as EV

    ckpt = Path(a.ckpt) if a.ckpt else ROOT / "output" / str(kv_get("emit_name") or "")
    if not (ckpt / "model.safetensors.index.json").exists():
        raise SystemExit(f"no checkpoint at {ckpt}")
    OUT.mkdir(parents=True, exist_ok=True)

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
    dev = "cuda"
    embed = lm.embed_tokens.to(dev, torch.bfloat16)
    norm = lm.norm.to(dev, torch.bfloat16)
    hc_head = lm.hc_head.to(dev, torch.bfloat16)
    head_w = reader.get("lm_head.weight").to(dev, torch.bfloat16)

    rows, _ = EV.load_heldout()
    # Long sequences only: a block drafter measured on 40-token stubs measures padding behaviour.
    rows = [r for r in rows if len(r[0]) >= a.max_len][:a.seqs]
    if not rows:
        raise SystemExit("no held-out rows long enough")
    log(f"capturing {len(rows)} sequences x {a.max_len} tokens from {ckpt.name}", "drafter_capture")

    states = []
    with torch.no_grad():
        for b, bucket in rows:
            ids = b[:a.max_len].long().unsqueeze(0).to(dev)
            ie = embed(ids)
            states.append({"ids": ids.cpu(), "bucket": bucket,
                           "hs": ie.unsqueeze(2).expand(-1, -1, tcfg.hc_mult, -1).contiguous().cpu(),
                           "topk": None, "taps": {}})
            del ie
    del shell
    gc.collect(); torch.cuda.empty_cache()

    t0 = time.time()
    for li in range(tcfg.num_hidden_layers):
        layer = _build_layer(tcfg, li, reader, torch.bfloat16)
        for st in states:
            with torch.no_grad():
                hs = st["hs"].to(dev)
                ids = st["ids"].to(dev)
                am = torch.ones(ids.shape[0], ids.shape[1], dtype=torch.bool, device=dev)
                pos = torch.arange(ids.shape[1], device=dev).unsqueeze(0)
                topk = st["topk"].to(dev) if st["topk"] is not None else None
                out, topk = layer(hs, attention_mask=am, position_ids=pos,
                                  position_embeddings=None, input_ids=ids,
                                  past_key_values=None, use_cache=False,
                                  prev_topk_indices=topk)
                if li in TAPS:
                    # DENSE, and hc-collapsed exactly as the model's own head does - this is
                    # hidden_states[li+1] in `extract_context_feature`'s indexing.
                    st["taps"][li] = hc_head(out).to(torch.bfloat16).cpu().clone()
                st["hs"] = out.cpu()
                st["topk"] = topk.cpu() if topk is not None else None
                del hs, out, ids, am, pos, topk
        del layer
        reader.release(); gc.collect(); torch.cuda.empty_cache()
        log(f"layer {li+1}/{tcfg.num_hidden_layers}  elapsed {(time.time()-t0)/60:.1f} min",
            "drafter_capture")

    n = 0
    with torch.no_grad():
        for i, st in enumerate(states):
            h = norm(hc_head(st["hs"].to(dev)))
            logits = torch.nn.functional.linear(h, head_w).float()
            greedy = logits.argmax(-1)[0].to(torch.int32).cpu()      # target's greedy next token
            taps = torch.cat([st["taps"][t] for t in TAPS], dim=-1)[0]   # [S, 20480]
            torch.save({"ids": st["ids"][0], "greedy": greedy, "taps": taps,
                        "bucket": st["bucket"], "taps_layers": TAPS},
                       OUT / f"seq_{i:03d}.pt")
            n += 1
            del h, logits, taps
    (OUT / "capture.json").write_text(json.dumps(
        {"checkpoint": ckpt.name, "sequences": n, "max_len": a.max_len, "taps": TAPS,
         "note": "greedy is the target's argmax teacher-forced on the ground-truth prefix"},
        indent=2))
    log(f"wrote {n} sequences to {OUT}", "drafter_capture")


if __name__ == "__main__":
    main()
