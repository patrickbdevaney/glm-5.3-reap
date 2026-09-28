"""Capture MTP teacher hidden states from the NVFP4 checkpoint.

WHY THIS EXISTS
---------------
The draft head consumes the SERVED model's final hidden state, so it has to be trained on states
from the quantisation it will actually run against. The IQ3_M traces already captured are for the
GGUF head; NVFP4 is a different function of the same weights and needs its own. IQ3_M vs IQ4_XS
came out at 0.891 mean cosine against a 0.999 same-quant floor, so "one head for every quant" is
not free, and NVFP4 is further from either than they are from each other.

Nothing can serve GLM-5.3 NVFP4 on this box - no vLLM model file, and model-level loading is
already documented as impossible here (nvfp4_tensor.py). So this streams the forward one decoder
layer at a time out of the shards, exactly as s03_saliency and drafter_capture already do. Only
the MoE experts are NVFP4-packed; attention, norms and eh_proj are plain bf16. `_build_layer`
reconstitutes both, and its NVFP4 arithmetic is verified bit-identical to compressed-tensors by
nvfp4_dequant_check.py.

WHAT IT WRITES
--------------
The same three files the llama.cpp capture writes, so quant_transfer.py and the fine-tune read
either source without knowing which produced it:

    <out>          fp16 [n_tokens, hidden]  final hidden state, post-norm
    <out>.ids      int32 [n_tokens]         the token at each row
    <out>.lens     int32 [n_seq]            rows per sequence

Post-norm is the point, not an incidental choice: vLLM's proposer receives the value returned by
the model's forward, which is after the final norm, and llama.cpp sets res->t_embd at the same
place. `norm(hc_head(h))` below is that tensor - hc_head collapses the hyper-connection dimension
the way the model's own head does, then the final RMSNorm.
"""
from __future__ import annotations

import argparse, gc, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

DEFAULT_CKPT = ROOT / "output" / "glm-5.3-flash-reap50-nvfp4-pass2"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--text", default=str(ROOT / "artifacts" / "qt_probe.txt"))
    ap.add_argument("--sep", default="<#sep#>")
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "qt_nvfp4.bin"))
    ap.add_argument("--max-len", type=int, default=2048,
                    help="matches the -c 2048 the llama.cpp capture ran at")
    ap.add_argument("--max-seqs", type=int, default=0, help="0 = every sequence in the file")
    a = ap.parse_args()

    from common import log
    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextForConditionalGeneration
    from accelerate import init_empty_weights
    import stream_saliency as SS
    from stages.s03_saliency import _build_layer

    ckpt = Path(a.ckpt)
    if not (ckpt / "model.safetensors.index.json").exists():
        raise SystemExit(f"no checkpoint at {ckpt}")

    cfg = AutoConfig.from_pretrained(ckpt)
    tcfg = cfg.text_config
    tok = AutoTokenizer.from_pretrained(ckpt)
    reader = SS.ShardReader(ckpt)

    texts = [t for t in Path(a.text).read_text(encoding="utf-8").split(a.sep) if t.strip()]
    if a.max_seqs:
        texts = texts[: a.max_seqs]
    if not texts:
        raise SystemExit(f"no sequences in {a.text}")

    # The llama.cpp side tokenised with the GGUF vocab and this side uses the HF one. They are the
    # same vocab, but a sequence that happens to tokenise differently would silently misalign every
    # row after it, so the ids are written out and quant_transfer.py matches per sequence and skips
    # any that disagree rather than trusting the offsets.
    seqs = [tok(t, add_special_tokens=True, truncation=True,
                max_length=a.max_len)["input_ids"] for t in texts]

    # Dense parts of the model that live outside the decoder layers. Loaded once and kept.
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

    n_tok = sum(len(s) for s in seqs)
    log(f"capturing {len(seqs)} sequences / {n_tok} tokens from {ckpt.name}", "nvfp4_capture")

    states = []
    with torch.no_grad():
        for s in seqs:
            ids = torch.tensor(s, dtype=torch.long, device=dev).unsqueeze(0)
            ie = embed(ids)
            states.append({
                "ids": ids.cpu(),
                # The layers carry hc_mult copies of the hidden state; the embedding is broadcast
                # into that shape exactly as the model's own forward does.
                "hs": ie.unsqueeze(2).expand(-1, -1, tcfg.hc_mult, -1).contiguous().cpu(),
                "topk": None,
            })
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
                st["hs"] = out.cpu()
                st["topk"] = topk.cpu() if topk is not None else None
                del hs, out, ids, am, pos, topk
        del layer
        reader.release(); gc.collect(); torch.cuda.empty_cache()
        log(f"layer {li+1}/{tcfg.num_hidden_layers}  elapsed {(time.time()-t0)/60:.1f} min",
            "nvfp4_capture")

    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Written to .part and renamed, so a killed run never leaves a truncated file that looks
    # complete to the comparison that reads it next.
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    lens, all_ids = [], []
    with torch.no_grad(), open(tmp, "wb") as fh:
        for st in states:
            h = norm(hc_head(st["hs"].to(dev)))[0]           # [S, hidden], post-final-norm
            fh.write(h.to(torch.float16).cpu().numpy().tobytes())
            ids = st["ids"][0].to(torch.int32).numpy()
            all_ids.append(ids)
            lens.append(len(ids))
            del h
    tmp.replace(out_path)
    np.concatenate(all_ids).astype(np.int32).tofile(str(out_path) + ".ids")
    np.asarray(lens, dtype=np.int32).tofile(str(out_path) + ".lens")

    log(f"wrote {out_path} ({out_path.stat().st_size/2**20:.1f} MiB), "
        f"{len(lens)} sequences, {sum(lens)} tokens", "nvfp4_capture")


if __name__ == "__main__":
    main()
