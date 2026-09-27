"""Measure where host memory actually goes when one decoder layer runs, step by step.

Four resets were spent modelling this. The model in mem_lookahead.py got S=16384-eager right and
still missed S=16384-sparse, so the model is not enough. This measures instead, with memfence
aborting before the box can wedge.

    python scripts/probe_layer_mem.py --seq 2048     # known-good control first
    python scripts/probe_layer_mem.py --seq 16384    # the arm that kills the box
"""
from __future__ import annotations

import argparse
import gc
import sys

import torch

sys.path.insert(0, "scripts")
sys.path.insert(0, "scripts/stages")
import memfence as MF  # noqa: E402

GIB = 2 ** 30


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, required=True)
    ap.add_argument("--layers", type=int, default=5, help="through the first DSA layer")
    a = ap.parse_args()

    from common import ROOT
    import stream_saliency as SS
    from s03_saliency import _build_layer
    from transformers import AutoConfig

    SRC = ROOT / "source" / "GLM-5.3-Flash"
    cfg = AutoConfig.from_pretrained(SRC)
    tcfg = getattr(cfg, "text_config", cfg)
    hid, hc = tcfg.hidden_size, getattr(tcfg, "hc_mult", 4)

    MF.checkpoint("start")
    reader = SS.ShardReader(SRC)
    SS.patch_experts_for_saliency()
    MF.checkpoint("reader + patch")

    # one row only: this is an allocation probe, not a statistics run
    ids = torch.randint(0, 1000, (1, a.seq), dtype=torch.long)
    hs_cpu = torch.zeros(1, a.seq, hc, hid, dtype=torch.bfloat16)
    MF.checkpoint(f"host tensors S={a.seq}")

    for li in range(a.layers):
        lt = tcfg.layer_types[li]
        # predicted cost of materialising this layer on the host, then again on the device
        is_moe = li >= tcfg.first_k_dense_replace
        w = (tcfg.n_routed_experts * 3 * tcfg.moe_intermediate_size * hid * 2 / GIB
             if is_moe else 3 * tcfg.intermediate_size * hid * 2 / GIB)
        MF.require(w * 2.2, f"layer {li} ({lt}) weights host+device")
        layer = _build_layer(tcfg, li, reader, torch.bfloat16)
        MF.checkpoint(f"L{li} {lt[:12]} built+to(DEV)")

        SS.set_current_layer(f"model.language_model.layers.{li}.mlp")
        SS.set_bucket("general")
        with torch.no_grad():
            hs = hs_cpu.to("cuda")
            i2 = ids.to("cuda")
            am = torch.ones(1, a.seq, dtype=torch.bool, device="cuda")
            pos = torch.arange(a.seq, device="cuda").unsqueeze(0)
            MF.checkpoint(f"L{li} inputs on device")
            out, _tk = layer(hs, attention_mask=am, position_ids=pos,
                             position_embeddings=None, input_ids=i2,
                             past_key_values=None, use_cache=False, prev_topk_indices=None)
            MF.checkpoint(f"L{li} FORWARD done")
            hs_cpu = out.cpu()
            del hs, out, i2, am, pos, _tk
        SS.set_current_layer(None)
        del layer
        reader.release()
        gc.collect()
        torch.cuda.empty_cache()
        MF.checkpoint(f"L{li} freed + empty_cache")

    MF.report()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except MF.MemFence as e:
        print(f"\nMEMFENCE ABORT: {e}", file=sys.stderr)
        MF.report()
        sys.exit(5)
