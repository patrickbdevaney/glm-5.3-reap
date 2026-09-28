"""Gate 2: how often would the UN-fine-tuned MTP head be accepted against the NVFP4 target?

The fine-tune is the fallback, not the default. GLM-5.3 ships a trained MTP module as layer 45,
and against the IQ3_M GGUF it was already accepted 46.5% of the time at depth 2 and 58.5% at
depth 3 with no training at all. NVFP4 is a higher-fidelity quantisation than IQ3_M in both the
weights and the states, so the head may well need little or nothing done to it - and a capture
plus fine-tune is days of GPU that should not be spent before that is checked.

It is checkable offline. Under greedy verification a drafted token is accepted exactly when it
equals the target's own argmax, and both sides are computable from the captured hidden states:

    target's next token at position t   = argmax(lm_head(h_t))
    head's proposal for that position   = argmax(lm_head(shared_head_norm(MTP(x_t, h_{t-1}))))

so acceptance is the rate at which those agree. No server, no speculative loop, no KV rollback.

The (x_t, h_{t-1}) pairing is the one the working llama.cpp path uses: the head is fed the newest
token together with the hidden state that PRECEDED it, and predicts the token after. Getting that
off by one would silently measure something else, so it is stated here rather than implied.

Layer 45 is built from the NVFP4 checkpoint at bf16. That is deliberate: quantising the head as
well would confound "is this head good enough" with "does this head survive 4-bit", and only the
first question is being asked here.
"""
from __future__ import annotations

import argparse, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

DEFAULT_CKPT = ROOT / "output" / "glm-5.3-flash-reap50-nvfp4-pass2"
MTP_LAYER = 45


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--states", default=str(ROOT / "artifacts" / "qt_nvfp4.bin"))
    ap.add_argument("--max-seqs", type=int, default=0)
    a = ap.parse_args()

    from common import log
    from transformers import AutoConfig
    import stream_saliency as SS
    from stages.s03_saliency import _build_layer

    ckpt = Path(a.ckpt)
    cfg = AutoConfig.from_pretrained(ckpt)
    tcfg = cfg.text_config

    # transformers has no entry for layer 45: every per-layer table has exactly
    # num_hidden_layers entries, because the reference implementation drops the MTP block on load.
    # Layer 45's own tensors say it is structurally the same as the DSA layers - kv_a_proj_with_mqa
    # + kv_b_proj, its own 7 indexer tensors, a sparse MoE, and no hc_* at all - and layer 43 is
    # the last of those. So extend every such table by copying index 43 rather than naming the
    # values one at a time; three tables need it today (layer_types, mlp_layer_types,
    # indexer_types) and this does not have to be revisited if a fourth appears.
    n_layer = tcfg.num_hidden_layers
    template = n_layer - 2                      # layer 43, the last deepseek_sparse_attention
    for k, v in list(vars(tcfg).items()):
        if isinstance(v, (list, tuple)) and len(v) == n_layer:
            setattr(tcfg, k, list(v) + [v[template]])
            log(f"extended cfg.{k} for the MTP layer with {v[template]!r}", "mtp_gate")

    reader = SS.ShardReader(ckpt)
    dev, dt = "cuda", torch.bfloat16

    # The hyper-connection parameters WILL be reported missing: the template layer has them and
    # layer 45 does not, because the MTP block is a plain pre-norm residual block. Nothing below
    # calls the decoder layer's own forward, only its sub-modules, so those stay unused.
    layer = _build_layer(tcfg, MTP_LAYER, reader, dt)

    p = f"model.language_model.layers.{MTP_LAYER}."
    def w(name):
        return reader.get(p + name).to(dev, dt)
    enorm_w, hnorm_w = w("enorm.weight"), w("hnorm.weight")
    eh_proj_w = w("eh_proj.weight")
    head_norm_w = w("shared_head.norm.weight")

    embed_w = reader.get("model.language_model.embed_tokens.weight").to(dev, dt)
    lm_head_w = reader.get("lm_head.weight").to(dev, dt)
    eps = getattr(tcfg, "rms_norm_eps", 1e-6)

    def rms(x, weight):
        v = x.float()
        v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps)
        return (v * weight.float()).to(dt)

    states = Path(a.states)
    H = tcfg.hidden_size
    hs_all = np.fromfile(states, dtype=np.float16).reshape(-1, H)
    ids_all = np.fromfile(str(states) + ".ids", dtype=np.int32)
    lens = np.fromfile(str(states) + ".lens", dtype=np.int32)
    if a.max_seqs:
        lens = lens[: a.max_seqs]
    log(f"gate2 on {len(lens)} sequences / {int(lens.sum())} tokens", "mtp_gate")

    off = 0
    n_tok = n_acc = n_true = 0
    per_seq = []
    with torch.no_grad():
        for si, L in enumerate(lens):
            L = int(L)
            h = torch.from_numpy(hs_all[off:off + L].copy()).to(dev, dt)     # [L, H] post-norm
            ids = torch.from_numpy(ids_all[off:off + L].copy().astype(np.int64)).to(dev)
            off += L
            if L < 3:
                continue

            # What the target itself would emit next at each position.
            tgt_next = (h @ lm_head_w.T).argmax(-1)                          # [L]

            # The head sees (x_t, h_{t-1}) for t = 1..L-1 and predicts x_{t+1}.
            e = rms(embed_w[ids[1:]], enorm_w)                               # [L-1, H]
            hh = rms(h[:-1], hnorm_w)                                        # [L-1, H]
            x = (torch.cat([e, hh], dim=-1) @ eh_proj_w.T).unsqueeze(0)      # [1, L-1, H]

            # This is the LOCAL BOOLEAN PADDING mask of shape [B, S], not a causal mask - the
            # module builds its own attention mask internally from the DSA indexer's top-k
            # selection (build_attention_mask_from_topk). Handing it a 4D additive causal mask
            # instead fails inside the indexer, where it is concatenated onto the key states.
            S = x.shape[1]
            mask = torch.ones(1, S, dtype=torch.bool, device=dev)

            r = x
            y = layer.self_attn(layer.input_layernorm(x), attention_mask=mask)
            y = y[0] if isinstance(y, tuple) else y
            x = r + y
            r = x
            x = r + layer.mlp(layer.post_attention_layernorm(x))

            out = rms(x[0], head_norm_w)
            draft_next = (out @ lm_head_w.T).argmax(-1)                      # [L-1]

            # Position t of the head lines up with position t+1 of the target: both are a
            # prediction of the token at t+2 in the original sequence... stated concretely,
            # head row j was built from (x_{j+1}, h_j) so it predicts x_{j+2}, and the target's
            # prediction of x_{j+2} is tgt_next[j+1].
            acc = (draft_next[:-1] == tgt_next[1:-1]).sum().item()
            tru = (draft_next[:-1] == ids[2:]).sum().item()
            m = draft_next[:-1].numel()
            n_acc += acc; n_true += tru; n_tok += m
            per_seq.append(acc / max(m, 1))
            del h, ids, e, hh, x, out, mask

    if not n_tok:
        sys.exit("no positions scored")
    ps = np.asarray(per_seq)
    print()
    print(f"positions scored          {n_tok}")
    print(f"ACCEPTANCE (vs target argmax)  {100*n_acc/n_tok:.2f}%   "
          f"per-seq p50 {100*np.percentile(ps,50):.1f}%  p10 {100*np.percentile(ps,10):.1f}%")
    print(f"top-1 vs ground truth          {100*n_true/n_tok:.2f}%")
    print()
    # Depth-k acceptance compounds: a chain of k drafted tokens survives only if every one of them
    # is accepted, so this is the optimistic reading of what depth buys, not a measurement of it.
    t = n_acc / n_tok
    print("if independent, expected accepted-per-iteration at depth k:")
    for k in (1, 2, 3, 4):
        exp = sum(t**j for j in range(1, k + 1))
        print(f"  depth {k}: {exp:.2f} drafted tokens accepted + 1 free = {1+exp:.2f} tokens")


if __name__ == "__main__":
    main()
