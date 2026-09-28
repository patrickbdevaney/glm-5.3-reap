"""Per-channel importance inside each expert, for uniform intra-expert narrowing.

WHY NARROWING, NOT DELETION. Whole-expert removal destroys general-text capability while
leaving code intact -- measured here (general dNLL 1.0027 / top-1 0.579 vs code 0.057/0.916)
and independently at +131% general perplexity (arXiv:2607.16721). Channel narrowing does not
produce that signature: across four fine-grained MoEs every expert-removing/merging method
loses ~20 HellaSwag points while narrowing loses 1-7 (arXiv:2606.18304 Table 3). The
granularity trend also runs toward us for once -- inter beats intra at 8 experts, ties at 60,
loses at 64 (arXiv:2411.01016), and we have 288.

WHY THIS IS DEPLOYABLE where non-uniform expert budgets were not. `moe_intermediate_size` is
a single config scalar, so every expert keeps the SAME NUMBER of channels -- but each keeps
its OWN best channels. That is just a per-expert row selection; tensor shapes stay uniform and
`num_local_experts` is untouched. Uniform-vs-learned costs 0.29-1.14 pts at 64 experts/top-8
versus 1.72-4.43 at 8 experts (arXiv:2606.27866 Table 8), so the scalar is nearly free at our
granularity.

CRITERION: activation-attribution, NOT magnitude. Magnitude-based narrowing is unsafe --
masking the top-12 Fisher channels out of 1.35M drops GSM8K 35.9 -> 0.8 (arXiv:2606.05538),
and <0.5% of experts carry 5600x activation outliers (arXiv:2507.23279). We accumulate the
sufficient statistics for several criteria on ONE forward, the same lesson pass 2 learned for
expert saliency: recomputing costs another 18-hour sweep.

    per (expert, channel), per domain bucket:
      s1   sum |a_c|              -> first-order Taylor / mean-abs activation
      s2   sum a_c^2              -> RMS activation, Fisher-style second moment
      cnt  tokens routed          -> conditional vs unconditional means
    plus the static ||W_down[:,c]||_2, which converts an activation into an OUTPUT contribution

`a_c` is the POST-SwiGLU intermediate, i.e. exactly the value multiplied by column c of
down_proj. Importance = RMS(a_c) * ||W_down[:,c]|| is the contribution norm of that channel.
"""
from __future__ import annotations
import torch
from pathlib import Path

_CH = {"acc": None, "layer": None, "bucket": 0, "valid": None, "nbucket": 8}


def set_layer(name, n_expert=None, inter=None, dev=None):
    _CH["layer"] = name
    if name is not None and _CH["acc"] is None and n_expert:
        z = lambda: torch.zeros(_CH["nbucket"], n_expert, inter, dtype=torch.float64, device=dev)
        _CH["acc"] = {"s1": z(), "s2": z(),
                      "cnt": torch.zeros(_CH["nbucket"], n_expert, dtype=torch.float64, device=dev)}


def reset():
    _CH["acc"] = None


def set_bucket(i): _CH["bucket"] = i
def set_valid(v):  _CH["valid"] = v


def patch_experts_for_channels():
    """Accumulate per-channel statistics on the POST-SwiGLU intermediate.

    Mirrors stream_saliency.patch_experts_for_saliency so both can run on the same sweep;
    the two accumulators are independent and neither perturbs the forward.
    """
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextExperts
    import torch.nn.functional as F
    if getattr(Glm5NextTextExperts, "_chan_patched", False):
        return
    orig = Glm5NextTextExperts.forward

    def forward(self, hidden_states, top_k_index, top_k_weights):
        acc = _CH["acc"]
        if acc is None or _CH["layer"] is None:
            return orig(self, hidden_states, top_k_index, top_k_weights)
        final = torch.zeros_like(hidden_states)
        n_exp = self.gate_up_proj.shape[0]
        with torch.no_grad():
            hit = torch.nn.functional.one_hot(top_k_index, num_classes=n_exp).permute(2, 1, 0)
        for expert_idx in range(n_exp):
            with torch.no_grad():
                _, token_idx = torch.where(hit[expert_idx])
            if token_idx.numel() == 0:
                continue
            cur = self._apply_gate(F.linear(hidden_states[token_idx],
                                            self.gate_up_proj[expert_idx]))   # [n, inter]
            f_j = F.linear(cur, self.down_proj[expert_idx])
            with torch.no_grad():
                v = cur
                if _CH["valid"] is not None:
                    keep = _CH["valid"][token_idx]
                    v = cur[keep]
                if v.shape[0]:
                    b = _CH["bucket"]
                    a = v.to(torch.float32)
                    acc["s1"][b, expert_idx] += a.abs().sum(0).double()
                    acc["s2"][b, expert_idx] += (a * a).sum(0).double()
                    acc["cnt"][b, expert_idx] += v.shape[0]
            gj = top_k_weights[token_idx, torch.where(hit[expert_idx])[0]]
            final.index_add_(0, token_idx, (f_j * gj[:, None]).to(final.dtype))
        return final

    Glm5NextTextExperts.forward = forward
    Glm5NextTextExperts._chan_patched = True


def dump(dirpath: Path, layer: str, down_proj: torch.Tensor, buckets: list[str]):
    """Persist the accumulators plus the STATIC down-proj column norms.

    Without ||W_down[:,c]|| an activation statistic says nothing about the channel's effect on
    the layer output -- a large activation into a near-zero column contributes nothing. Storing
    it here means the criterion can be re-fit offline without touching the weights again.
    """
    acc = _CH["acc"]
    if acc is None:
        return
    dirpath.mkdir(parents=True, exist_ok=True)
    # down_proj is [n_expert, hidden, inter]; column c of expert e is down_proj[e][:, c]
    wn = down_proj.to(torch.float32).pow(2).sum(dim=1).sqrt().cpu()      # [n_expert, inter]
    torch.save({"layer": layer, "buckets": buckets,
                "s1_by_bucket": acc["s1"].cpu(), "s2_by_bucket": acc["s2"].cpu(),
                "cnt_by_bucket": acc["cnt"].cpu(), "down_col_norm": wn},
               dirpath / (layer.replace(".", "__") + ".pt"))


# ---------------------------------------------------------------- streaming driver
def run(smoke_layers=0, max_len=1024, batch=2, out_dir="channels"):
    """Accumulate per-channel statistics over the UNPRUNED model, one layer resident at a time.

    Teacher trajectory: no mask, all 288 experts. Channel importance must be measured on the
    model as it actually is, not through a pruned router -- otherwise the ranking is
    conditioned on a keep-set we may not even use.

    PILOT BUDGET. This sweeps the held-out set, not the full 1.03B-token calibration corpus.
    That is deliberate: the point is to decide whether narrowing is the right AXIS before
    spending an 18-hour pass, and the research says calibration mixture swings expert-ranking
    results by up to 22 points, so a full pass on the OLD skewed corpus would be the wrong
    18 hours anyway. Treat the resulting ranking as directional.
    """
    import gc, time
    from pathlib import Path
    import stream_saliency as SS
    from stages.s03_saliency import _build_layer
    import stages.s09_eval as E
    from common import ROOT, ARTIFACTS, log
    from transformers import AutoConfig
    from accelerate import init_empty_weights
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextForConditionalGeneration
    import router_kd as RK

    DEV = "cuda"
    SRC = ROOT / "source" / "GLM-5.3-Flash"
    if not (SRC / "model.safetensors.index.json").exists():
        raise SystemExit(f"unpruned source not staged at {SRC}")
    outp = ARTIFACTS / out_dir
    patch_experts_for_channels()

    cfg = AutoConfig.from_pretrained(SRC); tcfg = cfg.text_config
    reader = SS.ShardReader(SRC)
    with init_empty_weights():
        shell = Glm5NextForConditionalGeneration(cfg)
    lm = shell.model.language_model
    reader.load_module("model.language_model.embed_tokens")
    embed = lm.embed_tokens.to(DEV)
    rows, _ = E.load_heldout()
    states = RK._states_from_rows(rows, tcfg, embed, DEV, max_len, batch)
    del shell; gc.collect(); torch.cuda.empty_cache()
    log(f"channel_saliency: {len(states)} batches, {len(rows)} rows", "channels")

    nlayer = smoke_layers if smoke_layers else tcfg.num_hidden_layers
    for li in range(nlayer):
        name = f"model.language_model.layers.{li}.mlp"
        t0 = time.time()
        layer = _build_layer(tcfg, li, reader, torch.bfloat16).to(DEV)
        experts = next((m for m in layer.modules()
                        if type(m).__name__ == "Glm5NextTextExperts"), None)
        if experts is not None:
            reset()
            set_layer(name, n_expert=experts.gate_up_proj.shape[0],
                      inter=experts.down_proj.shape[-1], dev=DEV)
        with torch.no_grad():
            for st in states:
                hs = st["hs"].to(DEV); ids = st["ids"].to(DEV)
                set_valid((ids != tcfg.pad_token_id))
                o, tk = RK._fwd(layer, hs, ids, st.get("topk"), DEV)
                st["hs"] = o.cpu(); st["topk"] = tk.cpu() if tk is not None else None
                set_valid(None)
                del hs, ids, o
        if experts is not None and _CH["acc"] is not None:
            dump(outp, name, experts.down_proj.detach(),
                 ["general","code","math","agentic","finance","science","bio","vision"])
            log(f"channel_saliency: {name} done {time.time()-t0:.0f}s", "channels")
        set_layer(None); reset()
        del layer; reader.release(); gc.collect(); torch.cuda.empty_cache()
    log(f"channel_saliency: wrote {outp}", "channels")
