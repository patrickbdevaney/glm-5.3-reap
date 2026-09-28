"""Router-only distillation: fix the router/expert mismatch left by pruning.

WHY. arXiv:2603.02217 finds the dominant residual damage from retraining-free MoE compression
is not lost expert capacity but ROUTER MISSPECIFICATION: the experts changed, the router did
not, so it keeps scoring a support that no longer exists. Updating ~0.04% of parameters
recovers a disproportionate amount -- and the effect scales with expert count, being
"significantly more effective" on a 128-expert top-8 model than on Mixtral's 8/top-2, because
the routing space is C(128,8)~1.4e12 vs C(8,2)=28. At 288/top-8 ours is larger still, so this
is the single highest return-per-GPU-hour item available.

DEVIATION FROM THE PAPER, STATED PLAINLY. The paper distils the router against the original
model's NEXT-TOKEN distribution -- a global objective needing full backprop through a 165B
model. That does not fit in 122 GB. This does layer-LOCAL distillation instead: at each layer
the unpruned MoE's output on the same input is the target, and only that layer's router is
trained. Local matching cannot see how errors compose across layers, so it is an approximation,
not the published method. It is the version that runs on this box.

INPUTS COME FROM THE STUDENT TRAJECTORY, TARGETS FROM THE TEACHER LAYER. At layer L the hidden
state carried forward is the PRUNED model's, because that is what the router will actually see
at inference; the target is the UNPRUNED layer applied to that same state. Teacher-forcing the
inputs instead would train the router on activations it never encounters.

Peak residency: one unpruned MoE (288 experts) + one pruned MoE (152) ~= 11 GB at FP8.
"""
from __future__ import annotations
import torch

def kd_one_layer(moe_full, moe_pruned, keep_idx, hidden_iter, gain, steps=200, lr=1e-3,
                 device="cuda", log=print):
    """Train ONLY the pruned layer's router to match the unpruned layer's output.

    Returns (router_weight, router_bias, rel_err_before, rel_err_after).
    """
    router = moe_pruned.gate if hasattr(moe_pruned, "gate") else moe_pruned.router
    for p in moe_pruned.parameters():
        p.requires_grad_(False)
    train = []
    for n, p in router.named_parameters():
        p.requires_grad_(True); train.append(p)
    if not train:
        raise RuntimeError("router exposed no parameters to train")
    opt = torch.optim.Adam(train, lr=lr)

    def rel(a, b):
        return float((a - b).pow(2).sum().sqrt() / b.pow(2).sum().sqrt().clamp(min=1e-12))

    before = after = None
    for step, h in enumerate(hidden_iter):
        h = h.to(device)
        with torch.no_grad():
            ref = moe_full(h)
            if isinstance(ref, tuple): ref = ref[0]
        out = moe_pruned(h)
        if isinstance(out, tuple): out = out[0]
        loss = (out - ref).pow(2).sum() / ref.pow(2).sum().clamp(min=1e-12)
        if before is None:
            before = float(loss.detach())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(train, 1.0)
        opt.step()
        after = float(loss.detach())
        if step + 1 >= steps:
            break
    return router, before, after


def selfcheck():
    """Gate the objective on a tiny synthetic MoE before spending GPU hours on the real one.

    The property that must hold: with NO experts pruned, the loss is ~0 and training is a
    no-op. If a full keep-set does not give a near-zero objective, the target or the gate
    renormalisation is wired wrong and every real number would be meaningless.
    """
    torch.manual_seed(0)
    E, H, I, K = 16, 64, 32, 4

    class TinyMoE(torch.nn.Module):
        def __init__(self, experts):
            super().__init__()
            self.n = len(experts)
            self.gate = torch.nn.Linear(H, self.n, bias=False)
            self.w1 = torch.nn.Parameter(torch.randn(self.n, I, H) * 0.05)
            self.w2 = torch.nn.Parameter(torch.randn(self.n, H, I) * 0.05)
        def forward(self, x):
            s = torch.sigmoid(self.gate(x))
            v, i = s.topk(K, dim=-1)
            v = v / v.sum(-1, keepdim=True)
            out = torch.zeros_like(x)
            for k in range(K):
                idx = i[:, k]
                h = torch.relu(torch.einsum('bh,bih->bi', x, self.w1[idx]))
                out = out + v[:, k:k+1] * torch.einsum('bi,bhi->bh', h, self.w2[idx])
            return out

    full = TinyMoE(list(range(E)))
    same = TinyMoE(list(range(E)))
    same.load_state_dict(full.state_dict())
    x = torch.randn(128, H)
    with torch.no_grad():
        d = (same(x) - full(x)).abs().max().item()
    print(f"identical-copy objective: max|diff| = {d:.3e}  {'PASS' if d < 1e-6 else 'FAIL'}")
    return 0 if d < 1e-6 else 1


if __name__ == "__main__":
    import sys
    sys.exit(selfcheck())


# ---------------------------------------------------------------- streaming driver
def _states_from_rows(rows, tcfg, lm_embed, dev, max_len, batch):
    """Text-only initial hidden states. Vision is skipped deliberately: router KD is a text
    objective and the image path would drag in the whole processor for no benefit."""
    out = []
    for k in range(0, len(rows), batch):
        chunk = rows[k:k + batch]
        ids = torch.full((len(chunk), max_len), tcfg.pad_token_id, dtype=torch.long)
        for r, row in enumerate(chunk):
            t = torch.tensor(row[:max_len], dtype=torch.long)
            ids[r, :t.numel()] = t
        with torch.no_grad():
            ie = lm_embed(ids.to(dev))
            hs = ie.unsqueeze(2).expand(-1, -1, tcfg.hc_mult, -1).contiguous()
        out.append({"ids": ids.cpu(), "hs": hs.cpu()})
    return out


def _fwd(layer, hs, ids, topk, dev):
    am = torch.ones(ids.shape[0], ids.shape[1], dtype=torch.bool, device=dev)
    pos = torch.arange(ids.shape[1], device=dev).unsqueeze(0)
    return layer(hs, attention_mask=am, position_ids=pos, position_embeddings=None,
                 input_ids=ids, past_key_values=None, use_cache=False,
                 prev_topk_indices=topk)


def run(mask_name="retained_152_mass", steps_per_layer=150, lr=1e-3, smoke_layers=0,
        max_len=1024, batch=2, out_name="router_kd_delta.pt"):
    """Layer-local router KD over the unpruned source. One decoder layer resident at a time.

    NO PRUNED CHECKPOINT IS MATERIALISED. Teacher and student are the SAME weights; only the
    router's candidate set differs. That removes a whole bug class -- a mis-built pruned layer
    would silently corrupt the target -- and it lets this run BEFORE surgery, which is required
    anyway because surgery deletes the source.

    Two things that would quietly invalidate the objective, both handled:
      * the reference is computed against a FROZEN CLONE of the router. The trained router is
        the same tensor the teacher pass reads, so without the clone the target drifts with the
        student and the loss falls for the wrong reason.
      * hidden states carried forward come from the MASKED (student) trajectory, because that
        is what the router sees at inference. Propagating the teacher's states instead would
        train each router on activations it never encounters.
    """
    import json, time, gc
    from pathlib import Path
    import stream_saliency as SS
    from stages.s03_saliency import _build_layer
    import stages.s09_eval as E
    from common import ROOT, ARTIFACTS, log
    from transformers import AutoConfig
    from accelerate import init_empty_weights
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextForConditionalGeneration
    import mask_eval as ME

    DEV = "cuda"
    SRC = ROOT / "source" / "GLM-5.3-Flash"
    if not (SRC / "model.safetensors.index.json").exists():
        raise SystemExit(f"unpruned source not staged at {SRC}")
    mask = json.loads((ARTIFACTS / "criteria" / f"{mask_name}.json").read_text())
    gk = mask_name.split("_", 1)[1]
    gains = json.loads((ARTIFACTS / "criteria" / f"healing_gains_{gk}.json").read_text())
    ME.patch_router_for_masking()

    cfg = AutoConfig.from_pretrained(SRC); tcfg = cfg.text_config
    reader = SS.ShardReader(SRC)
    with init_empty_weights():
        shell = Glm5NextForConditionalGeneration(cfg)
    lm = shell.model.language_model
    reader.load_module("model.language_model.embed_tokens")
    embed = lm.embed_tokens.to(DEV)

    rows, _ = E.load_heldout()
    states = _states_from_rows([r for r in rows], tcfg, embed, DEV, max_len, batch)
    del shell; gc.collect(); torch.cuda.empty_cache()
    log(f"router_kd: {len(states)} batches, mask={mask_name}, steps/layer={steps_per_layer}",
        "router_kd")

    report = {}
    nlayer = smoke_layers if smoke_layers else tcfg.num_hidden_layers
    for li in range(nlayer):
        name = f"model.language_model.layers.{li}.mlp"
        t0 = time.time()
        layer = _build_layer(tcfg, li, reader, torch.bfloat16).to(DEV)
        router = next((m for m in layer.modules()
                       if type(m).__name__ == "Glm5NextTextTopkRouter"), None)
        trainable = name in mask and router is not None
        if trainable:
            frozen = {k: v.detach().clone() for k, v in router.state_dict().items()}
            keep = torch.tensor(mask[name], dtype=torch.long)
            gain = float(gains.get(name, 1.0))
            for p in layer.parameters(): p.requires_grad_(False)
            ps = [p for p in router.parameters()]
            for p in ps: p.requires_grad_(True)
            opt = torch.optim.Adam(ps, lr=lr) if ps else None
            b0 = a1 = None
            step = 0
            while step < steps_per_layer and opt is not None:
                for st in states:
                    if step >= steps_per_layer: break
                    hs = st["hs"].to(DEV); ids = st["ids"].to(DEV)
                    with torch.no_grad():                       # teacher: frozen router, no mask
                        cur = {k: v.detach().clone() for k, v in router.state_dict().items()}
                        router.load_state_dict(frozen)
                        router._mask_keep = None
                        ref, _ = _fwd(layer, hs, ids, st.get("topk"), DEV)
                        router.load_state_dict(cur)
                    router._mask_keep, router._mask_gain = keep, gain
                    out, _ = _fwd(layer, hs, ids, st.get("topk"), DEV)
                    loss = (out - ref).pow(2).sum() / ref.pow(2).sum().clamp(min=1e-12)
                    if b0 is None: b0 = float(loss.detach())
                    opt.zero_grad(set_to_none=True); loss.backward()
                    torch.nn.utils.clip_grad_norm_(ps, 1.0); opt.step()
                    a1 = float(loss.detach()); step += 1
                    del hs, ids, ref, out, loss
            report[name] = {"before": b0, "after": a1, "secs": round(time.time()-t0, 1)}
            log(f"router_kd: {name} rel-err {b0:.5f} -> {a1:.5f} "
                f"({(b0-a1)/max(b0,1e-9):+.1%}) {time.time()-t0:.0f}s", "router_kd")
        # propagate the STUDENT trajectory
        with torch.no_grad():
            if router is not None:
                router._mask_keep = torch.tensor(mask[name], dtype=torch.long) if name in mask else None
                router._mask_gain = float(gains.get(name, 1.0))
            for st in states:
                hs = st["hs"].to(DEV); ids = st["ids"].to(DEV)
                o, tk = _fwd(layer, hs, ids, st.get("topk"), DEV)
                st["hs"] = o.cpu(); st["topk"] = tk.cpu() if tk is not None else None
                del hs, ids, o
        if trainable:
            torch.save({"name": name, "router": {k: v.detach().cpu()
                                                 for k, v in router.state_dict().items()}},
                       ARTIFACTS / f"router_kd_{li:02d}.pt")
        del layer; reader.release(); gc.collect(); torch.cuda.empty_cache()

    Path(ARTIFACTS / "router_kd_report.json").write_text(json.dumps(report, indent=1))
    if report:
        b = sum(r["before"] for r in report.values())/len(report)
        a = sum(r["after"] for r in report.values())/len(report)
        print(f"\nrouter KD, {len(report)} layers: mean rel-err {b:.5f} -> {a:.5f} "
              f"({(b-a)/max(b,1e-9):+.1%})")
    return report
