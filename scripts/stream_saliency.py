"""Path B - custom layer-streaming REAP saliency.

Used when the model cannot be placed by transformers/accelerate at all (see
wiki/96-implementation.md). Peak residency is ONE decoder layer.

Feasible because the entire inter-layer state in Glm5NextTextModel.forward is a single tensor
plus topk_indices - mHC's four residual streams live in the hc_mult axis of that tensor and
are manipulated inside each decoder layer, so there is no cross-layer bookkeeping to replicate.

Chunk by samples, sweeping all layers per chunk. The hc_mult=4 expansion makes activations 4x
larger than intuition suggests, and on this box read is 7x faster than write (3.4 GB/s vs
487 MB/s), so re-reading weights is far cheaper than re-writing activations.

PASS 2 INSTRUMENTATION
----------------------
Pass 1 accumulated exactly two numbers per expert (sum of g*||f||, and a token count) and threw
everything else away. Every alternative criterion then required another 14-hour pass. Pass 2
accumulates the sufficient statistics for ~8 offline criteria on the SAME forward, at negligible
marginal cost, plus a subsampled router-score cache that makes post-prune routing exactly
replayable offline.

What is accumulated, per layer, per expert, per bucket:
    sum, sq       first and second moment of g*||f||   -> REAP + variance-aware variants
    nrm, nsq      first and second moment of ||f||     -> separates expert magnitude from gate
    gat, gsq      first and second moment of g         -> separates gate from magnitude
    cnt           tokens routed                        -> conditional vs unconditional means
    hist          log-histogram of g*||f||             -> tail shape without storing tokens
    osum          sum of g*f_j as a VECTOR             -> lets P5 re-fit the healing gain from a
                                                          measured output norm rather than a
                                                          first-moment derivation

Two correctness fixes over pass 1, both of which silently biased the pass-1 statistics:
  * padding tokens were accumulated as if they were real tokens (see VALID below)
  * per-domain attribution was impossible after the fact, so the mixture could not be re-weighted
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch

# --------------------------------------------------------------------------------------------
# Accumulators
#
# Everything is [n_buckets, n_experts] float64 on the GPU: 16*288*8 = 37 KB per tensor per layer,
# so all seven scalar accumulators for all 42 layers cost ~11 MB. There is no reason to be
# stingy here, and pass 1's stinginess is what forced a second 14-hour pass.
# --------------------------------------------------------------------------------------------

# Bucket 0 is the catch-all so an unset bucket is never silently attributed to a real domain.
BUCKETS = ["general", "code", "math", "agentic", "finance", "science", "bio", "vision"]
BUCKET_ID = {b: i for i, b in enumerate(BUCKETS)}

HIST_BINS = 36
HIST_LO, HIST_HI = -6.0, 3.0        # log10(g*||f||); empirically ~[-3, 1] at 2048 tokens

ACC: dict[str, dict[str, torch.Tensor]] = {}

# Back-compat aliases. s04_sweep reads these; dump() keeps writing the same keys.
SAL_SUM: dict[str, torch.Tensor] = {}
SAL_CNT: dict[str, torch.Tensor] = {}

_CTX = {
    "layer": None,      # str  - which MoE layer is executing
    "bucket": 0,        # int  - domain bucket for the CURRENT batch (batches are homogeneous)
    "valid": None,      # BoolTensor [n_tokens] or None - False on padding
}

# Router score cache: [(layer, topk_scores fp16, topk_idx uint16), ...] for a token subsample.
ROUTER_CACHE: list[tuple[str, torch.Tensor, torch.Tensor]] = []
ROUTER_TOPK = 40
ROUTER_SUBSAMPLE = 0.20     # fraction of tokens cached; 5.5M tok * 42 layers * 40 * 4B * 0.2 = 7.4 GB


def set_bucket(name: str | None):
    """Set the domain bucket for the batch about to run.

    Batches must be homogeneous in bucket. The s03 driver groups samples by bucket before
    batching, which is free - it only changes the order samples are packed in.
    """
    _CTX["bucket"] = BUCKET_ID.get(name or "general", 0)


def set_valid_mask(valid: torch.Tensor | None):
    """Mark which flattened token positions are real.

    Pass 1 padded every sequence to a fixed 2048 and then accumulated saliency over the padding
    as though it were text. Padding routes like any other token, so this added a consistent,
    content-free signal to whichever experts happen to serve the pad embedding - and it did so
    most heavily for the SHORTEST documents. Masking it costs one gather.
    """
    _CTX["valid"] = valid


def _ensure(lname: str, n_experts: int, hidden: int, dev) -> dict[str, torch.Tensor]:
    a = ACC.get(lname)
    if a is None:
        nb = len(BUCKETS)
        z = lambda: torch.zeros(nb, n_experts, dtype=torch.float64, device=dev)
        a = {"sum": z(), "sq": z(), "cnt": z(), "nrm": z(), "nsq": z(), "gat": z(), "gsq": z(),
             "hist": torch.zeros(n_experts, HIST_BINS, dtype=torch.int64, device=dev),
             "osum": torch.zeros(n_experts, hidden, dtype=torch.float32, device=dev)}
        ACC[lname] = a
        SAL_SUM[lname] = a["sum"]
        SAL_CNT[lname] = a["cnt"]
    return a


def dequant_fp8_block(w: torch.Tensor, scale_inv: torch.Tensor,
                      block: int = 128, dtype=torch.bfloat16) -> torch.Tensor:
    """FP8 E4M3 + per-128x128-block F32 scale -> dtype."""
    out, inn = w.shape
    s = scale_inv.to(torch.float32)
    s = s.repeat_interleave(block, dim=0).repeat_interleave(block, dim=1)[:out, :inn]
    return (w.to(torch.float32) * s).to(dtype)


def patch_router_for_cache():
    """Cache the router's per-token expert SCORES so post-prune routing is replayable offline.

    This is what makes the expensive questions answerable without another forward pass. From
    `scores` plus `e_score_correction_bias` the entire selection path is reproducible exactly:
    selection uses `scores + bias` (group-masked, top-8), while the GATE that multiplies the
    expert output is `scores` alone, renormalised over the surviving top-8 and scaled by
    `routed_scaling_factor`. So for ANY candidate keep-set we can recompute, offline:
      * which experts each token would route to after pruning
      * the renormalised gate each would receive
    which is exactly what P5 (healing re-fit) and P9 (router-aware staged re-scoring) need, and
    what pass 1 could not do at any price short of re-running the pass.

    Only `scores` is cached, not `router_logits`: sigmoid is monotone so selection is identical,
    and scores are what the gate is read from.
    """
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextTopkRouter
    import torch.nn.functional as F

    if getattr(Glm5NextTextTopkRouter, "_reap_cached", False):
        return

    orig = Glm5NextTextTopkRouter.forward

    def forward(self, hidden_states):
        router_logits, topk_weights, topk_indices = orig(self, hidden_states)
        lname = _CTX["layer"]
        if lname is not None and ROUTER_SUBSAMPLE > 0:
            with torch.no_grad():
                scores = router_logits.sigmoid()
                valid = _CTX["valid"]
                if valid is not None:
                    scores = scores[valid]
                if scores.shape[0]:
                    n = max(1, int(scores.shape[0] * ROUTER_SUBSAMPLE))
                    # Deterministic stride, not RNG: reproducible across resumed chunks.
                    sel = torch.arange(0, scores.shape[0], max(1, scores.shape[0] // n),
                                       device=scores.device)[:n]
                    sc = scores[sel]
                    # Rank by the SELECTION criterion (scores + bias), not by scores. The bias
                    # reorders candidates, so a top-40 taken on scores alone can omit experts
                    # that would actually be selected - which would silently corrupt every
                    # offline replay. The VALUES stored are still `scores`, because that is
                    # what the gate is read from.
                    sfc = sc + self.e_score_correction_bias.to(sc.dtype)
                    i = sfc.topk(min(ROUTER_TOPK, sc.shape[1]), dim=-1).indices
                    v = sc.gather(1, i)
                    ROUTER_CACHE.append((lname, v.to(torch.float16).cpu(),
                                         i.to(torch.int32).cpu()))
        return router_logits, topk_weights, topk_indices

    Glm5NextTextTopkRouter.forward = forward
    Glm5NextTextTopkRouter._reap_cached = True


def patch_experts_for_saliency():
    """Replace Glm5NextTextExperts.forward with a copy that also accumulates REAP saliency.

    The arithmetic is the upstream forward verbatim; the only addition is that the expert
    output f_j is captured BEFORE it is scaled by the router gate, which is what REAP's
    S_j = mean(g_j * ||f_j||_2) is defined over.
    """
    import torch.nn.functional as F
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextExperts

    if getattr(Glm5NextTextExperts, "_reap_patched", False):
        return

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
            # ONE host sync for the hit list. Iterating a CUDA tensor in Python syncs on every
            # element - 288 syncs per batch, ~39k per layer.
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero().flatten().tolist()
        lname = _CTX["layer"]
        dev = hidden_states.device
        acc = None
        if lname is not None:
            acc = _ensure(lname, self.num_experts, hidden_states.shape[-1], dev)
            bkt = _CTX["bucket"]
            valid = _CTX["valid"]
            if valid is not None and valid.shape[0] != hidden_states.shape[0]:
                # Glm5NextTextMoE flattens with view(-1, H), so the mask is 1:1 with rows IFF the
                # MLP sees [B, S, H]. mHC's hc_mult axis is consumed inside the decoder layer
                # before the MLP, so it should. If that ever stops being true, a silently
                # misaligned mask would corrupt every statistic in the run while still looking
                # plausible - so refuse to guess how to broadcast it.
                raise RuntimeError(
                    f"valid mask length {valid.shape[0]} != {hidden_states.shape[0]} expert rows "
                    f"in {lname}; token flattening is not 1:1, fix the mask construction")
        for expert_idx in hit:
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(mask[expert_idx])
            cur = self._apply_gate(F.linear(hidden_states[token_idx],
                                            self.gate_up_proj[expert_idx]))
            f_j = F.linear(cur, self.down_proj[expert_idx])          # expert output, ungated
            g_j = top_k_weights[token_idx, top_k_pos]                 # router gate
            if acc is not None:
                with torch.no_grad():
                    if valid is not None:
                        keep = valid[token_idx]
                        fj_v, gj_v = f_j[keep], g_j[keep]
                    else:
                        fj_v, gj_v = f_j, g_j
                    if fj_v.shape[0]:
                        nrm = fj_v.to(torch.float32).norm(dim=-1)
                        gg = gj_v.to(torch.float32)
                        s = gg * nrm
                        sd = s.double()
                        acc["sum"][bkt, expert_idx] += sd.sum()
                        acc["sq"][bkt, expert_idx] += (sd * sd).sum()
                        acc["cnt"][bkt, expert_idx] += fj_v.shape[0]
                        nd = nrm.double()
                        acc["nrm"][bkt, expert_idx] += nd.sum()
                        acc["nsq"][bkt, expert_idx] += (nd * nd).sum()
                        gd = gg.double()
                        acc["gat"][bkt, expert_idx] += gd.sum()
                        acc["gsq"][bkt, expert_idx] += (gd * gd).sum()
                        # Vector sum of the GATED output. The layer's mean output is the sum of
                        # these over the keep-set, so P5 can measure the pre/post norm ratio
                        # instead of deriving it from first moments.
                        acc["osum"][expert_idx] += (fj_v * gj_v[:, None]).to(torch.float32).sum(0)
                        lg = torch.log10(s.clamp_min(1e-12))
                        bins = (((lg - HIST_LO) / (HIST_HI - HIST_LO)) * HIST_BINS
                                ).clamp_(0, HIST_BINS - 1).long()
                        acc["hist"][expert_idx].scatter_add_(
                            0, bins, torch.ones_like(bins, dtype=torch.int64))
            final.index_add_(0, token_idx, (f_j * g_j[:, None]).to(final.dtype))
        return final

    Glm5NextTextExperts.forward = forward
    Glm5NextTextExperts._reap_patched = True


class ShardReader:
    """Lazily mmap the source shards and hand back one layer's tensors at a time."""

    def __init__(self, src: Path):
        self.src = Path(src)
        idx = json.loads((self.src / "model.safetensors.index.json").read_text())
        self.map: dict[str, str] = idx["weight_map"]
        self._open: dict[str, object] = {}

    def _f(self, shard: str):
        from safetensors import safe_open
        if shard not in self._open:
            self._open[shard] = safe_open(str(self.src / shard), framework="pt", device="cpu")
        return self._open[shard]

    def names_for_layer(self, prefix: str) -> list[str]:
        return [k for k in self.map if k.startswith(prefix)]

    def get(self, name: str) -> torch.Tensor:
        return self._f(self.map[name]).get_tensor(name)

    def release(self) -> None:
        """Close every open shard handle.

        safe_open holds a live mmap of a ~5.3 GB shard. While the mapping exists its faulted-in
        pages CANNOT be reclaimed - drop_caches is a no-op against them - so keeping all 62
        handles open accumulates ~306 GB of unreclaimable page cache across the layer sweep.
        Callers must ensure no returned tensor still views the mapping (see copy=True below).
        """
        self._open.clear()

    def load_module(self, prefix: str, dtype=torch.bfloat16) -> dict[str, torch.Tensor]:
        """Return {relative_name: tensor}, dequantising any FP8+scale_inv pair."""
        out: dict[str, torch.Tensor] = {}
        names = self.names_for_layer(prefix)
        scales = {n for n in names if n.endswith("weight_scale_inv")}
        for n in names:
            if n in scales:
                continue
            rel = n[len(prefix):].lstrip(".")
            t = self.get(n)
            sname = n + "_scale_inv" if not n.endswith(".weight") else n[:-len("weight")] + "weight_scale_inv"
            if sname in scales:
                t = dequant_fp8_block(t, self.get(sname), dtype=dtype)   # already a new tensor
            else:
                # copy=True matters: .to(dtype) on a tensor ALREADY in that dtype returns self,
                # which is a view into the mmap. Releasing the handle under a live view would
                # be a use-after-unmap, and keeping the handle open is what pins the cache.
                t = t.to(dtype, copy=True) if t.is_floating_point() else t.clone()
            out[rel] = t
        return out

    def close(self):
        self._open.clear()


def set_current_layer(name: str | None):
    _CTX["layer"] = name


def reset_accumulators():
    ACC.clear()
    SAL_SUM.clear()
    SAL_CNT.clear()
    ROUTER_CACHE.clear()


def dump_router_cache(path: Path) -> int:
    """Flush the router-score cache for one chunk and clear it.

    Written per chunk, not per run: the cache is the one accumulator that grows with tokens
    rather than with experts, so it must not be allowed to accumulate across the whole pass.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not ROUTER_CACHE:
        return 0
    by: dict[str, list] = {}
    for lname, v, i in ROUTER_CACHE:
        by.setdefault(lname, []).append((v, i))
    out = {ln: {"scores": torch.cat([x[0] for x in rows]),
                "idx": torch.cat([x[1] for x in rows])} for ln, rows in by.items()}
    torch.save({"topk": ROUTER_TOPK, "subsample": ROUTER_SUBSAMPLE, "layers": out}, path)
    n = sum(v["scores"].shape[0] for v in out.values())
    ROUTER_CACHE.clear()
    return n


def dump_light(dirpath: Path) -> int:
    """Write a small CUMULATIVE snapshot of the ranking statistics only.

    Exists so the split-half stopping rule (P6) is computable at all. The keep-set depends only
    on `sum` and `cnt`, and a single running total cannot be split after the fact - so each chunk
    leaves a cumulative snapshot, and any chunk's own contribution is recovered by differencing
    two of them. Half A = chunks 0..5, half B = 6..11, and the overlap between the two keep-sets
    says whether the token budget was sufficient instead of guessing at a floor.

    Deliberately omits `osum` (288x4096 f32, ~4.7 MB/layer) and `hist`: at ~130 KB/layer this is
    ~5.5 MB per chunk, so keeping all 12 costs ~66 MB. The full accumulators are still written by
    dump() for resume.
    """
    dirpath = Path(dirpath)
    dirpath.mkdir(parents=True, exist_ok=True)
    for lname in sorted(ACC):
        a = ACC[lname]
        torch.save({"layer": lname,
                    "sum_saliency": a["sum"].sum(0).detach().cpu(),
                    "count": a["cnt"].sum(0).detach().cpu(),
                    "sum_by_bucket": a["sum"].detach().cpu(),
                    "cnt_by_bucket": a["cnt"].detach().cpu()},
                   dirpath / f"{lname.replace('.', '__')}.pt")
    return len(ACC)


def load_accumulators(dirpath: Path, device) -> int:
    """Restore accumulators from a previous run's per-layer dumps.

    Chunked calibration is only worth doing if a kill costs one chunk rather than the whole
    token budget, and that requires the accumulators to survive the process. They are already
    written per layer by dump(); this reads them back onto the device so the next chunk adds
    to them instead of starting from zero.
    """
    dirpath = Path(dirpath)
    n = 0
    for f in sorted(dirpath.glob("*.pt")):
        try:
            d = torch.load(f, weights_only=False)
        except Exception:
            continue
        lname = d.get("layer")
        if lname is None or "sum_by_bucket" not in d:
            continue          # a pass-1 dump: no per-bucket tensors, cannot be resumed into
        a = {
            "sum": d["sum_by_bucket"].to(device),
            "cnt": d["cnt_by_bucket"].to(device),
            "sq": d["sq_by_bucket"].to(device),
            "nrm": d["norm_sum_by_bucket"].to(device),
            "nsq": d["norm_sq_by_bucket"].to(device),
            "gat": d["gate_sum_by_bucket"].to(device),
            "gsq": d["gate_sq_by_bucket"].to(device),
            "hist": d["hist"].to(device),
            "osum": d["out_sum"].to(device),
        }
        ACC[lname] = a
        SAL_SUM[lname] = a["sum"]
        SAL_CNT[lname] = a["cnt"]
        n += 1
    return n


def dump(dirpath: Path, layer_name_fmt: str = "model.language_model.layers.{i}.mlp"):
    """Write per-layer accumulators.

    `sum_saliency` and `count` are kept as the bucket-summed 1D tensors pass 1 wrote, so
    s04_sweep and every existing artifact reader keep working unchanged. The per-bucket and
    higher-moment tensors are additive.
    """
    dirpath = Path(dirpath)
    dirpath.mkdir(parents=True, exist_ok=True)
    for lname in sorted(ACC):
        a = ACC[lname]
        rec = {
            "layer": lname,
            "num_experts": a["sum"].shape[1],
            "buckets": BUCKETS,
            # pass-1-compatible keys
            "sum_saliency": a["sum"].sum(0).detach().cpu(),
            "count": a["cnt"].sum(0).detach().cpu(),
            # pass-2 additions
            "sum_by_bucket": a["sum"].detach().cpu(),
            "cnt_by_bucket": a["cnt"].detach().cpu(),
            "sq_by_bucket": a["sq"].detach().cpu(),
            "norm_sum_by_bucket": a["nrm"].detach().cpu(),
            "norm_sq_by_bucket": a["nsq"].detach().cpu(),
            "gate_sum_by_bucket": a["gat"].detach().cpu(),
            "gate_sq_by_bucket": a["gsq"].detach().cpu(),
            "hist": a["hist"].detach().cpu(),
            "hist_range": (HIST_LO, HIST_HI, HIST_BINS),
            "out_sum": a["osum"].detach().cpu(),
        }
        torch.save(rec, dirpath / f"{lname.replace('.', '__')}.pt")
    return len(ACC)
