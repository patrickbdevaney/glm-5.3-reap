"""Offline replay of GLM-5.3 routing under an arbitrary expert keep-set.

This is what makes pass 2's expensive questions answerable without another forward pass.
`Glm5NextTextTopkRouter.forward` splits into two distinct uses of the router head, and REAP
depends on keeping them apart:

    scores            = sigmoid(router_logits)
    scores_for_choice = scores + e_score_correction_bias      <- SELECTION only
    topk_indices      = topk(scores_for_choice, k=8)
    topk_weights      = scores.gather(topk_indices)           <- the GATE excludes the bias
    topk_weights     /= topk_weights.sum()                    <- norm_topk_prob
    topk_weights     *= routed_scaling_factor

Pruning changes the support of that top-8. Because `norm_topk_prob` renormalises over whatever
survives, the gate a surviving expert receives after pruning is NOT the gate it received before -
it is strictly larger, since the same expert now divides by a smaller sum. That is the exact
effect P5 has to measure rather than derive, and the reason the pass-1 healing gain is `[OPEN]`.

`n_group == 1` and `topk_group == 1` in this checkpoint, so the grouped-topk path in the upstream
forward is an identity and is deliberately not reproduced here. Verified against config.json for
both the source and the pruned model.
"""
from __future__ import annotations

import torch


def simulate(scores: torch.Tensor, bias: torch.Tensor, keep: torch.Tensor | None = None,
             top_k: int = 8, norm_topk_prob: bool = True,
             routed_scaling_factor: float = 2.5) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (topk_index, topk_gate) for `scores` restricted to `keep`.

    scores: [T, E] sigmoid affinities. bias: [E]. keep: bool [E] or None for "keep everything".
    """
    sfc = scores.float() + bias.float()
    if keep is not None:
        sfc = sfc.masked_fill(~keep.to(sfc.device), float("-inf"))
    idx = torch.topk(sfc, k=top_k, dim=-1, sorted=False).indices
    w = scores.float().gather(1, idx)
    if norm_topk_prob:
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-20)
    return idx, w * routed_scaling_factor


def simulate_from_cache(cached_scores: torch.Tensor, cached_idx: torch.Tensor,
                        bias: torch.Tensor, keep: torch.Tensor | None = None,
                        top_k: int = 8, **kw) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Same, but from the top-K cache written by stream_saliency.

    Returns (global_expert_index, gate, enough) where `enough` is False for any token that had
    fewer than `top_k` survivors inside the cached candidate list. Those tokens are not wrong,
    they are *unknown* - the cache simply did not retain enough candidates to resolve them - and
    they must be excluded from any statistic rather than silently filled in. Reporting that rate
    is the honesty check on the whole replay.
    """
    T, K = cached_idx.shape
    E = bias.shape[0]
    sc = torch.full((T, E), float("nan"), dtype=torch.float32, device=cached_scores.device)
    sc.scatter_(1, cached_idx.long(), cached_scores.float())
    present = ~torch.isnan(sc)
    sc = torch.nan_to_num(sc, nan=0.0)
    avail = present if keep is None else (present & keep.to(present.device))
    enough = avail.sum(dim=-1) >= top_k
    sfc = sc + bias.float()
    sfc = sfc.masked_fill(~avail, float("-inf"))
    idx = torch.topk(sfc, k=top_k, dim=-1, sorted=False).indices
    w = sc.gather(1, idx)
    if kw.get("norm_topk_prob", True):
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-20)
    return idx, w * kw.get("routed_scaling_factor", 2.5), enough
