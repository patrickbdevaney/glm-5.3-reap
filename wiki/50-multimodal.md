# 50 — Vision preservation under expert pruning

Vision is a **first-class requirement** in this project, not an optional extra.

## The structural fact that reframes the whole problem

**GLM-5.3-Flash's vision tower contains no MoE.** It is a dense 24-layer ViT,
563.6M params, 0.18% of the model, entirely BF16
([10-target-model.md](10-target-model.md)).

**REAP therefore cannot damage the vision tower. It is structurally untouchable.** `[EST]`

The real vision risk is one level down: GLM-5.3-Flash is **natively multimodal**, pretrained
on a 30T-token multimodal corpus, so image tokens are projected into the *shared* text
backbone (`out_hidden_size 4096`) and routed through the *same* 288-expert MoE layers as
text. If image tokens route to experts that text calibration never activates, those experts
score `S_j = 0` — they are never in any `X_j` — and are deleted first.

> **A text-only calibration set does not "under-weight" vision experts. It makes them
> invisible, and REAP deletes invisible experts with certainty, at any prune ratio.** `[EXT,
> follows directly from the saliency definition]`

This is the sharpest, most actionable finding in the multimodal workstream, and it means the
15% multimodal calibration share is not a nice-to-have — it is load-bearing.

## Do vision tokens occupy distinct routing mass?

`[OPEN]` for GLM-5.3-Flash specifically. Nobody has published per-expert routing statistics
for `glm5_next`.

What the literature establishes generally: MoE-VLMs exhibit **genuine but partial** modality
specialisation. MoIIE (arXiv 2508.09779) finds enough intra-modality structure to be worth
architecting for, splitting experts into intra- and inter-modality pools plus a shared pool.
SMoES (arXiv 2604.23996) and long-tailed-router work (arXiv 2507.01351) both find vision and
language want *different expert loads*, motivating modality-specific routers. `[EST]`

So the expected structure is: a **shared committee** carrying most routing mass across both
modalities, plus a **thinner modality-specialised periphery**. That is the same shape the
directive cites for domain specialisation, and it has the same implication — the periphery
is what pruning destroys first, and the periphery is where vision-specific competence lives.

**This is directly measurable on our own box before any pruning, and it should be
measured.** The routing-mass instrumentation for the §3.6 proxies gives it for free: run the
calibration corpus, log per-expert routing mass **stratified by token modality** (image
tokens are identifiable by `image_token_id 154854` / `video_token_id 154855` and the
start/end markers). The output is a per-layer answer to "how much routing mass is
vision-only", which converts an `[OPEN]` into a measured number and directly sets the
required multimodal calibration share. **Do this in the sensitivity probe.**

## Resolving the ignore-list conflict (directive §3.2)

The directive flags llm-compressor's VL example passing
`ignore=["lm_head", "re:.*visual.*", "re:.*linear_attn.*"]` and asks whether to copy it.

**Answer: no, and the conflict dissolves once prune and quantise are separated.** `[EXT]`

llm-compressor's multimodal guidance is explicit about *why* it skips the tower:

> "Most examples do not demonstrate quantizing separate vision encoder parameters if they
> exist, as compressing these parameters offers little benefit."

That reasoning is about **quantisation**, and it applies here with even more force: the tower
is 0.18% of the model, so quantising it buys ~0.6 GB and risks first-class capability.
**Skip it — for the stated reason, not by imitation.**

Two distinct ignore lists are needed:

| Stage | ignore | rationale |
|---|---|---|
| **REAP prune** | *nothing needs listing* | the modifier only touches MoE layers it discovers; the tower, linear-attn and MLA have no experts and are never candidates |
| **NVFP4 quantise** | `re:.*visual.*`, `re:.*lm_head`, `re:.*mlp.gate.*` (routers), mHC (`re:.*hc_.*`, `mapping_proj`), `re:.*linear_attn.*` / KDA state params | tiny mass, high sensitivity, or numerically fragile |

The `linear_attn` exclusion in the upstream example is **correct and worth keeping**, for a
reason independent of vision: KDA carries recurrent state (`A_log`, `dt_bias`, gates, conv
kernels). Quantising a recurrence's state-transition parameters compounds error across the
sequence rather than averaging it out — the same reason Minitron-SSM finds SSM-state pruning
catastrophic ([40-hybrid-fragility.md](40-hybrid-fragility.md)). `[EXT, well-motivated]`

The GLM-5.2 precedent — **NVFP4 on MoE + FP8 on attention, >70% size reduction with GPQA
maintained** — is the right per-component policy shape and is what
[60-quantization.md](60-quantization.md) adopts. `[VEN]`

## Calibration and eval requirements

- The multimodal calibration slice must exercise **the routing paths vision actually uses**,
  which means real image-text pairs pushed through the real processor — not text descriptions
  of images.
- llm-compressor's generic `data_collator` handles most multimodal datasets, but
  `glm5_next` is new (`transformers 5.16.0`) and a model-specific collator may be required.
  **Verify early**; this is a plausible source of silent failure where images are dropped and
  calibration silently degenerates to text-only — the exact scenario that deletes vision.
  **Assert non-zero image-token count in the calibration stream before trusting any run.**
- All §3.6 proxy metrics must be computed on a **separate held-out image-text slice**.
  Averaging vision into a text-dominated mean would hide vision-specific collapse completely.

## Precedent

**Kimi-K3** (arXiv 2607.24653) is a 2.78T native-multimodal MoE — KDA + 896 experts + 1M
context, "text, images, and videos processed by a single shared backbone within one context,
with no post-hoc modality-alignment stage" — i.e. the same shared-backbone design as
GLM-5.3-Flash. Community REAP ports exist at **73% and 80%**
(`pipenetwork/Kimi-K3-REAP{73,80}-MLX-mxfp4-q8`). They are the only high-ratio REAP artifacts
in existence, and they are reported with **"noticeable degradation versus full K3"**.
No modality-stratified evaluation was published for them. `[VEN]`
