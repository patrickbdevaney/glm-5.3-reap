# Retraining DFlash 2 against the REAP target

**Decision (operator, 2026-08-27):** retrain `incoai/GLM-5.3-Flash-DFlash2` against our REAP model
on permissive data, and release it to a **private** HF repo. Private hosting is personal use, not
publication, so the upstream `cc-by-nc-nd-4.0` NoDerivatives term is satisfied. Revisit only if it
is ever made public.

Prior art: `~/deepseek-v4-flash-0731-cuda/wiki/draft-head-finetuning.md` (DSpark MTP head, 620
lines of measured discipline). Everything below either transfers a rule from it or records where
GLM-5.3 differs.

## 1. What we are training — read from the checkpoint, not the paper

The prior art's most expensive lesson is that a parameter count copied from a paper about a
different model failed at allocation *after* a multi-day capture had been paid for. So, from the
safetensors header:

| quantity | value |
|---|---|
| parameters | **1.171 B**, 80 tensors, **all BF16** |
| size | 2.18 GiB |
| backbone | 5 × qwen3 sliding-attention (`window 2048`), `is_causal: false` |
| `fc.weight` | 83.9 M = **20480 → 4096** — the feature projection over 5 concatenated taps |
| `candidate_selector.{predecessor,successor}_codebook` | 39.6 M each = vocab 154880 × rank 256 |
| MLP per layer | 251.7 M × 3 (`gate/up/down`) |
| two-tap dyn. conv | `attention_conv`, `mlp_conv` kernel projections, 21 M each |

**Memory is not a constraint here, unlike DSpark.** At 10 B/param (bf16 master + fp32 AdamW
moments) a **full** fine-tune of all 1.171 B is **~11.7 GiB** against a 122 GiB box. DSpark had to
freeze 12 B of MXFP4 experts; we do not have that problem, so we do not inherit that concession.
There is no `embed_tokens`/`lm_head` in this checkpoint to freeze either.

## 2. The capture trap, which applies here verbatim

DSpark's rule: **capture the raw taps, never the post-projection features**, because the feature
projection is trainable and capturing after it bakes in weights you are about to change.

`fc` is 20480→4096 and `target_layer_ids = [5, 14, 24, 33, 42]` — 5 × 4096 = 20480. So `fc` **is**
that projection, and the correct capture point is the five raw taps.

| capture point | bytes/token (bf16) | usable |
|---|---|---|
| post-`fc` features (4096) | 8 KB | **no** — `fc` is trainable |
| **5 raw taps (20480)** | **40 KB** | yes |
| **+ final hidden state** | **+8 KB** | yes — needed, see below |

**GLM-5.3 differs from DSpark on the target distribution.** DSpark recomputes `p` from its taps
because its taps (40/41/42) sit at the end of the stack. Ours end at **layer 42 of 45**, with
layers 43–44 and `model.norm` still to come, so the taps do **not** determine the final logits.
Store the final hidden state as well (+8 KB/token) and recompute `p = softmax(lm_head(norm(h)))`
during training.

Storing `p` itself is impossible and must not be attempted: vocab 154,880 × 4 B = **620 KB/token**.
`lm_head` is 154880×4096 bf16 = **1.27 GiB**, cheap to keep resident. The taps plus `h_final` are
the minimal sufficient statistic.

**48 KB/token total.**

## 3. Storage is a non-issue, because the recipe is one epoch

This is the rule that dissolves the "100M tokens = 4 TB" objection: at one epoch each sample is
used exactly once, so **capture a shard → train on it → delete → next shard**. Peak usage is one
shard, and corpus size becomes purely a time decision.

| shard | bf16 @ 48 KB/tok |
|---|---|
| 1 K samples × 1024 tok | 49 GB |
| 2 K samples | 98 GB |

Cap shards at ~1 K samples given the FP8 master also sits on disk until P12.

## 4. Recipe — the published defaults, adopted

| knob | value | source |
|---|---|---|
| loss | `Σ_k w_k [0.1·CE + 0.9·TV + 1.0·BCE(selector)]`, `w_k = exp(-(k-1)/5)` | DSpark / NeMo |
| LR | 5e-5 cosine, warmup 0.05 | FastMTP |
| optimiser | AdamW β=(0.9, 0.95) | FastMTP |
| global batch | 64 sequences | FastMTP |
| **epochs** | **1** — the head ships pretrained | Nebius, on DeepSeek-MTP from pretrained |
| precision | bf16 params, fp32 optimiser states | — |

**TV dominates for a measured reason, not a stylistic one.** Nebius's LK-losses result: KL reaches
50.2 % acceptance where TV reaches 60.2 % at identical capacity, because KL does not converge to
acceptance-maximising solutions under a capacity constraint — which is always the real case.
`α_tv = 0.9` already embodies that. The λ-schedule hybrid (+5.6 % on DeepSeek-V3 MTP fine-tuned
from pretrained, the closest published setting) is the one knob worth sweeping, after a baseline.

DFlash 2 drafts a **block** (`block_size 8`, 7 draft tokens/step) rather than a chain, so `k` in
`w_k` indexes position within the block. The `candidate_selector` is the natural target for the
third loss term, in the role DSpark's confidence head plays.

## 5. Data — regenerate, do not reuse human responses

DSpark, FastMTP, Nebius and NeMo independently agree: **take prompts from an instruction corpus,
discard the human responses, regenerate with the target.** NeMo states the reason plainly —
avoiding a train/inference distribution mismatch. Regeneration sampling (FastMTP, published):
`T=0.6, top-k 20, top-p 0.95, max 4096`.

Our target is the **REAP** model, which is the point: we want alignment to what we will serve.

Mix: adopt DSpark's, which is the one measured on a drafter — ~39 % math, ~39 % code, 18 % chat,
4 % IF. That is also where speculative decoding earns most. Draw prompts from our existing
permissive corpus, whose weighting is already code/agentic-heavy.

**This is the gating dependency.** Regeneration is autoregressive decode, not prefill, so it needs
the REAP NVFP4 actually *served* on Thor. Capture itself is teacher-forced prefill — one forward
pass per sequence — so capture throughput is prefill throughput, and the streaming harness already
does that shape of work.

## 6. Sequence

| # | step | depends on |
|---|---|---|
| D1 | pass 2 → REAP FP8 → NVFP4 | P14 |
| D2 | serve REAP NVFP4 on Thor (vLLM SM110 prior art) | D1 |
| D3 | regenerate responses from permissive prompts, `T=0.6/top-k 20/top-p 0.95` | D2 |
| D4 | teacher-forced capture: 5 taps + `h_final`, shard → train → delete | D3 |
| D5 | train 1 epoch, TV-dominant position-decayed loss | D4 |
| D6 | measure acceptance length vs. stock DFlash 2 **on the REAP target** | D5 |
| D7 | push to a **private** HF repo | D6 |

## 7. Two disciplines to carry over verbatim

1. **The promotion bar is `max(re-measured, registry)`, and both get printed.** A measurement
   passed forward *by value* goes stale the instant the thing it measured changes; a stale baseline
   nearly promoted a head that lost to what was already serving. Max can only refuse a promotion,
   never falsely grant one — the property you want in a ruler.
2. **Baseline first.** Measure stock DFlash 2's acceptance length against the REAP target *before*
   training anything. Without it there is no way to tell whether fine-tuning helped or whether REAP
   simply did not hurt the drafter much — and that number is also the honest answer to "was the
   retrain worth it".
