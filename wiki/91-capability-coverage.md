# 91 — Capability coverage: what the calibration set actually protects

*Opened 2026-09-26. Question asked directly: "with this third reap and mimo not only are we
preserving the multimodalities but also long context and all of the other needed model
capabilities that i may have not thought of correct".*

Short answer: **the named buckets are covered; long context and non-English are not, and
neither is measured.** Both gaps are in the *saliency* stage, not the corpus.

---

## 1. What the seven buckets do cover

`agentic, code, math, ballast(general), science, finance, multimodal(vision)` — plus audio and
video on MiMo. REAP keeps the experts these domains route to, and `protect_frac` reserves the
top-k of *every* domain separately so no bucket can be zeroed out by a louder one. Vision is
protected structurally, not incidentally: see [50-multimodal.md](50-multimodal.md).

That machinery works on any axis that has a bucket. The gaps below are axes that have **no
bucket**, and therefore no protection and no measurement.

---

## 2. Long context is never measured  `[EST]`

The corpus is built to carry long samples. The saliency pass throws them away.

```
scripts/corpus_spec.py:22    MAX_TOKENS = 16_384
scripts/corpus_spec.py:84    BAND_HARD  = (4_000, MAX_TOKENS)   # difficulty target 30%
scripts/stages/s03_saliency.py:65   MAX_LEN = kv_get("calib_max_len", 2048)
scripts/stages/s03_saliency.py:105  text_rows.append((it["input_ids"][:MAX_LEN], bucket))
```

MEASURED 2026-09-26 over `corpus/shards/text/*.pt` (10,141 samples, the real pass-3 corpus):

| statistic | value |
|---|---|
| mean sample length | 4,761 tokens |
| max sample length | 16,384 tokens |
| samples longer than 2,048 | **38.2 %** (3,875) |
| hard band (≥4,000 tok) | 31.0 % — spec target 30 %, built correctly |
| **tokens discarded by `MAX_LEN=2048`** | **35,471,933 of 48,283,959 = 73.5 %** |

So the corpus hits its difficulty spec and then `s03` deletes three quarters of it. Every
16,384-token agentic trajectory is scored as its **opening 2,048 tokens**. The hard band is not
under-weighted, it is *structurally erased* — after truncation a hard sample and a medium sample
are the same length.

**This does not prove long context is damaged.** It proves it is unmeasured. The question that
decides it is whether expert selection at position 100 K differs from position 1 K. If routing is
position-invariant, 2,048 is a free win and we should write that down. Nobody has checked.

### Why GLM is the worse case, and why MiMo's argument does not transfer  `[EXT]`

`scripts/exp_seqlen_saliency.py` in the MiMo repo asks exactly this question and **was never
run** — the source was deleted before it executed, so there is no result. Its reasoning was
MiMo-specific:

> 39 of 48 layers are SWA with a window of 128, so their attention context is capped at 128
> tokens NO MATTER WHAT S IS — structurally, S cannot change what those layers see.

That is a sound argument *for MiMo* and it predicts near-S-invariance. **It inverts on GLM.**
From `config.json`: `layer_types` is a hybrid, `linear_attn_config.kda_layers` covers roughly
three of every four layers, with `deepseek_sparse_attention` on the rest. KDA is linear attention
carrying a **recurrent state that accumulates over the entire context**. The MoE router reads the
hidden state that state produces. At token 100 K a KDA layer's state is a different distribution
from token 2 K — there is no window capping it.

Prediction with a known sign, per CLAUDE.md §6: **GLM saliency should be S-sensitive on KDA
layers, where MiMo's was predicted S-invariant on SWA layers.** Both directions are falsifiable.

### The test, and its cost  `[OPEN]`

Same tokens, several S (2,048 / 8,192 / 16,384), compare per-layer expert rankings
(Spearman ρ, and top-144 keep-set overlap — the quantity that actually decides the mask).

- ρ high and keep-sets agree → 2,048 is vindicated, document it and stop paying attention.
- keep-sets diverge on KDA layers → pass 3 should raise `calib_max_len`, and the published
  pass-1/pass-2 masks are calibrated for short context.

Cost is a partial saliency pass on a few layers, not a full run. It is the cheapest open
question on the board and it gates a claim we are implicitly making to downloaders.

---

## 3. Non-English is absent  `[EST]`

MEASURED 2026-09-26 — decoded 150 samples per bucket with the GLM tokenizer, counted samples
carrying more than 5 CJK codepoints:

| bucket | CJK-bearing |
|---|---|
| agentic | 0.7 % |
| ballast | 0.7 % |
| science | 0.7 % |
| code / finance / math | 0.0 % |
| **overall** | **0.3 %** |

GLM-5.3-Flash is a bilingual Chinese–English model. MoE experts are known to specialise by
language, which is the mechanism that makes this dangerous rather than merely untidy: a 99.7 %
English calibration set gives a language-specialised expert almost no routing mass, and REAP
prunes on routing mass. There is also **no eval bucket for it**, so the paired teacher/student
dNLL/flip gate cannot see the damage either — it would score a Chinese-crippled model as clean.

This is not an argument to rebalance the corpus. The user's calibration choice is deliberate and
defended in [80-calibration.md](80-calibration.md), and adding Chinese would dilute the domains
it exists to protect. It **is** an argument to either measure it or say so on the model card.
Cheapest honest move: a Chinese held-out slice in the *eval*, which costs no calibration budget
and converts an unknown into a number. `[EXT]`

---

## 4. Axes checked and judged covered  `[EXT]`

- **Instruction following / chat format** — carried by `agentic` and `ballast`; every sample is
  rendered through the chat template, so the routing pattern for turn structure is in-distribution.
- **Tool calling / structured output** — inside `agentic` by construction.
- **Reasoning depth / long CoT** — *partly* covered, and it is really the long-context gap wearing
  a different hat. `DIFFICULTY_TARGET` buys 30 % hard samples as a length proxy, and truncation to
  2,048 removes precisely the extended traces. Fixing §2 fixes this. `[OPEN]`
- **Safety / refusal behaviour** — no bucket, and no plan to add one. Refusals are a
  post-training behaviour concentrated in early-layer routing; 50 % pruning could plausibly move
  it either direction. Unmeasured, low stakes for this use, recorded so it is not a surprise.

---

## 5. Standing answer

> Multimodal: **yes, protected and measured.**
> Long context: **collected, then discarded at 2,048 — unmeasured, and GLM's KDA recurrence makes
> it the model most likely to be affected.**
> Non-English: **0.3 % of calibration, no eval bucket — unmeasured.**
> Everything else named: covered, or covered once long context is.

Nothing here says the shipped masks are bad. It says two claims we have not earned are easy to
mistake for claims we have. `[EST]`
