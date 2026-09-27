# 93b — Sequence-length ladder: the verdict

*MEASURED 2026-09-27. Arms: S=2048, S=8192, and a CONTROL that reruns S=2048 on **disjoint
tokens**. 131,072 tokens per arm, 12 layers, keep-set at the shipped 50% (top-144 of 288).*

## Result

| layer | type | CONTROL (same S, other tokens) | S8192 vs S2048 |
|---:|---|---:|---:|
| 3 | deepseek_sparse_attention | 0.8194 | 0.9792 |
| 4 | linear_attention | 0.8264 | 0.9653 |
| 5 | linear_attention | 0.8403 | 0.9792 |
| 6 | linear_attention | 0.8472 | 0.9722 |
| 7 | deepseek_sparse_attention | 0.8264 | 0.9792 |
| 8 | linear_attention | 0.8333 | 0.9583 |
| 9 | linear_attention | 0.8194 | 0.9931 |
| 10 | linear_attention | 0.8681 | 0.9861 |
| 11 | deepseek_sparse_attention | 0.8611 | 0.9583 |

```
CONTROL floor  (same S=2048, disjoint tokens): min 0.8194  mean 0.8380
S8192 vs S2048 (4x sequence length)          : min 0.9583  mean 0.9745
```

**A 4x change in sequence length moves the keep-set FAR LESS than changing which tokens you
sample.** 0.97 agreement across a 4x length change, against 0.84 for two different samples at the
same length. `MAX_LEN=2048` is vindicated, and the pre-registered verdict rule is met with room
to spare. `[EST]`

## The prediction was wrong, and in an informative direction

[92](92-long-context-program.md) §2 predicted, with a sign: *"GLM saliency should be S-SENSITIVE
on KDA layers, where MiMo's was predicted S-invariant"* — because KDA carries recurrent state
over the whole context with no window cap.

**Falsified.** Split by attention type:

| | S8192 vs S2048 | control |
|---|---:|---:|
| KDA (`linear_attention`) | 0.9757 | 0.8391 |
| DSA (`deepseek_sparse_attention`) | 0.9722 | 0.8356 |

KDA and DSA are **indistinguishable**, and both are far above their control. The recurrent state
does evolve over context, but whatever it does to the hidden state does not change *which experts
look salient*. The bounded-state saturation argument holds; the mechanism singled out for concern
was not the one that mattered. `[EST]`

## The more useful finding: token count, not sequence length

The control floor is the headline number nobody asked for. **~16% of the keep-set is resampling
noise at 131,072 tokens** — two honest runs over different tokens of the same corpus disagree on
one expert in six.

That reframes the calibration question asked in [91](91-capability-coverage.md):

> The lever is **how many tokens**, not how long they are. Sequence length is nearly free in time
> ([93](93-seqlen-cost-and-memory.md): 4x length costs +2.4% wall-time) and nearly irrelevant to
> the mask. Token count is neither.

Production calibration is 5.5M tokens, 42x this test, so the real noise floor is much lower —
sampling error falls as 1/sqrt(n), so roughly 6.5x tighter. This test does **not** say the
production mask is 16% noise. It says the ordering of levers is: tokens first, length second,
and length a distant second. `[EXT]`

## Consequences

1. **`calib_max_len` stays at 2048 for pass 3.** Raising it was the cheap lever
   ([93](93-seqlen-cost-and-memory.md)); it is now measured as not worth taking for mask quality.
   It remains the right choice if long-context *evaluation* ever needs in-distribution rows.
2. **The long-context concern in [91](91-capability-coverage.md) §2 is downgraded.** s03
   discarding 73.5% of collected tokens is still real, but it is a token-budget loss, not a
   long-context blindness: the experts that look salient at 2,048 are the same ones that look
   salient at 8,192.
3. **Untested above 8,192.** The corpus caps at 16,384 and the S=16384 arm could not run
   ([95](95-memory-lookahead.md)). Saturation is demonstrated across 2k->8k, not to 1M.
   `[OPEN]`
