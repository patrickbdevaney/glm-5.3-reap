# Speculative decoding: DFlash 2 and MTP

`[MEAS 2026-08-27 22:40]` unless noted.

## DFlash 2 is the better prize, and it is architecturally drop-in

`incoai/GLM-5.3-Flash-DFlash2` — 2.18 GiB, BF16, 4 files. A block-diffusion drafter: predicts a
whole block in one pass, keeps top candidates per position, and a lightweight selector traces one
coherent path. Decoding is lossless (greedy matches the target exactly).

Vendor numbers `[VEN]` (4× GB300, TP4, SGLang, block size 8 = 7 draft tokens/step):

| task | MTP accept len | DFlash 2 accept len | MTP tput | DFlash 2 tput |
|---|---|---|---|---|
| GSM8K | 5.06 | **5.78** | 1.93× | **2.42×** |
| MATH-500 | 4.95 | **5.86** | 2.05× | **2.79×** |
| HumanEval | 4.70 | **5.32** | 1.94× | **2.62×** |
| MBPP | 4.26 | **4.85** | 1.78× | **2.39×** |

**It beats the model's native MTP on every task.** Confirms the instinct to treat DFlash as the
prize and MTP as insurance.

### It stays compatible with our REAP

| draft head expects | source | REAP-50 | ok |
|---|---|---|---|
| `hidden_size` 4096 | 4096 | 4096 | yes |
| `num_target_layers` 45 | 45 | 45 | yes |
| `vocab_size` 154880 | 154880 | 154880 | yes |

REAP changes **expert count per layer**, not layer count, hidden size, or vocab. So the drafter is
architecturally drop-in against the pruned target; only its learned weights need adaptation.

### The line that matters: `target_layer_ids: [5, 14, 24, 33, 42]`

The drafter reads hidden states from **five specific target layers**. All five are MoE layers
(3–44), so **all five are pruned by REAP** and all five shift. That is exactly why acceptance
would degrade and why fine-tuning against the pruned target is the right move.

Config also carries: `block_size 8`, `conv_kernel_size 2` (two-tap dynamic convolutions keep the
draft from decaying late in the block), `selector_rank 256`, `selector_top_k 16`,
`sliding_window 2048`, 5 layers, `qwen3` backbone, `is_causal: false`.

**Pass-2 consequence (cheap, high value):** add per-layer hidden-state drift at layers
5/14/24/33/42 to the P13 evaluation. It directly predicts how much draft re-training the REAP
costs. We cannot *protect* those layers — `num_local_experts` is one scalar, so non-uniform
allocation is unloadable — but measuring is free and tells us what we are buying.

### Why fine-tuning it is feasible here when almost nothing else is

The drafter is **2.18 GiB** and the target stays **frozen**. No backward pass through a 165B
student — the thing that rules out router-KD and distillation healing. The target only has to run
*forward* to emit hidden states at five taps, which is precisely what the layer-streaming harness
already does. A REAP-50 NVFP4 target (~96 GiB, and smaller once the 1.45 GiB dequantisation waste
is fixed) is also the first version of this model that plausibly fits Thor's envelope at all.

Storage is the real constraint, not compute: 5 taps × 4096 × bf16 = **40 KB/token**, so 100M
training tokens would be 4 TB if materialised. Training must consume hidden states **online** —
generate a batch, train, discard — never write a corpus of them.

## LICENSE — flag before any upload `[MEAS]`

DFlash 2 is **`cc-by-nc-nd-4.0`**: NonCommercial **and NoDerivatives**.

A fine-tuned draft head is a derivative work. Under ND we may fine-tune and use it **privately**,
but **publishing** the adapted drafter is not permitted by that licence. This matters because the
project deliberately chose permissive-only calibration data to keep the output unencumbered —
shipping an NC-ND-derived drafter alongside it would reintroduce exactly the encumbrance that
choice avoided.

Options, in order of preference:
1. Keep the fine-tuned drafter **local/private**; publish only the REAP target. Costs nothing.
2. Train a drafter **from scratch** on the DFlash *architecture* (`github.com/z-lab/dflash` —
   check its code licence, which is separate from the weights') against our own permissive data.
   No NC-ND weights touched, so the result is ours to publish.
3. Ask inco.ai for a licence exception.

The target model's own licence still governs the REAP artifact independently.

## MTP: preserved as insurance (R14)

See `wiki/90-open-risks.md`. Pass 1 dropped layer 45; pass 2 preserves it, pruned by weight norm.
Even though DFlash 2 beats it, keeping it costs ~3.4 GiB (+2%) and preserves a fallback that needs
no third-party licence at all.
