# 85 — Corpus sourcing: full optimality pass, all seven buckets

[80-calibration.md](80-calibration.md) sets *size*, *difficulty* and *mixture policy*. This page
is the **source selection** — what actually goes in each bucket and why, after a per-bucket
optimality pass rather than an availability check.

All IDs verified against the HF API on **2026-08-27**. Licences as reported by the API.

> **Honest note on process.** In the Phase 0 sweep only the *finance* bucket received a real
> optimality pass; the other six were availability-checked — plausible candidates confirmed to
> resolve. The operator caught the inconsistency. This page brings all seven to the same
> standard. `[EST]`

---

## The two-corpus split (consequence of the NC-licence decision)

Operator decision 2026-08-27: **non-commercial-licensed data may be used for calibration but
never for healing SFT.**

The line is principled, not merely pragmatic. **REAP calibration does not update a single
weight** — it only decides which experts survive. Healing SFT *does* update weights, and this
checkpoint is intended for publication.

| | **Calibration corpus** | **Healing-SFT corpus** |
|---|---|---|
| Purpose | decide which experts survive | restore capability |
| Touches weights? | **no** | **yes** |
| NC licences (`cc-by-nc*`) | **permitted** | **forbidden** |
| Weighting principle | **fragility** — protect what pruning damages | **capability** — restore what the operator uses |

They share most sources; the SFT corpus is the permissive-licence subset, re-weighted.
**NC-tainted sources are tagged `NC` below and are calibration-only.**

---

## 1. Agentic coding trajectories

Supply is abundant — this bucket is not source-constrained. Value rose when **RLVR was dropped**
([70](70-healing.md)): agentic trajectories in *calibration* are now the primary protection for
long-horizon multi-step routing, and in *SFT* the primary repair.

| Source | Size | Licence | Role |
|---|---|---|---|
| **`togethercomputer/CoderForge-Preview`** | 258,134 traj (155,144 successful), **128K context** | ? verify | **Primary.** Reported best-performing coding-agent trajectory set; the 128K context is rare and directly exercises the hybrid attention |
| `nebius/SWE-rebench` | 10K–100K | CC-BY-4.0 | Decontaminated, continuously refreshed real-world SWE tasks |
| `thoughtworks/agentic-coding-trajectories` | 15K sessions / 618K turns | other | Unified multi-turn corpus; directive's original pick |
| `SWE-bench/SWE-smith-trajectories` | 10K–100K | MIT | Synthetic SWE trajectories |
| `SWE-Gym/SWE-Gym` | 1K–10K | MIT | Executable-environment tasks |
| `Salesforce/xlam-function-calling-60k` | 60K | CC-BY-4.0 | Dense multi-call tool use |
| operator's own hermes-max / Claude Code logs | — | own | **Highest-value per token** — the actual target distribution |

**Sampled, not taken whole:** `AlienKevin/SWE-ZERO-12M-trajectories` (111B tokens) and
`open-thoughts/AgentTrove` (19B tokens). Both dwarf our entire ~200M-token budget; they are
reservoirs to sample the medium-hard band from, never bulk-loaded.

**Caveat.** Agent trajectories are token-inefficient for calibration: tool-call scaffolding,
diffs and stack traces repeat heavily, so a token of trajectory protects fewer distinct experts
than a token of dense reasoning. Filter for **turn diversity** and drop boilerplate-dominated
turns. `[EXT]`

## 2. Code — multi-language, systems, CUDA, IaC, repo-scale

| Source | Size | Licence | Role |
|---|---|---|---|
| **`nvidia/OpenCodeReasoning-2`** | 1M–10M | CC-BY-4.0 | **Primary.** Reasoning-annotated competitive code — medium-hard by construction |
| `nvidia/OpenCodeInstruct` | 1M–10M | CC-BY-4.0 | Broad instruction-style coverage |
| **`GPUMODE/KernelBook`** | 10K–100K | other | **CUDA/Triton kernels** — the directive names kernels explicitly, and this is directly adjacent to the downstream kernel work |
| **`SakanaAI/AI-CUDA-Engineer-Archive`** | 10K–100K | CC-BY-4.0 | 30K+ CUDA kernels with verification outcomes |
| `bigcode/the-stack-v2-dedup` | 1B–10B files | other (agreement) | **Repo-scale long-context** — sample whole repos, do not bulk-load |
| `bigcode/commitpackft` | — | MIT | Commit-level edits; real diffs |

**IaC (Terraform/K8s/Ansible)** has no strong dedicated instruction dataset. Cover by filtering
`the-stack-v2-dedup` on `.tf`/`.yaml`/`.hcl` paths. `[OPEN]`, minor.

## 3. Multimodal

**The load-bearing bucket** (risk R3: vision-only experts score `S_j = 0` under text-only
calibration and are deleted with *certainty*).

| Source | Size | Licence | Role |
|---|---|---|---|
| **`nvidia/Nemotron-VLM-Dataset-v2`** | 1M–10M | CC-BY-4.0 | **Primary.** Large, permissive, modern VLM blend — usable in **both** corpora |
| `HuggingFaceM4/the_cauldron` | 1M–10M | ? verify | 50-dataset aggregate (ChartQA, DocVQA, AI2D…) — broadest single coverage |
| `HuggingFaceM4/Docmatix` | 1M–10M | MIT | Document understanding at scale |
| `lmms-lab/LLaVA-OneVision-Data` | 1M–10M | Apache-2.0 | General VL instruction |
| `allenai/pixmo-docs` / `pixmo-cap` | 100K–1M | ODC-BY | Charts, figures, dense captions |
| `ServiceNow/BigDocs-Bench` | 100K–1M | CC-BY-4.0 | Document layout |
| **`xlangai/aguvis-stage2`** | — | Apache-2.0 | **GUI/screenshot agent grounding** — the directive names screenshots and UI state |
| `TIGER-Lab/VisualWebInstruct` | 1M–10M | Apache-2.0 | Web/visual reasoning |
| `osunlp/UGround-V1-Data` `NC` | 1M–10M | CC-BY-NC-SA-4.0 | GUI grounding — **calibration only** |
| `sujet-ai/Sujet-Finance-QA-Vision-100k`, `TheFinAI/FinMR` | 100K | Apache-2.0 / verify | Financial charts — **double duty** with the finance bucket |

## 4. Math + algorithm/architecture synthesis

| Source | Size | Licence | Role |
|---|---|---|---|
| **`nvidia/OpenMathReasoning`** | 1M–10M | CC-BY-4.0 | **Primary.** AIMO-winning corpus, reasoning traces |
| `zwhe99/DeepMath-103K` | 100K–1M | MIT | Difficulty-graded, decontaminated |
| `open-r1/OpenR1-Math-220k` | 100K–1M | Apache-2.0 | Verified R1 traces |
| `AI-MO/NuminaMath-1.5` | 100K–1M | Apache-2.0 | Competition maths breadth |
| `internlm/Lean-Workbook` | 10K–100K | Apache-2.0 | **Formal proofs** — a genuinely distinct routing profile from natural-language maths |
| `nvidia/Nemotron-PrismMath` | 1M–10M | CC-BY-4.0 | Directive's original pick |

**This bucket is over-supplied relative to its risk.** The Kimi-Linear-REAP-30 reference shows
maths is among the *most robust* capabilities under expert pruning (AIME25 **+10.0**,
MATH-500 −1.0, GSM8k −1.5). See the re-weighting argument below.

## 5. Hard science & engineering, incl. bio/biotech + biomedical literature

Operator scope decision 2026-08-27: **bio/biotech + biomedical literature, no clinical QA.**
Diagnosis and patient-case QA stay excluded per the directive; molecular/genomic/protein
science and the primary literature are in.

| Source | Size | Licence | Role |
|---|---|---|---|
| **`open-thoughts/OpenThoughts3-1.2M`** | 1M–10M | Apache-2.0 | **Primary.** Large permissive science+maths reasoning — usable in **both** corpora |
| `nvidia/sft_datablend_v1` | 100K–1M | CC-BY-4.0 | Directive's original pick |
| `EricLu/SCP-116K` `NC` | 274K | CC-BY-NC-SA-4.0 | Physics/chem/bio/maths problem–solution — **calibration only** |
| `camel-ai/physics`, `/chemistry`, `/biology` `NC` | ~20K each | CC-BY-NC-4.0 | Domain-specific — **calibration only** |
| `jablonkagroup/ChemBench` | 1K–10K | MIT | Graduate chemistry reasoning |
| `TIGER-Lab/MMLU-Pro` | 10K–100K | MIT | Broad hard STEM knowledge; **knowledge-dense, FRAMES-shaped** |
| **`tattabio/OG`** | 1M–10M | CC-BY-SA-4.0 | **Open Genome** — genomic sequence/annotation, permissive |
| `InstaDeepAI/multi_species_genomes` | — | ? verify | Multi-species genomic sequence |
| **`ncbi/pubmed`** | 10M–100M | other (NCBI terms) | **Biomedical literature** — abstracts, sampled. *Literature, not clinical QA* |
| arXiv `physics`/`cond-mat`/`q-bio`/`econ.EM` slices | — | mixed | Long-form technical prose; also carries the econometrics residual |

**Explicitly excluded per directive + operator:** `qiaojin/PubMedQA`, `bigbio/pubmed_qa`,
MedQA-style clinical reasoning. Both resolve on HF and were rejected on scope, not availability.

## 6. Finance / quant / business

Resolved in [80-calibration.md](80-calibration.md#financequantbusinesseconometrics--bucket-resolved-r9-closed).
Led by `kensho/DocFinQA` (~123k-word contexts) — the only source in the whole corpus that
exercises genuine long context.

## 7. General coherence ballast

| Source | Size | Licence | Role |
|---|---|---|---|
| `HuggingFaceFW/fineweb-edu` | 1B–10B | ODC-BY | Directive's pick; educational web prose |
| `allenai/tulu-3-sft-mixture` | 100K–1M | ODC-BY | Broad instruction-following |
| `HuggingFaceTB/smoltalk2` | 1M–10M | ? verify | Modern general chat/instruction |
| `HuggingFaceTB/finemath` | 10M–100M | ODC-BY | Maths-flavoured web ballast |

**Do not cut.** The domain-invariant "standing committee" of experts carries the majority of
routing mass across *all* domains; starving it damages connective tissue every target domain
routes through ([80](80-calibration.md)).

---

## Unavailable / rejected on access

| Source | Status |
|---|---|
| `nvidia/Nemotron-CC-Math` | gated (401) — `Nemotron-PrismMath` substitutes |
| `TheFinAI/MultiFinBen` | gated (401) — would have added multilingual finance |
| `mlfoundations/MINT-1T` | gated (401) — interleaved multimodal |
| `nvidia/Nemotron-Agentic-Trajectories-v1` | 401 — CoderForge substitutes |
| `GleghornLab/ProteinLMBench` | 401 — `tattabio/OG` substitutes for protein/genomic |
| `liuganghuggingface/MolTextNet` | 307 redirect — resolve canonical ID if wanted |
| `zhihz0535/X-Reasoner-SFT` | 401 |
