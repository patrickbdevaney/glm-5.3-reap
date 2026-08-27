"""Calibration corpus specification.

Mixture and licence policy are operator-approved (2026-08-27) and reasoned about in
wiki/80-calibration.md. Summary of the two decisions encoded here:

  * Mixture is weighted for "a strong coder/agent that stays empirically knowledgeable",
    so code-adjacent (agentic + code + math) is 60%, world knowledge sits at a sufficiency
    floor of 18%, and multimodal holds at 15% because R3 is the one risk where failure is
    certain rather than probable.
  * Permissive licences only. GLM-5.3-Flash is MIT and a derivative can be MIT; that property
    is irreversible if lost. Every NC source had a permissive replacement, the main one 4x
    larger. Sources excluded on licence/scope are listed at the bottom so the exclusion stays
    visibly deliberate.

`text_fn` maps a raw row to a training string. Returning None drops the row.
"""
from __future__ import annotations

import json

TOTAL_SAMPLES = 12_288
MAX_TOKENS = 16_384
HELDOUT_FRACTION = 0.08          # stratified, per bucket, for the section-8 proxies

MIXTURE = {                       # bucket -> share
    "agentic":    0.24,
    "code":       0.21,
    "math":       0.15,
    "multimodal": 0.15,
    "science":    0.10,
    "finance":    0.08,
    "ballast":    0.07,
}
assert abs(sum(MIXTURE.values()) - 1.0) < 1e-9

# Difficulty policy: medium-hard, NOT maximal. Hard-only calibration degrades general
# perplexity 6.2-12.1% vs 1.5-4.2% mixed (arXiv 2510.10618), which would damage exactly the
# connective tissue the ballast slice exists to protect.
DIFFICULTY_TARGET = {"medium": 0.60, "hard": 0.30, "easy": 0.10}
# Token-length bands used as the difficulty proxy (cheap, and correlates with reasoning depth).
BAND_EASY = (0, 700)
BAND_MEDIUM = (700, 4_000)
BAND_HARD = (4_000, MAX_TOKENS)


def _first(row, *keys):
    for k in keys:
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return None


def _messages(row, key="messages"):
    """Handle {role,content} and ShareGPT {from,value}, and columns that hold a JSON *string*.

    CoderForge stores its whole trajectory as a JSON string in `messages`; treating that as a
    list silently yields nothing, which is how 30,000 rows produced zero samples.
    """
    msgs = row.get(key)
    if isinstance(msgs, str):
        try:
            msgs = json.loads(msgs)
        except Exception:
            return None
    if not isinstance(msgs, list):
        return None
    parts = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        c = m.get("content", m.get("value"))
        if isinstance(c, list):
            c = " ".join(str(x.get("text", "")) for x in c if isinstance(x, dict))
        if c:
            parts.append(f"{m.get('role', m.get('from', 'user'))}: {c}")
    return "\n\n".join(parts) or None


def _any_messages(row):
    for k in ("messages", "conversations", "conversation", "turns"):
        v = _messages(row, k)
        if v:
            return v
    return None


def _qa(row, q_keys, a_keys):
    q = _first(row, *q_keys)
    a = _first(row, *a_keys)
    if not q:
        return None
    return f"{q}\n\n{a}" if a else q


# Each entry: (hf_id, config, split, weight_within_bucket, text_fn)
SOURCES: dict[str, list[tuple]] = {
    "agentic": [
        ("togethercomputer/CoderForge-Preview", "trajectories", "filtered_reward1", 0.30, lambda r: _any_messages(r) or _first(r, "text", "trajectory")),
        ("nebius/SWE-rebench", None, "test", 0.20, lambda r: _qa(r, ["problem_statement", "text"], ["patch", "solution"])),
        ("SWE-bench/SWE-smith-trajectories", None, "tool", 0.20, lambda r: _any_messages(r) or _first(r, "text")),
        ("SWE-Gym/SWE-Gym", None, "train", 0.10, lambda r: _qa(r, ["problem_statement"], ["patch"])),
        ("arcee-ai/agent-data", None, "train", 0.10, lambda r: _any_messages(r) or _qa(r, ["query", "instruction"], ["answers", "output"])),
        ("open-thoughts/AgentTrove", None, "train", 0.10, lambda r: _any_messages(r) or _first(r, "text")),
    ],
    "code": [
        ("nvidia/OpenCodeReasoning-2", "train", "python", 0.30, lambda r: _qa(r, ["input", "question", "problem"], ["output", "solution", "r1_generation"])),
        ("nvidia/OpenCodeReasoning-2", "train", "cpp", 0.15, lambda r: _qa(r, ["input", "question", "problem"], ["output", "solution", "r1_generation"])),
        ("nvidia/OpenCodeInstruct", None, "train", 0.25, lambda r: _qa(r, ["input", "instruction"], ["output", "response"])),
        ("GPUMODE/KernelBook", None, "train", 0.15, lambda r: _first(r, "python_code", "triton_code", "code", "text")),
        ("SakanaAI/AI-CUDA-Engineer-Archive", None, "level_1", 0.10, lambda r: _first(r, "CUDA_Code", "Kernel_Code", "cuda_code")),
        
    ],
    "math": [
        ("nvidia/OpenMathReasoning", "default", "cot", 0.35, lambda r: _qa(r, ["problem", "question"], ["generated_solution", "solution", "answer"])),
        ("zwhe99/DeepMath-103K", None, "train", 0.25, lambda r: _qa(r, ["question", "problem"], ["r1_solution_1", "final_answer", "solution"])),
        ("open-r1/OpenR1-Math-220k", None, "train", 0.20, lambda r: _qa(r, ["problem", "question"], ["solution", "answer"])),
        ("AI-MO/NuminaMath-1.5", None, "train", 0.10, lambda r: _qa(r, ["problem"], ["solution"])),
        ("internlm/Lean-Workbook", None, "train", 0.10, lambda r: _qa(r, ["natural_language_statement", "problem"], ["formal_statement", "answer"])),
    ],
    "science": [
        ("open-thoughts/OpenThoughts3-1.2M", None, "train", 0.45, lambda r: _any_messages(r) or _qa(r, ["problem", "question"], ["solution", "answer"])),
        ("nvidia/sft_datablend_v1", None, "train", 0.25, lambda r: _any_messages(r)),
        ("TIGER-Lab/MMLU-Pro", None, "test", 0.15, lambda r: _qa(r, ["question"], ["cot_content", "answer"])),
        ("jablonkagroup/ChemBench", "organic_chemistry", "train", 0.15, lambda r: _qa(r, ["question", "input"], ["answer", "target"])),
        
    ],
    "finance": [
        ("kensho/DocFinQA", "default", "train", 0.15, lambda r: _qa(r, ["context", "question"], ["answer", "program"])),
        ("ChanceFocus/flare-finqa", None, "train", 0.20, lambda r: _qa(r, ["query", "question", "text"], ["answer", "choices"])),
        ("G4KMU/t2-ragbench", "finqa", "train", 0.15, lambda r: _qa(r, ["question", "context"], ["answer"])),
        ("sujet-ai/Sujet-Financial-RAG-EN-Dataset", None, "train", 0.10, lambda r: _qa(r, ["question", "context"], ["answer"])),
        ("kensho/bizbench", "program_synthesis", "train", 0.20, lambda r: _qa(r, ["question", "context"], ["answer", "program"])),
        ("next-tat/tat-llm-instructions", None, "train", 0.20, lambda r: _qa(r, ["instruction", "input"], ["output", "response"])),
        ("TheFinAI/Fino1_Reasoning_Path_FinQA", None, "train", 0.15, lambda r: _qa(r, ["Open-ended Verifiable Question", "question"], ["Complex_CoT", "Response", "answer"])),
    ],
    "ballast": [
        ("HuggingFaceFW/fineweb-edu", "sample-10BT", "train", 0.50, lambda r: _first(r, "text")),
        ("allenai/tulu-3-sft-mixture", None, "train", 0.30, lambda r: _messages(r)),
        ("HuggingFaceTB/finemath", "finemath-3plus", "train", 0.20, lambda r: _first(r, "text")),
    ],
}

# Multimodal is handled by a separate loader: rows must carry real images, which are pushed
# through the real processor. Text descriptions of images route like text and protect nothing.
MM_SOURCES = [
    # Configs are EXPLICIT here. Nemotron-VLM-Dataset-v2 has 46 configs of which the first
    # alphabetically is `wiki_de` - German Wikipedia text. Letting the loader auto-correct a
    # missing config would silently calibrate the R3-critical bucket on text.
    ("nvidia/Nemotron-VLM-Dataset-v2", "chartqa_cot", "train", 0.10),
    ("nvidia/Nemotron-VLM-Dataset-v2", "docvqa_cot", "train", 0.10),
    ("nvidia/Nemotron-VLM-Dataset-v2", "llava_cot_100k", "train", 0.10),
    ("nvidia/Nemotron-VLM-Dataset-v2", "infographicsvqa_cot", "train", 0.06),
    ("nvidia/Nemotron-VLM-Dataset-v2", "plotqa_cot", "train", 0.06),
    ("nvidia/Nemotron-VLM-Dataset-v2", "fintabnet_cot", "train", 0.05),
    ("nvidia/Nemotron-VLM-Dataset-v2", "visual_web_instruct_cot", "train", 0.05),
    ("HuggingFaceM4/the_cauldron", "chartqa", "train", 0.08),
    ("HuggingFaceM4/the_cauldron", "ai2d", "train", 0.06),
    ("HuggingFaceM4/the_cauldron", "docvqa", "train", 0.06),
    ("HuggingFaceM4/Docmatix", "images", "train", 0.08),
    ("lmms-lab/LLaVA-OneVision-Data", "CLEVR-Math(MathV360K)", "train", 0.05),
    ("allenai/pixmo-docs", "charts", "train", 0.05),
    ("xlangai/aguvis-stage2", "guiact-web-single", "train", 0.05),
    ("ServiceNow/BigDocs-Bench", None, "train", 0.05),
]

EXCLUDED = {
    "licence_nc": ["EricLu/SCP-116K", "camel-ai/physics", "camel-ai/chemistry",
                   "camel-ai/biology", "osunlp/UGround-V1-Data"],
    "licence_sharealike": ["tattabio/OG"],
    "scope_clinical": ["qiaojin/PubMedQA", "bigbio/pubmed_qa"],
    "gated_401": ["nvidia/Nemotron-CC-Math", "TheFinAI/MultiFinBen", "mlfoundations/MINT-1T"],
}
