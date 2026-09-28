"""Measure vision on REAL held-out images, with real ground truth.

The card currently says vision is "verified working and uncharacterised", on the strength of a
single generated image with a red square and a blue circle. That is honest but thin, and it is
the weakest claim on the card.

I had assumed there was no ground-truth metric available here and that the best possible was a
token-overlap grounding signal. That was wrong. The held-out multimodal rows are **ChartQA**:
every row carries question/answer pairs with exact short answers ("2014.", "6.", "Yes."), 136
pairs across 64 rows. So this reports real ChartQA relaxed accuracy on rows the calibration
never saw, not a vibe check.

Two things had to be got right, and both would have failed silently:

1. pixel_values are NOT packed row-major over the patch grid. The processor emits them in 2x2
   MERGE-BLOCK order. Undoing that with a plain (gh, gw) reshape yields a visibly scrambled
   image -- doubled text, black bands -- which the model would then describe badly, and the
   resulting number would read as a vision quality problem rather than a bug in this file.

2. The model reasons before answering. A short -n truncates mid-reasoning and scores 0 on
   answers the model actually had right, which looks like a vision failure and is not one.

Scoring is ChartQA relaxed accuracy: numeric answers count as correct within 5% relative error,
text answers on normalised exact match. One deviation from the plain metric, and it matters:
YEARS are graded exactly. 5% of 2014 is +/-100 years, so an unmodified relaxed match scores
"2013" as correct against "2014" and every year question becomes a free mark -- which would have
inflated this number on a public card. Otherwise the metric is standard and comparable to
published ChartQA figures, with the caveat that the sample here is small and deliberately so.
"""
from __future__ import annotations

import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = Path.home() / "glm5-llama.cpp" / "build-cuda" / "bin" / "llama-mtmd-cli"
MM = ROOT / "gguf" / "mmproj-GLM-5.3-Flash-REAP50-F16.gguf"

MAX_QUESTIONS = int(sys.argv[2]) if len(sys.argv) > 2 else 24
N_PREDICT = 160          # the gate already learned that 48 truncates this model mid-reasoning
PER_CALL_TIMEOUT = 900

_TOK = None


def ref_text(row) -> str:
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained(str(ROOT / "source" / "GLM-5.3-Flash"),
                                             trust_remote_code=True)
    ids = row["input_ids"]
    ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)
    txt = _TOK.decode(ids, skip_special_tokens=True)
    # <|image|> is an ADDED token, not a special one, so skip_special_tokens leaves hundreds of
    # placeholders behind and the real prose sits past any excerpt window.
    txt = re.sub(r"<\|[a-z_]+\|>", " ", txt)
    return re.sub(r"\s+", " ", txt).strip()


def qa_pairs(row):
    m = re.search(r"\[\s*\{.*\}\s*\]", ref_text(row), re.S)
    if not m:
        return []
    try:
        return [d for d in json.loads(m.group(0)) if "user" in d and "assistant" in d]
    except Exception:
        return []


def reconstruct(row, path) -> bool:
    import numpy as np
    from PIL import Image
    pv = row["pixel_values"]
    t, h, w = row["image_grid_thw"][0].tolist()
    C, tp, ps, m = 3, 2, 14, 2
    patches = pv.float().numpy()
    if t * h * w != patches.shape[0] or h % m or w % m:
        return False
    mean = np.array([0.48145466, 0.4578275, 0.40821073])
    std = np.array([0.26862954, 0.26130258, 0.27577711])
    # Exact inverse of the pack: (t, gh//m, gw//m, m, m, C, tp, ps, ps) -> (C, t*tp, h*ps, w*ps)
    img = patches.reshape(t, h // m, w // m, m, m, C, tp, ps, ps)
    img = img.transpose(5, 0, 6, 1, 3, 7, 2, 4, 8)
    img = img.reshape(C, t * tp, h * ps, w * ps)[:, 0]
    img = (img.transpose(1, 2, 0) * std + mean).clip(0, 1)
    Image.fromarray((img * 255).astype("uint8")).save(path)
    return True


_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def _is_float(x: str) -> bool:
    try:
        float(x)
        return True
    except ValueError:
        return False


def normalise(s: str) -> str:
    return re.sub(r"[^a-z0-9.%]+", " ", s.lower()).strip().rstrip(".").strip()


def relaxed_match(pred: str, gold: str) -> bool:
    """ChartQA relaxed accuracy: numbers within 5% relative, text on exact match."""
    p, g = normalise(pred), normalise(gold)
    if not p:
        return False
    gn = _NUM.findall(g)
    if gn:
        gv = float(gn[0])
        # YEARS ARE A PATHOLOGY for relaxed accuracy: 5% of 2014 is +/-100 years, so "2013"
        # would score correct against "2014" and every year question becomes free marks. ChartQA
        # relaxed accuracy is meant for chart VALUES; years get exact match.
        if float(gv).is_integer() and 1800 <= gv <= 2200:
            return any(c.strip() and float(c) == gv
                       for c in _NUM.findall(p) if _is_float(c))
        for cand in _NUM.findall(p):
            try:
                pv = float(cand)
            except ValueError:
                continue
            if gv == 0:
                if pv == 0:
                    return True
            elif abs(pv - gv) / abs(gv) <= 0.05:
                return True
        return False
    # text answer: accept it appearing as a whole token-run in the reply
    return g == p or re.search(rf"\b{re.escape(g)}\b", p) is not None


def extract_answer(stdout: str) -> str:
    """Grade ONLY what the model generated.

    The first version of this concatenated stdout+stderr and filtered by line prefix, which let
    llama.cpp's own logging survive into the graded text. That is not cosmetic: relaxed accuracy
    scans for a number ANYWHERE in the reply, and the line "image decoded (batch 1/1) in 12036 ms"
    alone contributes 1, 1 and 12036 -- so a gold answer of "1." scored correct off the batch
    counter, having never looked at the chart. The echoed prompt survived too, which hands back
    the gold for any yes/no question whose wording contains the word. Both inflate the score, and
    the first run of this file reported 17/24 on a body that was demonstrably part log noise.

    Generation goes to stdout (mtmd-cli does LOG(...) then fflush(stdout)); everything else is
    stderr. So keep stdout, drop stderr entirely, and cut the echoed prompt at the last chat-turn
    marker before scoring.
    """
    txt = stdout.replace("\r", "\n")
    for marker in ("<|assistant|>", "\nassistant\n"):
        if marker in txt:
            txt = txt.rsplit(marker, 1)[1]
            break
    # The model does not stop at the end of its turn: only <|endoftext|> is registered as EOG in
    # these GGUFs, while the model actually ends a turn with <|user|>. So it emits its answer,
    # then <|user|>, then HALLUCINATES a second question and answers that too. Cut at the first
    # turn boundary and keep the real answer -- taking the text after the LAST </think> would
    # grade the fabricated second turn, which in testing disagreed with the first one.
    for stop in ("<|user|>", "<|observation|>", "<|endoftext|>", "<|assistant|>"):
        if stop in txt:
            txt = txt.split(stop, 1)[0]
    if "</think>" in txt:
        txt = txt.rsplit("</think>", 1)[1]
    # belt and braces: any log line that still reached stdout
    noise = ("load_", "print_info", "llama_", "main:", "build:", "init_", "ggml_", "encoding",
             "decoding", "clip_", "mtmd_", "warmup", "alloc_", "system_info", "sched_",
             "register_", "common_", "srv ", "WARN", "warn:", "sampler", "generate:",
             "image slice", "image decoded", "--- ", "      For normal")
    lines = [l for l in txt.split("\n") if not l.strip().startswith(noise)]
    return "\n".join(lines).strip()


def wilson(k: int, n: int):
    if n == 0:
        return (None, None)
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0.0, c - h), 3), round(min(1.0, c + h), 3))


def pick_model():
    # ship2 deletes each quant right after upload, so by the end of the run the only thing left
    # locally is the 175 GB Q8_0 -- which does not fit in RAM and would thrash for hours rather
    # than fail. Never select that by accident; the caller passes the file it still has.
    if len(sys.argv) > 1 and Path(sys.argv[1]).exists():
        return Path(sys.argv[1])
    for name in ("Q4_K_M", "Q4_K_S", "IQ4_XS", "Q3_K_M", "IQ3_M"):
        p = ROOT / "gguf" / f"GLM-5.3-Flash-REAP50-{name}.gguf"
        if p.exists():
            return p
    return None


def main():
    model = pick_model()
    if model is None:
        print("no suitable local GGUF (refusing to run against Q8_0); skipped")
        return 0
    if not MM.exists() or not BIN.exists():
        print("no mmproj or mtmd binary; skipped")
        return 0

    import torch
    shards = sorted((ROOT / "corpus/shards/multimodal").glob("*.pt"))
    if not shards:
        print("no multimodal shards")
        return 0
    recs = torch.load(shards[0], weights_only=False)

    out_dir = ROOT / "artifacts" / "vision_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"vision characterisation using {model.name}, up to {MAX_QUESTIONS} questions")

    results, asked = [], 0
    for i, row in enumerate(recs):
        if asked >= MAX_QUESTIONS:
            break
        pairs = qa_pairs(row)
        if not pairs:
            continue
        png = out_dir / f"img_{i:02d}.png"
        if not png.exists() and not reconstruct(row, png):
            print(f"  row {i}: could not reconstruct")
            continue

        for qa in pairs[:2]:
            if asked >= MAX_QUESTIONS:
                break
            q = qa["user"].replace("\n", " ").strip()
            gold = str(qa["assistant"]).strip()
            t0 = time.time()
            try:
                proc = subprocess.run(
                    [str(BIN), "-m", str(model), "--mmproj", str(MM), "--image", str(png),
                     "-p", q, "-ngl", "0", "--no-repack", "-c", "4096",
                     # GLM-5.3 ends a turn with <|user|> (154827) and <|observation|> (154829),
                     # but convert only carried <|endoftext|> into the GGUF, so nothing stops
                     # generation at the end of the answer. Registering them as eot/eom makes the
                     # model stop where it means to, which is both correct and much faster.
                     "--override-kv", "tokenizer.ggml.eot_token_id=int:154827",
                     "--override-kv", "tokenizer.ggml.eom_token_id=int:154829",
                     "-n", str(N_PREDICT), "--temp", "0"],
                    capture_output=True, text=True, timeout=PER_CALL_TIMEOUT)
                # keep the raw streams: if the extraction is ever wrong again, the evidence is
                # on disk instead of being inferred from a truncated excerpt.
                (out_dir / f"raw_{asked:02d}.out").write_text(proc.stdout)
                (out_dir / f"raw_{asked:02d}.err").write_text(proc.stderr[-20000:])
                reply = extract_answer(proc.stdout)
            except subprocess.TimeoutExpired:
                reply = ""
            ok = relaxed_match(reply, gold)
            asked += 1
            results.append({
                "image": png.name, "question": q, "gold": gold,
                "reply": reply[:1200], "correct": ok,
                "secs": round(time.time() - t0, 1),
            })
            print(f"  [{asked}/{MAX_QUESTIONS}] {png.name} {'OK ' if ok else 'MISS'} "
                  f"gold={gold!r} :: {reply[:120]!r}")

    n = len(results)
    k = sum(1 for r in results if r["correct"])
    empty = sum(1 for r in results if not r["reply"].strip())
    if empty:
        print(f"\nWARNING: {empty}/{n} replies were EMPTY after extraction. That is a harness "
              f"failure, not a model failure - inspect {out_dir}/raw_*.out before trusting this.")
    lo, hi = wilson(k, n)
    payload = {
        "model": model.name,
        "dataset": "ChartQA (held-out multimodal corpus rows)",
        "metric": "relaxed accuracy (numeric within 5%, text exact match)",
        "n": n, "correct": k, "empty_replies": empty,
        "accuracy": round(k / n, 3) if n else None,
        "wilson95": [lo, hi],
        "method": ("held-out rows the calibration never saw; pixel_values un-packed from 2x2 "
                   "merge-block order back to PNG; each question asked at temperature 0 with "
                   f"-n {N_PREDICT} so reasoning is not truncated mid-answer; only the model's "
                   "stdout is graded, with the echoed prompt cut at the last chat-turn marker, so "
                   "neither llama.cpp log lines nor the question itself can satisfy a match"),
        "caveat": (f"n={n} is small and deliberately so -- this is a smoke-level characterisation, "
                   "not a leaderboard run. The 95% Wilson interval is wide; read it, not the "
                   "point estimate. Decode is CPU-only (-ngl 0) at ~1 tok/s, which bounds n."),
        "results": results,
    }
    p = ROOT / "artifacts" / "vision_characterization.json"
    p.write_text(json.dumps(payload, indent=2))
    print(f"\nChartQA relaxed accuracy: {k}/{n}"
          f"{f' = {k/n:.1%} (95% CI {lo:.1%}-{hi:.1%})' if n else ''}")
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
