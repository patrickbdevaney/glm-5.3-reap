"""Needle-in-a-haystack for the GGUF, so context guidance on the card is measured.

WHY THIS MATTERS HERE SPECIFICALLY. llama.cpp does not implement DSA - the indexer tensors load
and are never used, so the 11 sparse-attention layers run DENSE. Dense is a superset of what DSA
selects (the model sees its top-2048 plus extra low-relevance tokens, which softmax down-weights),
so quality should degrade gracefully rather than break. But "should" is not a measurement, and the
model advertises a 1M window it cannot possibly serve through an O(n^2) attention path.

So: find the length where retrieval actually falls over, and put THAT number on the card instead
of the config's 1048576.

Deliberately small - three depths at each length, a handful of minutes per length. This is a
context-limit probe, not a long-context benchmark.
"""
from __future__ import annotations

import argparse, json, random, re, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = Path.home() / "glm5-llama.cpp" / "build-cuda" / "bin" / "llama-completion"

FILLER = (
    "The city council met on Tuesday to discuss the new drainage plan. Rainfall in the "
    "northern district has exceeded the seasonal average for three consecutive years. "
    "Engineers presented four options, each with different costs and timelines. "
    "The committee agreed to review the proposals again next month. "
)


def build_haystack(n_tokens: int, depth: float, code: str) -> str:
    # ~4 chars/token is close enough for a length probe; the exact count does not matter,
    # only that it brackets the regime we care about.
    n_chars = n_tokens * 4
    reps = max(1, n_chars // len(FILLER))
    body = FILLER * reps
    needle = f"\n\nThe secret passcode is {code}. Remember it.\n\n"
    cut = int(len(body) * depth)
    return body[:cut] + needle + body[cut:]


def run(model: Path, prompt: str, ctx: int, ngl: int, timeout: int) -> tuple[str, float]:
    t0 = time.time()
    p = subprocess.run(
        [str(BIN), "-m", str(model), "-c", str(ctx), "-n", "24", "--temp", "0",
         "-ngl", str(ngl), "--no-warmup", "--no-repack", "-p", prompt],
        capture_output=True, text=True, timeout=timeout)
    return (p.stdout + p.stderr).replace("\r", "\n"), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--lengths", default="2000,8000,32000")
    ap.add_argument("--depths", default="0.1,0.5,0.9")
    ap.add_argument("--ngl", type=int, default=0)   # unified memory: offload is non-evictable and OOMs
    ap.add_argument("--timeout", type=int, default=3600)
    a = ap.parse_args()

    rng = random.Random(0)
    results = []
    for L in [int(x) for x in a.lengths.split(",")]:
        for d in [float(x) for x in a.depths.split(",")]:
            code = "".join(rng.choice("0123456789") for _ in range(6))
            hay = build_haystack(L, d, code)
            prompt = (hay + "\n\nQuestion: What is the secret passcode mentioned in the text "
                             "above?\nAnswer: The secret passcode is")
            ctx = max(4096, int(L * 1.4))
            try:
                out, secs = run(Path(a.model), prompt, ctx, a.ngl, a.timeout)
                found = code in out.split("Answer: The secret passcode is")[-1][:200]
            except subprocess.TimeoutExpired:
                out, secs, found = "", a.timeout, False
                print(f"  L={L:>6} depth={d:.1f}  TIMEOUT after {a.timeout}s")
            results.append({"tokens": L, "depth": d, "found": bool(found), "secs": round(secs, 1)})
            if not (results[-1].get("secs") == a.timeout and not found):
                print(f"  L={L:>6} depth={d:.1f}  {'FOUND' if found else 'MISS '}  {secs:6.1f}s")
            sys.stdout.flush()

    out_p = ROOT / "artifacts" / f"niah_{Path(a.model).stem}.json"
    out_p.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_p}")
    for L in sorted({r["tokens"] for r in results}):
        rs = [r for r in results if r["tokens"] == L]
        hit = sum(r["found"] for r in rs)
        print(f"  {L:>6} tokens: {hit}/{len(rs)} found, "
              f"median {sorted(r['secs'] for r in rs)[len(rs)//2]:.0f}s")


if __name__ == "__main__":
    main()
