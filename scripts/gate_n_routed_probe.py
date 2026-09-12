"""Fixed greedy prompt set for the N_ROUTED gate. Deterministic by construction: temperature 0,
a pinned reasoning_effort (the two servers disagree on the default, which voided a head-to-head
on 2026-09-11), and a fixed max_tokens so a run that diverges late still diverges inside the
window. Asserts the served model identity and a non-empty reply, because a gate that passes
against a dead engine is worse than no gate."""
import json, sys, urllib.request

BASE = "http://127.0.0.1:8080"
PROMPTS = [
    "What is the capital of France? One word.",
    "List the first 20 prime numbers, comma separated.",
    "Write a Python function reverse_words(s) that reverses the word order. Code only.",
    "What is 17 * 23? Just the number.",
    "Explain in two sentences why a MoE layer routes each token to only a few experts.",
    "Sort [5,3,9,1,7] ascending. Just the list.",
    "Write a Python one-liner that sums the squares of 1..n.",
    "What is the chemical symbol for gold? One word.",
]

def post(path, obj):
    r = urllib.request.Request(BASE + path, json.dumps(obj).encode(),
                               {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=900).read())

served = post("/v1/chat/completions", {"model": "gate", "temperature": 0.0, "max_tokens": 4,
                                       "reasoning_effort": "low",
                                       "messages": [{"role": "user", "content": "hi"}]})
model = served.get("model", "?")
print(f"served model: {model}", flush=True)
assert model and model != "?", "server did not report a model identity"

out = []
for p in PROMPTS:
    r = post("/v1/chat/completions",
             {"model": "gate", "temperature": 0.0, "max_tokens": 256,
              "reasoning_effort": "low", "messages": [{"role": "user", "content": p}]})
    m = r["choices"][0]["message"]
    txt = (m.get("reasoning_content") or "") + "\x00" + (m.get("content") or "")
    assert txt.strip("\x00").strip(), f"empty reply for {p!r} -- engine is not generating"
    out.append({"prompt": p, "text": txt, "model": model,
                "tokens": int(r.get("usage", {}).get("completion_tokens", 0))})
    print(f"  {r['usage']['completion_tokens']:4d} tok  {p[:50]!r}", flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
