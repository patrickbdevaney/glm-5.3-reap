#!/usr/bin/env python3
"""Why does GLM burn 8192 tokens on a 3-line HumanEval problem?

The budget pilot showed completion_tokens=8192, finish_reason="length", and content="" --
every token went into reasoning_content and none into the answer. include/openai_api.h:63:
an absent `reasoning_effort` renders as Max. The eval never set it. This probe separates
the two candidate causes:
  (a) effort -- Max reasoning is simply unbounded on easy problems; low/high terminate.
  (b) degenerate loop -- greedy decoding repeating forever, in which case NO effort setting
      saves it and the finding is a sampling bug, not a budget one.
Repetition is measured directly: the fraction of 40-char windows in the tail that recur.
"""
import glob, json, re, sys, urllib.request, collections

BASE = "http://127.0.0.1:8080"
HE = "/home/patrickd/.cache/huggingface/hub/datasets--openai--openai_humaneval/**/*.parquet"

def post(obj, timeout=1800):
    r = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(obj).encode(),
                               {"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as fh:
        return json.loads(fh.read())

def rep_frac(t, w=40):
    """Fraction of w-char windows that appear more than once. >0.5 == looping."""
    if len(t) < 4 * w: return 0.0
    win = [t[i:i+w] for i in range(0, len(t) - w, w)]
    c = collections.Counter(win)
    return sum(n for n in c.values() if n > 1) / len(win)

import pyarrow.parquet as pq
rows = pq.read_table(sorted(glob.glob(HE, recursive=True))[0]).to_pylist()
probs = {r["task_id"]: r for r in rows}

CONFIGS = [("max(default)", {}), ("low", {"reasoning_effort": "low"}),
           ("high", {"reasoning_effort": "high"}), ("nothink", {"nothink": True})]

print(f"{'task':14} {'effort':13} {'ctok':>6} {'content':>8} {'reason':>8} {'finish':>7} "
      f"{'rep':>5} {'def?':>5}")
out = []
for tid in ("HumanEval/0", "HumanEval/2"):
    p = probs[tid]
    base = ("Complete this Python function. Reply with the full function in a single "
            "```python code block, no explanation.\n\n" + p["prompt"])
    for name, extra in CONFIGS:
        prompt = ("/nothink\n" + base) if extra.pop("nothink", False) else base
        try:
            r = post({"model": "glm5", "temperature": 0.0, "max_tokens": 8192,
                      "messages": [{"role": "user", "content": prompt}], **extra})
            m = r["choices"][0]["message"]
            c, rc = m.get("content", "") or "", m.get("reasoning_content", "") or ""
            ct = int(r.get("usage", {}).get("completion_tokens", 0))
            fin = r["choices"][0].get("finish_reason")
        except Exception as e:
            c, rc, ct, fin = "", "", -1, repr(e)[:60]
        rf = rep_frac(rc or c)
        hd = f"def {p['entry_point']}" in (c + rc)
        print(f"{tid:14} {name:13} {ct:6} {len(c):8} {len(rc):8} {str(fin):>7} "
              f"{rf:5.2f} {str(hd):>5}", flush=True)
        out.append({"task": tid, "effort": name, "ctok": ct, "n_content": len(c),
                    "n_reasoning": len(rc), "finish": fin, "rep_frac": round(rf, 3),
                    "has_def": hd, "reason_tail": rc[-600:], "content_head": c[:300]})
json.dump(out, open("/home/patrickd/glm-5.3-reap/artifacts/pilot/effort_probe.json", "w"), indent=1)
print("\nwrote artifacts/pilot/effort_probe.json")
