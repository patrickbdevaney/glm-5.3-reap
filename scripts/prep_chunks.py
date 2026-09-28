#!/usr/bin/env python
"""Build llama-embedding input chunks for draft-head teacher extraction.

The shards hold input_ids, but llama-embedding consumes text, so every sequence is decoded
and re-tokenized. The re-tokenized ids are NOT the training targets - the teacher writes the
ids it actually tokenized alongside the hidden states, so alignment is true by construction.
What we produce here is only the text the teacher reads, plus the bookkeeping needed to know
which emitted rows are trustworthy.

Long documents are WINDOWED, not dropped. The first version dropped anything over MAX_SEQ,
which threw away 90.4% of the corpus tokens - and not uniformly: agentic kept 1014/2949 seqs
and science 487/1229, while code kept 2531/2580. The surviving mix was mostly short code
snippets, which is close to the opposite of the intended capability blend.

Each continuation window carries OVERLAP tokens of preceding context. Those tokens are real
input to the teacher but must be masked when training the draft head: their hidden states are
fine, they are simply duplicates of rows already present in the previous window. The count is
written to chunk_NNNN.cold.npy, one entry per sequence, in the same order as the prompts.
"""
import os, glob, json, torch, numpy as np
from transformers import AutoTokenizer

TOKD     = os.environ.get("TOKENIZER", "output/glm-5.3-flash-reap50-nvfp4-pass2")
TARGET   = int(os.environ.get("TARGET_TOKENS", 30_000_000))
CHUNK    = int(os.environ.get("CHUNK_TOKENS", 250_000))
MAXSEQ   = int(os.environ.get("MAX_SEQ", 1900))      # must stay under -c/-b
OVERLAP  = int(os.environ.get("OVERLAP", 128))       # context carried into a continuation
MINSEQ   = int(os.environ.get("MIN_SEQ", 8))
BUCKET_CAP = float(os.environ.get("BUCKET_CAP", 0.40))   # no single bucket over 40% of target
SEP      = os.environ.get("SEP", "<#sep#>")

tok = AutoTokenizer.from_pretrained(TOKD, trust_remote_code=True)
os.makedirs("chunks", exist_ok=True)

# ---- load, then window -------------------------------------------------------------------
per_bucket = {}
for f in sorted(glob.glob("corpus/shards/text/*.pt")):
    bucket = os.path.basename(f)[:-3]
    out = []
    for rec in torch.load(f, map_location="cpu", weights_only=False):
        ids = rec.get("input_ids")
        if ids is None:
            continue
        ids = ids.tolist()
        if len(ids) < MINSEQ:
            continue
        if len(ids) <= MAXSEQ:
            out.append((ids, 0))
            continue
        stride = MAXSEQ - OVERLAP
        start = 0
        while start < len(ids):
            w = ids[start:start + MAXSEQ]
            if len(w) < MINSEQ:
                break
            out.append((w, 0 if start == 0 else OVERLAP))
            if start + MAXSEQ >= len(ids):
                break
            start += stride
    per_bucket[bucket] = out
    print(f"{bucket:10s} windows {len(out):6d} tokens {sum(len(w) for w,_ in out):10d}", flush=True)

# ---- budget: cap any one bucket so agentic (53% of the corpus) cannot swamp the blend -----
cap = int(TARGET * BUCKET_CAP)
budget = {}
for b, ws in per_bucket.items():
    have = sum(len(w) for w, _ in ws)
    budget[b] = min(have, cap)
short = TARGET - sum(budget.values())
if short > 0:   # redistribute the slack to buckets that still have material
    for b in sorted(budget, key=lambda x: -(sum(len(w) for w, _ in per_bucket[x]) - budget[x])):
        have = sum(len(w) for w, _ in per_bucket[b])
        take = min(short, have - budget[b])
        budget[b] += take; short -= take
        if short <= 0:
            break
print("budget:", {b: budget[b] for b in sorted(budget)}, flush=True)

# ---- round-robin so every chunk is capability-mixed ---------------------------------------
cursor = {b: 0 for b in per_bucket}
spent  = {b: 0 for b in per_bucket}
order  = sorted(per_bucket)
seqs   = []
live   = True
while live:
    live = False
    for b in order:
        ws = per_bucket[b]
        i  = cursor[b]
        if i >= len(ws) or spent[b] >= budget[b]:
            continue
        seqs.append((b,) + ws[i]); cursor[b] = i + 1
        spent[b] += len(ws[i][0]); live = True

# ---- emit ---------------------------------------------------------------------------------
n_chunk = 0; cur_txt = []; cur_cold = []; cur_lens = []; total = 0; dropped = 0

def flush():
    global n_chunk, cur_txt, cur_cold, cur_lens
    if not cur_txt:
        return
    base = f"chunks/chunk_{n_chunk:04d}"
    with open(base + ".txt", "w", encoding="utf-8") as fh:
        fh.write(SEP.join(cur_txt))
    np.save(base + ".cold.npy", np.array(cur_cold, dtype=np.int32))
    np.save(base + ".lens.npy", np.array(cur_lens, dtype=np.int32))
    n_chunk += 1; cur_txt = []; cur_cold = []; cur_lens = []

pending = 0
for bucket, ids, cold in seqs:
    if total >= TARGET:
        break
    text = tok.decode(ids, skip_special_tokens=True)
    if not text.strip() or SEP in text:
        dropped += 1; continue
    n = len(tok(text, add_special_tokens=True)["input_ids"])
    if not (MINSEQ <= n <= MAXSEQ + 64):     # slack: decode/encode is not exactly length-preserving
        dropped += 1; continue
    cur_txt.append(text); cur_lens.append(n); cur_cold.append(cold)
    total += n; pending += n
    if pending >= CHUNK:
        flush(); pending = 0
flush()

meta = {"chunks": n_chunk, "tokens": total, "dropped": dropped, "sep": SEP,
        "max_seq": MAXSEQ, "overlap": OVERLAP, "bucket_cap": BUCKET_CAP,
        "spent": spent, "bytes_fp16": total * 4096 * 2}
json.dump(meta, open("chunks/manifest.json", "w"), indent=2)
print(json.dumps(meta, indent=2))
