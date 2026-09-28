#!/usr/bin/env python
"""Does a draft head trained on one quant's hidden states transfer to another quant?

The traces are being captured from IQ3_M. A draft head consumes the BASE MODEL'S hidden state,
and that state is not the same tensor across quantisations - IQ3_M and IQ4_XS are different
functions of the same weights. So "train on IQ3_M, serve on IQ4_XS" is a train/serve
distribution shift, and whether it costs acceptance is an empirical question, not something
that can be asserted from the fact that both are the same REAP checkpoint.

This compares hidden states for the SAME tokens under two quants. Rows are matched by the ids
each run emitted, so a tokenisation difference can never silently misalign rows.

Matching is done PER SEQUENCE, not over the concatenated stream. Both runs see the same prompts,
but the probe file is rebuilt by splitting and rejoining the chunk on the separator, and that
round-trip can normalise a line ending here and there. One such sequence shifts every row after
it, which would make the whole capture look incomparable when in fact only that sequence is.
So each sequence is checked on its own and only the ones whose ids agree exactly are compared -
the rest are reported as skipped, since a partial sample is a real measurement and a
whole-stream abort is not.

High cosine (>~0.99) means one head serves every quant. Low cosine means the head is
quant-specific and the honest options are a per-quant head or a mixed-quant capture.
"""
import sys, numpy as np

ref_bin, ref_ids, probe_bin, probe_ids = sys.argv[1:5]
ref_lens, probe_lens = sys.argv[5], sys.argv[6]
D = 4096

a_ids  = np.fromfile(ref_ids,    dtype=np.int32)
b_ids  = np.fromfile(probe_ids,  dtype=np.int32)
a_len  = np.fromfile(ref_lens,   dtype=np.int32)
b_len  = np.fromfile(probe_lens, dtype=np.int32)

nseq = min(len(a_len), len(b_len))
a_off = np.concatenate([[0], np.cumsum(a_len)])
b_off = np.concatenate([[0], np.cumsum(b_len)])

# Read only the rows we actually keep. The bins are 250 MB each; there is no reason to hold
# both in full when most of the file is sequences we may end up skipping anyway.
A_ref = np.memmap(ref_bin,   dtype=np.float16, mode="r").reshape(-1, D)
B_ref = np.memmap(probe_bin, dtype=np.float16, mode="r").reshape(-1, D)

keep_a, keep_b, skipped = [], [], []
for i in range(nseq):
    la, lb = int(a_len[i]), int(b_len[i])
    ia, ib = int(a_off[i]), int(b_off[i])
    if la != lb or not np.array_equal(a_ids[ia:ia+la], b_ids[ib:ib+lb]):
        skipped.append(i)
        continue
    keep_a.append(np.arange(ia, ia+la))
    keep_b.append(np.arange(ib, ib+lb))

if not keep_a:
    print("FAIL: no sequence tokenised identically under both quants - not comparable")
    sys.exit(1)

ka = np.concatenate(keep_a); kb = np.concatenate(keep_b)
n = len(ka)
print(f"matched {nseq-len(skipped)}/{nseq} sequences, {n} tokens"
      + (f"; skipped seqs {skipped} (tokenisation drift in the probe rebuild)" if skipped else ""))

A = np.asarray(A_ref[ka], dtype=np.float32)
B = np.asarray(B_ref[kb], dtype=np.float32)

cos = (A*B).sum(1) / (np.linalg.norm(A,axis=1)*np.linalg.norm(B,axis=1) + 1e-9)
rel = np.linalg.norm(A-B,axis=1) / (np.linalg.norm(A,axis=1) + 1e-9)

print(f"cosine   mean {cos.mean():.5f}  p50 {np.percentile(cos,50):.5f}  "
      f"p5 {np.percentile(cos,5):.5f}  min {cos.min():.5f}")
print(f"rel L2   mean {rel.mean():.5f}  p95 {np.percentile(rel,95):.5f}  max {rel.max():.5f}")
print(f"rows with cosine < 0.99: {100*(cos<0.99).mean():.2f}%")
print(f"rows with cosine < 0.95: {100*(cos<0.95).mean():.2f}%")

m = cos.mean()
print()
if m >= 0.99:
    print("VERDICT: states track closely across quants. One head should serve all of them;")
    print("         confirm with a real acceptance measurement once a head exists.")
elif m >= 0.95:
    print("VERDICT: measurable drift. A single head is likely usable but will lose some")
    print("         acceptance off its training quant. Prefer capturing from the quant you")
    print("         will actually serve, or mix quants in the capture.")
else:
    print("VERDICT: states differ substantially. A head trained on one quant should NOT be")
    print("         advertised as optimal for the others - capture per-quant or mix.")
