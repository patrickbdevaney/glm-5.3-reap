# Rejected artifacts

## vision_characterization.CONTAMINATED.json
First ChartQA run against IQ4_XS, reported 17/24 = 70.8%. **Not usable, never published.**

The harness graded `stdout + stderr` after a line-prefix filter. llama.cpp log lines that did not
match any filter prefix survived into the graded text, and ChartQA relaxed accuracy scans for a
number *anywhere* in the reply. So `image decoded (batch 1/1) in 12036 ms` supplied the numbers
1, 1 and 12036 for free, and a gold answer of `1.` scored correct without the model ever looking
at the chart. The echoed prompt survived too, handing back the gold for yes/no questions whose
wording contained the word.

Fixed in `scripts/vision_characterize.py`: grade stdout only, cut the echoed prompt at the last
chat-turn marker, and write the raw streams to `artifacts/vision_eval/raw_*.{out,err}` so the
next disagreement is settled by evidence rather than by inference from a truncated excerpt.
