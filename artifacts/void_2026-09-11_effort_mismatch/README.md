# VOID — unpinned reasoning_effort (2026-09-11)

These results are kept for the record and must not be cited. See `../../EVAL_INTEGRITY.md`.

Reported:

    GLM-5.3-Flash-REAP50    pass@1  68/164 = 41.5%   BFCL 70/100
    DeepSeek-0731-REAP      pass@1 127/164 = 77.4%   BFCL 79/100

The serving was correct this time (the 2026-09-10 orphan bug was fixed, and each arm logged its
served model id). The *generation config* was not. The harness sent no `reasoning_effort`, and the
two servers default differently — dsv4 to `"low"`, glm5 to Max — and glm5 at Max never terminates.
92 of GLM's 96 failures emitted no `def` at all; **zero were wrong answers**. Re-run with effort
pinned to `high` at the same budget, GLM scored 150/164 = 91.5%.

`humaneval_glm.json` here is the 2026-09-10 first arm, which was the one arm the orphan bug spared.
It is void for this second, independent reason.
