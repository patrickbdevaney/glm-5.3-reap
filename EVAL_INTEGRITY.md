# Eval integrity: two bugs that voided the same head-to-head twice

The GLM-vs-DeepSeek head-to-head decides whether the GLM compression line is worth continuing,
and whether the 311 GB GLM source can be deleted. It produced a confident, wrong answer twice
before it produced a real one. Both failures were in the *harness*, not the models, and both
produced plausible-looking numbers. This file records what to check before believing a score.

## Bug 1 (2026-09-10) — the orphaned server: two arms re-measured the same model

`scripts/run_gpu_chain.sh` launched each server as

```bash
( cd "$dir" && "$@" > log 2>&1 & echo $! > /tmp/.he_pid )
```

`$!` there is the intermediate **subshell**, not the server. `kill "$pid"` killed the wrapper; the
server reparented to init and kept :8080 forever. `wait_health` then curled `/health` *before*
checking liveness, so the surviving orphan answered on the first attempt for a server that never
started — and the next two arms scored the previous model.

Proof, not inference:
- pid 848579 = `glm5-server`, **PPID 1**, up 15h15m
- live `/health` returned `glm-5.3-flash-reap50-nvfp4` in the middle of the "vexp" arm
- `bfcl_dsv4.json` byte-identical to `bfcl_glm5.json` (md5 `c2b41294543c5212c7864b6b6e1a5e14`)
- the dsv4 server log ends mid-load

Fixed: `( cd "$dir" && exec "$@" ) &` so `$!` is the server; liveness-check before curl; abort if
:8080 is already held; log the **served model id** on every result line; FATAL if the port is still
held after teardown. Void results: `artifacts/void_2026-09-10_server_orphan_bug/`.

## Bug 2 (2026-09-11) — unpinned reasoning_effort: a looping GLM vs a low-effort DeepSeek

The corrected chain ran cleanly and reported:

```
GLM-5.3-Flash-REAP50    pass@1  68/164 = 41.5%
DeepSeek-0731-REAP      pass@1 127/164 = 77.4%
```

That is not a capability gap. Classifying every failure:

```
GLM   fails 96   no `def` emitted 92   truncated  4   actually wrong  0
DSV4  fails 37   no `def` emitted  3   truncated 33   actually wrong  1
```

**GLM produced zero wrong answers.** The harness sent no `reasoning_effort`, and the two servers
disagree on the default — `deepseek-v4-flash-0731-cuda/include/openai_api.h:34` -> `"low"`,
`glm-5.3-flash-cuda-server/include/openai_api.h:63` -> **Max**. Measured at the ceiling:

| task | effort | ctok | content | reasoning | finish | rep |
|---|---|---|---|---|---|---|
| HumanEval/0 | max | 8192 | **0** | 18574 | length | **0.61** |
| HumanEval/0 | low | 174 | 550 | 36 | stop | 0.00 |
| HumanEval/0 | high | 172 | 542 | 19 | stop | 0.00 |
| HumanEval/2 | max | 8192 | **0** | 15640 | length | 0.10 |
| HumanEval/2 | low | 99 | 374 | 36 | stop | 0.00 |
| HumanEval/2 | high | 141 | 397 | 208 | stop | 0.00 |

Two *different* pathologies with one symptom: HumanEval/0 is a true repetition loop (`rep` = the
fraction of recurring 40-char windows, 0.61); HumanEval/2 is 15,640 characters of **non**-repeating
reasoning that still reaches no answer. Both end with everything in `reasoning_content` and
`content` **empty**, which a harness reading only `content` scores as a wrong answer.

**Raising `max_tokens` would not have fixed this** — the obvious fix, and the one originally
planned. The overrun is unbounded: a 32k budget buys 4x the GPU time and the same 41.5%. Pinning
effort, at the same budget, moved GLM from 41.5% to 91.5%.

Void results: `artifacts/void_2026-09-11_effort_mismatch/`. Nothing was deleted.

## What the harnesses now do

- `--reasoning-effort` on both harnesses, default `high`, sent explicitly on every request, so no
  arm inherits a server default. `low`/`high` only: anything else renders as Max on glm5.
- Every row records `completion_tokens`, `finish_reason`, `truncated`, `has_def`, and head/tail of
  the reply. Every summary prints the token distribution and a loud
  `TRUNCATED n/N <-- not a capability number`.
- `truncated` is derived from `completion_tokens >= max_tokens`, **not** `finish_reason` — dsv4's
  server hardcodes `"stop"` and never reports `"length"`.
- Result files record the effort and budget they were produced under, so a stale file cannot be
  silently compared against a fresh one.

## Checklist before believing any eval number

1. **Assert the served identity** in every result line. A model id is one curl away; an orphan is not.
2. **Check liveness before polling health.** A port answering is not proof your process is up.
3. **Distrust equal or byte-identical scores across models.** Check the md5s.
4. **Pin generation config explicitly** — effort, temperature, budget — for every arm.
5. **Classify the failures before reading the score.** All-`NameError` means you measured the
   harness. Zero wrong answers means you measured the harness.
6. **Read the truncation count first, the score second.** A pass@1 with no truncation column cannot
   distinguish a weak model from a starved one.
