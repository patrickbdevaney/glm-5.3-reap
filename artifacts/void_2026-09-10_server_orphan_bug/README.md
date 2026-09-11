# VOID — these files do not measure what their names say

All three were produced against an ORPHANED `glm5-server` still bound to :8080.
They are GLM-5.3-Flash-REAP50 results mislabelled as DeepSeek.

Proof:
- `bfcl_dsv4.json` was byte-identical (md5 c2b41294543c5212c7864b6b6e1a5e14) to `bfcl_glm5.json`.
- `humaneval_dsv4.json` scored exactly 68/164, same as GLM (differs in md5 only because
  HumanEval samples at T>0).
- At 2026-09-11T00:15, mid-"vexp" run, `/health` on :8080 returned
  `glm-5.3-flash-reap50-nvfp4`; the serving process was pid 848579, PPID 1, up 15h15m.
- The real `dsv4-server` log ended mid-load; it never bound the port.

Root cause: `scripts/run_gpu_chain.sh:27` captured `$!` of the subshell wrapping
`cd "$dir" && "$@"`, not of the server. `kill "$pid"` killed the wrapper; the server
reparented to init and kept :8080. `wait_health` then curled /health BEFORE checking
process liveness, so the orphan answered on the first attempt and every subsequent arm
evaluated the previous model.

Only `humaneval_glm.json` (68/164 = 41.5%) and `bfcl_glm5.json` (70/100 = 70.0%) are valid —
they were the first arm, so the server was the right one.
