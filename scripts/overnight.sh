#!/usr/bin/env bash
# Draft-head teacher extraction. Unattended, STOPPABLE, RESUMABLE. Wall clock only.
#   ./scripts/overnight_ctl.sh {stop|resume|status}
# A step is marked done only on rc=0; interrupted or failed steps re-run on resume.
# Never uses `pgrep -f <pattern>` - that self-matched earlier and idled two jobs for 2.5h.
set -uo pipefail
cd /home/patrickd/glm-5.3-reap
LOG=logs/overnight.log
STATE=artifacts/overnight_state
BIN=$HOME/glm5-llama.cpp/build-cuda/bin
M=gguf/GLM-5.3-Flash-REAP50-IQ3_M.gguf
NGL=16                 # 24 peaked in bench but thrashed under real page-cache load; 16 finishes
SEP='<#sep#>'          # keeps newlines inside prompts intact (matters for code/math)
MIN_FREE_GB=40         # stop before filling the disk
mkdir -p "$STATE" traces chunks
# Exactly one extractor, ever. Two runners would race on the same chunk and each would see the
# other's half-written bin. This matters now that a systemd unit can start the chain at boot
# while a hand-launched one is still going.
if [ -f "$STATE/ALL_DONE" ]; then exit 0; fi
exec 9>"$STATE/lock"
if ! flock -n 9; then
  echo "[$(date -Is)] overnight: another run holds the lock; exiting" >> "$LOG"
  exit 0
fi
echo $$ > "$STATE/pid"
say() { echo "[$(date -Is)] overnight: $*" >> "$LOG"; }
stop_requested() { [ -f "$STATE/STOP" ]; }

run_guarded() {   # honour STOP mid-step, not just between steps
  "$@" & local cp=$!
  while kill -0 "$cp" 2>/dev/null; do
    if stop_requested; then kill -TERM "$cp" 2>/dev/null; sleep 3; kill -KILL "$cp" 2>/dev/null; return 130; fi
    sleep 5
  done
  wait "$cp"
}
step() {
  local n=$1; shift
  if [ -f "$STATE/$n.done" ]; then say "skip $n (already done)"; return 0; fi
  if stop_requested; then say "STOP honoured; halting before $n"; exit 0; fi
  say "=== $n ==="
  "$@"; local rc=$?
  if [ $rc -eq 130 ]; then say "$n interrupted by STOP - will re-run on resume"; exit 0; fi
  if [ $rc -eq 0 ]; then touch "$STATE/$n.done"; say "$n complete"
  else say "$n failed rc=$rc - will retry on resume"; fi
  return 0
}

prep_chunks() {
  [ -f chunks/manifest.json ] && { say "  chunks already prepared"; return 0; }
  # 20M, not 30M: measured prefill is 76 tok/s (llama-bench's 125 was one hot contiguous
  # batch, not packed ~1500-token sequences), so 30M would be 110h of wall clock.
  # MAX_SEQ stays under -c 2048. A -c 8192 probe was tried and was worse, not better:
  # the larger compute buffers crowd the 67 GB mmap and the GPU sat idle ~40% of samples.
  TARGET_TOKENS=20000000 CHUNK_TOKENS=250000 MAX_SEQ=1900 OVERLAP=128 \
    run_guarded ./.venv/bin/python scripts/prep_chunks.py >> "$LOG" 2>&1
}

extract_traces() {
  local done_tok=0 t_start=$SECONDS
  for TXT in chunks/chunk_*.txt; do
    stop_requested && return 130
    local base bin rows ntok hf_tok
    base=$(basename "$TXT" .txt); bin="traces/$base.bin"

    # A finished chunk is one where the ids companion exists and its token count matches the
    # rows in the bin. The bin is written before the ids, so the ids file existing at the right
    # size is itself the completion marker - no separate bookkeeping to get out of sync.
    if [ -f "$bin" ] && [ -f "$bin.ids" ]; then
      rows=$(( $(stat -c%s "$bin") / 8192 ))
      ntok=$(( $(stat -c%s "$bin.ids") / 4 ))
      if [ "$rows" -eq "$ntok" ] && [ "$ntok" -gt 0 ]; then done_tok=$((done_tok+ntok)); continue; fi
      say "  $base: rows $rows != ids $ntok - redoing"
    fi

    local free_gb; free_gb=$(df --output=avail -BG / | tail -1 | tr -dc '0-9')
    if [ "$free_gb" -lt "$MIN_FREE_GB" ]; then
      say "  stopping: only ${free_gb} GB free (floor ${MIN_FREE_GB} GB). Extracted so far is intact."
      return 0
    fi

    local t0=$SECONDS
    # 4h, not 2h. A timeout throws away the entire chunk because the bin is only written after
    # the last prompt, so a too-tight limit burns two hours of GPU and keeps nothing.
    LLAMA_EMBD_BIN="$bin" run_guarded timeout 14400 "$BIN/llama-embedding" -m "$M" -ngl $NGL \
        -c 2048 -b 2048 -ub 512 --pooling none --embd-normalize -1 \
        --embd-separator "$SEP" -f "$TXT" > /dev/null 2>> logs/extract.err
    local rc=$?; [ $rc -eq 130 ] && return 130
    local dt=$((SECONDS-t0))

    # Durability, not tidiness. fwrite+fclose only reaches the page cache, so a power cut could
    # lose the tail of a chunk that every marker calls complete - and the ids file would be lost
    # the same way, so the row/ids check would still agree and nothing downstream would notice.
    # Force both to stable storage before anything inspects them.
    sync "$bin" "$bin.ids" "$bin.lens" 2>/dev/null

    rows=0; ntok=0
    [ -f "$bin" ]      && rows=$(( $(stat -c%s "$bin") / 8192 ))
    [ -f "$bin.ids" ]  && ntok=$(( $(stat -c%s "$bin.ids") / 4 ))
    if [ "$rc" -ne 0 ] || [ "$ntok" -eq 0 ] || [ "$rows" -ne "$ntok" ]; then
      say "  $base FAILED rc=$rc rows=$rows ids=$ntok after ${dt}s (see logs/extract.err)"
      mkdir -p traces/bad && mv -f "$bin" "traces/bad/$base.bin" 2>/dev/null
      rm -f "$bin.ids" "$bin.lens"
      continue
    fi

    # Hand the page cache back. The bin is write-once and never re-read during extraction;
    # leaving 2 GB of it resident per chunk is what starved the model mmap.
    ./.venv/bin/python -c "
import os,sys
for f in sys.argv[1:]:
    fd=os.open(f,os.O_RDONLY)
    try: os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
    finally: os.close(fd)
" "$bin" "$bin.ids" 2>/dev/null

    local nseq=0
    [ -f "$bin.lens" ] && nseq=$(( $(stat -c%s "$bin.lens") / 4 ))
    done_tok=$((done_tok+ntok))
    say "  $base ok: $ntok tok / $nseq seq in ${dt}s = $((ntok/(dt>0?dt:1))) tok/s; cumulative $((done_tok/1000))k; free ${free_gb}G"
  done
  say "  extraction pass done: $((done_tok/1000))k tokens in $(( (SECONDS-t_start)/60 )) min"
  # Nothing outstanding means a restart has no work to do. Say so once, so the supervisor
  # can bring the chain back after a reboot without spinning on a finished job forever.
  local n_txt n_ok
  n_txt=$(ls chunks/chunk_*.txt 2>/dev/null | wc -l)
  n_ok=$(ls traces/chunk_*.bin.ids 2>/dev/null | wc -l)
  if [ "$n_txt" -gt 0 ] && [ "$n_ok" -eq "$n_txt" ]; then
    touch "$STATE/ALL_DONE"; say "  ALL_DONE: $n_ok/$n_txt chunks extracted"
  fi
  return 0
}

# The whole dataset rests on the claim that the ids file the teacher writes lines up with the
# rows it writes. Three minutes to prove that on a toy prompt is cheaper than discovering it
# an hour into the first real chunk.
gate_ids() {
  local t=artifacts/gate_ids.txt b=artifacts/gate_ids.bin
  printf 'The capital of France is Paris.<#sep#>def add(a, b):\n    return a + b\n' > "$t"
  rm -f "$b" "$b.ids" "$b.lens"
  LLAMA_EMBD_BIN="$b" run_guarded timeout 1800 "$BIN/llama-embedding" -m "$M" -ngl $NGL \
      -c 2048 -b 2048 -ub 512 --pooling none --embd-normalize -1 \
      --embd-separator "$SEP" -f "$t" > /dev/null 2>> logs/gate_ids.err
  local rc=$?; [ $rc -eq 130 ] && return 130
  [ -f "$b" ] && [ -f "$b.ids" ] || { say "  gate_ids: no output (rc=$rc)"; return 1; }
  local rows ntok nseq
  rows=$(( $(stat -c%s "$b") / 8192 )); ntok=$(( $(stat -c%s "$b.ids") / 4 ))
  nseq=$(( $(stat -c%s "$b.lens") / 4 ))
  say "  gate_ids: rows=$rows ids=$ntok seqs=$nseq"
  [ "$rows" -eq "$ntok" ] && [ "$ntok" -gt 0 ] && [ "$nseq" -eq 2 ]
}

say "=========== start (pid $$) ==========="
step gate_ids        gate_ids
step prep_chunks     prep_chunks
step extract_traces  extract_traces
say "=========== done ==========="
rm -f "$STATE/pid"
