set -u
cd /home/patrickd/glm-5.3-reap
PY=./.venv/bin/python
fail=0
# The model is validated against outcomes ALREADY OBSERVED on this box on 2026-09-26.
# A predictor that cannot reproduce the three runs we actually watched is not a predictor.
chk(){ # seq attn expect_rc label
  $PY scripts/mem_lookahead.py --seq "$1" --attn "$2" --quiet >/dev/null 2>&1; rc=$?
  if [ "$rc" = "$3" ]; then echo "PASS: $4"; else echo "FAIL: $4 (rc=$rc want $3)"; fail=1; fi
}
chk 2048  eager  0 "S=2048 eager predicted to FIT   -- it ran to completion"
chk 8192  eager  0 "S=8192 eager predicted to FIT   -- it ran to completion"
chk 16384 eager  4 "S=16384 eager predicted to FAIL -- it wedged the box entering the DSA layer"
chk 16384 sparse 0 "S=16384 sparse predicted to FIT -- the fix, not the naive path"
# monotone in sequence length: a longer sequence can never be predicted cheaper
prev=0
for S in 2048 4096 8192 16384; do
  v=$($PY scripts/mem_lookahead.py --seq $S --attn eager 2>/dev/null | awk '/TOTAL/{print int($2)}')
  if [ "$v" -lt "$prev" ]; then echo "FAIL: not monotone at S=$S ($v < $prev)"; fail=1; fi
  prev=$v
done
[ $fail -eq 0 ] && echo "PASS: peak is monotone non-decreasing in sequence length"
echo "---"; [ $fail -eq 0 ] && echo "GATE PASS" || echo "GATE FAIL"
