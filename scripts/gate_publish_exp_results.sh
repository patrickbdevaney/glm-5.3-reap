set -u
cd /home/patrickd/glm-5.3-reap
BK=artifacts/exp_seqlen/comparison.json.realbak
[ -f artifacts/exp_seqlen/comparison.json ] && cp artifacts/exp_seqlen/comparison.json $BK
mkdir -p artifacts/exp_seqlen
fail=0

# 1: refuses a table with no control arm (no noise floor -> no verdict possible)
cat > artifacts/exp_seqlen/comparison.json <<'J'
[{"arm":"S8192","layer":3,"type":"deepseek_sparse_attention","spearman":0.97,"keep_overlap":0.95}]
J
if python3 scripts/publish_exp_results.py 2>/dev/null; then echo "FAIL 1: published without a control"; fail=1; else echo "PASS 1: refuses table with no control arm"; fi

# 2: refuses when the file is absent
mv artifacts/exp_seqlen/comparison.json /tmp/ci_gone.json
if python3 scripts/publish_exp_results.py 2>/dev/null; then echo "FAIL 2: published with no input"; fail=1; else echo "PASS 2: refuses absent input"; fi
mv /tmp/ci_gone.json artifacts/exp_seqlen/comparison.json

# 3: an arm ABOVE the floor is called WITHIN; an arm BELOW is called a REAL EFFECT
cat > artifacts/exp_seqlen/comparison.json <<'J'
[{"arm":"S8192","layer":3,"type":"deepseek_sparse_attention","spearman":0.99,"keep_overlap":0.97},
 {"arm":"S16384","layer":3,"type":"linear_attention","spearman":0.80,"keep_overlap":0.61},
 {"arm":"control_S2048_disjoint","layer":3,"type":"deepseek_sparse_attention","spearman":0.98,"keep_overlap":0.93}]
J
python3 scripts/publish_exp_results.py >/dev/null 2>&1
if grep -q '| S8192 | 0.9700 | +0.0400 | WITHIN noise floor |' wiki/93-seqlen-ladder-results.md; then echo "PASS 3a: above-floor arm called WITHIN"; else echo "FAIL 3a"; fail=1; fi
if grep -q 'S16384 | 0.6100 | -0.3200 | \*\*BELOW floor' wiki/93-seqlen-ladder-results.md; then echo "PASS 3b: below-floor arm called REAL EFFECT"; else echo "FAIL 3b"; fail=1; fi
if grep -q 'worst keep-overlap: \*\*0.9300\*\*' wiki/93-seqlen-ladder-results.md; then echo "PASS 3c: floor taken from the control arm"; else echo "FAIL 3c"; fail=1; fi
# the control arm must not be scored against itself
if grep -q '| control_S2048_disjoint | 0' wiki/93-seqlen-ladder-results.md; then echo "FAIL 3d: control scored as an arm"; fail=1; else echo "PASS 3d: control excluded from verdicts"; fi

rm -f wiki/93-seqlen-ladder-results.md artifacts/exp_seqlen/comparison.json
[ -f $BK ] && mv $BK artifacts/exp_seqlen/comparison.json
echo "---"; [ $fail -eq 0 ] && echo "GATE PASS" || echo "GATE FAIL"
