set -u
# Extract the victim-selection block from the patched memguard and exercise it against REAL
# processes whose start order we control. Fake decoys are named to match the licence patterns.
NEW=/tmp/claude-2002/-home-patrickd/3952824c-51a0-4b40-aa8e-acf3b3b02ab9/scratchpad/mg.new
START=$(grep -n '^    cands=""' $NEW | cut -d: -f1)
END=$(grep -n 'newest=\$st; pid=\$q; fi' $NEW | cut -d: -f1)
sed -n "${START},$((END+1))p" $NEW > /tmp/sel.sh
grep -q 'newest' /tmp/sel.sh || { echo "FAIL: could not extract selection block"; exit 1; }
fail=0

mkdir -p /tmp/mgt/scripts
cat > /tmp/mgt/run_stage.py <<'P'
import time; time.sleep(300)
P
cp /tmp/mgt/run_stage.py /tmp/mgt/scripts/exp_seqlen_saliency.py

# OLD process first (mimics the pipeline stage, running a long time)
python3 /tmp/mgt/run_stage.py & OLD=$!
sleep 2
# NEW process second (mimics the experiment that started later)
python3 /tmp/mgt/scripts/exp_seqlen_saliency.py & NEW_P=$!
sleep 1

pid=""; . /tmp/sel.sh
echo "candidates found, selected pid=$pid (old=$OLD new=$NEW_P)"
if [ "$pid" = "$NEW_P" ]; then echo "PASS 1: selects the NEWEST licensed job, not the long-running stage"; else echo "FAIL 1: selected $pid, expected newest $NEW_P"; fail=1; fi
case " $cands " in *" $NEW_P "*) echo "PASS 2: exp_seqlen_saliency.py is inside the licence";; *) echo "FAIL 2: experiment not licensed"; fail=1;; esac
case " $cands " in *" $OLD "*) echo "PASS 3: run_stage.py still licensed";; *) echo "FAIL 3"; fail=1;; esac
case " $cands " in *" $$ "*) echo "FAIL 4: selected its own shell"; fail=1;; *) echo "PASS 4: never selects itself";; esac

kill $OLD $NEW_P 2>/dev/null
# 5: with nothing licensed running, it must select nothing rather than something arbitrary
sleep 1
pid=""; cands=""; . /tmp/sel.sh
REALS=$(pgrep -f 'run_stage\.py' | head -1)
if [ -n "$REALS" ]; then
  if [ "$pid" = "$REALS" ]; then echo "PASS 5: with decoys gone, falls back to the real licensed stage $REALS"; else echo "FAIL 5: picked $pid, real licensed stage is $REALS"; fail=1; fi
else
  if [ -z "$pid" ]; then echo "PASS 5: no licensed job -> no victim (never 'biggest RSS')"; else echo "FAIL 5: picked $pid with nothing licensed"; fail=1; fi
fi
rm -rf /tmp/mgt /tmp/sel.sh
echo "---"; [ $fail -eq 0 ] && echo "GATE PASS" || echo "GATE FAIL"
