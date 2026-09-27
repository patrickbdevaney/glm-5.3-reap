"""Predict PEAK host memory for a full end-to-end run, before a single byte is allocated.

Written 2026-09-26 after the third box wedge in one evening. Every one of them was a memory
ceiling discovered by hitting it. The recovery mechanisms all exist and all failed in turn:

  * memguard's tier 1 cannot reclaim a live allocation.
  * memguard's tier 2 killed the right PID twice, 39 s apart, and MemAvailable kept FALLING
    (125MB -> 97MB -> 49MB). A process blocked in the GPU driver does not die on SIGKILL until
    the driver returns, so killing is not a recovery mechanism for this workload.
  * `MemoryMax=72G` on the cgroup did not bind, because Tegra unified/nvmap allocations are not
    charged to the process cgroup that requested them.

When detection, killing and cgroup limits all fail, the only thing left is not starting a run
that cannot fit. That is what this does.

USAGE
    python scripts/mem_lookahead.py --seq 16384 --batch 1 --layers 12 --attn eager
    -> exits 0 and prints the budget if it fits, exits 4 with the binding term if it does not.

It is a MODEL, so it is validated against outcomes already observed rather than trusted:
`gate_mem_lookahead.sh` requires it to reproduce S=2048 OK, S=8192 OK, S=16384 FAIL.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GIB = 2 ** 30

# MemTotal is 122 GiB. The usable transient budget is far below that and the reasons are measured:
#   * streaming the shards keeps page cache at tens of GiB (MEASURED 2026-09-26: a drop_caches
#     released 88 GiB mid-run, and it refilled within seconds). It is reclaimable in principle
#     and NOT reclaimable fast enough to satisfy a driver allocation -- the documented Tegra
#     failure that killed s03 six times with no oom-kill line.
#   * the nvmap driver pool is attributed to nobody in /proc/meminfo and only a full shrinker
#     pass releases it.
# So the budget is set against OBSERVED outcomes, not against MemTotal.
BUDGET_GIB = 55.0


def load_cfg():
    c = json.loads((ROOT / "source" / "GLM-5.3-Flash" / "config.json").read_text())
    return c.get("text_config", c)


def peak(seq: int, batch: int, layers: int, attn: str, tokens: int, cfg) -> dict:
    H = cfg["num_attention_heads"]
    hid = cfg["hidden_size"]
    n_exp = cfg["n_routed_experts"]
    moe_i = cfg["moe_intermediate_size"]
    vocab = cfg["vocab_size"]
    hc = cfg.get("hc_mult", 4)
    topk = cfg.get("index_topk", 2048)

    terms = {}
    # resident for the whole run
    terms["embed_table"] = vocab * hid * 2 / GIB
    terms["hidden_states_held"] = tokens * hid * hc * 2 / GIB

    # one decoder layer materialised bf16 -- the experts dominate
    terms["moe_layer_weights"] = n_exp * 3 * moe_i * hid * 2 / GIB

    # attention workspace, the term that actually varies with sequence length
    if attn == "eager":
        # The DSA indexer must SCORE every key before it can select the top-`index_topk`, so the
        # eager path materialises [B, heads, L, L]. `index_topk` caps what is ATTENDED, not what
        # is SCORED -- which is where the earlier "DSA never reaches the quadratic regime" claim
        # was wrong. x3 for scores + softmax + a transient copy.
        terms["dsa_attn_eager"] = 3 * batch * H * seq * seq * 2 / GIB
    else:
        terms["dsa_attn_sparse"] = 3 * batch * H * seq * topk * 2 / GIB
    # KDA is linear: fixed-size recurrent state, plus the per-token projection
    terms["kda_workspace"] = batch * seq * H * cfg.get("linear_attn_config", {}).get(
        "head_dim", 128) * 2 * 2 / GIB

    total = sum(terms.values())
    binding = max(terms, key=terms.get)
    return {"terms": terms, "total": total, "binding": binding,
            "fits": total <= BUDGET_GIB}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, required=True)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--attn", default="eager", choices=["eager", "sparse"])
    ap.add_argument("--tokens", type=int, default=131072, help="tokens held across the sweep")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    r = peak(a.seq, a.batch, a.layers, a.attn, a.tokens, load_cfg())
    if not a.quiet:
        print(f"peak-memory lookahead: seq={a.seq} batch={a.batch} attn={a.attn} "
              f"tokens={a.tokens:,}")
        for k, v in sorted(r["terms"].items(), key=lambda kv: -kv[1]):
            mark = "  <-- binding" if k == r["binding"] else ""
            print(f"    {k:22s} {v:8.2f} GiB{mark}")
        print(f"    {'TOTAL':22s} {r['total']:8.2f} GiB   budget {BUDGET_GIB:.0f} GiB")
    if r["fits"]:
        if not a.quiet:
            print("FITS")
        return 0
    print(f"WILL NOT FIT: {r['total']:.1f} GiB > {BUDGET_GIB:.0f} GiB budget; "
          f"binding term is {r['binding']} ({r['terms'][r['binding']]:.1f} GiB)",
          file=sys.stderr)
    return 4


if __name__ == "__main__":
    sys.exit(main())
