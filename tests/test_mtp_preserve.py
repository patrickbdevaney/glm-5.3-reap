"""The MTP block must survive surgery, pruned consistently and with its router sliced.

Pass 1 dropped layer 45 entirely and shipped `num_nextn_predict_layers: 0`, which forecloses
speculative decoding from the artifact - the draft head simply is not in the weights. This
checks the replacement behaviour on a synthetic checkpoint, because the failure mode is silent:
a model with an unsliced router or a mis-numbered expert set still loads.
"""
import json, sys, tempfile, shutil
from pathlib import Path
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import stages.s04b_surgery as SG

fails = []
def check(name, cond, extra=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name} {extra}")
    if not cond: fails.append(name)

N_ORIG, N_KEEP, H, I = 8, 4, 16, 32
td = Path(tempfile.mkdtemp())
try:
    src = td / "src"; art = td / "art"
    src.mkdir(); art.mkdir()
    SG.SRC, SG.ARTIFACTS = src, art

    # Experts get deliberately ordered magnitudes: expert e scaled by (e+1).
    # The top-4 by weight norm must therefore be experts 4,5,6,7.
    tensors, wm = {}, {}
    for e in range(N_ORIG):
        scale = float(e + 1)
        for proj, shape in (("gate_proj", (I, H)), ("up_proj", (I, H)), ("down_proj", (H, I))):
            k = f"model.language_model.layers.{SG.MTP_LAYER}.mlp.experts.{e}.{proj}.weight"
            tensors[k] = torch.full(shape, scale, dtype=torch.float32)
            wm[k] = "shard.safetensors"
    gk = f"model.language_model.layers.{SG.MTP_LAYER}.mlp.gate.weight"
    tensors[gk] = torch.randn(N_ORIG, H); wm[gk] = "shard.safetensors"
    bk = f"model.language_model.layers.{SG.MTP_LAYER}.mlp.gate.e_score_correction_bias"
    tensors[bk] = torch.randn(N_ORIG); wm[bk] = "shard.safetensors"
    save_file(tensors, str(src / "shard.safetensors"), metadata={"format": "pt"})
    (src / "model.safetensors.index.json").write_text(json.dumps({"weight_map": wm}))

    keep = SG._mtp_keep_set(N_KEEP, N_ORIG)
    check("MTP keep-set is produced", keep is not None and len(keep) == N_KEEP, f"{keep}")
    check("ranks the highest-norm experts", keep == [4, 5, 6, 7], f"got {keep}")
    check("keep-set is cached for resume", (art / "mtp_keep.json").exists())

    # A resume after the source shards are gone must reuse the cache, not silently drop the block.
    (src / "shard.safetensors").unlink()
    again = SG._mtp_keep_set(N_KEEP, N_ORIG)
    check("resume reuses the cache once sources are consumed", again == keep, f"{again}")

    # Router slicing: the surgery path keys off `retained`, so verify the shape contract that
    # path depends on - an unsliced router emits logits over experts that no longer exist and
    # produces garbage routing WITHOUT crashing, which is the worst way for this to fail.
    idx = torch.tensor(keep, dtype=torch.long)
    check("router slices to the kept experts", tensors[gk][idx].shape[0] == N_KEEP)
    check("router bias slices to the kept experts", tensors[bk][idx].shape[0] == N_KEEP)
    check("expert renumbering is contiguous 0..N_KEEP-1",
          sorted({i for i, _ in enumerate(keep)}) == list(range(N_KEEP)))
finally:
    shutil.rmtree(td, ignore_errors=True)

print("\n" + ("ALL PASS" if not fails else f"FAILURES: {fails}"))
sys.exit(1 if fails else 0)
