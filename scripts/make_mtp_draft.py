#!/usr/bin/env python
"""Repackage GLM-5.3's native MTP block (blk.45) as a standalone draft GGUF.

GLM-5.3 ships a Multi-Token Prediction module inside every GGUF: blk.45, with nextn.eh_proj /
enorm / hnorm / shared_head_norm plus a full MLA+DSA attention and a 144-expert MoE. Nothing
executes it - transformers drops it on load, llama.cpp loads it and leaves it out of the graph.
It is roughly 1.5 GiB of every quant that currently does no work.

This lifts it into a model llama.cpp can actually run as a speculative draft. The repackaging is
byte-preserving: quantised blocks are copied verbatim, never dequantised and requantised, so the
draft is exactly the tensor that shipped in the parent file.

Three renames carry the design:

  blk.45.*                          -> blk.0.*      the MTP block becomes the draft's only layer
  blk.45.nextn.shared_head_norm     -> output_norm  it IS the final norm before the shared lm_head
  (token_embd, output)              -> copied       MTP has no embedding or head of its own; the
                                                    reference shares the parent's, so the draft
                                                    must carry a copy to stand alone

The result is ~2.3 GiB at IQ3_M and pairs with exactly one parent quant. It is NOT interchangeable
across quants: hidden states drift hard between quantisations (IQ3_M vs IQ4_XS cosine 0.891
against a 0.999 same-quant floor), so a draft built from one parent must be served with that
parent.
"""
import argparse, logging, os, sys
from pathlib import Path

# Prefer an installed gguf, so this script is usable by anyone who has `pip install gguf`, and fall
# back to a checkout beside this tree for the case where it is not installed. GGUF_PY_DIR overrides
# both, for a llama.cpp checked out somewhere else entirely.
try:
    import gguf
except ImportError:  # noqa: E722
    _candidates = []
    if os.environ.get("GGUF_PY_DIR"):
        _candidates.append(Path(os.environ["GGUF_PY_DIR"]))
    _here = Path(__file__).resolve()
    _candidates += [
        _here.parent.parent.parent / "glm5-llama.cpp" / "gguf-py",
        _here.parent.parent.parent / "llama.cpp" / "gguf-py",
        _here.parent.parent / "llama.cpp" / "gguf-py",
    ]
    for _c in _candidates:
        if (_c / "gguf").is_dir():
            sys.path.insert(0, str(_c))
            break
    else:
        sys.exit("cannot find the gguf python package - `pip install gguf`, "
                 "or set GGUF_PY_DIR to a llama.cpp gguf-py directory")
    import gguf  # noqa: E402
from tqdm import tqdm  # noqa: E402

logger = logging.getLogger("make_mtp_draft")

SRC_ARCH = "glm5-next"
DST_ARCH = "glm5-next-mtp"

# Keys that describe machinery the MTP block does not have. blk.45 carries no hc_* tensors (it is
# a plain pre-norm residual block, unlike the 45 hyper-connected layers ahead of it) and no ssm_*
# tensors (it is MLA, not KDA), so advertising either would make the loader look for weights that
# are not in the file.
DROP_SUFFIXES = (
    ".attention.hc.mult", ".attention.hc.sinkhorn_iters", ".attention.hc.eps",
    ".ssm.conv_kernel", ".ssm.gate_lower_bound", ".kda.head_dim",
    ".nextn_predict_layers",
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path, help="parent GGUF containing blk.45")
    ap.add_argument("dst", type=Path, help="draft GGUF to write")
    ap.add_argument("--mtp-layer", type=int, default=None,
                    help="source block index of the MTP module (default: block_count-1)")
    ap.add_argument("--force", action="store_true", help="overwrite dst if it exists")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s: %(message)s")

    if args.dst.exists() and not args.force:
        logger.error("%s exists; pass --force to overwrite", args.dst)
        sys.exit(1)

    reader = gguf.GGUFReader(args.src, "r")

    arch_f = reader.get_field("general.architecture")
    arch = arch_f.contents() if arch_f else None
    if arch != SRC_ARCH:
        logger.error("expected general.architecture=%s, found %r", SRC_ARCH, arch)
        sys.exit(1)

    bc = reader.get_field(f"{SRC_ARCH}.block_count")
    nextn = reader.get_field(f"{SRC_ARCH}.nextn_predict_layers")
    n_block = int(bc.contents())
    n_nextn = int(nextn.contents()) if nextn else 0
    if n_nextn != 1:
        logger.error("this script handles exactly one MTP layer, file declares %d", n_nextn)
        sys.exit(1)
    mtp_il = args.mtp_layer if args.mtp_layer is not None else n_block - 1
    src_prefix = f"blk.{mtp_il}."
    logger.info("source block_count=%d, MTP module at blk.%d", n_block, mtp_il)

    writer = gguf.GGUFWriter(path=None, arch=DST_ARCH, endianess=reader.endianess)

    # ---- key/value metadata -------------------------------------------------------------
    # Hyperparameters are re-prefixed rather than restated, so anything the parent knows about
    # its own MLA, MoE and indexer geometry reaches the draft without being retyped here (and
    # without silently drifting from the parent if the converter ever changes).
    n_copied = 0
    for field in reader.fields.values():
        name = field.name
        if name in ("GGUF.version", "GGUF.tensor_count", "GGUF.kv_count"):
            continue
        if name == "general.architecture":
            continue
        if any(name == SRC_ARCH + s for s in DROP_SUFFIXES):
            logger.debug("dropping %s (not present in the MTP block)", name)
            continue

        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        value = field.contents()

        if name.startswith(SRC_ARCH + "."):
            suffix = name[len(SRC_ARCH):]
            if suffix == ".block_count":
                value = 1
            elif suffix == ".leading_dense_block_count":
                # The MTP block is a MoE block; there are no dense layers ahead of it here.
                value = 0
            elif suffix == ".attention.head_count_kv":
                # Per-layer in the parent (0 on KDA layers, 1 on MLA). The draft has one MLA layer.
                value = [1]
                sub_type = gguf.GGUFValueType.INT32
            name = DST_ARCH + suffix
        elif name == "general.name":
            value = str(value) + " MTP Draft"
        elif name.startswith("quantize.imatrix."):
            # The parent's imatrix provenance describes the parent, not this file.
            continue

        writer.add_key_value(name, value, val_type, sub_type=sub_type)
        n_copied += 1

    writer.add_key_value(f"{DST_ARCH}.mtp.parent_block", mtp_il, gguf.GGUFValueType.UINT32)
    logger.info("copied %d kv pairs", n_copied)

    # ---- tensors ------------------------------------------------------------------------
    renames: dict[str, str] = {}
    for t in reader.tensors:
        if t.name in ("token_embd.weight", "output.weight"):
            renames[t.name] = t.name
        elif t.name.startswith(src_prefix):
            tail = t.name[len(src_prefix):]
            if tail == "nextn.shared_head_norm.weight":
                renames[t.name] = "output_norm.weight"
            else:
                renames[t.name] = "blk.0." + tail

    required = {
        "token_embd.weight", "output.weight", "output_norm.weight",
        "blk.0.nextn.eh_proj.weight", "blk.0.nextn.enorm.weight", "blk.0.nextn.hnorm.weight",
        "blk.0.attn_norm.weight", "blk.0.attn_output.weight", "blk.0.ffn_norm.weight",
        "blk.0.ffn_gate_inp.weight", "blk.0.ffn_down_exps.weight",
    }
    produced = set(renames.values())
    missing = required - produced
    if missing:
        logger.error("source is missing required MTP tensors: %s", sorted(missing))
        sys.exit(1)

    keep = [t for t in reader.tensors if t.name in renames]
    total = 0
    for t in keep:
        dst_name = renames[t.name]
        writer.add_tensor_info(dst_name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
        total += t.n_bytes
        logger.debug("%-46s -> %s", t.name, dst_name)

    logger.info("writing %d tensors, %.2f GiB to %s", len(keep), total / 2**30, args.dst)

    tmp = args.dst.with_suffix(args.dst.suffix + ".part")
    writer.open_output_file(tmp)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    bar = tqdm(desc="writing", total=total, unit="B", unit_scale=True)
    for t in keep:
        writer.write_tensor_data(t.data, tensor_endianess=reader.endianess)
        bar.update(t.n_bytes)
    bar.close()
    writer.close()

    # Land the file only once it is complete on disk. A draft GGUF truncated by a power cut
    # would otherwise sit there looking like a valid artifact.
    os.replace(tmp, args.dst)
    logger.info("done: %s (%.2f GiB)", args.dst, args.dst.stat().st_size / 2**30)


if __name__ == "__main__":
    main()
