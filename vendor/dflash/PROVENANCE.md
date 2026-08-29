# Vendored from github.com/z-lab/dflash

`model.py`, `benchmark.py`, `cli.py`, `__init__.py` copied verbatim (`--depth 1` clone,
2026-08-29). Upstream **LICENSE is MIT** (see `LICENSE`) and is reproduced unmodified.

This is the **code**. It is a different licence from the **weights**: `incoai/GLM-5.3-Flash-DFlash2`
is `cc-by-nc-nd-4.0` (NonCommercial + NoDerivatives), which is why any retrained drafter goes to a
**private** repo. Vendoring MIT code carries no such restriction.

Vendored rather than pip-installed because the package pulls a serving stack this box does not
have, and because the offline acceptance harness needs to call `DFlash2DraftModel.forward` and
`.propose` directly rather than through `dflash_generate`, which requires a resident target doing
autoregressive decode. `model_mlx.py` is dropped - Apple Silicon backend, irrelevant here.

Nothing here is modified. Local behaviour lives in `scripts/drafter_*.py`.
