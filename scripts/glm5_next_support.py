"""glm5_next support shims for llm-compressor.

llm-compressor has no registration for `glm5_next` (it is absent from
`modeling/moe/conversion_mappings.py::ARCH_TO_IMPORT_PATHS`). Most of the path works anyway,
because `LinearExperts2D.get_linear_experts_cls` can auto-derive a class from any experts
module carrying the `@use_experts_implementation` decorator -- which `Glm5NextTextExperts` does.

One thing does not survive that auto-derivation:

    Glm5NextTextExperts._apply_gate  ->  reads `self.swiglu_limit`

but `LinearExperts2D.__init__` stores that same value as `self.limit`
(`MoEConfig.from_config` maps config.swiglu_limit -> limit). The derived class therefore has
the value under the wrong name and `_apply_gate` raises AttributeError on the first forward.

`register()` is idempotent and must be called before `linearize_moe`. Upstreamable as either
an alias in the derived class or a proper `ARCH_TO_IMPORT_PATHS` entry for glm5_next.
"""
from __future__ import annotations

__all__ = ["register"]


def register():
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextExperts
    from llmcompressor.modeling.moe.linear_experts import LinearExperts2D

    existing = LinearExperts2D._registry.get(Glm5NextTextExperts)
    if existing is not None and getattr(existing, "_glm5_next_shim", False):
        return existing

    # Drop any cached auto-derived class, re-derive to inherit the correct classvars
    # (has_gate / is_transposed / is_concatenated / has_bias / _apply_gate), then subclass.
    LinearExperts2D._registry.pop(Glm5NextTextExperts, None)
    derived = LinearExperts2D.get_linear_experts_cls(Glm5NextTextExperts)

    class Glm5NextLinearExperts(derived):  # type: ignore[misc, valid-type]
        _glm5_next_shim = True

        def __init__(self, config, *args, **kwargs):
            super().__init__(config, *args, **kwargs)
            # alias for Glm5NextTextExperts._apply_gate
            self.swiglu_limit = self.limit

    LinearExperts2D._registry[Glm5NextTextExperts] = Glm5NextLinearExperts
    return Glm5NextLinearExperts
