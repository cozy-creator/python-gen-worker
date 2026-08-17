"""The 3D boundary models: two EAGER declarations, both layout-undeclarable.

pgw#1346 B5, and this pair is the clearest case for why K2's
``layouts_undeclarable`` had to move onto the declaration rather than be
dropped with ``Slot``. Neither of these two can name a registered contract
handle, and the reasons are different and both real:

* **trellis-3d** consumes an overlay-indirected, source-built tree. Which
  layout it actually consumes is not determinable from evidence in this repo,
  and the endpoint says exactly that — *reported, not guessed*.
* **hunyuan3d-2.1** ships pickle ``.ckpt`` weights. The header-reading
  derivation cannot classify them, and READING one to find out is the banned
  act, so the honest declaration is that no registered handle names these
  bytes.

``ModelSpec``'s alternative is ``DEFAULT_LAYOUT = "bf16"``, which would state a
contract for both of these and be wrong twice — which is the K2 finding.
"""

from __future__ import annotations

from typing import Final

from ..spec import ModelSpec, TunedValues


class Hunyuan3dTuned(TunedValues, frozen=True):
    """Hunyuan3D's recipe, migrated from ``@family("hunyuan3d")``.

    Field names are the WIRE names (pgw#654 gap #4): ``RuntimeFormula``
    resolves ``a + b*num_shape_steps`` by same-named lookup, payload over the
    stamped values.
    """

    num_shape_steps: int = 50
    guidance_scale: float = 5.0


#: TRELLIS-2 image-to-3D. No tuned schema: the endpoint declares no ``@family``
#: and reads no ``ctx.defaults``, so K8 is inapplicable and no tensorhub PR is
#: owed for this name.
TRELLIS_3D: Final = ModelSpec(
    name="trellis_3d",
    layouts_undeclarable=(
        "overlay-indirected source-built tree — the consumed layout is NOT "
        "determinable from evidence in this repo; reported, not guessed"
    ),
)

#: Hunyuan3D 2.1 image-to-3D (shape + optional PBR texture paint).
HUNYUAN3D: Final = ModelSpec(
    name="hunyuan3d",
    tuned=Hunyuan3dTuned,
    layouts_undeclarable=(
        "pickle .ckpt — unclassifiable by the header-reading derivation, and "
        "reading one to find out is the banned act"
    ),
)


__all__ = ["HUNYUAN3D", "TRELLIS_3D", "Hunyuan3dTuned"]
