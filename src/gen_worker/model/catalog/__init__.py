"""The first-party model catalog: import a family, not diffusers.

pgw#1326's ruling, made an import path. An endpoint writes

```python
from gen_worker.model.catalog import Sdxl

@endpoint(families={"sdxl": Sdxl})
class Generate:
    def generate(self, ctx: RequestContext, p: In, sdxl: Sdxl) -> Out: ...
```

and never names a pipeline class, a component, or a graph. The names exported
here are GENERATED types — the class is the type, a value of it is a resolved
instance carrying weights and catalog-stamped tuned values.

**This package is SDK surface and the worker runtime never imports it.** Two
reasons, both structural: importing a declaration would put model code on an
adopt-only serve pod that is CI-fenced against having any (pgw#1328), and the
bare ``gen_worker.families`` registry is asserted empty-by-design, so a catalog
that registered itself on a plain ``import gen_worker`` would make the library
ship families it has no business shipping.

**Resolution is lazy, and that is not an optimisation.** A generated binding
imports its own declaration module to expose ``SPEC``, so an eager package
``__init__`` would import every family's declaration to satisfy one, and — worse
— could not be run at all before the first ``gen-worker model generate``,
which is the command whose own import path goes through here. PEP 562 breaks
the cycle: importing this package costs nothing, and asking for a name costs
exactly that one family.

Adding a family is: write the declaration module, ``gen-worker model export``
it, ``gen-worker model generate`` its bindings, add two lines below. That is a
CATALOG-ONLY diff by construction — no non-catalog module changes, so a new
family is a no-op for every existing endpoint (greenfield B9).
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:  # pragma: no cover - the eager spelling, for type checkers only
    from ._generated.chatterbox import Chatterbox
    from ._generated.ernie import Ernie, ErnieBatch, ErnieLayout, ErnieShape
    from ._generated.flex2_preview import Flex2Preview
    from ._generated.flux1_dev import Flux1Dev, Flux1DevLayout, Flux1DevResolution
    from ._generated.flux1_schnell import (
        Flux1Schnell,
        Flux1SchnellLayout,
        Flux1SchnellTokens,
    )
    from ._generated.flux2_klein_4b import (
        Flux2Klein4b,
        Flux2Klein4bLayout,
        Flux2Klein4bTokens,
    )
    from ._generated.flux2_klein_9b import (
        Flux2Klein9b,
        Flux2Klein9bLayout,
        Flux2Klein9bTokens,
    )
    from ._generated.foundation_1 import Foundation1
    from ._generated.hunyuan3d import Hunyuan3d
    from ._generated.internvl_u import InternvlU
    from ._generated.joycaption import Joycaption
    from ._generated.ltx23 import Ltx23, Ltx23AudioTokens, Ltx23Layout, Ltx23VideoTokens
    from ._generated.minimax_h3 import MinimaxH3
    from ._generated.musicgen import Musicgen
    from ._generated.qwen36_27b_mtp import Qwen3627bMtp
    from ._generated.qwen36_35b_a3b import Qwen3635bA3b
    from ._generated.qwen_image import QwenImage, QwenImageLayout, QwenImageShape
    from ._generated.sd2 import Sd2, Sd2DeclaredSampler, Sd2Layout, Sd2Shape
    from ._generated.sd15 import Sd15, Sd15DeclaredSampler, Sd15Layout, Sd15Shape
    from ._generated.sdxl import Sdxl, SdxlDeclaredSampler, SdxlLayout, SdxlShape
    from ._generated.stable_audio_open import StableAudioOpen
    from ._generated.trellis_3d import Trellis3d
    from ._generated.z_image import ZImage, ZImageBranches, ZImageLayout
    from ._generated.wan22_i2v_a14b import (
        Wan22I2vA14b,
        Wan22I2vA14bFrames,
        Wan22I2vA14bLayout,
        Wan22I2vA14bShape,
    )
    from ._generated.wan22_t2v_a14b import (
        Wan22T2vA14b,
        Wan22T2vA14bFrames,
        Wan22T2vA14bLayout,
        Wan22T2vA14bShape,
    )
    from ._generated.wan22_ti2v_5b import (
        Wan22Ti2v5b,
        Wan22Ti2v5bFrames,
        Wan22Ti2v5bLayout,
        Wan22Ti2v5bShape,
    )

#: Exported name -> the generated module that defines it. THE catalog index:
#: adding a family adds rows here and nothing else changes.
#:
#: The EAGER models (pgw#1346 B5) contribute ONE name each and no ``Layout`` or
#: bucket alias, because they declare no graph classes to have traced variants
#: of. The shorter row is the tier showing through the index, not an omission.
_FAMILIES: Final[dict[str, str]] = {
    "Chatterbox": "chatterbox",
    "Ernie": "ernie",
    "ErnieBatch": "ernie",
    "ErnieLayout": "ernie",
    "ErnieShape": "ernie",
    "Flex2Preview": "flex2_preview",
    "Flux1Dev": "flux1_dev",
    "Flux1DevLayout": "flux1_dev",
    "Flux1DevResolution": "flux1_dev",
    "Flux1Schnell": "flux1_schnell",
    "Flux1SchnellLayout": "flux1_schnell",
    "Flux1SchnellTokens": "flux1_schnell",
    "Flux2Klein4b": "flux2_klein_4b",
    "Flux2Klein4bLayout": "flux2_klein_4b",
    "Flux2Klein4bTokens": "flux2_klein_4b",
    "Flux2Klein9b": "flux2_klein_9b",
    "Flux2Klein9bLayout": "flux2_klein_9b",
    "Flux2Klein9bTokens": "flux2_klein_9b",
    "Foundation1": "foundation_1",
    "Hunyuan3d": "hunyuan3d",
    "InternvlU": "internvl_u",
    "Joycaption": "joycaption",
    "MinimaxH3": "minimax_h3",
    "Musicgen": "musicgen",
    "Qwen3627bMtp": "qwen36_27b_mtp",
    "Qwen3635bA3b": "qwen36_35b_a3b",
    "QwenImage": "qwen_image",
    "QwenImageLayout": "qwen_image",
    "QwenImageShape": "qwen_image",
    "Ltx23": "ltx23",
    "Ltx23AudioTokens": "ltx23",
    "Ltx23Layout": "ltx23",
    "Ltx23VideoTokens": "ltx23",
    "Sd2": "sd2",
    "Sd2DeclaredSampler": "sd2",
    "Sd2Layout": "sd2",
    "Sd2Shape": "sd2",
    "Sd15": "sd15",
    "Sd15DeclaredSampler": "sd15",
    "Sd15Layout": "sd15",
    "Sd15Shape": "sd15",
    "Sdxl": "sdxl",
    "SdxlDeclaredSampler": "sdxl",
    "SdxlLayout": "sdxl",
    "SdxlShape": "sdxl",
    "StableAudioOpen": "stable_audio_open",
    "Trellis3d": "trellis_3d",
    "ZImage": "z_image",
    "ZImageBranches": "z_image",
    "ZImageLayout": "z_image",
    "Wan22I2vA14b": "wan22_i2v_a14b",
    "Wan22I2vA14bFrames": "wan22_i2v_a14b",
    "Wan22I2vA14bLayout": "wan22_i2v_a14b",
    "Wan22I2vA14bShape": "wan22_i2v_a14b",
    "Wan22T2vA14b": "wan22_t2v_a14b",
    "Wan22T2vA14bFrames": "wan22_t2v_a14b",
    "Wan22T2vA14bLayout": "wan22_t2v_a14b",
    "Wan22T2vA14bShape": "wan22_t2v_a14b",
    "Wan22Ti2v5b": "wan22_ti2v_5b",
    "Wan22Ti2v5bFrames": "wan22_ti2v_5b",
    "Wan22Ti2v5bLayout": "wan22_ti2v_5b",
    "Wan22Ti2v5bShape": "wan22_ti2v_5b",
}


def __getattr__(name: str) -> Any:
    module = _FAMILIES.get(name)
    if module is None:
        raise AttributeError(
            f"the catalog has no {name!r}; it carries {sorted(_FAMILIES)!r}"
        )
    return getattr(import_module(f"{__name__}._generated.{module}"), name)


def __dir__() -> list[str]:
    return sorted(_FAMILIES)


__all__ = [
    "Chatterbox",
    "Ernie",
    "ErnieBatch",
    "ErnieLayout",
    "ErnieShape",
    "Flex2Preview",
    "Flux1Dev",
    "Flux1DevLayout",
    "Flux1DevResolution",
    "Flux1Schnell",
    "Flux1SchnellLayout",
    "Flux1SchnellTokens",
    "Flux2Klein4b",
    "Flux2Klein4bLayout",
    "Flux2Klein4bTokens",
    "Flux2Klein9b",
    "Flux2Klein9bLayout",
    "Flux2Klein9bTokens",
    "Foundation1",
    "Hunyuan3d",
    "InternvlU",
    "Joycaption",
    "MinimaxH3",
    "Musicgen",
    "Qwen3627bMtp",
    "Qwen3635bA3b",
    "QwenImage",
    "QwenImageLayout",
    "QwenImageShape",
    "Ltx23",
    "Ltx23AudioTokens",
    "Ltx23Layout",
    "Ltx23VideoTokens",
    "Sd2",
    "Sd2DeclaredSampler",
    "Sd2Layout",
    "Sd2Shape",
    "Sd15",
    "Sd15DeclaredSampler",
    "Sd15Layout",
    "Sd15Shape",
    "Sdxl",
    "SdxlDeclaredSampler",
    "SdxlLayout",
    "SdxlShape",
    "StableAudioOpen",
    "Trellis3d",
    "ZImage",
    "ZImageBranches",
    "ZImageLayout",
    "Wan22I2vA14b",
    "Wan22I2vA14bFrames",
    "Wan22I2vA14bLayout",
    "Wan22I2vA14bShape",
    "Wan22T2vA14b",
    "Wan22T2vA14bFrames",
    "Wan22T2vA14bLayout",
    "Wan22T2vA14bShape",
    "Wan22Ti2v5b",
    "Wan22Ti2v5bFrames",
    "Wan22Ti2v5bLayout",
    "Wan22Ti2v5bShape",
]
