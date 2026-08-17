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
    from ._generated.flux1_dev import Flux1Dev, Flux1DevLayout, Flux1DevResolution
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
    from ._generated.sd2 import Sd2, Sd2Layout, Sd2Shape
    from ._generated.sd15 import Sd15, Sd15Layout, Sd15Shape
    from ._generated.sdxl import Sdxl, SdxlLayout, SdxlShape

#: Exported name -> the generated module that defines it. THE catalog index:
#: adding a family adds rows here and nothing else changes.
_FAMILIES: Final[dict[str, str]] = {
    "Flux1Dev": "flux1_dev",
    "Flux1DevLayout": "flux1_dev",
    "Flux1DevResolution": "flux1_dev",
    "Flux2Klein4b": "flux2_klein_4b",
    "Flux2Klein4bLayout": "flux2_klein_4b",
    "Flux2Klein4bTokens": "flux2_klein_4b",
    "Flux2Klein9b": "flux2_klein_9b",
    "Flux2Klein9bLayout": "flux2_klein_9b",
    "Flux2Klein9bTokens": "flux2_klein_9b",
    "Sd2": "sd2",
    "Sd2Layout": "sd2",
    "Sd2Shape": "sd2",
    "Sd15": "sd15",
    "Sd15Layout": "sd15",
    "Sd15Shape": "sd15",
    "Sdxl": "sdxl",
    "SdxlLayout": "sdxl",
    "SdxlShape": "sdxl",
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
    "Flux1Dev",
    "Flux1DevLayout",
    "Flux1DevResolution",
    "Flux2Klein4b",
    "Flux2Klein4bLayout",
    "Flux2Klein4bTokens",
    "Flux2Klein9b",
    "Flux2Klein9bLayout",
    "Flux2Klein9bTokens",
    "Sd2",
    "Sd2Layout",
    "Sd2Shape",
    "Sd15",
    "Sd15Layout",
    "Sd15Shape",
    "Sdxl",
    "SdxlLayout",
    "SdxlShape",
]
