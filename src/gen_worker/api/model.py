"""``model=`` — the payload-level checkpoint placement key (pgw#509).

Model selection is a runtime PAYLOAD ARGUMENT, not a build-time fan-out.
The three-boundary rule:

* **class = memory-sharing unit** — weights load once in ``setup()``, resident
  per instance.
* **method = wire contract** — every ``@endpoint`` method is its own routable
  function with its own payload schema.
* **arg = checkpoint** — a handler whose payload declares a field typed with a
  ``ModelChoice`` subclass selects, per request, which curated checkpoint runs
  against the resident base. One ``generate(model=)`` replaces N near-identical
  functions.

A ``ModelChoice`` is authored as DATA: a ``str``-valued :class:`enum.Enum`
whose members are :class:`Model` rows, each carrying its :class:`ModelRef`
binding + a typed per-model :class:`ModelDefaults`::

    class SdxlDefaults(ModelDefaults, frozen=True):
        scheduler: Literal["euler_a", "dpmpp_2m_karras"]
        steps: int
        guidance: float

    class SdxlModel(ModelChoice[SdxlDefaults], enum.Enum):
        WAI = Model("wai-illustrious", Hub("tensorhub/wai-illustrious"),
                    SdxlDefaults("euler_a", 28, 6.0))
        PONY = Model("cyberrealistic-pony", Hub("tensorhub/cyberrealistic-pony"),
                     SdxlDefaults("dpmpp_2m_karras", 30, 5.0))

    class TextToImage(msgspec.Struct):
        prompt: str
        model: SdxlModel = SdxlModel.WAI          # curated-only

The handler reads the pick as fully-typed data — no ``ctx.models`` string
sniffing::

    def generate(self, ctx, p: TextToImage) -> ...:
        d = p.model.defaults          # typed SdxlDefaults
        steps = d.steps

On the wire a curated pick is just its id string (``"wai-illustrious"``); the
JSON schema is a closed ``enum`` — the curated allowlist, introspectable by
discovery.

**BYOM is the FIELD TYPE.** A field typed ``SdxlModel`` is curated-only; a
field typed ``SdxlModel | ModelRef`` additionally accepts an arbitrary
client-supplied :class:`ModelRef` (bring-your-own-model). No decorator, no
``sources=``/``scope=`` — architecture compatibility is derived downstream
from the pipeline the endpoint's ``models=`` loads. Per-method policy falls
straight out of method=contract: ``generate`` can be BYOM-open while
``generate_turbo`` (fixed distillation LoRA) stays curated.
"""

from __future__ import annotations

import enum
from typing import Any, Generic, Optional, TypeVar

import msgspec

from .binding import ModelRef


class ModelDefaults(msgspec.Struct, frozen=True):
    """Base for a ``ModelChoice``'s per-model inference-default row. Authors
    subclass with their own fields (steps, guidance, scheduler, prompt
    dialect, ...). Frozen + typed so a pick's defaults are read as data, and
    the whole row is manifest-exportable (the catalog/UI can render the
    effective ``steps: 28`` before submit — a gap ``Optional[int] = None``
    payload fields hide)."""


D = TypeVar("D", bound=ModelDefaults)


class Model(Generic[D]):
    """One curated checkpoint row: a stable ``id`` (its wire value + the
    scheduler's warm-pool key), the structured :class:`ModelRef` it resolves
    to, and its typed :class:`ModelDefaults`. Optional ``hot`` (keep a warm
    pool for it) and ``price`` (per-model pricing) hints ride along into the
    manifest.

    Assign one per :class:`ModelChoice` enum member; the base's ``__new__``
    unpacks it (``_value_ = id``)."""

    __slots__ = ("id", "ref", "defaults", "hot", "price")

    id: str
    ref: ModelRef
    defaults: D
    hot: bool
    price: Optional[float]

    def __init__(
        self,
        id: str,
        ref: ModelRef,
        defaults: D,
        *,
        hot: bool = False,
        price: Optional[float] = None,
    ) -> None:
        cid = str(id or "").strip()
        if not cid:
            raise ValueError("Model requires a non-empty id")
        if not isinstance(ref, ModelRef):
            raise TypeError(
                f"Model({cid!r}) ref must be a ModelRef (Hub/HF/Civitai/"
                f"ModelScope), got {type(ref).__name__}"
            )
        if not isinstance(defaults, ModelDefaults):
            raise TypeError(
                f"Model({cid!r}) defaults must be a ModelDefaults subclass, "
                f"got {type(defaults).__name__}"
            )
        self.id = cid
        self.ref = ref
        self.defaults = defaults
        self.hot = bool(hot)
        self.price = price


class ModelChoice(Generic[D]):
    """Mixin for a curated model enum. Combine with ``str`` + ``enum.Enum``
    and parametrize with the endpoint's :class:`ModelDefaults` subclass::

        class SdxlModel(ModelChoice[SdxlDefaults], enum.Enum): ...

    Every member is a :class:`Model` row; the mixin exposes it as typed
    attributes (``.ref``, ``.defaults``, ``.hot``, ``.price``) and provides
    :meth:`rows` for discovery introspection. The enum's wire value is the
    member id, so a payload field typed with the subclass encodes to that
    string and its JSON schema is a closed ``enum`` (the curated allowlist)."""

    _model_: "Model[D]"

    def __new__(cls, model: "Model[D]") -> "ModelChoice[D]":
        if not isinstance(model, Model):
            raise TypeError(
                f"{cls.__name__} members must be Model(...) rows, got "
                f"{type(model).__name__}"
            )
        obj = object.__new__(cls)
        obj._value_ = model.id  # type: ignore[attr-defined]
        obj._model_ = model
        return obj

    @property
    def id(self) -> str:
        return self._model_.id

    @property
    def ref(self) -> ModelRef:
        """The structured :class:`ModelRef` this pick resolves to."""
        return self._model_.ref

    @property
    def defaults(self) -> D:
        """This checkpoint's typed per-model inference defaults."""
        return self._model_.defaults

    @property
    def hot(self) -> bool:
        return self._model_.hot

    @property
    def price(self) -> Optional[float]:
        return self._model_.price

    @classmethod
    def rows(cls) -> "tuple[Model[Any], ...]":
        """Every curated :class:`Model` row, declaration order — the set
        discovery emits and the scheduler warm-pools over."""
        return tuple(member._model_ for member in cls.__members__.values())  # type: ignore[attr-defined]


def is_model_choice(t: Any) -> bool:
    """True when ``t`` is a concrete :class:`ModelChoice` enum subclass."""
    return isinstance(t, type) and issubclass(t, ModelChoice) and issubclass(t, enum.Enum)


__all__ = ["Model", "ModelChoice", "ModelDefaults", "ModelRef"]
