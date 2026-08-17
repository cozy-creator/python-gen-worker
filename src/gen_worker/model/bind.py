"""``Bind`` — the three axes that belong to the ENDPOINT, not to the model.

pgw#1346 K3. When the retired ``Slot`` was split, most of its fields turned out
to describe the model's own CODE — which tensor layouts it can execute, what
executing them needs of the machine — and those went onto
:class:`~gen_worker.model.spec.ModelSpec`, where two endpoints binding one model
state one demand.

Three did not, because they name something the model cannot know:

* ``selected_by`` names a field of THIS endpoint's payload type. Two endpoints
  binding ``Flux1`` may branch on differently-named fields, and a catalog model
  has no payload to name.
* ``default_checkpoint`` is a code-side bootstrap ref for hub-less runs. It is
  a property of this deployment, not of the architecture.
* ``root`` marks which of several bound models is THE one for an endpoint that
  binds more than one. "Which of my models is primary" is a sentence only the
  endpoint can say.

So the mapping value stays a bare model class in the common case, and grows a
wrapper exactly when one of those three is needed::

    @endpoint(models={"pipeline": Flux1Dev})                      # the common case
    @endpoint(models={"pipeline": Bind(Krea2, selected_by="model")})

This is deliberately NOT ``Slot`` under a new name. It carries no
``pipeline_cls`` (the declaration is the component tree), no ``layouts``
(the model's), and no ``optional`` (still DERIVED from the handler parameter's
own default, so the signature stays the single source of truth).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from .errors import ModelError, ModelRefusal

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..api.binding import ModelRef
    from .runtime import Model

M = TypeVar("M", bound="Model")


@dataclass(frozen=True, slots=True)
class Bind(Generic[M]):
    """One entry of ``@endpoint(models={...})``, with its endpoint-coupled axes.

    ``Bind(Flux1Dev)`` and a bare ``Flux1Dev`` mean exactly the same thing; the
    wrapper exists to carry the three fields below and nothing else.
    """

    #: The generated model class. The handler parameter's annotation must be
    #: this same class — the decorator checks it, so a `Bind` cannot quietly
    #: bind a parameter to a different model than the one it is typed as.
    model: type[M]

    #: The ``str``-typed payload field that branches which checkpoint serves
    #: this parameter. Validated at registration against the handler's payload
    #: type: the field must exist and be plain ``str``. The enum of legal
    #: values is overlaid live by the hub and is never baked into the SDK.
    selected_by: str = ""

    #: The code-side bootstrap ref, and the ONLY resolution source in hub-less
    #: mode. A live hub mapping always wins when present.
    default_checkpoint: "ModelRef | None" = None

    #: THE root model, for the residual ``ctx`` questions that still resolve
    #: against one — ``ctx.defaults``, ``ctx.for_request``. Marking none is
    #: fine and is the normal case: a handler names every model it binds, by
    #: parameter, so unlike a `Slot` declaration there is no ambiguity here for
    #: a root to settle. Marking TWO is a decoration-time error.
    root: bool = False

    def __post_init__(self) -> None:
        from .runtime import Model as _Model

        if not (isinstance(self.model, type) and issubclass(self.model, _Model)):
            raise ModelError(
                ModelRefusal.FAMILY_INVALID,
                f"Bind(model=...) must be a generated model class (a subclass of "
                f"Model), got {self.model!r}. A declaration — ModelSpec / "
                "GraphModelSpec — is what you EXPORT; what an endpoint binds is the "
                "class generated from it.",
            )
        object.__setattr__(self, "selected_by", str(self.selected_by or "").strip())
        object.__setattr__(self, "root", bool(self.root))
        if self.default_checkpoint is not None:
            from ..api.binding import ModelRef as _ModelRef

            if not isinstance(self.default_checkpoint, _ModelRef):
                raise ModelError(
                    ModelRefusal.FAMILY_INVALID,
                    f"Bind(default_checkpoint=...) must be a ModelRef (Hub/HF/"
                    f"Civitai/ModelScope), got {type(self.default_checkpoint).__name__}",
                )


def as_bind(value: Any) -> Bind[Any]:
    """Normalize one ``models={}`` value to a :class:`Bind`.

    A bare model class is the common case and stays the common case; this is
    the ONE place that reading is applied, so no downstream reader has to know
    that a mapping value has two shapes.
    """

    if isinstance(value, Bind):
        return value
    return Bind(model=value)


__all__ = ["Bind", "as_bind"]
