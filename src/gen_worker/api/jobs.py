"""``@job`` — run-once functions.

A job is an arbitrary one-off long-running execution for ONE user: a
quantization, a fine-tune, a dataset bake. It is submitted, not invoked; it
runs once on a pod with its own lifecycle; it is billed per GPU-second::

    @job(resources=Resources(gpu=True), env=("HF_TOKEN",), resumable=True)
    def h3_modulation_bake(ctx: JobContext, spec: BakeInput) -> BakeResult: ...

Module-level function, one job per function, its own ``.py`` file encouraged.
**No class and no ``setup()``**: the lifecycle is run-once — the worker parent
(which holds the hub stream) forks a fresh child per job, the child loads what
it needs, executes, publishes, exits. Nothing survives in-process between jobs
by contract; disk caches survive and are the held-open-pod win.

**Portability with ``@endpoint`` is a REQUIREMENT, not a style.** Both
decorators share the ``(ctx, payload: msgspec.Struct) -> Struct`` contract, the
same schema machinery, and the same context surface
(:class:`~gen_worker.request_context.JobContext` is a strict superset of what a
producer-shaped endpoint handler may use, under the same names). Promoting a
job to a bounded-recipe serverless endpoint is wrapping the SAME body in
``@endpoint`` and pricing it — zero body edits. ``tests/test_job_decorator_pgw1294.py``
registers one body under both decorators and runs it under both harnesses.

``publishes=True`` is the ONE declaration that a function may write
tensors/repos to the hub, on ``@job`` and ``@endpoint`` alike. It says MAY
write; the request still says WHERE. Undeclared code reaching the publisher
surface is refused typed at the SDK before a byte moves.

``emits_media=True`` is its SIBLING, and jobs only: this job writes MEDIA (a
report, a contact sheet, a sample grid) rather than — or as well as — a repo.
An endpoint's product IS media, so it declares nothing; a job's is not, so the
hub mints the ``upload_media`` grant only for a job whose release declares it
(th#2069). The two are independent: an eval job that writes no repo declares
``emits_media=True, publishes=False``, and a quality-matrix job declares both.
"""

from __future__ import annotations

import inspect
import typing
from typing import Any, Callable, Mapping, Optional, Sequence, TypeVar, overload

import msgspec

from ..model.bind import Bind
from ..model.runtime import Model
from .decorators import (
    Resources,
    _normalize_families,
    _validate_env_decl,
    _validate_family_params,
)

T = TypeVar("T", bound=Callable[..., Any])

JOB_ATTR = "__gen_worker_job__"

#: Private by default. ``public`` exposes the job's source via the catalog AND
#: third-party submission (payer = the submitting org). Nothing goes public
#: until an abuse/review story exists for third-party-submittable code, but the
#: declaration ships so the gate has something to read.
VISIBILITIES = ("private", "public")


class JobDecl(msgspec.Struct, frozen=True, kw_only=True):
    """Metadata attached by ``@job``; read by the registry walker."""

    name: str
    resources: Resources = msgspec.field(default_factory=Resources)
    env: tuple[str, ...] = ()
    #: The job's retry may reattach its network volume and resume from the
    #: chunk it reached, instead of restarting from zero.
    resumable: bool = False
    visibility: str = "private"
    #: This job MAY write tensors/repos to the hub. The hub mints the
    #: worker-capability grant off this declaration; the SDK refuses the
    #: publisher surface without it.
    publishes: bool = False
    #: This job MAY write media. Same shape, different grant
    #: (``upload_media``), and independent of ``publishes``.
    emits_media: bool = False
    #: pgw#1332: handler parameter -> the generated family TYPE bound to it,
    #: exactly as on ``@endpoint``. Portability is a REQUIREMENT (see the
    #: module docstring), so a body that binds a family instance has to promote
    #: between the two decorators unchanged — a job that could not take one
    #: would make the promotion a rewrite for precisely the endpoints most
    #: likely to want it.
    families: Mapping[str, "Bind[Any]"] = msgspec.field(default_factory=dict)


def _qualname_owner(fn: Callable[..., Any]) -> str:
    """The enclosing CLASS name for a method, else ``""``.

    ``Klass.method`` -> ``"Klass"``; ``outer.<locals>.inner`` -> ``""`` (a
    closure is not a class body).
    """
    parts = str(getattr(fn, "__qualname__", "") or "").split(".")
    for segment in reversed(parts[:-1]):
        if segment == "<locals>":
            return ""
        return segment
    return ""


def _validate_job_shape(
    name: str, fn: Callable[..., Any], families: frozenset[str] = frozenset()
) -> tuple[type, type]:
    """-> (payload_type, output_type). Every refusal names the rule."""
    if inspect.isclass(fn):
        raise TypeError(
            f"@job {name!r}: a job is a module-level FUNCTION, never a class. "
            "Jobs have no setup() and no instance state — the run-once "
            "lifecycle forks a fresh child per job, so nothing survives "
            "in-process between jobs by contract. One job per function, its "
            "own .py file encouraged."
        )
    if not inspect.isfunction(fn):
        raise TypeError(
            f"@job requires a function, got {type(fn).__name__}"
        )
    if inspect.isgeneratorfunction(fn) or inspect.isasyncgenfunction(fn):
        raise TypeError(
            f"@job {name!r}: a job must not be a generator. A job writes "
            "files locally, publishes explicitly (ctx.save_checkpoint / "
            "gen_worker.convert.publish_flavors) and returns ONE result "
            "struct; streaming generators are inference-only."
        )
    owner = _qualname_owner(fn)
    if owner:
        raise TypeError(
            f"@job {name!r}: declared inside class {owner!r}. A job is a "
            "module-level function — there is no setup() and no instance to "
            "carry state, so a job method's `self` could only ever be "
            "meaningless. Move it to module level (its own .py file "
            "encouraged)."
        )

    params = list(inspect.signature(fn).parameters.values())
    extra = [p.name for p in params[2:] if p.name not in families]
    if len(params) < 2 or extra:
        raise TypeError(
            f"@job {name!r}: must accept (ctx, payload) plus declared family "
            f"instances — got params {[p.name for p in params]}. This is the SAME "
            "contract @endpoint functions take, which is what makes a job "
            "promotable to a serverless endpoint without a body edit."
        )
    try:
        hints = typing.get_type_hints(fn, include_extras=False)
    except Exception as exc:  # pragma: no cover - defensive
        raise TypeError(
            f"@job {name!r}: could not resolve type hints: {exc}"
        ) from exc

    payload_type = hints.get(params[1].name)
    if not (isinstance(payload_type, type) and issubclass(payload_type, msgspec.Struct)):
        raise TypeError(
            f"@job {name!r}: payload param {params[1].name!r} must be "
            f"annotated with a msgspec.Struct (got {payload_type!r})"
        )
    output_type = hints.get("return")
    if not (isinstance(output_type, type) and issubclass(output_type, msgspec.Struct)):
        raise TypeError(
            f"@job {name!r}: return type must be a msgspec.Struct (got "
            f"{output_type!r}). Artifacts go through the publish path; the "
            "returned struct is the small result record."
        )
    return payload_type, output_type


@overload
def job(target: T) -> T: ...


@overload
def job(
    *,
    resources: Optional[Resources] = ...,
    env: Optional[Sequence[str]] = ...,
    resumable: bool = ...,
    visibility: str = ...,
    publishes: bool = ...,
    emits_media: bool = ...,
    name: Optional[str] = ...,
    families: Optional[Mapping[str, "type[Model] | Bind[Any]"]] = ...,
) -> Callable[[T], T]: ...


def job(
    target: Optional[T] = None,
    *,
    resources: Optional[Resources] = None,
    env: Optional[Sequence[str]] = None,
    resumable: bool = False,
    visibility: str = "private",
    publishes: bool = False,
    emits_media: bool = False,
    name: Optional[str] = None,
    families: Optional[Mapping[str, "type[Model] | Bind[Any]"]] = None,
) -> Any:
    """The one job decorator. See the module docstring for the shape."""
    if resources is not None and not isinstance(resources, Resources):
        raise TypeError(
            f"@job resources= must be a Resources, got {type(resources).__name__}"
        )
    vis = str(visibility or "").strip().lower()
    if vis not in VISIBILITIES:
        raise ValueError(
            f"@job visibility must be one of {VISIBILITIES}, got {visibility!r}"
        )

    def apply(obj: Any) -> Any:
        declared = str(name or getattr(obj, "__name__", "") or "")
        family_map = _normalize_families(f"job {declared or '<job>'!r}", families)
        _validate_job_shape(declared or "<job>", obj, frozenset(family_map))
        _validate_family_params(
            f"job {declared or '<job>'!r}", obj, family_map, is_method=False
        )
        if getattr(obj, JOB_ATTR, None) is not None:
            raise ValueError(f"@job: {declared!r} already carries a job declaration")
        decl = JobDecl(
            name=declared,
            resources=resources if resources is not None else Resources(),
            env=_validate_env_decl(declared, env),
            resumable=bool(resumable),
            visibility=vis,
            publishes=bool(publishes),
            emits_media=bool(emits_media),
            families=family_map,
        )
        setattr(obj, JOB_ATTR, decl)
        return obj

    if target is not None:
        return apply(target)
    return apply


__all__ = ["JOB_ATTR", "VISIBILITIES", "JobDecl", "job"]
