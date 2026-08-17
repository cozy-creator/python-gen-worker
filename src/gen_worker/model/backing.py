"""The three backings a typed runner callable can have, under one signature.

A typed callable's CONTRACT is its ingress digest. Its BACKING is whichever of
these is live, and **handler code never knows which** — that is the whole of
pgw#1326's dual backing, expressed as a type rather than as a branch every
handler has to remember to write.

* :class:`EagerBacking` calls the family's own constructor output — the same
  module the mint traces. First serve of a family on a trusted eager-capable
  pod runs here and mints as a side effect (DESIGN-RULINGS §4.28), so nothing
  waits on a compile and nobody requests one.
* :class:`CompiledBacking` calls an adopted AOTI runner through the declared
  ingress's own flattening. Feeds are resolved by REPLAYING each declared
  input's recorded param/path coordinate, never by searching — a dict carrying
  a colliding key must not be able to decide a bind.
* :class:`FakeBacking` returns shape-correct deterministic tensors derived from
  the declaration (greenfield B8). It needs no hub, no weights and no GPU, so
  payload validation, bucket mapping, composition, streaming and asset saving
  are all testable in CI, and cozy-local parity falls out.

:class:`DualBacking` is the pair: compiled where a runner is armed, eager
everywhere else, decided per (runner, bucket, layout). That is exactly today's
``arm_compiled_graph`` hot-swap and its per-graph-class sticky de-arm, moved
behind the typed surface instead of monkey-patching ``module.forward``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from .._vendor.torchcg.ingress import IngressError
from .errors import ModelError, ModelRefusal
from .snapshot import ExportedVariant


class BackingKind(StrEnum):
    """Which implementation is answering a typed runner call."""

    EAGER = "eager"
    COMPILED = "compiled"
    FAKE = "fake"


@runtime_checkable
class Backing(Protocol):
    """What every backing must be able to do: answer one declared call."""

    @property
    def kind(self) -> BackingKind: ...

    def serves(self, runner: str, variant: ExportedVariant) -> bool:
        """Whether this backing can answer THIS variant, not merely this runner."""
        ...

    def invoke(
        self,
        runner: str,
        variant: ExportedVariant,
        args: tuple[object, ...],
        kwargs: Mapping[str, object],
    ) -> Any: ...


class EagerBacking:
    """Calls the family's own eager module for a runner.

    The module is the SAME object the declaration's ``build`` produces and the
    mint traces, which is what makes eager and compiled two readings of one
    declaration rather than two implementations that can drift. Numerics are
    close, not bitwise — today's accepted trade (pgw#1326 escape hatch 0).
    """

    def __init__(self, modules: Mapping[str, Any]) -> None:
        self._modules = dict(modules)

    @property
    def kind(self) -> BackingKind:
        return BackingKind.EAGER

    @property
    def runners(self) -> tuple[str, ...]:
        return tuple(sorted(self._modules))

    def serves(self, runner: str, variant: ExportedVariant) -> bool:
        # An eager module is bucket- and layout-agnostic: it is the whole
        # architecture, and a bucket is a shape the call carries, not a
        # different object. So the question is only whether the module is here.
        del variant
        return runner in self._modules

    def invoke(
        self,
        runner: str,
        variant: ExportedVariant,
        args: tuple[object, ...],
        kwargs: Mapping[str, object],
    ) -> Any:
        del variant
        module = self._modules.get(runner)
        if module is None:
            raise ModelError(
                ModelRefusal.BACKING_MISSING,
                f"no eager module bound for runner {runner!r}; it has "
                f"{sorted(self._modules)!r}",
            )
        return module(*args, **dict(kwargs))


class CompiledBacking:
    """Calls adopted AOTI runners, one per (runner, bucket, layout).

    The key is the VARIANT, not the runner: a compiled backing accepts exactly
    its traced layout and exactly its traced bucket (torchcg G15), so a
    runner armed at 1024 does not answer a 512 call — the dual backing serves
    that one eagerly instead of quietly running the wrong graph.
    """

    def __init__(
        self,
        runners: Mapping[tuple[str, tuple[tuple[str, int], ...], str], Any],
    ) -> None:
        self._runners = dict(runners)

    @property
    def kind(self) -> BackingKind:
        return BackingKind.COMPILED

    @staticmethod
    def key(
        runner: str, variant: ExportedVariant
    ) -> tuple[str, tuple[tuple[str, int], ...], str]:
        return (
            runner,
            tuple((str(name), value) for name, value in variant.bucket),
            str(variant.layout),
        )

    @property
    def armed(self) -> tuple[tuple[str, tuple[tuple[str, int], ...], str], ...]:
        return tuple(sorted(self._runners))

    def serves(self, runner: str, variant: ExportedVariant) -> bool:
        return self.key(runner, variant) in self._runners

    def invoke(
        self,
        runner: str,
        variant: ExportedVariant,
        args: tuple[object, ...],
        kwargs: Mapping[str, object],
    ) -> Any:
        compiled = self._runners.get(self.key(runner, variant))
        if compiled is None:
            raise ModelError(
                ModelRefusal.BACKING_MISSING,
                f"runner {runner!r} is not armed at bucket "
                f"{ {str(k): v for k, v in variant.bucket} !r} layout {str(variant.layout)!r}",
            )
        try:
            feeds = variant.ingress.feeds(args, dict(kwargs))
        except IngressError as exc:
            raise ModelError(
                ModelRefusal.CALL_INVALID,
                f"runner {runner!r} call does not satisfy its declared ingress: {exc}",
            ) from exc
        return compiled(*feeds)


class FakeBacking:
    """Shape-correct deterministic tensors, derived from the declaration.

    Greenfield B8. Values are a function of (family, runner, bucket, layout,
    output index) and nothing else — no clock, no RNG state, no global seed — so
    two runs of one handler produce identical bytes and a test can assert on
    them. They are NOT model outputs and must never be presented as any: what
    this backing tests is the handler, not the model.

    Symbolic output dimensions are bound from the CALL's own inputs, by
    replaying the ingress's recorded coordinates. A symbol the call does not
    determine is a refusal rather than a guess: a fabricated dimension would
    make every downstream shape assertion vacuous.
    """

    def __init__(self, family: str, *, seed: int = 0) -> None:
        self._family = str(family)
        self._seed = int(seed)

    @property
    def kind(self) -> BackingKind:
        return BackingKind.FAKE

    def serves(self, runner: str, variant: ExportedVariant) -> bool:
        del runner, variant
        return True

    def invoke(
        self,
        runner: str,
        variant: ExportedVariant,
        args: tuple[object, ...],
        kwargs: Mapping[str, object],
    ) -> Any:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - torch is an extra
            raise ModelError(
                ModelRefusal.BACKING_MISSING,
                "the fake backing materializes tensors and needs PyTorch",
            ) from exc
        symbols = self._bind_symbols(runner, variant, args, kwargs)
        results: list[Any] = []
        for index, output in enumerate(variant.outputs):
            shape = tuple(
                dim if isinstance(dim, int) else symbols[dim] for dim in output.shape
            )
            dtype = getattr(torch, output.dtype, None)
            if dtype is None:
                raise ModelError(
                    ModelRefusal.BACKING_MISSING,
                    f"runner {runner!r} output {index} declares dtype {output.dtype!r}, "
                    "which this torch build does not carry",
                )
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self._stream(runner, variant, index))
            count = 1
            for dim in shape:
                count *= int(dim)
            flat = torch.rand(count, generator=generator, dtype=torch.float32)
            results.append(flat.reshape(shape).to(dtype))
        return tuple(results) if len(results) != 1 else results[0]

    def _stream(self, runner: str, variant: ExportedVariant, index: int) -> int:
        material = "|".join(
            (
                self._family,
                runner,
                str(variant.layout),
                ",".join(f"{name}={value}" for name, value in variant.bucket),
                str(index),
                str(self._seed),
            )
        )
        return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big")

    @staticmethod
    def _bind_symbols(
        runner: str,
        variant: ExportedVariant,
        args: tuple[object, ...],
        kwargs: Mapping[str, object],
    ) -> dict[str, int]:
        wanted = {dim for output in variant.outputs for dim in output.shape if isinstance(dim, str)}
        if not wanted:
            return {}
        bound: dict[str, int] = {}
        for spec in variant.ingress.inputs:
            found, value = spec.resolve(args, dict(kwargs))
            shape = getattr(value, "shape", None) if found else None
            if shape is None:
                continue
            for declared, actual in zip(spec.shape, tuple(shape), strict=False):
                if isinstance(declared, str) and declared in wanted:
                    bound.setdefault(declared, int(actual))
        missing = sorted(wanted - set(bound))
        if missing:
            raise ModelError(
                ModelRefusal.CALL_INVALID,
                f"runner {runner!r} returns a tensor whose dimension {missing[0]!r} the call "
                "does not determine; the fake backing refuses rather than invent one",
            )
        return bound


class DualBacking:
    """Compiled where armed, eager everywhere else — decided per variant.

    This is the object that makes ``@endpoint(families=...)`` work on a pod with
    no artifacts and on one with a full keyset, from ONE handler body. It never
    mints and never asks anybody to: a miss is served eagerly here (trusted,
    eager-capable pod) or refused by the adopt-only role, and the mint happens
    as a side effect of serving (DESIGN-RULINGS §4.28/§4.30).
    """

    def __init__(self, eager: Backing | None, compiled: Backing | None) -> None:
        if eager is None and compiled is None:
            raise ModelError(
                ModelRefusal.BACKING_MISSING,
                "a dual backing needs at least one of an eager or a compiled half",
            )
        self._eager = eager
        self._compiled = compiled

    @property
    def kind(self) -> BackingKind:
        return BackingKind.COMPILED if self._compiled is not None else BackingKind.EAGER

    @property
    def eager(self) -> Backing | None:
        return self._eager

    @property
    def compiled(self) -> Backing | None:
        return self._compiled

    def kind_for(self, runner: str, variant: ExportedVariant) -> BackingKind:
        """Which half answers THIS variant. Observability, never a branch point.

        Handlers must not read this to decide anything — the whole contract is
        that they cannot tell. It exists so a receipt can say which half ran.
        """

        if self._compiled is not None and self._compiled.serves(runner, variant):
            return self._compiled.kind
        if self._eager is not None and self._eager.serves(runner, variant):
            return self._eager.kind
        raise ModelError(
            ModelRefusal.BACKING_MISSING,
            f"neither half of this backing serves runner {runner!r} at bucket "
            f"{ {str(k): v for k, v in variant.bucket} !r} layout {str(variant.layout)!r}",
        )

    def serves(self, runner: str, variant: ExportedVariant) -> bool:
        return (self._compiled is not None and self._compiled.serves(runner, variant)) or (
            self._eager is not None and self._eager.serves(runner, variant)
        )

    def invoke(
        self,
        runner: str,
        variant: ExportedVariant,
        args: tuple[object, ...],
        kwargs: Mapping[str, object],
    ) -> Any:
        if self._compiled is not None and self._compiled.serves(runner, variant):
            return self._compiled.invoke(runner, variant, args, kwargs)
        if self._eager is not None and self._eager.serves(runner, variant):
            return self._eager.invoke(runner, variant, args, kwargs)
        raise ModelError(
            ModelRefusal.BACKING_MISSING,
            f"no backing serves runner {runner!r} at bucket "
            f"{ {str(k): v for k, v in variant.bucket} !r} layout {str(variant.layout)!r}; "
            "an adopt-only pod refuses and the hub routes — it never mints on demand",
        )


__all__ = [
    "Backing",
    "BackingKind",
    "CompiledBacking",
    "DualBacking",
    "EagerBacking",
    "FakeBacking",
]
