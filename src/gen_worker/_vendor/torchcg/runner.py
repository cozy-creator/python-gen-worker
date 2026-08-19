"""Verified loading and constant binding for one compiled graph."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, cast

from .identity import GRAPH_SPECIALIZATION_BLOCK
from .storage import StoredCompiledGraph


class ConstantBindingError(RuntimeError):
    """A loaded graph cannot be made safe to call with the supplied constants."""

    def __init__(self, reason: str, detail: str) -> None:
        self.reason = str(reason)
        super().__init__(detail)


@dataclass(frozen=True, slots=True)
class _Constant:
    fqn: str
    source: str


def _constants(graph: StoredCompiledGraph) -> tuple[_Constant, ...]:
    graph_specialization = graph.metadata[GRAPH_SPECIALIZATION_BLOCK]
    assert isinstance(graph_specialization, Mapping)  # artifact validation owns this boundary
    rows = graph_specialization["constants"]
    assert isinstance(rows, list)
    return tuple(
        _Constant(str(row["fqn"]), str(row["source"])) for row in rows if isinstance(row, Mapping)
    )


def _load_package(path: Path, name: str) -> Any:
    try:
        module = import_module("torch._inductor.package")
        loader = vars(module)["load_package"]
        return cast(Any, loader)(str(path), model_name=name)
    except (ImportError, KeyError, RuntimeError, OSError) as exc:
        raise ConstantBindingError(
            "package_load_failed",
            f"cannot load compiled graph {name!r}: {type(exc).__name__}: {exc}",
        ) from exc


def _load_literals(path: Path | None, device: str) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        module = import_module("safetensors.torch")
        loader = vars(module)["load_file"]
        return dict(cast(Any, loader)(str(path), device=device))
    except (ImportError, KeyError, RuntimeError, OSError) as exc:
        raise ConstantBindingError(
            "literal_load_failed",
            f"cannot load compiled-graph literals: {type(exc).__name__}: {exc}",
        ) from exc


def _is_device_oom(exc: BaseException) -> bool:
    if type(exc).__name__ in ("OutOfMemoryError", "CUDAOutOfMemoryError"):
        return True
    if not isinstance(exc, RuntimeError):
        return False
    detail = str(exc).lower()
    return any(
        phrase in detail
        for phrase in (
            "out of memory",
            "cuda oom",
            "cublas_status_alloc_failed",
            "cudnn_status_alloc_failed",
        )
    )


def _oom_in_chain(exc: BaseException) -> BaseException | None:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        if _is_device_oom(current):
            return current
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return None


class CompiledGraphRunner:
    """One exact AOTI package that cannot run before a complete bind.

    The worker owns ingress selection and eager fallback.  This class owns the
    dangerous AOTI boundary: exact constant-table equality, full update, and
    the lifetime of every by-reference value.
    """

    key: str
    name: str
    calls: int
    _graph: StoredCompiledGraph
    _constants: tuple[_Constant, ...]
    _package: Any
    _bound_values: dict[str, Any]
    _retained_values: dict[str, Any]
    _bound: bool
    _failed: bool

    def __init__(self) -> None:
        raise TypeError("compiled-graph runners are created only by Engine.runner")

    @classmethod
    def _from_verified_graph(cls, graph: StoredCompiledGraph) -> CompiledGraphRunner:
        """Load one graph after Engine has resolved and admitted its exact bytes."""

        self = object.__new__(cls)
        graph_specialization = graph.metadata[GRAPH_SPECIALIZATION_BLOCK]
        assert isinstance(graph_specialization, Mapping)
        self.key = graph.key
        self.name = str(graph_specialization["name"])
        self._graph = graph
        self._constants = _constants(graph)
        try:
            self._package = _load_package(graph.package, self.name)
        except ConstantBindingError as exc:
            oom = _oom_in_chain(exc)
            if oom is None:
                raise
            raise ConstantBindingError(
                "out_of_memory",
                f"loading compiled graph {self.name!r} exhausted device memory "
                f"({type(oom).__name__}: {oom})",
            ) from exc
        self._bound_values = {}
        self._retained_values = {}
        self._bound = False
        self._failed = False
        self.calls = 0
        return self

    @property
    def bound(self) -> bool:
        return self._bound

    @property
    def bound_fqns(self) -> tuple[str, ...]:
        return tuple(sorted(self._bound_values))

    @property
    def declared_fqns(self) -> tuple[str, ...]:
        return tuple(constant.fqn for constant in self._constants)

    def bind(self, state: Mapping[str, Any], *, device: str) -> None:
        """Bind the complete constant table by reference, exactly once."""

        if self._bound:
            raise ConstantBindingError("already_bound", "compiled graph is already bound")
        if self._failed:
            raise ConstantBindingError(
                "binding_failed",
                "compiled graph had a failed partial bind and must be loaded again",
            )
        requested_device = str(device).strip()
        if not requested_device:
            raise ConstantBindingError("device_missing", "constant binding requires a device")

        try:
            actual = {str(name) for name in self._package.get_constant_fqns()}
        except Exception as exc:
            self._failed = True
            oom = _oom_in_chain(exc)
            if oom is not None:
                raise ConstantBindingError(
                    "out_of_memory",
                    f"reading the artifact constant table exhausted device memory "
                    f"({type(oom).__name__}: {oom})",
                ) from exc
            raise ConstantBindingError(
                "constant_table_unreadable",
                f"artifact will not report its constant FQNs: {type(exc).__name__}: {exc}",
            ) from exc
        declared = {constant.fqn for constant in self._constants}
        if actual != declared:
            self._failed = True
            raise ConstantBindingError(
                "constant_set_mismatch",
                "artifact constant table != declared manifest; "
                f"declared-only={sorted(declared - actual)[:6]!r} "
                f"artifact-only={sorted(actual - declared)[:6]!r}",
            )

        try:
            literals = _load_literals(self._graph.literals, requested_device)
        except Exception as exc:
            self._failed = True
            oom = _oom_in_chain(exc)
            if oom is not None:
                raise ConstantBindingError(
                    "out_of_memory",
                    f"loading literal constants exhausted device memory "
                    f"({type(oom).__name__}: {oom})",
                ) from exc
            if isinstance(exc, ConstantBindingError):
                raise
            raise ConstantBindingError(
                "literal_load_failed",
                f"cannot load compiled-graph literals: {type(exc).__name__}: {exc}",
            ) from exc
        values: dict[str, Any] = {}
        missing: list[str] = []
        for constant in self._constants:
            if constant.source == "computed":
                continue
            table = state if constant.source == "state_dict" else literals
            value = table.get(constant.fqn)
            if value is None:
                missing.append(f"{constant.fqn} (source={constant.source})")
                continue
            if constant.source == "state_dict":
                try:
                    if not bool(value.is_contiguous()):
                        value = value.detach().contiguous()
                except AttributeError:
                    pass
                except Exception as exc:
                    self._failed = True
                    oom = _oom_in_chain(exc)
                    reason = "out_of_memory" if oom is not None else "normalization_failed"
                    cause = oom or exc
                    raise ConstantBindingError(
                        reason,
                        f"cannot normalize constant {constant.fqn!r} for by-reference "
                        f"binding ({type(cause).__name__}: {cause})",
                    ) from exc
            values[constant.fqn] = value
        if missing:
            self._failed = True
            raise ConstantBindingError(
                "constant_unresolved",
                f"{len(missing)} declared constant(s) have no value: {sorted(missing)[:6]!r}",
            )

        # AOTI may install some raw pointers before reporting a later failure.
        # Retain every attempted value even on that failure so no installed
        # user-managed pointer can outlive its Python owner.
        self._retained_values = values
        try:
            self._package.load_constants(
                values,
                check_full_update=True,
                user_managed=True,
            )
        except Exception as exc:
            self._failed = True
            oom = _oom_in_chain(exc)
            reason = "out_of_memory" if oom is not None else "injection_failed"
            cause = oom or exc
            raise ConstantBindingError(
                reason,
                f"artifact refused its complete constant update ({type(cause).__name__}: {cause})",
            ) from exc

        self._bound_values = values
        self._bound = True

    def __call__(self, *feeds: object) -> Any:
        if not self._bound:
            raise ConstantBindingError(
                "constants_unbound",
                f"refusing to invoke graph {self.name!r} before complete binding",
            )
        result = self._package(*feeds)
        self.calls += 1
        return result


__all__ = ["CompiledGraphRunner", "ConstantBindingError"]
