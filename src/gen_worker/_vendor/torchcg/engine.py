"""Local-first compile, store, resolve, and admission lifecycle."""

from __future__ import annotations

import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

from tensorfs import LocalCAS

from .artifact import build_metadata, pack_artifact, read_metadata
from .compiler import _compile_exported_program, _package_compiled_files
from .declaration import (
    GraphClassDeclaration,
    GraphClassSpec,
    RuntimeCompatibility,
    _literal_digest_for,
)
from .host_isa import HostISAError, _admit_host
from .identity import GRAPH_CLASS_BLOCK, CompiledGraphKey, toolchain_axis_digest
from .introspection import DeclaredConstant, declared_constants
from .runner import CompiledGraphRunner
from .storage import (
    QuarantinedArtifact,
    StorageError,
    StoredCompiledGraph,
    StoreOutcome,
    StoreResult,
    _CompiledGraphStore,
)


class AdmissionError(StorageError):
    """Materialized bytes do not exactly satisfy their requested graph plan."""


class EnsureOutcome(StrEnum):
    REUSED = "reused"
    MINTED = "minted"


@dataclass(frozen=True, slots=True)
class _GraphClassPlan:
    """The one declaration and exact key used by both resolve and mint."""

    spec: GraphClassSpec
    declaration: GraphClassDeclaration
    runtime: RuntimeCompatibility
    key: CompiledGraphKey


@dataclass(frozen=True, slots=True)
class EnsureResult:
    outcome: EnsureOutcome
    compiled_graph: StoredCompiledGraph
    publication: StoreOutcome | None = None


def _literal_value(program: object, constant: DeclaredConstant) -> Any:
    values = getattr(program, "constants", {}) or {}
    if constant.fqn in values:
        return values[constant.fqn]
    if constant.name in values:
        return values[constant.name]
    raise AdmissionError(f"declared literal constant {constant.fqn!r} has no exported value")


def _write_literals(
    program: object, constants: tuple[DeclaredConstant, ...], target: Path
) -> str:
    literals = sorted(
        (constant for constant in constants if constant.source == "literal"),
        key=lambda constant: constant.fqn,
    )
    if not literals:
        return ""
    try:
        from safetensors import TensorSpec, serialize_file
    except ImportError as exc:  # pragma: no cover - package dependency
        raise AdmissionError("safetensors is required for literal graph constants") from exc
    tensors = {
        constant.fqn: _literal_value(program, constant).detach().cpu().contiguous()
        for constant in literals
    }
    try:
        specs = {
            name: TensorSpec(
                dtype=str(tensor.dtype).removeprefix("torch."),
                shape=list(tensor.shape),
                data_ptr=tensor.data_ptr(),
                data_len=tensor.numel() * tensor.element_size(),
            )
            for name, tensor in tensors.items()
        }
        serialize_file(specs, str(target))
    except Exception as exc:
        raise AdmissionError(f"cannot serialize literal graph constants: {exc}") from exc
    return _literal_digest_for(program, (constant.fqn for constant in literals))


def _compile_package(plan: _GraphClassPlan, workspace: Path) -> Path:
    files = _compile_exported_program(plan.spec.program)
    return _package_compiled_files(
        plan.declaration.graph_class,
        files,
        workspace / "model.pt2",
    )


def _program_state_dict_fqns(program: object) -> set[str]:
    signature = getattr(program, "graph_signature", None)
    return {
        str(name)
        for field in ("parameters", "buffers")
        for name in (getattr(signature, field, ()) or ())
    }


def _admit_constant_table(plan: _GraphClassPlan, constants: tuple[DeclaredConstant, ...]) -> None:
    """Prove the package did not invent a constant or compile in a weight."""

    stated = plan.declaration.graph.get("constant_fqns")
    if not isinstance(stated, list) or not all(isinstance(name, str) for name in stated):
        raise AdmissionError("graph declaration has no canonical constant FQN table")
    program = set(stated)
    package = {constant.fqn for constant in constants}
    bindable = {constant.fqn for constant in constants if constant.source != "computed"}
    package_only = sorted(bindable - program)
    if package_only:
        # The two sides are INDEPENDENT witnesses -- the exported program's
        # signature and the AOTI package's own table -- which is exactly what
        # makes this check worth having. But independent tools also NAME
        # things differently: a module attribute torch lifts as `const` can
        # come back from the package as `_tensor_constant0`. So the refusal
        # states whose vocabulary each side is speaking, instead of implying
        # the package invented a constant.
        raise AdmissionError(
            f"compiled package declares {len(package_only)} constant(s) that are not "
            f"in the exported program's lifted set: {package_only[:6]!r}. "
            f"Package vocabulary: {sorted(package)[:6]!r}; export vocabulary: "
            f"{sorted(program)[:6]!r}. A pure NAMING divergence (torch's "
            f"`const` vs inductor's `_tensor_constant0`) is a bare, unnamed "
            f"module attribute; a named attribute keeps its fqn on both sides."
        )
    folded_weights = sorted(_program_state_dict_fqns(plan.spec.program) - package)
    if folded_weights:
        raise AdmissionError(
            f"compiled package eliminated {len(folded_weights)} state-dict constant(s): "
            f"{folded_weights[:6]!r}; their checkpoint values may be compiled into code"
        )


class Engine:
    """One local compiled-graph engine backed exclusively by one tensorfs store."""

    def __init__(self, cas: LocalCAS) -> None:
        self._store = _CompiledGraphStore(cas)

    @staticmethod
    def _plan(spec: GraphClassSpec, runtime: RuntimeCompatibility) -> _GraphClassPlan:
        declaration = spec.declare()
        return _GraphClassPlan(spec, declaration, runtime, runtime.key(declaration))

    @staticmethod
    def _admit_host_metadata(metadata: Mapping[str, object]) -> None:
        host_isa = metadata.get("host_isa")
        if not isinstance(host_isa, Mapping):
            raise AdmissionError("artifact records no host_isa")
        try:
            _admit_host(cast(Mapping[str, str], host_isa))
        except HostISAError as exc:
            raise AdmissionError(f"artifact host ISA is unsupported: {exc}") from exc

    @staticmethod
    def _admit(plan: _GraphClassPlan, compiled_graph: StoredCompiledGraph) -> None:
        metadata = compiled_graph.metadata
        graph_class = metadata.get(GRAPH_CLASS_BLOCK)
        if not isinstance(graph_class, Mapping):
            raise AdmissionError("compiled graph records no graph_class")
        expected_graph_class: dict[str, object] = {
            "name": plan.declaration.graph_class,
            "target": plan.declaration.target,
            "class_hash": plan.declaration.class_hash,
            "graph": dict(plan.declaration.graph),
            "graph_witness": plan.declaration.graph_witness,
            "range_digest": plan.declaration.range_digest,
            "fork": [[name, value] for name, value in plan.declaration.fork],
            "class_dims": [[name, value] for name, value in plan.declaration.class_dims],
            "strict": plan.declaration.strict,
            "lora_bucket": plan.declaration.lora_bucket,
            "literal_values": plan.declaration.literal_values,
            "placement": list(plan.declaration.placement),
        }
        mismatches = [
            field
            for field, expected in expected_graph_class.items()
            if graph_class.get(field) != expected
        ]
        if metadata.get("sm") != plan.runtime.sm:
            mismatches.append("sm")
        stored_toolchain = metadata.get("toolchain")
        if not isinstance(stored_toolchain, Mapping) or toolchain_axis_digest(
            stored_toolchain
        ) != toolchain_axis_digest(plan.runtime.toolchain):
            mismatches.append("toolchain")
        if metadata.get("compiled_graph_key") != str(plan.key):
            mismatches.append("compiled_graph_key")
        if mismatches:
            raise AdmissionError(
                "artifact does not satisfy the exact graph plan: " + ", ".join(mismatches)
            )

    def resolve(
        self, key: str | CompiledGraphKey, destination: str | Path
    ) -> StoredCompiledGraph | None:
        """Resolve and verify an exact key without importing Torch or building a program."""

        graph = self._store.resolve(key, destination)
        if graph is None:
            return None
        self._admit_host_metadata(graph.metadata)
        return graph

    @staticmethod
    def _admit_request(
        compiled_graph: StoredCompiledGraph,
        *,
        target: str,
        toolchain: Mapping[str, str],
    ) -> None:
        requested_target = str(target).strip().lower()
        requested_toolchain = {str(name): str(value) for name, value in toolchain.items()}
        if (
            not requested_target
            or not requested_toolchain
            or any(
                not name or name != name.strip() or not value or value != value.strip()
                for name, value in requested_toolchain.items()
            )
        ):
            raise AdmissionError("ensure requires target and a recorded toolchain block")
        stored_target = compiled_graph.metadata.get("sm")
        target_matches = stored_target == requested_target or (
            requested_target == "cpu"
            and isinstance(stored_target, str)
            and stored_target.startswith("cpu-")
        )
        if not target_matches:
            raise AdmissionError(
                f"stored target {stored_target!r} does not match requested {requested_target!r}"
            )
        stored_toolchain = compiled_graph.metadata.get("toolchain")
        if not isinstance(stored_toolchain, Mapping) or toolchain_axis_digest(
            stored_toolchain
        ) != toolchain_axis_digest(requested_toolchain):
            raise AdmissionError(
                "stored toolchain axis does not match the requested compiler-content facts"
            )

    def import_artifact(self, key: str | CompiledGraphKey, artifact: str | Path) -> StoreResult:
        """Fully verify and attach bytes admitted to the store under one exact key."""

        self._admit_host_metadata(read_metadata(artifact))
        return self._store.store(key, artifact)

    def export_artifact(self, key: str | CompiledGraphKey, destination: str | Path) -> Path:
        """Export a fully verified artifact without exposing the store's ref layout."""

        return self._store.export_artifact(key, destination)

    def runner(
        self, key: str | CompiledGraphKey, destination: str | Path
    ) -> CompiledGraphRunner | None:
        """Resolve the exact selected artifact and load its gated AOTI runner."""

        graph = self.resolve(key, destination)
        return None if graph is None else CompiledGraphRunner._from_verified_graph(graph)

    def _mint(self, plan: _GraphClassPlan) -> StoreResult:
        """Compile, package, verify, and publish one plan into the local CAS."""

        with tempfile.TemporaryDirectory(prefix="torchcg-mint-") as raw:
            workspace = Path(raw)
            package = _compile_package(plan, workspace)
            constants = declared_constants(package, plan.declaration.graph_class)
            _admit_constant_table(plan, constants)
            literals = workspace / "constants.safetensors"
            literal_payload_values = _write_literals(plan.spec.program, constants, literals)
            if plan.spec.declare() != plan.declaration:
                raise AdmissionError(
                    "exported program changed during compilation, packaging, "
                    "or literal serialization"
                )
            metadata = build_metadata(
                graph_class={
                    "name": plan.declaration.graph_class,
                    "target": plan.declaration.target,
                    "class_hash": plan.declaration.class_hash,
                    "graph": dict(plan.declaration.graph),
                    "graph_witness": plan.declaration.graph_witness,
                    "range_digest": plan.declaration.range_digest,
                    "fork": [[name, value] for name, value in plan.declaration.fork],
                    "class_dims": [[name, value] for name, value in plan.declaration.class_dims],
                    "strict": plan.declaration.strict,
                    "lora_bucket": plan.declaration.lora_bucket,
                    "literal_values": plan.declaration.literal_values,
                    "literal_payload_values": literal_payload_values,
                    "placement": list(plan.declaration.placement),
                    "constants": [constant.as_manifest_row() for constant in constants],
                },
                sm=plan.runtime.sm,
                toolchain=plan.runtime.toolchain,
                host_isa=plan.runtime.host_isa,
            )
            artifact = pack_artifact(
                package,
                workspace / "compiled_graph.tar.gz",
                metadata,
                literals=literals if literals.is_file() else None,
            )
            return self._store.store(plan.key, artifact)

    def compile(
        self,
        spec: GraphClassSpec,
        runtime: RuntimeCompatibility,
        destination: str | Path,
    ) -> EnsureResult:
        """Compile or reuse one self-derived graph class under sealed policy."""

        plan = self._plan(spec, runtime)
        try:
            existing = self.resolve(plan.key, destination)
        except QuarantinedArtifact:
            existing = None
        if existing is not None:
            self._admit(plan, existing)
            return EnsureResult(EnsureOutcome.REUSED, existing)

        publication = self._mint(plan)
        resolved = self.resolve(plan.key, destination)
        if resolved is None:
            raise StorageError(f"compiled graph {plan.key} disappeared after publication")
        self._admit(plan, resolved)
        outcome = (
            EnsureOutcome.REUSED
            if publication.outcome == StoreOutcome.DIVERGENT
            else EnsureOutcome.MINTED
        )
        return EnsureResult(outcome, resolved, publication.outcome)

    def ensure(
        self,
        key: str | CompiledGraphKey,
        destination: str | Path,
        *,
        target: str,
        toolchain: Mapping[str, str],
        recipe: Callable[[], GraphClassSpec],
    ) -> EnsureResult:
        """Reuse an admitted exact key, compiling only on a miss or quarantine."""

        requested = str(key)
        try:
            existing = self.resolve(requested, destination)
        except QuarantinedArtifact:
            existing = None
        if existing is not None:
            self._admit_request(
                existing,
                target=target,
                toolchain=toolchain,
            )
            return EnsureResult(EnsureOutcome.REUSED, existing)

        spec = recipe()
        runtime = RuntimeCompatibility(target, toolchain=toolchain)
        plan = self._plan(spec, runtime)
        if str(plan.key) != requested:
            raise AdmissionError(f"lazy recipe derives {plan.key}, not requested key {requested!r}")
        publication = self._mint(plan)
        resolved = self.resolve(plan.key, destination)
        if resolved is None:
            raise StorageError(f"compiled graph {plan.key} disappeared after publication")
        self._admit(plan, resolved)
        outcome = (
            EnsureOutcome.REUSED
            if publication.outcome == StoreOutcome.DIVERGENT
            else EnsureOutcome.MINTED
        )
        return EnsureResult(outcome, resolved, publication.outcome)
