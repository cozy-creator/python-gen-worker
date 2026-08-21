"""``Resources`` — the machine envelope ONE declaration states."""

from __future__ import annotations

from typing import Any, Dict

import msgspec

from ..models.tensor_layout_contract import (
    LayoutRequirements, parse_layout_requirements,
)


def force(obj: Any, field: str, value: Any) -> None:
    """Assign through a frozen msgspec.Struct's normalization."""
    object.__setattr__(obj, field, value)


_PARALLEL_MECHANISMS = frozenset({"sequence", "cfg"})


class Resources(msgspec.Struct, frozen=True, omit_defaults=True):
    """Hardware envelope for one function: ONLY what the endpoint CANNOT RUN WITHOUT — ``Resources(gpu, gpu_count, libraries, vcpus)``."""

    gpu: bool = False
    gpu_count: int = 1
    libraries: tuple[str, ...] = ()
    vcpus: int | None = None
    requires: Any = None
    max_gpu_count: int | None = None
    max_gpus_per_execution_group: int | None = None
    parallel: tuple[str, ...] = ()

    def manifest_dict(self) -> Dict[str, Any]:
        """The manifest ``resources{}`` projection."""
        raw = msgspec.to_builtins(self)
        out: Dict[str, Any] = dict(raw) if isinstance(raw, dict) else {}
        out.pop("requires", None)
        requirement = self.requirement()
        if requirement is not None:
            out["requires"] = requirement.manifest_row()
        return out

    def requirement(self) -> LayoutRequirements | None:
        """The parsed function-scope requirement, or None if undeclared."""
        return self.requires if isinstance(
            self.requires, LayoutRequirements) else None

    def __post_init__(self) -> None:
        force = msgspec.structs.force_setattr
        if self.libraries:
            force(self, "libraries", tuple(
                str(x).strip() for x in self.libraries if str(x).strip()
            ))
        n_gpu = int(self.gpu_count)
        if n_gpu <= 0:
            raise ValueError(f"gpu_count must be positive, got {self.gpu_count}")
        force(self, "gpu_count", n_gpu)
        if self.requires is not None:
            force(self, "requires", parse_layout_requirements(
                self.requires, where="Resources(requires=)"))
        requirement = self.requirement()
        implies_gpu = requirement is not None and bool(
            requirement.min_terms().min_sm
            or requirement.min_terms().min_vram_gb
            or requirement.recommended_terms().min_sm
            or requirement.recommended_terms().min_vram_gb)
        if n_gpu > 1 or implies_gpu:
            force(self, "gpu", True)
        if self.vcpus is not None:
            c = int(self.vcpus)
            if c <= 0:
                raise ValueError(f"vcpus must be positive, got {c}")
            force(self, "vcpus", c)
        if self.max_gpu_count is not None:
            m = int(self.max_gpu_count)
            if m != self.max_gpu_count or m <= 0:
                raise ValueError(
                    f"max_gpu_count must be a positive whole number of "
                    f"devices, got {self.max_gpu_count}")
            if m < n_gpu:
                raise ValueError(
                    f"max_gpu_count {m} is below gpu_count {n_gpu}")
            force(self, "max_gpu_count", m)
            force(self, "gpu", True)
        if self.max_gpus_per_execution_group is not None:
            d = int(self.max_gpus_per_execution_group)
            if d != self.max_gpus_per_execution_group:
                raise ValueError(
                    f"max_gpus_per_execution_group must be a whole number of "
                    f"devices, got {self.max_gpus_per_execution_group}")
            if d < 2:
                raise ValueError(
                    f"max_gpus_per_execution_group {d} declares a sharding "
                    f"width that does not shard; omit the field to let the hub "
                    f"use the whole max_gpu_count as one group")
            if self.max_gpu_count is None or d > self.max_gpu_count:
                raise ValueError(
                    f"max_gpus_per_execution_group {d} exceeds max_gpu_count "
                    f"{self.max_gpu_count} — one request cannot span more GPUs "
                    f"than the pod may hold")
            force(self, "max_gpus_per_execution_group", d)
            force(self, "gpu", True)
        if self.parallel:
            mechanisms = tuple(
                str(x).strip().lower() for x in self.parallel if str(x).strip()
            )
            unknown = [m for m in mechanisms if m not in _PARALLEL_MECHANISMS]
            if unknown:
                raise ValueError(
                    f"parallel mechanism(s) {unknown} not implemented by the "
                    f"platform; known: {sorted(_PARALLEL_MECHANISMS)}")
            if len(set(mechanisms)) != len(mechanisms):
                raise ValueError(f"parallel repeats a mechanism: {mechanisms}")
            if self.max_gpu_count is None or self.max_gpu_count <= n_gpu:
                raise ValueError(
                    f"parallel {list(mechanisms)} declared without a "
                    f"max_gpu_count above gpu_count {n_gpu} — a mechanism "
                    f"is only reachable with device headroom")
            force(self, "parallel", mechanisms)
            force(self, "gpu", True)
        elif self.max_gpus_per_execution_group is not None:
            raise ValueError(
                f"max_gpus_per_execution_group "
                f"{self.max_gpus_per_execution_group} declared without a "
                f"parallel mechanism — nothing shards a group, so the "
                f"declaration is inert")




__all__ = ["Resources"]
