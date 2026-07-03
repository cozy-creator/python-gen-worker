from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomllib
import re


_RE_CLAUSE = re.compile(r"^\s*(>=|<=|==|~=|>|<)?\s*([0-9]+(?:\.[0-9]+)*)\s*$")
_RE_VERSION_PREFIX = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)*)")
_RE_MODEL_SEGMENT = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


@dataclass(frozen=True)
class EndpointResources:
    """Endpoint-level hardware spec from endpoint.toml [resources] (tensorhub #232).

    Single source of truth for hardware. Size axes (vram_gb, gpu_count,
    memory_gb, cpu_cores, disk_gb) are invoker-overridable for training
    invocations; architecture axes (accelerator, min_compute_capability) are
    always pinned at the endpoint level.
    """
    accelerator: str = ""                    # "cuda" | "none" | ""
    min_compute_capability: str = ""         # e.g. "8.0"
    vram_gb: int = 0
    gpu_count: int = 0
    memory_gb: int = 0
    cpu_cores: int = 0
    disk_gb: int = 0
    # gen-orchestrator #350: GPU class preference + per-request override
    # policy. Both ride the same [resources] block as the size axes.
    gpu_tier: str = ""                       # e.g. "RTX 4090", "A100-80", "H100"
    invoker_compute_override: str = ""       # "" (default = size-only), "allowed", "size-only", "none"

    def to_dict(self) -> dict[str, Any]:
        """Serialize non-default fields for the manifest wire shape."""
        out: dict[str, Any] = {}
        if self.accelerator:
            out["accelerator"] = self.accelerator
        if self.min_compute_capability:
            out["min_compute_capability"] = self.min_compute_capability
        if self.vram_gb:
            out["vram_gb"] = self.vram_gb
        if self.gpu_count:
            out["gpu_count"] = self.gpu_count
        if self.memory_gb:
            out["memory_gb"] = self.memory_gb
        if self.cpu_cores:
            out["cpu_cores"] = self.cpu_cores
        if self.disk_gb:
            out["disk_gb"] = self.disk_gb
        if self.gpu_tier:
            out["gpu_tier"] = self.gpu_tier
        if self.invoker_compute_override:
            out["invoker_compute_override"] = self.invoker_compute_override
        return out


@dataclass(frozen=True)
class EndpointToml:
    schema_version: int
    name: str
    main: str
    cuda: str | None
    compute_capabilities: tuple[str, ...]
    resources: EndpointResources


def _validate_constraint(raw: str, *, field: str) -> None:
    s = (raw or "").strip()
    if not s:
        raise ValueError(f"{field} constraint cannot be empty")
    for clause in s.split(","):
        clause = clause.strip()
        if not clause:
            continue
        if not _RE_CLAUSE.match(clause):
            raise ValueError(f"invalid {field} constraint clause: {clause!r}")

def _parse_compute_capabilities(raw: Any, *, field: str) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list) or not all(isinstance(x, str) for x in raw):
        raise ValueError(f"{field} must be a list of strings")
    out: list[str] = []
    for item in raw:
        token = item.strip()
        if not token:
            continue
        # Exact/wildcard major forms: 8.0, 8.x
        if re.match(r"^[0-9]+\.(?:[0-9]+|x)$", token):
            out.append(token)
            continue
        # Range/comparator forms: >=12.0,<13.0
        ok = True
        for clause in token.split(","):
            clause = clause.strip()
            if not clause:
                continue
            if not _RE_CLAUSE.match(clause):
                ok = False
                break
        if not ok:
            raise ValueError(
                f"invalid {field} entry {token!r}; expected forms like '8.0', '8.x', or '>=12.0,<13.0'"
            )
        out.append(token)
    return tuple(out)


def load_endpoint_toml(path: Path) -> EndpointToml:
    out, _ = load_endpoint_toml_with_warnings(path)
    return out


def load_endpoint_toml_with_warnings(path: Path) -> tuple[EndpointToml, list[str]]:
    """Same as load_endpoint_toml but also returns a list of non-fatal
    warnings (e.g. deprecated `dtypes=[...]` shape usage). The warnings flow
    into publish-time diagnostics in tensorhub.
    """
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("endpoint.toml must be a TOML table at root")
    warnings: list[str] = []

    schema_version = data.get("schema_version")
    if schema_version != 1:
        raise ValueError("endpoint.toml schema_version must be 1")

    name = str(data.get("name") or "").strip()
    main = str(data.get("main") or "").strip()
    if not main:
        raise ValueError("endpoint.toml missing main")

    cuda: str | None = None
    compute_capabilities: tuple[str, ...] = ()
    host = data.get("host")
    if isinstance(host, dict):
        req = host.get("requirements")
        if isinstance(req, dict):
            raw = req.get("cuda")
            if isinstance(raw, str) and raw.strip():
                cuda = raw.strip()
                _validate_constraint(cuda, field="cuda")
            if "cuda_compute_capabilities" in req:
                raise ValueError(
                    "host.requirements.cuda_compute_capabilities is no longer supported; "
                    "use host.requirements.compute_capabilities"
                )
            compute_capabilities = _parse_compute_capabilities(
                req.get("compute_capabilities"),
                field="host.requirements.compute_capabilities",
            )

    if "models" in data:
        # 0.7.0 hard cut: [models] and [models.<fn>] toml tables were replaced
        # by the Python-side @inference(models={...}) binding kwarg.
        # Fail loud — tenants must migrate.
        raise ValueError(
            "endpoint.toml: [models] / [models.<fn>] tables were removed in "
            "gen-worker 0.7.0. Declare model bindings in Python via "
            "@inference(models={...}) with Repo / Dispatch."
        )

    # Reject legacy endpoint.toml blocks that became platform-controlled.
    _REJECTED_ENDPOINT_BLOCKS = {
        "scaling": (
            "autoscaling is platform-controlled; remove the [scaling] block from "
            "endpoint.toml. If you need policy knobs, request them via platform config."
        ),
    }
    for block_name, message in _REJECTED_ENDPOINT_BLOCKS.items():
        if block_name in data:
            raise ValueError(f"endpoint.toml: [{block_name}] is no longer accepted — {message}")

    if "functions" in data:
        raise ValueError(
            "endpoint.toml: function metadata sections are no longer accepted; "
            "declare function metadata in Python decorators"
        )

    # Endpoint-level [resources] is the default.
    raw_resources = data.get("resources")
    res_kwargs: dict[str, Any] = {}
    if isinstance(raw_resources, dict):
        # gen-orchestrator #350: gpu_tier + invoker_compute_override are
        # string fields alongside accelerator/min_compute_capability.
        for k in ("accelerator", "min_compute_capability", "gpu_tier", "invoker_compute_override"):
            if k in raw_resources:
                res_kwargs[k] = str(raw_resources[k])
        # Validate invoker_compute_override values per the orch's
        # ResourceRequirements.InvokerComputeOverride contract:
        # "" (default = size-only), "allowed", "size-only", "none".
        if res_kwargs.get("invoker_compute_override") not in (None, "", "allowed", "size-only", "none"):
            raise ValueError(
                f"resources.invoker_compute_override must be one of "
                f"'allowed', 'size-only', 'none' (or omitted); "
                f"got {res_kwargs['invoker_compute_override']!r}"
            )
        for k in ("vram_gb", "gpu_count", "memory_gb", "cpu_cores", "disk_gb"):
            if k in raw_resources:
                res_kwargs[k] = int(raw_resources[k])
        # ``ram_gb`` is the legacy name for memory_gb — accept as alias when
        # memory_gb isn't explicitly set.
        if "memory_gb" not in res_kwargs and "ram_gb" in raw_resources:
            res_kwargs["memory_gb"] = int(raw_resources["ram_gb"])
        if "max_inflight_requests" in raw_resources:
            raise ValueError(
                "resources.max_inflight_requests is no longer accepted; "
                "Tensorhub learns scheduling concurrency from runtime observations"
            )
    resources = EndpointResources(**res_kwargs)

    return (
        EndpointToml(
            schema_version=1,
            name=name,
            main=main,
            cuda=cuda,
            compute_capabilities=compute_capabilities,
            resources=resources,
        ),
        warnings,
    )
