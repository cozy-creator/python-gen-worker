from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import tomllib
import re

from .names import slugify_name

_RE_CLAUSE = re.compile(r"^\s*(>=|<=|==|~=|>|<)?\s*([0-9]+(?:\.[0-9]+)*)\s*$")
_RE_VERSION_PREFIX = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)*)")
_RE_MODEL_SEGMENT = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


@dataclass(frozen=True)
class TensorhubModelSpec:
    """Endpoint.toml [models] entry.

    `flavor` selects a named checkpoint inside the resolved checkpoint group.
    `flavors` is an ordered fallback list. The concrete selector fields are
    kept explicit; there is no free-form attributes bag in endpoint.toml.
    """

    ref: str
    flavor: str = ""
    flavors: tuple[str, ...] = ()
    dtype: str = ""
    file_layout: str = ""
    file_type: str = ""


@dataclass(frozen=True)
class EndpointResources:
    """Endpoint-level hardware spec from endpoint.toml [resources] (tensorhub #232).

    Single source of truth for hardware. Size axes (vram_gb, gpu_count,
    memory_gb, cpu_cores, disk_gb) are invoker-overridable for training
    invocations; architecture axes (accelerator, cuda_compute_min) are
    always pinned at the endpoint level.
    """
    accelerator: str = ""                    # "cuda" | "none" | ""
    cuda_compute_min: str = ""               # e.g. "8.0"
    vram_gb: int = 0
    gpu_count: int = 0
    memory_gb: int = 0
    cpu_cores: int = 0
    disk_gb: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize non-default fields for the manifest wire shape."""
        out: dict[str, Any] = {}
        if self.accelerator:
            out["accelerator"] = self.accelerator
        if self.cuda_compute_min:
            out["cuda_compute_min"] = self.cuda_compute_min
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
        return out


@dataclass(frozen=True)
class EndpointToml:
    schema_version: int
    name: str
    main: str
    cuda: str | None
    compute_capabilities: tuple[str, ...]
    models: dict[str, TensorhubModelSpec]
    # top-level fixed keyspace: model_key -> spec
    function_models: dict[str, dict[str, TensorhubModelSpec]]
    # function payload keyspace: function_name -> model_key -> spec
    function_resources: dict[str, dict[str, Any]]  # function_name -> runtime/resource hints
    function_batch_dimensions: dict[str, str]  # function_name -> payload-root batch dimension path
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

def _parse_version_tuple(raw: str) -> tuple[int, ...]:
    m = _RE_VERSION_PREFIX.match(raw or "")
    if not m:
        raise ValueError(f"invalid version: {raw!r}")
    parts = tuple(int(x) for x in m.group(1).split(".") if x != "")
    if not parts:
        raise ValueError(f"invalid version: {raw!r}")
    return parts


def _cmp_versions(a: tuple[int, ...], b: tuple[int, ...]) -> int:
    n = max(len(a), len(b))
    aa = a + (0,) * (n - len(a))
    bb = b + (0,) * (n - len(b))
    if aa < bb:
        return -1
    if aa > bb:
        return 1
    return 0


def _compatible_upper_bound(v: tuple[int, ...]) -> tuple[int, ...]:
    # PEP 440 compatible release:
    # - ~=1.4    => <2.0
    # - ~=1.4.5  => <1.5.0
    if len(v) == 1:
        return (v[0] + 1,)
    prefix = list(v[:-1])
    prefix[-1] += 1
    return tuple(prefix) + (0,)


def constraint_satisfied(spec: str, version: str) -> bool:
    """
    Best-effort evaluator for simple comparator sets like:
      - ">=0.2.0,<0.3.0"
      - "~=0.2.1"
      - "0.2.1" (treated as ==0.2.1)

    This is intentionally minimal and intended for dev validation only.
    """
    v = _parse_version_tuple(version)
    s = (spec or "").strip()
    if not s:
        return True
    for clause in s.split(","):
        clause = clause.strip()
        if not clause:
            continue
        m = _RE_CLAUSE.match(clause)
        if not m:
            return False
        op = m.group(1) or "=="
        target = _parse_version_tuple(m.group(2))

        c = _cmp_versions(v, target)
        if op == "==":
            if c != 0:
                return False
        elif op == ">=":
            if c < 0:
                return False
        elif op == "<=":
            if c > 0:
                return False
        elif op == ">":
            if c <= 0:
                return False
        elif op == "<":
            if c >= 0:
                return False
        elif op == "~=":
            upper = _compatible_upper_bound(target)
            if _cmp_versions(v, target) < 0:
                return False
            if _cmp_versions(v, upper) >= 0:
                return False
        else:
            return False
    return True


def _parse_model_spec(v: Any, *, warnings: list[str] | None = None) -> TensorhubModelSpec:
    """Parse a single endpoint.toml [models] entry.

    Two accepted input shapes:
      1. Bare string "owner/repo" or "owner/repo#flavor".
      2. Table: {ref, flavor?, flavors?, dtype?, file_layout?, file_type?}.
    """
    del warnings
    if isinstance(v, str):
        ref, flavor = _split_model_ref_flavor(v.strip(), field="model ref")
        _validate_endpoint_model_ref(ref, field="model ref")
        return TensorhubModelSpec(ref=ref, flavor=flavor)

    if isinstance(v, Mapping):
        ref, ref_flavor = _split_model_ref_flavor(
            str(v.get("ref") or "").strip(),
            field="model spec ref",
        )
        _validate_endpoint_model_ref(ref, field="model spec ref")

        if "dtypes" in v:
            raise ValueError(
                "model spec 'dtypes' field removed — use flavor/flavors or dtype/file_layout/file_type"
            )
        if "attributes" in v:
            raise ValueError(
                "model spec 'attributes' field removed — use flavor/flavors or dtype/file_layout/file_type"
            )

        flavor = _string_field(v, "flavor")
        if ref_flavor and flavor:
            raise ValueError("model spec cannot set both ref#flavor and flavor")
        flavors = _string_list_field(v, "flavors")
        if (ref_flavor or flavor) and flavors:
            raise ValueError("model spec cannot set both flavor and flavors")

        return TensorhubModelSpec(
            ref=ref,
            flavor=flavor or ref_flavor,
            flavors=tuple(flavors),
            dtype=_string_field(v, "dtype"),
            file_layout=_string_field(v, "file_layout"),
            file_type=_string_field(v, "file_type"),
        )

    raise ValueError("model spec must be a string or a table {ref=...}")


def _split_model_ref_flavor(raw: str, *, field: str) -> tuple[str, str]:
    if "#" not in raw:
        return raw, ""
    if raw.count("#") != 1:
        raise ValueError(f"{field} has invalid flavor selector: {raw!r}")
    ref, flavor = raw.rsplit("#", 1)
    flavor = flavor.strip()
    if not flavor:
        raise ValueError(f"{field} has empty flavor selector: {raw!r}")
    return ref.strip(), flavor


def _string_field(raw: Mapping[str, Any], key: str) -> str:
    if key not in raw:
        return ""
    value = raw.get(key)
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"model spec {key!r} must be a string")
    return value.strip()


def _string_list_field(raw: Mapping[str, Any], key: str) -> list[str]:
    if key not in raw:
        return []
    value = raw.get(key)
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"model spec {key!r} must be a list of strings")
    out: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"model spec {key!r} entries must be strings")
        s = item.strip()
        if not s:
            raise ValueError(f"model spec {key!r} entries must be non-empty")
        if s not in out:
            out.append(s)
    return out


def _validate_endpoint_model_ref(ref: str, *, field: str) -> None:
    raw = (ref or "").strip()
    if not raw:
        raise ValueError(f"{field} cannot be empty")
    if raw.lower() != raw:
        raise ValueError(f"{field} must be lowercase: {raw!r}")
    if raw.startswith("cozy:") or raw.startswith("hf:"):
        raise ValueError(
            f"{field} must not include a scheme prefix; use plain owner/repo form (got {raw!r})"
        )

    base = raw
    if "@" in base:
        if base.count("@") != 1:
            raise ValueError(f"{field} has invalid digest selector: {raw!r}")
        base, digest = base.split("@", 1)
        if not digest.strip():
            raise ValueError(f"{field} has empty digest selector: {raw!r}")

    if ":" in base:
        base, tag = base.rsplit(":", 1)
        if not tag.strip():
            raise ValueError(f"{field} has empty tag selector: {raw!r}")

    parts = [p.strip() for p in base.split("/")]
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(
            f"{field} must be 'owner/repo' (optionally with :tag or @digest), got {raw!r}"
        )
    owner, repo = parts
    if not _RE_MODEL_SEGMENT.match(owner):
        raise ValueError(f"{field} has invalid owner segment: {owner!r}")
    if not _RE_MODEL_SEGMENT.match(repo):
        raise ValueError(f"{field} has invalid repo segment: {repo!r}")


def _parse_function_resource_hints(v: Any) -> dict[str, Any]:
    if not isinstance(v, Mapping):
        return {}
    if "max_concurrency" in v or "max_inflight_requests" in v:
        raise ValueError(
            "function-level concurrency hints are not supported in endpoint.toml; "
            "Tensorhub learns scheduling concurrency from runtime observations"
        )
    out: dict[str, Any] = {}
    for key in (
        "batch_size_min",
        "batch_size_target",
        "batch_size_max",
        "prefetch_depth",
        "max_wait_ms",
        "memory_hint_mb",
    ):
        if key not in v:
            continue
        raw = v.get(key)
        if raw is None:
            continue
        try:
            iv = int(raw)
        except Exception:
            raise ValueError(f"function resource hint {key} must be an integer")
        if iv <= 0:
            raise ValueError(f"function resource hint {key} must be > 0")
        out[key] = iv

    if "stage_profile" in v or "stage_traits" in v:
        raise ValueError(
            "function resource hints stage_profile/stage_traits were removed; "
            "use a single string hint: kind"
        )
    if "kind" in v:
        kind = str(v.get("kind") or "").strip()
        if kind:
            out["kind"] = kind

    if "accelerator" in v:
        accelerator = str(v.get("accelerator") or "").strip().lower()
        if accelerator == "gpu":
            accelerator = "cuda"
        elif accelerator == "cpu":
            accelerator = "none"
        if accelerator and accelerator not in {"none", "cuda"}:
            raise ValueError("function resource hint accelerator must be 'none' or 'cuda'")
        if accelerator:
            out["accelerator"] = accelerator

    if "accelerator_preference" in v:
        preference = str(v.get("accelerator_preference") or "").strip().lower()
        if preference and preference not in {"required", "preferred"}:
            raise ValueError("function resource hint accelerator_preference must be 'required' or 'preferred'")
        if preference:
            out["accelerator_preference"] = preference

    if "requires_gpu" in v:
        raw = v.get("requires_gpu")
        if isinstance(raw, bool):
            out["requires_gpu"] = raw
        else:
            raise ValueError("function resource hint requires_gpu must be a boolean")

    compute_min_raw = v.get("compute_capability_min")
    if compute_min_raw is None:
        compute_min_raw = v.get("cuda_compute_min")
    if compute_min_raw is not None:
        raw = str(compute_min_raw or "").strip()
        if raw:
            try:
                parsed = float(raw)
            except Exception:
                raise ValueError("function resource hint compute_capability_min must be numeric")
            if parsed <= 0:
                raise ValueError("function resource hint compute_capability_min must be > 0")
            out["compute_capability_min"] = f"{parsed:.1f}"
            if "cuda_compute_min" in v:
                out["cuda_compute_min"] = f"{parsed:.1f}"

    if "min_vram_gb" in v:
        raw = v.get("min_vram_gb")
        try:
            val = float(raw)
        except Exception:
            raise ValueError("function resource hint min_vram_gb must be numeric")
        if val <= 0:
            raise ValueError("function resource hint min_vram_gb must be > 0")
        out["min_vram_gb"] = val

    if "vram_multiplier" in v:
        raw = v.get("vram_multiplier")
        try:
            val = float(raw)
        except Exception:
            raise ValueError("function resource hint vram_multiplier must be numeric")
        if val <= 0:
            raise ValueError("function resource hint vram_multiplier must be > 0")
        out["vram_multiplier"] = val

    for key in ("supported_conversion_profiles", "supported_precisions", "required_libraries"):
        if key not in v:
            continue
        raw = v.get(key)
        if not isinstance(raw, list):
            raise ValueError(f"function resource hint {key} must be a list of strings")
        items: list[str] = []
        for item in raw:
            if not isinstance(item, str):
                raise ValueError(f"function resource hint {key} must be a list of strings")
            trimmed = item.strip()
            if trimmed:
                items.append(trimmed)
        if items:
            out[key] = items
    return out


def _parse_batch_dimension_path(raw: Any, *, field: str) -> str | None:
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise ValueError(f"{field} must be a string")
    s = raw.strip()
    if not s:
        return None
    if s.startswith("input."):
        raise ValueError(
            f"{field} must be payload-root relative (for example, 'items' instead of 'input.items')"
        )
    return s


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


def _parse_function_batch_dimension(v: Any, *, field_prefix: str) -> str | None:
    if not isinstance(v, Mapping):
        return None

    for unsupported in (
        "models",
        "request_contract",
        "batch_dimension_path",
        "request_mode",
        "max_items_per_request",
        "dynamic_batching",
        "partitioning",
    ):
        if unsupported in v:
            raise ValueError(
                f"{field_prefix}.{unsupported} is not supported; "
                "only batch_dimension may be configured by endpoints"
            )

    return _parse_batch_dimension_path(v.get("batch_dimension"), field=f"{field_prefix}.batch_dimension")


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

    models: dict[str, TensorhubModelSpec] = {}
    function_models: dict[str, dict[str, TensorhubModelSpec]] = {}
    raw_models = data.get("models")
    if raw_models is not None:
        if not isinstance(raw_models, Mapping):
            raise ValueError("models must be a table")
        for key_raw, value_raw in raw_models.items():
            key = str(key_raw).strip()
            if not key:
                continue
            # [models.<function_name>] subtable support: maps without any
            # model-spec keys are treated as function keyspaces.
            if (
                isinstance(value_raw, Mapping)
                and "ref" not in value_raw
                and "dtypes" not in value_raw
                and "attributes" not in value_raw
                and "flavor" not in value_raw
                and "flavors" not in value_raw
                and "dtype" not in value_raw
                and "file_layout" not in value_raw
                and "file_type" not in value_raw
            ):
                fn = slugify_name(key)
                if not fn:
                    raise ValueError(f"invalid function keyspace name under [models]: {key!r}")
                keyspace: dict[str, TensorhubModelSpec] = {}
                for model_key_raw, model_spec_raw in value_raw.items():
                    model_key = str(model_key_raw).strip()
                    if not model_key:
                        continue
                    keyspace[model_key] = _parse_model_spec(model_spec_raw, warnings=warnings)
                if keyspace:
                    function_models[fn] = keyspace
                continue
            models[key] = _parse_model_spec(value_raw, warnings=warnings)

    # Reject legacy endpoint.toml blocks that became platform-controlled.
    # Function-scoped [resources] is accepted for mixed CPU/GPU endpoints;
    # concurrency inside it remains rejected by _parse_function_resource_hints.
    _REJECTED_ENDPOINT_BLOCKS = {
        "scaling": (
            "autoscaling is platform-controlled; remove the [scaling] block from "
            "endpoint.toml. If you need policy knobs, request them via platform config."
        ),
    }
    _REJECTED_FUNCTION_BLOCKS = {
        "runtime": (
            "per-function [functions.<fn>.runtime] numerics (batch_size_max, prefetch_depth, etc) "
            "were removed. Tensorhub learns scheduling behavior from runtime observations."
        ),
        "compute_envelope": (
            "compute_envelope (min/max/default) was removed in tensorhub #232. "
            "Declare a single hardware default at endpoint-level [resources]; invokers "
            "override size axes at runtime via wire-payload `compute`."
        ),
        "concurrency": (
            "per-function [functions.<fn>.concurrency] was removed. "
            "Tensorhub learns scheduling concurrency from runtime observations."
        ),
    }
    for block_name, message in _REJECTED_ENDPOINT_BLOCKS.items():
        if block_name in data:
            raise ValueError(f"endpoint.toml: [{block_name}] is no longer accepted — {message}")

    function_resources: dict[str, dict[str, Any]] = {}
    function_batch_dimensions: dict[str, str] = {}
    raw_functions = data.get("functions")
    if isinstance(raw_functions, dict):
        for fn_name, fn_cfg in raw_functions.items():
            fn = slugify_name(str(fn_name).strip())
            if not fn or not isinstance(fn_cfg, dict):
                continue

            for rejected_key, reject_msg in _REJECTED_FUNCTION_BLOCKS.items():
                if rejected_key in fn_cfg:
                    raise ValueError(
                        f"endpoint.toml: [functions.{fn}.{rejected_key}] is no longer accepted — {reject_msg}"
                    )

            batch_path = _parse_function_batch_dimension(fn_cfg, field_prefix=f"functions.{fn}")
            if batch_path:
                function_batch_dimensions[fn] = batch_path
            if "resources" in fn_cfg:
                parsed = _parse_function_resource_hints(fn_cfg.get("resources"))
                if parsed:
                    function_resources[fn] = parsed

    # Endpoint-level [resources] is the default. Function-level [resources]
    # may narrow/override hardware for mixed CPU/GPU endpoints.
    raw_resources = data.get("resources")
    res_kwargs: dict[str, Any] = {}
    if isinstance(raw_resources, dict):
        for k in ("accelerator", "cuda_compute_min"):
            if k in raw_resources:
                res_kwargs[k] = str(raw_resources[k])
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
            models=models,
            function_models=function_models,
            function_resources=function_resources,
            function_batch_dimensions=function_batch_dimensions,
            resources=resources,
        ),
        warnings,
    )
