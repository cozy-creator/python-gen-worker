from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import tomllib
import re

from .names import slugify_function_name

_DEFAULT_DTYPES: tuple[str, ...] = ("fp16", "bf16")

_RE_CLAUSE = re.compile(r"^\s*(>=|<=|==|~=|>|<)?\s*([0-9]+(?:\.[0-9]+)*)\s*$")
_RE_VERSION_PREFIX = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)*)")
_RE_MODEL_SEGMENT = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


@dataclass(frozen=True)
class TensorhubModelSpec:
    """Endpoint.toml [models] entry.

    `attributes` is the tensorhub #229 variant-attribute selector. Each value
    is an ordered preference list: ("bf16",) means strict bf16; ("bf16", "fp16")
    means prefer bf16 but fall back to fp16 if no bf16 variant exists. At most
    ONE attribute per entry may have more than one preference — the others
    must be single-valued. Stored as a hashable tuple-of-(key, tuple-of-values)
    so the dataclass stays frozen + equality-comparable.

    Use `.attributes_as_dict()` for dict-keyed-by-str access (values are
    lists). Use `.strict_attributes_as_dict()` when you know every attribute
    is single-valued (raises if any has preferences); the tensorhub resolver
    API accepts single-valued selectors directly.

    `dtypes` is populated from `attributes["dtype"]` preferences for the
    backward-compat migration window. New code should read `attributes`
    instead; dtypes will be removed two releases after the attributes-map
    shape ships.
    """

    ref: str
    attributes: tuple[tuple[str, tuple[str, ...]], ...] = ()
    # DEPRECATED: derived from attributes["dtype"] during the migration window.
    # For multi-preference dtype lists, contains the full list in declared
    # order. Kept populated so downstream consumers relying on the old shape
    # keep working. Will be removed in a future release.
    dtypes: tuple[str, ...] = _DEFAULT_DTYPES

    def attributes_as_dict(self) -> dict[str, list[str]]:
        """Canonical dict form: key → ordered preference list."""
        return {k: list(v) for k, v in self.attributes}

    def strict_attributes_as_dict(self) -> dict[str, str]:
        """Dict form assuming every attribute is single-valued. Raises
        ValueError if any attribute has multiple preferences."""
        out: dict[str, str] = {}
        for k, v in self.attributes:
            if len(v) != 1:
                raise ValueError(
                    f"attribute {k!r} has multiple preferences {list(v)!r}; "
                    "caller expected single-valued attributes"
                )
            out[k] = v[0]
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
    resources: dict[str, Any]


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
      1. Bare string "owner/repo" → Attributes empty.
      2. Table: {ref, attributes={...}} — the tensorhub #229 attributes-map shape.
    """
    del warnings  # no deprecation warnings emitted — 'dtypes' is hard-rejected below
    if isinstance(v, str):
        ref = v.strip()
        _validate_endpoint_model_ref(ref, field="model ref")
        return TensorhubModelSpec(ref=ref, attributes=(), dtypes=())

    if isinstance(v, Mapping):
        ref = str(v.get("ref") or "").strip()
        _validate_endpoint_model_ref(ref, field="model spec ref")

        if "dtypes" in v:
            raise ValueError(
                "model spec 'dtypes' field removed — use attributes={dtype=[...]} instead"
            )

        if "attributes" in v:
            attrs = _parse_attributes_map(v["attributes"])
            dtypes = tuple(attrs.get("dtype", []))
            return TensorhubModelSpec(
                ref=ref,
                attributes=_freeze_attributes(attrs),
                dtypes=dtypes,
            )

        return TensorhubModelSpec(ref=ref, attributes=(), dtypes=())

    raise ValueError("model spec must be a string or a table {ref=..., attributes={...}}")


def _parse_attributes_map(raw: Any) -> dict[str, list[str]]:
    """Decode a tensorhub #229 attribute selector. Each value is either a
    string (strict match) or a list of strings (ordered preferences: first
    match wins). AT MOST ONE attribute per entry may be list-valued; the rest
    must be single-valued. Normalized to dict[str, list[str]] where
    single-valued entries become 1-element lists.

    Unknown keys are accepted (forward-compat with new #229 axes); commit-time
    validation on tensorhub enforces per-family required keys.
    """
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError("attributes must be a table of string → (string | list[string])")
    out: dict[str, list[str]] = {}
    list_valued_keys: list[str] = []
    for k_raw, v_raw in raw.items():
        k = str(k_raw).strip()
        if not k:
            raise ValueError("attributes keys must be non-empty strings")
        if isinstance(v_raw, str):
            v = v_raw.strip()
            if not v:
                raise ValueError(f"attributes[{k!r}] must be a non-empty string")
            out[k] = [v]
        elif isinstance(v_raw, list):
            values: list[str] = []
            for item in v_raw:
                if not isinstance(item, str):
                    raise ValueError(
                        f"attributes[{k!r}] preference-list entries must be strings "
                        f"(got {type(item).__name__})"
                    )
                item_trimmed = item.strip()
                if not item_trimmed:
                    raise ValueError(f"attributes[{k!r}] preference entries must be non-empty strings")
                if item_trimmed in values:
                    # De-duplicate while preserving first-seen order.
                    continue
                values.append(item_trimmed)
            if not values:
                raise ValueError(f"attributes[{k!r}] preference list cannot be empty")
            out[k] = values
            if len(values) > 1:
                list_valued_keys.append(k)
        else:
            raise ValueError(
                f"attributes[{k!r}] must be a string or list of strings "
                f"(got {type(v_raw).__name__})"
            )
    if len(list_valued_keys) > 1:
        raise ValueError(
            "at most one attribute per [models] entry may carry a multi-value "
            "preference list; declare separate keyspace entries for multi-axis "
            f"preferences (got list-valued: {sorted(list_valued_keys)!r})"
        )
    return out


def _freeze_attributes(attrs: dict[str, list[str]]) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Hashable, deterministically-ordered representation of an attribute
    selector so TensorhubModelSpec can stay `@dataclass(frozen=True)`."""
    return tuple((k, tuple(attrs[k])) for k in sorted(attrs))


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
            "set endpoint concurrency only via [resources].max_inflight_requests"
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

    if "requires_gpu" in v:
        raw = v.get("requires_gpu")
        if isinstance(raw, bool):
            out["requires_gpu"] = raw
        else:
            raise ValueError("function resource hint requires_gpu must be a boolean")

    if "compute_capability_min" in v:
        raw = str(v.get("compute_capability_min") or "").strip()
        if raw:
            try:
                parsed = float(raw)
            except Exception:
                raise ValueError("function resource hint compute_capability_min must be numeric")
            if parsed <= 0:
                raise ValueError("function resource hint compute_capability_min must be > 0")
            out["compute_capability_min"] = f"{parsed:.1f}"

    if "min_vram_gb" in v:
        raw = v["min_vram_gb"]
        try:
            val = float(raw)
        except Exception:
            raise ValueError("function resource hint min_vram_gb must be numeric")
        if val <= 0:
            raise ValueError("function resource hint min_vram_gb must be > 0")
        out["min_vram_gb"] = val

    if "vram_multiplier" in v:
        raw = v["vram_multiplier"]
        try:
            val = float(raw)
        except Exception:
            raise ValueError("function resource hint vram_multiplier must be numeric")
        if val <= 0:
            raise ValueError("function resource hint vram_multiplier must be > 0")
        out["vram_multiplier"] = val

    for key in ("supported_conversion_profiles", "supported_precisions"):
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
    if not name:
        raise ValueError("endpoint.toml missing name")
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
            # [models.<function_name>] subtable support:
            # entries that are maps without any model-spec keys (ref / attributes
            # / dtypes) are treated as function keyspaces.
            if (
                isinstance(value_raw, Mapping)
                and "ref" not in value_raw
                and "attributes" not in value_raw
                and "dtypes" not in value_raw
            ):
                fn = slugify_function_name(key)
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

    function_resources: dict[str, dict[str, Any]] = {}
    function_batch_dimensions: dict[str, str] = {}
    raw_functions = data.get("functions")
    if isinstance(raw_functions, dict):
        for fn_name, fn_cfg in raw_functions.items():
            fn = slugify_function_name(str(fn_name).strip())
            if not fn or not isinstance(fn_cfg, dict):
                continue

            merged_hints: dict[str, Any] = {}
            runtime_hints = _parse_function_resource_hints(fn_cfg.get("runtime"))
            if runtime_hints:
                merged_hints.update(runtime_hints)
            resource_hints = _parse_function_resource_hints(fn_cfg.get("resources"))
            if resource_hints:
                merged_hints.update(resource_hints)
            if merged_hints:
                function_resources[fn] = merged_hints

            batch_path = _parse_function_batch_dimension(fn_cfg, field_prefix=f"functions.{fn}")
            if batch_path:
                function_batch_dimensions[fn] = batch_path

    resources: dict[str, Any] = {}
    raw_resources = data.get("resources")
    if isinstance(raw_resources, dict):
        for k in ("vram_gb", "ram_gb", "cpu_cores", "disk_gb", "max_inflight_requests"):
            if k in raw_resources:
                val = raw_resources[k]
                if k == "max_inflight_requests":
                    try:
                        iv = int(val)
                    except Exception:
                        raise ValueError("resources.max_inflight_requests must be an integer")
                    if iv <= 0:
                        raise ValueError("resources.max_inflight_requests must be > 0")
                    resources[k] = iv
                else:
                    resources[k] = val
    # Endpoint-level inflight default: sequential by default unless explicitly raised.
    if "max_inflight_requests" not in resources:
        resources["max_inflight_requests"] = 1

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
