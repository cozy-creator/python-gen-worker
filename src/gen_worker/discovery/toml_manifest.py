from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import tomllib
import re

from .names import slugify_function_name

_DEFAULT_DTYPES: tuple[str, ...] = ("fp16", "bf16")
_ALLOWED_DTYPES: frozenset[str] = frozenset({"fp16", "bf16", "fp8", "fp32", "int8", "int4"})

_RE_CLAUSE = re.compile(r"^\s*(>=|<=|==|~=|>|<)?\s*([0-9]+(?:\.[0-9]+)*)\s*$")
_RE_VERSION_PREFIX = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)*)")
_RE_MODEL_SEGMENT = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


@dataclass(frozen=True)
class TensorhubModelSpec:
    ref: str
    dtypes: tuple[str, ...] = _DEFAULT_DTYPES


@dataclass(frozen=True)
class TensorhubToml:
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


def _parse_model_spec(v: Any) -> TensorhubModelSpec:
    if isinstance(v, str):
        ref = v.strip()
        _validate_endpoint_model_ref(ref, field="model ref")
        return TensorhubModelSpec(ref=ref, dtypes=_DEFAULT_DTYPES)

    if isinstance(v, Mapping):
        ref = str(v.get("ref") or "").strip()
        _validate_endpoint_model_ref(ref, field="model spec ref")
        dtypes_raw = v.get("dtypes", None)
        if dtypes_raw is None:
            return TensorhubModelSpec(ref=ref, dtypes=_DEFAULT_DTYPES)
        if not isinstance(dtypes_raw, list) or not all(isinstance(x, str) for x in dtypes_raw):
            raise ValueError("model spec dtypes must be a list of strings")
        dtypes = tuple(x.strip() for x in dtypes_raw if x.strip())
        if not dtypes:
            raise ValueError("model spec dtypes cannot be empty")
        bad = sorted({x for x in dtypes if x not in _ALLOWED_DTYPES})
        if bad:
            raise ValueError(f"model spec has invalid dtypes: {bad} (allowed: {sorted(_ALLOWED_DTYPES)})")
        return TensorhubModelSpec(ref=ref, dtypes=dtypes)

    raise ValueError("model spec must be a string or a table {ref=..., dtypes=[...]}")


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

    if "stage_profile" in v:
        prof = str(v.get("stage_profile") or "").strip()
        if prof:
            out["stage_profile"] = prof
    if "stage_traits" in v:
        raw_traits = v.get("stage_traits")
        if not isinstance(raw_traits, list) or not all(isinstance(x, str) for x in raw_traits):
            raise ValueError("function resource hint stage_traits must be a list of strings")
        traits = [x.strip() for x in raw_traits if x.strip()]
        if traits:
            out["stage_traits"] = traits
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


def load_tensorhub_toml(path: Path) -> TensorhubToml:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("endpoint.toml must be a TOML table at root")

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
            # entries that are maps without model-spec keys are treated as function keyspaces.
            if isinstance(value_raw, Mapping) and "ref" not in value_raw and "dtypes" not in value_raw:
                fn = slugify_function_name(key)
                if not fn:
                    raise ValueError(f"invalid function keyspace name under [models]: {key!r}")
                keyspace: dict[str, TensorhubModelSpec] = {}
                for model_key_raw, model_spec_raw in value_raw.items():
                    model_key = str(model_key_raw).strip()
                    if not model_key:
                        continue
                    keyspace[model_key] = _parse_model_spec(model_spec_raw)
                if keyspace:
                    function_models[fn] = keyspace
                continue
            models[key] = _parse_model_spec(value_raw)

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

    return TensorhubToml(
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
    )
