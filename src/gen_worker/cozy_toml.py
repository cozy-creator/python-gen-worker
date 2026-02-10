from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import tomllib
import re


_DEFAULT_DTYPES: tuple[str, ...] = ("fp16", "bf16")
_ALLOWED_DTYPES: frozenset[str] = frozenset({"fp16", "bf16", "fp8", "fp32", "int8", "int4"})

_RE_CLAUSE = re.compile(r"^\s*(>=|<=|==|~=|>|<)?\s*([0-9]+(?:\.[0-9]+)*)\s*$")
_RE_VERSION_PREFIX = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)*)")


@dataclass(frozen=True)
class CozyModelSpec:
    ref: str
    dtypes: tuple[str, ...] = _DEFAULT_DTYPES


@dataclass(frozen=True)
class CozyToml:
    schema_version: int
    name: str
    main: str
    gen_worker: str
    cuda: str | None
    models: dict[str, CozyModelSpec]
    endpoint_models: dict[str, dict[str, CozyModelSpec]]  # endpoint_name -> model_key -> spec
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


def _parse_model_spec(v: Any) -> CozyModelSpec:
    if isinstance(v, str):
        ref = v.strip()
        if not ref:
            raise ValueError("model ref cannot be empty")
        return CozyModelSpec(ref=ref, dtypes=_DEFAULT_DTYPES)

    if isinstance(v, Mapping):
        ref = str(v.get("ref") or "").strip()
        if not ref:
            raise ValueError("model spec missing ref")
        dtypes_raw = v.get("dtypes", None)
        if dtypes_raw is None:
            return CozyModelSpec(ref=ref, dtypes=_DEFAULT_DTYPES)
        if not isinstance(dtypes_raw, list) or not all(isinstance(x, str) for x in dtypes_raw):
            raise ValueError("model spec dtypes must be a list of strings")
        dtypes = tuple(x.strip() for x in dtypes_raw if x.strip())
        if not dtypes:
            raise ValueError("model spec dtypes cannot be empty")
        bad = sorted({x for x in dtypes if x not in _ALLOWED_DTYPES})
        if bad:
            raise ValueError(f"model spec has invalid dtypes: {bad} (allowed: {sorted(_ALLOWED_DTYPES)})")
        return CozyModelSpec(ref=ref, dtypes=dtypes)

    raise ValueError("model spec must be a string or a table {ref=..., dtypes=[...]}")


def load_cozy_toml(path: Path) -> CozyToml:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("cozy.toml must be a TOML table at root")

    schema_version = data.get("schema_version")
    if schema_version != 1:
        raise ValueError("cozy.toml schema_version must be 1")

    name = str(data.get("name") or "").strip()
    main = str(data.get("main") or "").strip()
    gen_worker = str(data.get("gen_worker") or "").strip()
    if not name:
        raise ValueError("cozy.toml missing name")
    if not main:
        raise ValueError("cozy.toml missing main")
    if not gen_worker:
        raise ValueError("cozy.toml missing gen_worker")
    _validate_constraint(gen_worker, field="gen_worker")

    cuda: str | None = None
    host = data.get("host")
    if isinstance(host, dict):
        req = host.get("requirements")
        if isinstance(req, dict):
            raw = req.get("cuda")
            if isinstance(raw, str) and raw.strip():
                cuda = raw.strip()
                _validate_constraint(cuda, field="cuda")

    models: dict[str, CozyModelSpec] = {}
    raw_models = data.get("models")
    if isinstance(raw_models, dict):
        for k, v in raw_models.items():
            key = str(k).strip()
            if not key:
                continue
            models[key] = _parse_model_spec(v)

    endpoint_models: dict[str, dict[str, CozyModelSpec]] = {}
    raw_endpoints = data.get("endpoints")
    if isinstance(raw_endpoints, dict):
        for ep_name, ep_cfg in raw_endpoints.items():
            ep = str(ep_name).strip()
            if not ep or not isinstance(ep_cfg, dict):
                continue
            ep_models_raw = ep_cfg.get("models")
            if not isinstance(ep_models_raw, dict):
                continue
            m: dict[str, CozyModelSpec] = {}
            for k, v in ep_models_raw.items():
                key = str(k).strip()
                if not key:
                    continue
                m[key] = _parse_model_spec(v)
            if m:
                endpoint_models[ep] = m

    resources: dict[str, Any] = {}
    raw_resources = data.get("resources")
    if isinstance(raw_resources, dict):
        for k in ("vram_gb", "ram_gb", "cpu_cores", "disk_gb"):
            if k in raw_resources:
                resources[k] = raw_resources[k]

    return CozyToml(
        schema_version=1,
        name=name,
        main=main,
        gen_worker=gen_worker,
        cuda=cuda,
        models=models,
        endpoint_models=endpoint_models,
        resources=resources,
    )
