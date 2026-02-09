from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml


COZY_PIPELINE_LOCK_FILENAME = "cozy.pipeline.lock.yaml"
COZY_PIPELINE_FILENAME = "cozy.pipeline.yaml"
DIFFUSERS_MODEL_INDEX_FILENAME = "model_index.json"


@dataclass(frozen=True)
class CozyPipelineSpec:
    source_path: Path
    raw: Dict[str, Any]

    @property
    def pipe_class(self) -> str:
        pipe = self.raw.get("pipe") or {}
        return str(pipe.get("class") or "").strip()

    @property
    def custom_pipeline_path(self) -> Optional[str]:
        pipe = self.raw.get("pipe") or {}
        v = pipe.get("custom_pipeline_path")
        if v is None:
            return None
        s = str(v).strip()
        return s or None


def _safe_child_path(root: Path, rel: str) -> Path:
    # Ensure rel doesn't escape root (best-effort).
    p = (root / rel).resolve()
    r = root.resolve()
    if r == p or r in p.parents:
        return p
    raise ValueError("path escapes model root")


def load_cozy_pipeline_spec(model_root: Path) -> Optional[CozyPipelineSpec]:
    """
    Load the Cozy pipeline spec, preferring the lockfile when present.

    This is a worker-side helper used during pipeline loading to implement:
    - prefer `cozy.pipeline.lock.yaml` when present
    - fall back to `cozy.pipeline.yaml` otherwise
    """
    root = Path(model_root)
    lock_path = root / COZY_PIPELINE_LOCK_FILENAME
    spec_path = lock_path if lock_path.exists() else (root / COZY_PIPELINE_FILENAME)
    if not spec_path.exists():
        return None

    raw = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("invalid cozy pipeline spec (expected mapping)")
    api = str(raw.get("apiVersion") or "").strip()
    kind = str(raw.get("kind") or "").strip()
    if api and api != "v1":
        raise ValueError(f"unsupported cozy pipeline apiVersion: {api!r}")
    if kind and kind != "DiffusersPipeline":
        raise ValueError(f"unsupported cozy pipeline kind: {kind!r}")

    return CozyPipelineSpec(source_path=spec_path, raw=raw)


def cozy_custom_pipeline_arg(model_root: Path, spec: CozyPipelineSpec) -> Optional[str]:
    """
    Compute the diffusers `custom_pipeline=` argument from the Cozy pipeline spec.

    Spec contract: `custom_pipeline_path` (when present) is a directory inside
    the artifact root which must contain pipeline.py.
    """
    rel = spec.custom_pipeline_path
    if not rel:
        return None
    root = Path(model_root)
    path = _safe_child_path(root, rel)
    if path.is_dir() and (path / "pipeline.py").exists():
        return path.as_posix()
    # Best-effort: allow the caller to try even if the file isn't present; but
    # keep the error surface clear.
    raise ValueError(f"custom_pipeline_path is invalid or missing pipeline.py: {rel!r}")


def generate_diffusers_model_index_from_cozy(spec: CozyPipelineSpec) -> Dict[str, Any]:
    """
    Convert a Cozy pipeline spec into a minimal diffusers `model_index.json` mapping.

    This is used as a compatibility shim for diffusers' from_pretrained loader
    when an artifact only ships Cozy pipeline YAML.
    """
    pipe_class = spec.pipe_class
    if not pipe_class:
        raise ValueError("cozy pipeline spec missing pipe.class")

    out: Dict[str, Any] = {"_class_name": pipe_class}
    # Diffusers uses this for informational checks; include when available.
    try:
        import diffusers  # type: ignore

        v = str(getattr(diffusers, "__version__", "") or "").strip()
        if v:
            out["_diffusers_version"] = v
    except Exception:
        pass

    comps = spec.raw.get("components") or {}
    if not isinstance(comps, dict):
        raise ValueError("cozy pipeline spec components must be a mapping")

    for comp_name, comp in comps.items():
        name = str(comp_name or "").strip()
        if not name:
            continue
        if not isinstance(comp, dict):
            raise ValueError(f"component {name!r} must be a mapping")

        if "ref" in comp and comp.get("ref") not in (None, ""):
            # In published artifacts, lockfiles should not contain refs.
            raise ValueError(f"component {name!r} uses ref; expected lockfile-expanded path")

        path = str(comp.get("path") or name).strip()
        lib = str(comp.get("library") or "").strip()
        cls = str(comp.get("class") or "").strip()
        if not path or "/" in path:
            raise ValueError(f"component {name!r} has invalid path {path!r} (must be root-level)")
        if not lib or not cls:
            raise ValueError(f"component {name!r} missing library/class")

        out[name] = [lib, cls]

    return out


def ensure_diffusers_model_index_json(model_root: Path) -> Optional[Path]:
    """
    If `model_index.json` is missing but Cozy pipeline YAML exists, generate a
    minimal `model_index.json` from it.

    Returns the path to model_index.json when present/created, otherwise None.
    """
    root = Path(model_root)
    mi = root / DIFFUSERS_MODEL_INDEX_FILENAME
    if mi.exists():
        return mi

    spec = load_cozy_pipeline_spec(root)
    if spec is None:
        return None

    model_index = generate_diffusers_model_index_from_cozy(spec)
    tmp = root / f".{DIFFUSERS_MODEL_INDEX_FILENAME}.tmp.{os.getpid()}"
    tmp.write_text(json.dumps(model_index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        os.replace(tmp, mi)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
    return mi

