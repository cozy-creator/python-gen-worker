"""Machine-readable endpoint introspection (``gen-worker run --list``).

Builds a stable JSON document describing the endpoint WITHOUT loading any
model: every routable function with its input JSON Schema, output type,
generator-ness, and model bindings, plus ``protocol_version`` +
``capabilities``. ``serve --list-functions --json`` emits the same
``functions`` array (one shared builder, one shape).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

import msgspec

from .protocol import CAPABILITIES, PROTOCOL_VERSION, gen_worker_version

if TYPE_CHECKING:
    from .run import _SelectedFunction


def describe_binding(binding: Any) -> Dict[str, Any]:
    """Introspect one model binding into a JSON-able dict (no model load)."""
    from ..api.binding import ModelRef

    if isinstance(binding, ModelRef):
        out: Dict[str, Any] = {
            "type": binding.source,
            "provider": binding.provider,
            "ref": binding.ref,
        }
        for key in ("tag", "flavor", "revision", "dtype", "subfolder", "version", "storage_dtype"):
            v = getattr(binding, key, "")
            if v:
                out[key] = v
        if getattr(binding, "allow_lora", False):
            out["allow_lora"] = True
        files = tuple(getattr(binding, "files", ()) or ())
        if files:
            out["files"] = list(files)
        return out
    return {"type": type(binding).__name__}


def _function_input_schema(payload_type: Optional[type]) -> Dict[str, Any]:
    """JSON Schema for a function's payload Struct, top-level $ref inlined."""
    if payload_type is None:
        return {}
    try:
        s = msgspec.json.schema(payload_type)
    except Exception:
        return {}
    ref = s.get("$ref")
    defs = s.get("$defs")
    if isinstance(ref, str) and ref.startswith("#/$defs/") and isinstance(defs, dict):
        key = ref.split("/")[-1]
        target = defs.get(key)
        if isinstance(target, dict):
            merged = dict(target)
            rest = {k: v for k, v in defs.items() if k != key}
            if rest:
                merged["$defs"] = rest
            return merged
    return s


def _describe_resources(resources: Any) -> Dict[str, Any]:
    """JSON-able view of a function's ``Resources`` declaration."""
    if resources is None:
        return {}
    try:
        d = msgspec.to_builtins(resources)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def detected_capabilities() -> Dict[str, Any]:
    """This machine's capabilities block (#380): SM, torch/cuda, installed
    quant libs, free VRAM. Imports torch; loads no model."""
    from ..models.hub_policy import detect_worker_capabilities
    from ..models.memory import get_available_vram_gb

    caps = detect_worker_capabilities()
    out = caps.to_dict()
    out["free_vram_gb"] = round(get_available_vram_gb(), 2)
    return out


def _gguf_probe(
    binding: Any, caps: Any, free_gb: float,
) -> Optional[Dict[str, Any]]:
    """cl#27: would the local ladder serve this function via a stored
    #gguf-<qtype> flavor? One hub resolve per bare tensorhub binding, only
    consulted when the declared-Resources verdict already says the function
    can't run natively. Quiet + fail-open: any miss/failure returns None and
    the declared verdict stands."""
    import os

    if os.getenv("GEN_WORKER_NO_PRECISION_LADDER", "").strip() == "1":
        return None
    if getattr(binding, "provider", "") != "tensorhub":
        return None
    if getattr(binding, "flavor", "") or getattr(binding, "storage_dtype", ""):
        return None  # author override — never laddered
    try:
        from ..api.binding import wire_ref
        from ..models.hub_client import resolve_repo
        from ..models.ladder import (
            gguf_fit_bounds,
            ladder_model_from_resolved,
            resolve,
            resolve_local_gguf,
        )
        from ..models.memory import cpu_offload_forbidden
        from ..models.refs import parse_model_ref

        thref = parse_model_ref(wire_ref(binding)).tensorhub
        if thref is None or thref.digest:
            return None
        model = ladder_model_from_resolved(resolve_repo(thref, timeout=5.0))
        gpu_sm = int(getattr(caps, "gpu_sm", 0) or 0)
        native = resolve(
            model, gpu_sm=gpu_sm, free_vram_gb=free_gb,
            libs=tuple(getattr(caps, "installed_libs", ()) or ()),
        )
        if not native.refusal:
            return None  # a native rung fits — the ladder never reaches gguf
        pick = resolve_local_gguf(
            model, gpu_sm=gpu_sm, free_vram_gb=free_gb,
            allow_te_offload=not cpu_offload_forbidden())
        if pick is None:
            return None
        row = next((r for r in model.flavors if r.token == pick.flavor), None)
        est, te_offload = 0.0, False
        if row is not None:
            resident, offloaded = gguf_fit_bounds(model, row.size_gb)
            te_offload = resident > free_gb
            est = offloaded if te_offload else resident
        return {"flavor": pick.flavor, "est_gb": est, "te_offload": te_offload}
    except Exception:
        return None


def function_entries(
    candidates: List["_SelectedFunction"],
    *,
    detected: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """The stable ``functions`` array, shared by ``run --list`` and
    ``serve --list-functions --json``. With ``detected`` (the
    ``detected_capabilities()`` block), each entry carries a ``fit`` verdict
    (``fits | offload | incompatible``) computed from its ``Resources``."""
    from ..models.hub_policy import (
        FIT_EMERGENCY,
        FIT_EMERGENCY_FP8,
        FIT_OFFLOAD,
        TensorhubWorkerCapabilities,
        variant_fit,
    )

    caps: Optional[TensorhubWorkerCapabilities] = None
    free_gb = 0.0
    if detected is not None:
        caps = TensorhubWorkerCapabilities(
            cuda_version=str(detected.get("cuda_version") or ""),
            gpu_sm=int(detected.get("gpu_sm") or 0),
            torch_version=str(detected.get("torch_version") or ""),
            installed_libs=list(detected.get("installed_libs") or []),
        )
        free_gb = float(detected.get("free_vram_gb") or 0.0)

    out: List[Dict[str, Any]] = []
    for c in sorted(candidates, key=lambda c: c.fn_name):
        output_type = getattr(c, "output_type", None)
        entry: Dict[str, Any] = {
            "name": c.fn_name,
            "class": getattr(c.cls, "__name__", None) if c.cls is not None else None,
            "method": getattr(c, "attr_name", None) or None,
            "kind": getattr(c, "kind", ""),
            "is_generator": bool(getattr(c, "is_generator", False)),
            "input_schema": _function_input_schema(getattr(c, "payload_type", None)),
            "output": getattr(output_type, "__name__", None) if output_type else None,
            "models": {
                param: describe_binding(binding)
                for param, binding in (getattr(c, "bindings", {}) or {}).items()
            },
        }
        resources = getattr(c, "resources", None)
        res_dict = _describe_resources(resources)
        if res_dict:
            entry["resources"] = res_dict
        variant_of = str(getattr(c, "variant_of", "") or "")
        if variant_of:
            entry["variant_of"] = variant_of
        if caps is not None:
            from ..models.serve_fit import plan_serve

            primary = next(iter((getattr(c, "bindings", {}) or {}).values()), None)
            fit, reason = variant_fit(resources, caps, free_gb, binding=primary)
            entry["fit"] = fit
            if reason:
                entry["fit_reason"] = reason
            # th#683 P3 honest-guidance: how it will actually run on this card +
            # the realistic latency trade + the ideal hardware.
            plan = plan_serve(resources, caps, free_gb, binding=primary)
            entry["serveable"] = plan.serveable
            entry["run_mode"] = plan.run_mode
            if plan.est_latency_multiplier and plan.est_latency_multiplier != 1.0:
                entry["est_latency_multiplier"] = round(plan.est_latency_multiplier, 2)
            if plan.recommended_vram_gb:
                entry["recommended_vram_gb"] = plan.recommended_vram_gb
            if plan.warning:
                entry["advisory"] = plan.warning
            elif plan.reason and not plan.serveable:
                entry["advisory"] = plan.reason
            # cl#27: when nothing native fits, the local ladder may serve a
            # stored #gguf-<qtype> flavor instead of emergency-nf4/offload —
            # surface the SAME pick the setup-time walk will make.
            if fit in (FIT_EMERGENCY_FP8, FIT_EMERGENCY, FIT_OFFLOAD):
                gguf = _gguf_probe(primary, caps, free_gb)
                if gguf is not None:
                    from ..models.hub_policy import FIT_GGUF
                    from ..models.ladder import gguf_qtype

                    qtype = gguf_qtype(gguf["flavor"]) or gguf["flavor"]
                    entry["fit"] = FIT_GGUF
                    entry["fit_flavor"] = gguf["flavor"]
                    resident_note = (
                        "denoiser resident, TEs offloaded"
                        if gguf.get("te_offload") else "fully resident"
                    )
                    entry["fit_reason"] = (
                        f"runs via GGUF {qtype} (local-only, reduced quality): "
                        f"~{gguf['est_gb']:.1f} GB {resident_note}, "
                        f"{free_gb:.1f} GB free"
                    )
                    entry["serveable"] = True
                    entry["run_mode"] = FIT_GGUF
                    entry["est_latency_multiplier"] = 1.75
                    entry["advisory"] = (
                        "pre-quantized GGUF rung: fit regime only — "
                        "no compile/TRT speedups, quality below the stored "
                        "bf16/fp8 flavors"
                    )
        out.append(entry)
    return out


def build_description(
    *,
    main_module: str,
    candidates: List["_SelectedFunction"],
) -> Dict[str, Any]:
    """Assemble the full description document (no model load)."""
    classes = sorted({
        c.cls.__name__ for c in candidates if c.cls is not None
    })
    kinds = sorted({str(getattr(c, "kind", "") or "") for c in candidates if getattr(c, "kind", "")})
    detected = detected_capabilities()
    return {
        "protocol_version": PROTOCOL_VERSION,
        "gen_worker_version": gen_worker_version(),
        "capabilities": list(CAPABILITIES),
        "detected": detected,
        "endpoint": {
            "main_module": main_module,
            "kind": kinds[0] if len(kinds) == 1 else kinds,
            "classes": classes,
        },
        "functions": function_entries(candidates, detected=detected),
    }
