"""Layout repackaging — the generic engine that executes declared maps.

singlefile <-> diffusers. This module owns the *mechanics*: locating weight
files under diffusers' several naming conventions, applying an ordered set of
key-rewrite rules, casting dtypes, driving the single-file pipeline loader, and
refusing a converter whose family disagrees with the tree on disk.

It owns **no family knowledge**. Every rename table, component signature,
pipeline class and config donor arrives as a :class:`~.repack_spec.RepackageFamily`
declaration registered by the endpoint that owns the family. Hand-ported
per-family tensor surgery here means two copies of a family signature that can
disagree, and a tree handed to the wrong converter with no exception raised.
"""

from __future__ import annotations

import importlib
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from .registry import (
    normalize_family,
    registered_repackage_families,
    repackage_family,
    require_repackage_family,
)
from .repack_spec import ComponentRepack, DtypePolicy, RepackageFamily
from .writer import (
    NEVER_SHARD_MAX_SIZE,
    ConversionImplementationError,
    assert_one_file_per_component,
)

if TYPE_CHECKING:
    import torch


class NotImplementedFamilyError(ConversionImplementationError):
    pass


class ConverterSignatureError(ValueError):
    """The chosen converter does not match the tree's component signature."""


class AmbiguousFamilyError(ValueError):
    """More than one registered family claims this tree."""


def _read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text("utf-8")))


def _torch() -> Any:
    import torch

    return torch


def _st_load(path: Path) -> dict[str, Any]:
    from safetensors.torch import load_file as st_load_file

    return cast(dict[str, Any], st_load_file(str(path), device="cpu"))


def _st_save(state_dict: dict[str, Any], output_path: Path) -> None:
    from safetensors.torch import save_file as st_save_file

    st_save_file(state_dict, str(output_path))


def _load_sharded_safetensors(index_json: Path) -> dict[str, torch.Tensor]:
    idx = _read_json(index_json)
    weight_map = idx.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("invalid_safetensors_index")
    # Load each shard once, merge dicts. This is memory heavy but conversion needs full weights anyway.
    shard_names: list[str] = sorted({str(v) for v in weight_map.values() if isinstance(v, str) and v.strip()})
    out: dict[str, torch.Tensor] = {}
    for name in shard_names:
        shard_path = index_json.parent / name
        out.update(_st_load(shard_path))
    return out


def _load_component_state_dict(
    component_dir: Path,
    *,
    safetensors_bases: list[str] | tuple[str, ...],
    bin_base: str | None = None,
) -> dict[str, torch.Tensor]:
    """
    Loads a diffusers component state_dict from common weight naming patterns.

    Supported safetensors names:
    - <base>.safetensors
    - <base>.safetensors.index.json (sharded)
    - <base>.bin.safetensors (common in some HF repos)
    - <base>.bin.safetensors.index.json (rare; but cheap to support)
    - <base>.<variant>.safetensors and its sharded index (HF dtype-variant
      downloads keep the suffix — e.g. diffusion_pytorch_model.fp16.safetensors
      in repo-cas mirrors cloned with a dtype preference)

    A component that offers only a pickle (``<bin_base>.bin``) is REFUSED:
    pickles are banned platform-wide, and the remedy is to mirror the source
    repo without the pickle rather than to unpickle it here.
    """
    for raw_base in safetensors_bases:
        base = str(raw_base or "").strip()
        if not base:
            continue
        st_path = component_dir / f"{base}.safetensors"
        st_index = component_dir / f"{base}.safetensors.index.json"
        if st_path.exists():
            return _st_load(st_path)
        if st_index.exists():
            return _load_sharded_safetensors(st_index)

        st_path = component_dir / f"{base}.bin.safetensors"
        st_index = component_dir / f"{base}.bin.safetensors.index.json"
        if st_path.exists():
            return _st_load(st_path)
        if st_index.exists():
            return _load_sharded_safetensors(st_index)

        # Dtype-variant names, exact-name misses only.
        for st in sorted(component_dir.glob(f"{base}.*.safetensors")):
            return _st_load(st)
        for idx in sorted(component_dir.glob(f"{base}.safetensors.*.index.json")) + sorted(
            component_dir.glob(f"{base}.*.safetensors.index.json")
        ):
            return _load_sharded_safetensors(idx)

    if bin_base:
        bin_path = component_dir / f"{bin_base}.bin"
        if bin_path.exists():
            # The clone lane feeds this ARBITRARY tenant-submitted repos, so a
            # `.bin` here is an untrusted pickle and reading it is arbitrary
            # code execution inside a pod holding hub credentials and other
            # tenants' work (cozy_snapshot.py:285-292). Pickles are banned
            # platform-wide; `classifier.py` already refuses a pickle-only repo
            # with `pickle_only`, and this is the same refusal at the component
            # door.
            raise ConversionImplementationError(
                f"pickle_only:{bin_path.name}: this component offers only a "
                f"pickle, and pickles are refused. Mirror the source repo "
                f"without the pickle (safetensors) and convert that."
            )
    raise FileNotFoundError(f"missing weights for {component_dir.name}")


def _maybe_cast_tensor(t: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    torch_mod = _torch()
    if not isinstance(t, torch_mod.Tensor):
        return t
    if t.is_floating_point():
        return t.to(dtype=out_dtype)
    return t


def _cast_state_dict(sd: dict[str, torch.Tensor], out_dtype: torch.dtype) -> dict[str, torch.Tensor]:
    return {k: _maybe_cast_tensor(v, out_dtype) for k, v in sd.items()}


_DTYPE_ALIASES: dict[str, str] = {
    "fp16": "fp16", "f16": "fp16", "float16": "fp16", "half": "fp16",
    "bf16": "bf16", "bfloat16": "bf16",
    "fp32": "fp32", "f32": "fp32", "float32": "fp32", "full": "fp32",
}


def _canonical_dtype(name: str | None) -> str:
    return _DTYPE_ALIASES.get(str(name or "").strip().lower(), "")


def _torch_dtype(canonical: str) -> Any:
    torch_mod = _torch()
    return {"fp16": torch_mod.float16, "bf16": torch_mod.bfloat16, "fp32": torch_mod.float32}[canonical]


def _repackage_torch_dtype(dtype: str | None, policy: DtypePolicy | None = None) -> Any:
    """Map a requested output dtype onto the torch dtype the single-file pipeline
    is materialized in. Dtypes the policy marks ``preserve`` are loaded as-is —
    an fp16 source loaded bf16 rounds a 10-bit mantissa to 7 bits and the later
    fp16 cast cannot recover it. Everything else (quant targets) loads at the
    policy default."""
    pol = policy or DtypePolicy()
    canonical = _canonical_dtype(dtype)
    if canonical and canonical in pol.preserve:
        return _torch_dtype(canonical)
    return _torch_dtype(pol.load_default)


def present_components(model_dir: Path) -> frozenset[str]:
    """Top-level component directories present in a diffusers tree."""
    if not model_dir.is_dir():
        return frozenset()
    return frozenset(p.name for p in model_dir.iterdir() if p.is_dir())


def detect_diffusers_family(model_dir: Path) -> str:
    """Detect the family of a diffusers tree from the *declared* signatures.

    There is exactly ONE copy of each family's signature — the one in its
    declaration, which is also what the converter guard checks. That is what
    makes a wrong-converter divergence structurally impossible rather than
    merely fixed: detection and the guard cannot disagree, because they
    read the same field.

    Two families claiming one tree is a declaration bug and is refused by name,
    never resolved by pick-the-first.
    """
    present = present_components(model_dir)
    claims: list[str] = []
    for name in registered_repackage_families():
        spec = repackage_family(name)
        if spec is None:
            continue
        sig = spec.signature
        if not (sig.requires_all or sig.requires_any or sig.forbids):
            continue
        if not sig.violations(present):
            claims.append(spec.family)
    if len(claims) > 1:
        raise AmbiguousFamilyError(
            f"{model_dir}: component set {sorted(present)!r} satisfies the signature of "
            f"MORE THAN ONE registered family ({sorted(claims)!r}) — refusing to guess. "
            "Tighten the declarations (a family that must not carry a component declares "
            "it in `forbids`)."
        )
    if claims:
        return claims[0]
    return _family_from_pipeline_class(model_dir)


def _family_from_pipeline_class(model_dir: Path) -> str:
    """Fallback: ``model_index.json``'s ``_class_name`` against declared targets.

    Still declaration-driven — a family is claimed only if it declares that
    pipeline class in ``to_diffusers``.
    """
    model_index = model_dir / "model_index.json"
    if not model_index.exists():
        return "unknown"
    try:
        cls = str(_read_json(model_index).get("_class_name") or "").strip()
    except Exception:
        return "unknown"
    if not cls:
        return "unknown"
    for name in registered_repackage_families():
        spec = repackage_family(name)
        if spec is None:
            continue
        for target in spec.to_diffusers:
            if target.pipeline_class == cls:
                return spec.family
    return "unknown"


def assert_converter_matches_signature(model_dir: Path, spec: RepackageFamily) -> None:
    """Refuse to run a converter whose declared signature disagrees with disk.

    The permanent guard. Whichever way the family arrives — detected
    here or passed in by a caller — the component set on disk is ground truth,
    and a mismatch is a named refusal rather than a wrong output file.
    """
    violations = spec.signature.violations(present_components(model_dir))
    if violations:
        raise ConverterSignatureError(
            f"converter/signature mismatch: family={spec.family!r} but the tree at "
            f"{model_dir} " + "; ".join(violations) + " — refusing to run this family's "
            "converter against a tree it does not describe"
        )


def _ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


# diffusers wraps SingleFileComponentError with a remedy snippet naming both
# the component and its class: "{name} = {class_name}.from_pretrained('...')".
_MISSING_COMPONENT_RE = re.compile(r"(\w+) = (\w+)\.from_pretrained\('\.\.\.'\)")


def _load_component_from_config_repo(
    config: str, name: str, class_name: str, torch_dtype: Any
) -> Any:
    """Load a pipeline component the checkpoint doesn't carry from the family's
    declared config repo. DiT-only fine-tunes (common on civitai) ship no
    text-encoder/VAE weights — sourcing them from the declared donor is exactly
    the CAS dedup story: the fine-tune tree stores only its denoiser."""

    comp_cls = None
    for lib in ("transformers", "diffusers"):
        comp_cls = getattr(importlib.import_module(lib), class_name, None)
        if comp_cls is not None:
            break
    if comp_cls is None:
        raise ConversionImplementationError(f"component_class_unavailable:{class_name}")
    try:
        return comp_cls.from_pretrained(config, subfolder=name, torch_dtype=torch_dtype)
    except TypeError:  # non-torch components (tokenizers etc.) reject torch_dtype
        return comp_cls.from_pretrained(config, subfolder=name)


def _load_singlefile_pipeline(
    *, input_path: Path, pipeline_class: str, config: str | None, torch_dtype: Any,
) -> Any:
    import diffusers
    from diffusers.loaders.single_file import SingleFileComponentError

    cls = getattr(diffusers, pipeline_class, None)
    if cls is None:
        raise ValueError(f"pipeline_class_unavailable:{pipeline_class}")

    kwargs: dict[str, Any] = {"torch_dtype": torch_dtype}
    if config:
        kwargs["config"] = config

    if hasattr(cls, "from_single_file"):
        loader = cls.from_single_file
    else:
        from diffusers.loaders.single_file import FromSingleFileMixin

        fn = getattr(FromSingleFileMixin.from_single_file, "__func__", None)
        if fn is None:
            raise ValueError(f"pipeline_not_implemented:{pipeline_class}")

        def loader(path: str, **kw: Any) -> Any:
            return fn(cls, path, **kw)

    # Components missing from the checkpoint are pulled from the config repo
    # (bounded: a pipeline has a handful of components).
    for _ in range(6):
        try:
            return loader(str(input_path), **kwargs)
        except SingleFileComponentError as exc:
            if not config:
                raise
            m = _MISSING_COMPONENT_RE.search(str(exc))
            if m is None or m.group(1) in kwargs:
                raise
            name, class_name = m.group(1), m.group(2)
            kwargs[name] = _load_component_from_config_repo(
                config, name, class_name, torch_dtype
            )
    raise ConversionImplementationError(
        f"singlefile_missing_component_loop:{pipeline_class}:{config}"
    )


def _singlefile_attempts(spec: RepackageFamily) -> list[tuple[str, str | None]]:
    """Declared pipeline classes: each tried bare first, then with each donor config."""
    attempts: list[tuple[str, str | None]] = []
    seen: set[tuple[str, str]] = set()
    for target in spec.to_diffusers:
        for cfg in (None, *target.config_repos):
            key = (target.pipeline_class, cfg or "")
            if key in seen:
                continue
            seen.add(key)
            attempts.append((target.pipeline_class, cfg))
    return attempts


def singlefile_to_diffusers(
    input_path: Path, output_dir: Path, *, model_family: str, output_dtype: str | None = None,
) -> dict[str, str]:
    spec = require_repackage_family(model_family)
    attempts = _singlefile_attempts(spec)
    if not attempts:
        raise ConversionImplementationError(
            f"unsupported_family_for_layout_conversion:{spec.family}:singlefile_to_diffusers"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    for pipeline_class, config in attempts:
        try:
            pipe = _load_singlefile_pipeline(
                input_path=input_path,
                pipeline_class=pipeline_class,
                config=config,
                torch_dtype=_repackage_torch_dtype(output_dtype, spec.dtypes),
            )
            pipe.save_pretrained(str(output_dir), safe_serialization=True,
                                 max_shard_size=NEVER_SHARD_MAX_SIZE)
            assert_one_file_per_component(output_dir, producer="layout repackage")
            return {
                "pipeline_class": pipeline_class,
                "pipeline_config": str(config or ""),
                "output_dir": str(output_dir),
            }
        except Exception as exc:  # noqa: BLE001
            cfg = f" config={config}" if config else ""
            # single-line: the worker fatal path keeps only the first line,
            # which hid 5 of 6 attempt errors (e2e #112 flux diagnosis).
            flat = str(exc).replace("\n", " | ")
            errors.append(f"{pipeline_class}{cfg}: {flat}")
            continue

    raise ConversionImplementationError(
        "singlefile_to_diffusers_failed:" + "; ".join(errors[:5])
    )


def _repack_component(model_dir: Path, comp: ComponentRepack) -> dict[str, torch.Tensor]:
    """Load ONE component and rewrite its keys per the matching declared variant."""
    component_dir = model_dir / comp.name
    if not component_dir.is_dir():
        if comp.required:
            raise ConverterSignatureError(
                f"required component {comp.name!r} is absent from {model_dir}"
            )
        return {}

    state_dict = _load_component_state_dict(
        component_dir,
        safetensors_bases=comp.safetensors_bases,
        bin_base=comp.bin_base,
    )
    variant = comp.variant_for(frozenset(state_dict))

    out: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        for rule in variant.rules:
            new_key = rule.apply(new_key)
        out[variant.out_prefix + new_key] = value

    for key in variant.transpose_keys:
        full = variant.out_prefix + key
        if full in out:
            out[full] = out[full].t().contiguous()
    return out


def convert_diffusers_to_singlefile(
    model_dir: Path,
    output_path: Path,
    *,
    family: str | None = None,
    output_dtype: str = "fp16",
) -> None:
    """Drive loop: resolve the declaration, guard it, execute it, cast, save."""
    requested = str(family or "").strip().lower()
    if requested and requested != "unknown":
        # A caller that NAMES a family is refused by name when nothing declares
        # it — never quietly re-detected into some other family's converter.
        spec = require_repackage_family(requested)
    else:
        detected = detect_diffusers_family(model_dir)
        if detected == "unknown":
            raise NotImplementedFamilyError(
                f"cannot identify a registered family for {model_dir}; registered: "
                f"{', '.join(registered_repackage_families()) or '<none>'}"
            )
        spec = require_repackage_family(detected)
    if not spec.supports_diffusers_to_singlefile:
        raise NotImplementedFamilyError(
            f"diffusers_to_singlefile not implemented for family={spec.family} "
            "(its declaration carries no `to_singlefile` components)"
        )

    canonical = _canonical_dtype(output_dtype)
    if not canonical or canonical not in spec.dtypes.allowed_output:
        raise ValueError("invalid_output_dtype")

    # The guard, kept independent of detection: a converter must never run
    # against a component set its declaration does not describe.
    assert_converter_matches_signature(model_dir, spec)

    state_dict: dict[str, torch.Tensor] = {}
    for comp in spec.to_singlefile:
        state_dict.update(_repack_component(model_dir, comp))
    if not state_dict:
        raise ConversionImplementationError(
            f"repackage produced an EMPTY state dict for family={spec.family}"
        )

    state_dict = _cast_state_dict(state_dict, _torch_dtype(canonical))
    _ensure_parent(output_path)
    _st_save(cast(dict[str, Any], state_dict), output_path)


def diffusers_to_singlefile(input_dir: Path, output_path: Path, *, model_family: str) -> dict[str, str]:
    convert_diffusers_to_singlefile(
        input_dir,
        output_path,
        family=model_family,
        output_dtype="bf16",
    )
    return {
        "model_family": normalize_family(model_family),
        "output_path": str(output_path),
    }


__all__ = [
    "AmbiguousFamilyError",
    "ConverterSignatureError",
    "NotImplementedFamilyError",
    "assert_converter_matches_signature",
    "convert_diffusers_to_singlefile",
    "detect_diffusers_family",
    "diffusers_to_singlefile",
    "present_components",
    "singlefile_to_diffusers",
]
