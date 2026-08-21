"""Small HF repo classifier."""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

from gen_worker.api.errors import ValidationError
from ..models.file_layout import MULTI_FILE, SINGLE_FILE

_PICKLE_EXTS = {".bin", ".pt", ".pth", ".ckpt", ".pickle", ".pkl"}
_JUNK_EXTS = {
    ".onnx", ".onnx_data", ".pb", ".h5", ".tflite", ".engine", ".plan",
    ".mlmodel", ".mlpackage", ".msgpack", ".nemo", ".xml",
}
_ALWAYS_INCLUDE = {"readme.md", "license", "license.md", "license.txt", "notice", "usage_policy.md"}

_SIZE_REFUSE_BYTES = 100 * 1024 * 1024 * 1024

_DTYPE_TAGS: dict[str, tuple[str, ...]] = {
    "bf16": ("bf16",),
    "fp16": ("fp16", "f16", "half"),
    "fp32": ("fp32", "f32"),
    "fp8": ("fp8", "fp8_e4m3fn", "fp8-e4m3", "fp8_e5m2"),
}
_VARIANT_TAG_RE = re.compile(r"\.([a-z0-9_\-]+)\.safetensors$")
_KNOWN_VARIANT_TAGS = frozenset(t for tags in _DTYPE_TAGS.values() for t in tags)
_SHARD_SUFFIX_RE = re.compile(r"-\d{5}-of-\d{5}$")
_OFFICIAL_INDEX_VARIANT_RE = re.compile(
    r"\.safetensors\.index\.([a-z0-9_-]+)\.json$",
)
_LEGACY_INDEX_VARIANT_RE = re.compile(
    r"\.([a-z0-9_-]+)\.safetensors\.index\.json$",
)

_GGUF_QUANT_PREFERENCE = (
    "q8_0", "q6_k", "q5_k_m", "q5_k_s", "q4_k_m", "q4_k_s", "q4_0",
    "q3_k_m", "q3_k_s", "q2_k", "f16", "bf16", "f32",
)

_GGUF_QTYPE_RE = re.compile(r"(?:ud-)?(?:i?q\d[0-9a-z_]*|bf16|f16|f32)")
_DIFFUSERS_COMPONENT_WEIGHT_RE = re.compile(
    r"^diffusion_pytorch_model(?:\.([a-z0-9_-]+?))?"
    r"(?:-\d{5}-of-\d{5})?\.safetensors$",
)


class RepoRefusal(ValidationError):
    """The repo can't be ingested."""

    def __init__(self, reason: str, *, files_seen: Sequence[str] = (), detail: str = "") -> None:
        self.reason = str(reason)
        self.files_seen = list(files_seen)[:32]
        msg = f"repo refused: {reason}"
        if detail:
            msg += f" ({detail})"
        super().__init__(msg)


class SourceIncludeError(ValidationError):
    """One or more ``source_include`` globs matched no file in the source repo's listing."""

    def __init__(
        self,
        unmatched: Sequence[str],
        *,
        matched: Mapping[str, Sequence[str]],
        all_paths: Sequence[str],
    ) -> None:
        self.unmatched = list(unmatched)
        self.matched = {str(k): list(v) for k, v in matched.items()}
        self.all_paths = list(all_paths)
        detail = "; ".join(
            f"{g!r} -> {len(v)} file(s)" for g, v in self.matched.items() if g not in unmatched
        )
        super().__init__(
            f"source_include glob(s) matched no file: {self.unmatched!r} "
            f"(other globs: {detail or 'none'}; {len(self.all_paths)} file(s) in repo)"
        )


def apply_source_include(paths: Sequence[str], include: Sequence[str]) -> list[str]:
    """Filter repo-relative paths to only those matching at least one glob in ``include`` (``fnmatch`` against the full repo-relative path, e.g."""
    if not include:
        return list(paths)
    matched: dict[str, list[str]] = {}
    keep: set[str] = set()
    for pattern in include:
        hits = [p for p in paths if fnmatch.fnmatch(p, pattern)]
        matched[pattern] = hits
        keep.update(hits)
    unmatched = [g for g, hits in matched.items() if not hits]
    if unmatched:
        raise SourceIncludeError(unmatched, matched=matched, all_paths=paths)
    return sorted(keep)


@dataclass(frozen=True)
class RepoClassification:
    """Result of :func:`classify_repo` + :func:`select_files`."""

    strategy: str
    runtime_library: str
    allow_patterns: list[str]
    attrs: dict[str, str] = field(default_factory=dict)
    detection_reason: str = ""
    component_subfolder: str = ""


def _norm(p: str) -> str:
    return str(p or "").strip().replace("\\", "/").lstrip("/")


def _ext(p: str) -> str:
    base = p.rsplit("/", 1)[-1].lower()
    return "." + base.rsplit(".", 1)[-1] if "." in base else ""


def _is_safetensors_index(path: str) -> bool:
    lower = path.lower()
    return lower.endswith(".safetensors.index.json") or (
        ".safetensors.index." in lower and lower.endswith(".json")
    )


def _root(paths: Sequence[str]) -> list[str]:
    return [p for p in paths if "/" not in p]


def _variant_tag(path: str) -> str:
    name = path.rsplit("/", 1)[-1].lower()
    if name.endswith(".safetensors"):
        stem = _SHARD_SUFFIX_RE.sub("", name.removesuffix(".safetensors"))
        name = stem + ".safetensors"
    m = _VARIANT_TAG_RE.search(name)
    if m is None:
        return ""
    tag = m.group(1)
    return tag if tag in _KNOWN_VARIANT_TAGS else ""


def _index_variant_tag(path: str) -> str:
    name = path.rsplit("/", 1)[-1].lower()
    tag = ""
    match = _OFFICIAL_INDEX_VARIANT_RE.search(name)
    if match is not None:
        tag = match.group(1)
    else:
        match = _LEGACY_INDEX_VARIANT_RE.search(name)
        if match is not None:
            tag = match.group(1)
    return tag if tag in _KNOWN_VARIANT_TAGS else ""


def _dtype_of_tag(tag: str) -> str:
    for dtype, tags in _DTYPE_TAGS.items():
        if tag in tags:
            return dtype
    return ""


def _pick_weight_set(
    safetensors_paths: Sequence[str],
    dtype_pref: Sequence[str],
) -> tuple[list[str], str]:
    by_tag: dict[str, list[str]] = {}
    for p in safetensors_paths:
        by_tag.setdefault(_variant_tag(p), []).append(p)
    if not by_tag:
        return [], ""
    for want in dtype_pref:
        for tag, group in by_tag.items():
            if tag and _dtype_of_tag(tag) == want:
                return sorted(group), want
    if "" in by_tag:
        return sorted(by_tag[""]), ""
    tag = sorted(by_tag)[0]
    return sorted(by_tag[tag]), _dtype_of_tag(tag)


def _pick_diffusers_component_weight_set(
    root_paths: Sequence[str],
    dtype_pref: Sequence[str],
) -> tuple[list[str], list[str], str]:
    by_tag: dict[str, list[str]] = {}
    for path in root_paths:
        match = _DIFFUSERS_COMPONENT_WEIGHT_RE.match(path.lower())
        if match is not None:
            by_tag.setdefault(match.group(1) or "", []).append(path)
    if not by_tag:
        return [], [], ""

    selected_tag = ""
    for want in dtype_pref:
        selected_tag = next(
            (tag for tag in by_tag if tag and _dtype_of_tag(tag) == want), "")
        if selected_tag:
            break
    if not selected_tag and "" not in by_tag:
        selected_tag = sorted(by_tag)[0]
    weights = sorted(by_tag[selected_tag])

    prefix = "diffusion_pytorch_model"
    index_names = {
        f"{prefix}.safetensors.index.json" if not selected_tag
        else f"{prefix}.{selected_tag}.safetensors.index.json",
        f"{prefix}.safetensors.index.{selected_tag}.json" if selected_tag else "",
    }
    indexes = sorted(path for path in root_paths if path.lower() in index_names)
    return weights, indexes, _dtype_of_tag(selected_tag)


def _subfolder_component_candidates(
    paths: Sequence[str],
    component_configs: Mapping[str, Mapping[str, object]] | None,
    dtype_pref: Sequence[str],
) -> dict[str, tuple[str, list[str], list[str], str]]:
    by_subdir: dict[str, list[str]] = {}
    for p in paths:
        head, _, rest = p.partition("/")
        if not rest or "/" in rest:
            continue
        by_subdir.setdefault(head, []).append(rest)

    out: dict[str, tuple[str, list[str], list[str], str]] = {}
    for sub, names in by_subdir.items():
        if "config.json" not in {n.lower() for n in names}:
            continue
        cfg = (component_configs or {}).get(sub) or {}
        cls = str(cfg.get("_class_name") or "").strip()
        if not cls:
            continue
        weights, indexes, dtype = _pick_diffusers_component_weight_set(names, dtype_pref)
        if not weights:
            continue
        out[sub] = (cls, [f"{sub}/{n}" for n in weights],
                    [f"{sub}/{n}" for n in indexes], dtype)
    return out


def classify_repo(
    files: Sequence[str],
    *,
    sizes: Mapping[str, int] | None = None,
    config_json: Mapping[str, object] | None = None,
    component_configs: Mapping[str, Mapping[str, object]] | None = None,
    safetensors_metadata: Mapping[str, str] | None = None,
    readme_tags: Sequence[str] = (),
    dtype_pref: Sequence[str] = ("bf16", "fp16", "fp32"),
    gguf_quant: Optional[str] = None,
) -> RepoClassification:
    """Classify a HF repo from its file listing and build allow_patterns."""
    paths = [_norm(p) for p in files if _norm(p)]
    sizes = dict(sizes or {})
    root = _root(paths)
    root_set = {p.lower() for p in root}

    configs = [
        p for p in paths
        if _ext(p) in {".json", ".txt", ".model", ".jinja", ".yaml", ".yml"}
        and _ext(p) not in _JUNK_EXTS
        and not _is_safetensors_index(p)
    ]
    always = [p for p in root if p.lower() in _ALWAYS_INCLUDE]

    def _finish(strategy: str, library: str, weights: list[str], indexes: list[str],
                attrs: dict[str, str], reason: str, *,
                config_paths: Sequence[str] | None = None,
                component_subfolder: str = "") -> RepoClassification:
        allow = sorted(
            set(weights) | set(indexes)
            | set(configs if config_paths is None else config_paths)
            | set(always))
        selected = sum(int(sizes.get(p, 0)) for p in allow)
        if selected > _SIZE_REFUSE_BYTES:
            raise RepoRefusal("too_large", files_seen=paths,
                              detail=f"{selected} bytes selected")
        return RepoClassification(
            strategy=strategy, runtime_library=library, allow_patterns=allow,
            attrs=attrs, detection_reason=reason,
            component_subfolder=component_subfolder,
        )

    if "modules.json" in root_set or "config_sentence_transformers.json" in root_set:
        weights, dtype = _pick_weight_set(
            [p for p in paths if p.lower().endswith(".safetensors")], dtype_pref)
        if not weights:
            raise RepoRefusal("missing_safetensors", files_seen=paths)
        return _finish("sentence_transformers", "sentence-transformers", weights, [],
                       {"dtype": dtype or "fp32"}, "modules.json at root")

    if "adapter_config.json" in root_set:
        weights = sorted(p for p in root if p.lower().endswith(".safetensors"))
        if not weights:
            raise RepoRefusal("missing_safetensors", files_seen=paths)
        return _finish("peft", "peft", weights, [], {}, "adapter_config.json at root")

    if "model_index.json" in root_set:
        d_weights: list[str] = []
        d_indexes: list[str] = []
        resolved: dict[str, str] = {}
        by_component: dict[str, list[str]] = {}
        for p in paths:
            if not p.lower().endswith(".safetensors"):
                continue
            comp = p.split("/", 1)[0] if "/" in p else ""
            by_component.setdefault(comp, []).append(p)
        for comp, group in by_component.items():
            if not comp:
                continue
            comp_weights, comp_dtype = _pick_weight_set(group, dtype_pref)
            d_weights.extend(comp_weights)
            selected_tag = _variant_tag(comp_weights[0]) if comp_weights else ""
            d_indexes.extend(sorted(
                p for p in paths
                if _is_safetensors_index(p)
                and (p.split("/", 1)[0] if "/" in p else "") == comp
                and _index_variant_tag(p) == selected_tag
            ))
            if comp and comp_dtype:
                resolved[comp] = comp_dtype
        if not d_weights:
            raise RepoRefusal("missing_safetensors", files_seen=paths)
        dtypes = sorted(set(resolved.values()))
        return _finish("diffusers", "diffusers", d_weights, d_indexes,
                       {"dtype": dtypes[0] if len(dtypes) == 1 else "",
                        "file_layout": MULTI_FILE},
                       "model_index.json at root")

    diffusers_class = str((config_json or {}).get("_class_name") or "").strip()
    component_weights, component_indexes, component_dtype = (
        _pick_diffusers_component_weight_set(root, dtype_pref)
    )
    if diffusers_class and component_weights:
        component_attrs = {
            "file_layout": SINGLE_FILE,
            "architecture": diffusers_class,
        }
        if component_dtype:
            component_attrs["dtype"] = component_dtype
        return _finish(
            "diffusers_component",
            "diffusers",
            component_weights,
            component_indexes,
            component_attrs,
            f"diffusers component config ({diffusers_class})",
        )

    if "config.json" not in root_set:
        candidates = _subfolder_component_candidates(paths, component_configs, dtype_pref)
        if len(candidates) == 1:
            sub, (sub_class, sub_weights, sub_indexes, sub_dtype) = next(
                iter(candidates.items()))
            sub_attrs = {"file_layout": MULTI_FILE, "architecture": sub_class}
            if sub_dtype:
                sub_attrs["dtype"] = sub_dtype
            return _finish(
                "diffusers_component", "diffusers", sub_weights, sub_indexes,
                sub_attrs, f"diffusers component config in {sub}/ ({sub_class})",
                config_paths=[p for p in configs if p.startswith(f"{sub}/")],
                component_subfolder=sub,
            )

    has_st = any(p.lower().endswith(".safetensors") for p in paths)
    has_st_index = any(_is_safetensors_index(p) for p in paths)

    if "pipeline.json" in root_set and has_st:
        pt_weights = sorted(p for p in paths if p.lower().endswith(".safetensors"))
        return _finish("pipeline_tree", "trellis2", pt_weights, [],
                       {"file_layout": SINGLE_FILE}, "pipeline.json at root")

    if config_json is not None and "config.json" in root_set and (has_st or has_st_index):
        t_indexes = [p for p in root if _is_safetensors_index(p)]
        t_weights, t_dtype = _pick_weight_set(
            [p for p in root if p.lower().endswith(".safetensors")], dtype_pref)
        attrs: dict[str, str] = {"file_layout": SINGLE_FILE}
        if t_dtype:
            attrs["dtype"] = t_dtype
        arch = config_json.get("architectures")
        if isinstance(arch, list) and arch:
            attrs["architecture"] = str(arch[0])
        if config_json.get("quantization_config") is not None:
            attrs["subtype"] = "quantized"
        return _finish("transformers", "transformers", t_weights, t_indexes, attrs,
                       "config.json + safetensors")

    gguf_files = [p for p in root if p.lower().endswith(".gguf")]
    if gguf_files:
        def _quant_of(p: str) -> str:
            base = p.rsplit("/", 1)[-1].lower()
            for q in _GGUF_QUANT_PREFERENCE:
                if q in base:
                    return q
            m = _GGUF_QTYPE_RE.search(base)
            return m.group(0) if m else ""
        gguf_pick: Optional[str] = None
        if gguf_quant:
            want = str(gguf_quant).strip().lower()
            gguf_pick = next((p for p in gguf_files if want in p.lower()), None)
            if gguf_pick is None:
                raise RepoRefusal("gguf_quant_not_found", files_seen=gguf_files, detail=want)
        else:
            for q in _GGUF_QUANT_PREFERENCE:
                gguf_pick = next((p for p in sorted(gguf_files) if _quant_of(p) == q), None)
                if gguf_pick is not None:
                    break
            gguf_pick = gguf_pick or sorted(gguf_files)[0]
        return _finish("gguf", "llama-cpp", [gguf_pick], [],
                       {"dtype": f"gguf:{_quant_of(gguf_pick) or 'unknown'}",
                        "file_type": "gguf", "file_layout": SINGLE_FILE},
                       f"{len(gguf_files)} *.gguf at root")

    st_root = [p for p in root if p.lower().endswith(".safetensors")]
    if st_root:
        md = dict(safetensors_metadata or {})
        is_lora = any(k.startswith("ss_") for k in md)
        if not is_lora:
            tags = {str(t or "").strip().lower() for t in readme_tags}
            is_lora = bool(tags & {"lora", "lycoris", "loha", "lokr", "adapter"})
        if not is_lora and sizes:
            gb = 1024 ** 3
            is_lora = all(int(sizes.get(p, 0)) < gb for p in st_root) and bool(
                all(sizes.get(p) is not None for p in st_root))
        if is_lora:
            attrs = {"file_layout": SINGLE_FILE}
            for k in ("ss_base_model_version", "ss_sd_model_name", "ss_network_module"):
                if md.get(k):
                    attrs[f"kohya_{k}"] = str(md[k])
            return _finish("native_lora", "diffusers-lora", sorted(st_root), [], attrs,
                           "root safetensors + LoRA signature")
        if len(st_root) == 1:
            return _finish("aio_singlefile", "diffusers-single-file", st_root, [],
                           {"file_layout": SINGLE_FILE},
                           "single safetensors at root, no structured signals")
        weights, dtype = _pick_weight_set(st_root, dtype_pref)
        return _finish("aio_singlefile", "diffusers-single-file", weights, [],
                       {"file_layout": SINGLE_FILE, "dtype": dtype},
                       "root safetensors, dtype-variant pick")

    exts = {_ext(p) for p in paths}
    if exts & _PICKLE_EXTS:
        raise RepoRefusal("pickle_only", files_seen=[p for p in paths if _ext(p) in _PICKLE_EXTS])
    if ".onnx" in exts or ".onnx_data" in exts:
        raise RepoRefusal("onnx_only", files_seen=paths)
    if ".h5" in exts or ".pb" in exts or ".tflite" in exts:
        raise RepoRefusal("tf_only", files_seen=paths)
    if ".msgpack" in exts:
        raise RepoRefusal("flax_only", files_seen=paths)
    if ".mlmodel" in exts or ".mlpackage" in exts:
        raise RepoRefusal("coreml_only", files_seen=paths)
    if ".engine" in exts or ".plan" in exts:
        raise RepoRefusal("tensorrt_only", files_seen=paths)
    raise RepoRefusal("unknown_shape", files_seen=paths)


__all__ = [
    "RepoClassification", "RepoRefusal", "classify_repo",
    "SourceIncludeError", "apply_source_include",
]
