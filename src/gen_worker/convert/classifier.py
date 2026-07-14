"""Small HF repo classifier (replaces gen_worker.conversion.hf_classifier).

Given a repo file listing (`HfApi.list_repo_files`) plus a couple of tiny
config fetches, decide:

  1. the repo's shape / runtime library (``RepoKind``), and
  2. the ``allow_patterns`` list for ``snapshot_download`` — one weight set
     per component (dtype-variant preference), all configs/tokenizers,
     never pickle/ONNX/TF/demo junk.

One refusal exception (``RepoRefusal(reason=...)``) replaces the old 14-class
hierarchy; ``reason`` is a stable machine token.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

_PICKLE_EXTS = {".bin", ".pt", ".pth", ".ckpt", ".pickle", ".pkl"}
_JUNK_EXTS = {
    ".onnx", ".onnx_data", ".pb", ".h5", ".tflite", ".engine", ".plan",
    ".mlmodel", ".mlpackage", ".msgpack", ".nemo", ".xml",
}
_DEMO_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".mp4", ".mp3", ".wav", ".avif"}
_ALWAYS_INCLUDE = {"readme.md", "license", "license.md", "license.txt", "notice", "usage_policy.md"}

_SIZE_REFUSE_BYTES = 100 * 1024 * 1024 * 1024

# dtype token -> filename variant tags it matches (``model.fp16.safetensors``)
_DTYPE_TAGS: dict[str, tuple[str, ...]] = {
    "bf16": ("bf16",),
    "fp16": ("fp16", "f16", "half"),
    "fp32": ("fp32", "f32"),
    "fp8": ("fp8", "fp8_e4m3fn", "fp8-e4m3", "fp8_e5m2"),
}
_VARIANT_TAG_RE = re.compile(r"\.([a-z0-9_\-]+)\.safetensors$")

_GGUF_QUANT_PREFERENCE = (
    "q8_0", "q6_k", "q5_k_m", "q5_k_s", "q4_k_m", "q4_k_s", "q4_0",
    "q3_k_m", "q3_k_s", "q2_k", "f16", "bf16", "f32",
)

# Full quant token in a gguf FILENAME, incl. unsloth-dynamic ("UD-Q4_K_XL")
# and i-quant ("IQ4_XS") forms the preference list doesn't name.
_GGUF_QTYPE_RE = re.compile(r"(?:ud-)?(?:i?q\d[0-9a-z_]*|bf16|f16|f32)")


class RepoRefusal(RuntimeError):
    """The repo can't be ingested. ``reason`` is a stable machine token:

    pickle_only | onnx_only | tf_only | flax_only | coreml_only |
    tensorrt_only | unknown_shape | too_large
    """

    def __init__(self, reason: str, *, files_seen: Sequence[str] = (), detail: str = "") -> None:
        self.reason = str(reason)
        self.files_seen = list(files_seen)[:32]
        msg = f"repo refused: {reason}"
        if detail:
            msg += f" ({detail})"
        super().__init__(msg)


@dataclass(frozen=True)
class RepoClassification:
    """Result of :func:`classify_repo` + :func:`select_files`."""

    strategy: str          # diffusers | transformers | peft | sentence_transformers |
                           # gguf | native_lora | aio_singlefile
    runtime_library: str   # diffusers | transformers | peft | sentence-transformers |
                           # llama-cpp | diffusers-single-file | diffusers-lora
    allow_patterns: list[str]
    attrs: dict[str, str] = field(default_factory=dict)
    detection_reason: str = ""


def _norm(p: str) -> str:
    return str(p or "").strip().replace("\\", "/").lstrip("/")


def _ext(p: str) -> str:
    base = p.rsplit("/", 1)[-1].lower()
    return "." + base.rsplit(".", 1)[-1] if "." in base else ""


def _root(paths: Sequence[str]) -> list[str]:
    return [p for p in paths if "/" not in p]


def _variant_tag(path: str) -> str:
    m = _VARIANT_TAG_RE.search(path.rsplit("/", 1)[-1].lower())
    return m.group(1) if m else ""


def _dtype_of_tag(tag: str) -> str:
    for dtype, tags in _DTYPE_TAGS.items():
        if tag in tags:
            return dtype
    return ""


def _pick_weight_set(
    safetensors_paths: Sequence[str],
    dtype_pref: Sequence[str],
) -> tuple[list[str], str]:
    """Pick ONE dtype-variant weight set from a component's safetensors files.

    Groups files by variant tag (sharded sets share a tag), then picks the
    first preferred dtype present; untagged files are the fallback group.
    Returns (paths, resolved_dtype) — dtype "" when untagged.
    """
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
    # Only tagged variants none of which matched the preference — take the
    # first preferred-order tag deterministically.
    tag = sorted(by_tag)[0]
    return sorted(by_tag[tag]), _dtype_of_tag(tag)


def classify_repo(
    files: Sequence[str],
    *,
    sizes: Mapping[str, int] | None = None,
    config_json: Mapping[str, object] | None = None,
    safetensors_metadata: Mapping[str, str] | None = None,
    readme_tags: Sequence[str] = (),
    dtype_pref: Sequence[str] = ("bf16", "fp16", "fp32"),
    gguf_quant: Optional[str] = None,
) -> RepoClassification:
    """Classify a HF repo from its file listing and build allow_patterns.

    ``config_json`` is the parsed root ``config.json`` when present (needed
    only to distinguish transformers repos). ``safetensors_metadata`` is the
    ``__metadata__`` block of the largest root safetensors (kohya LoRA
    detection; pass ``huggingface_hub.get_safetensors_metadata`` output).
    Raises :class:`RepoRefusal` when the repo has no ingestable weights.
    """
    paths = [_norm(p) for p in files if _norm(p)]
    sizes = dict(sizes or {})
    root = _root(paths)
    root_set = {p.lower() for p in root}

    configs = [
        p for p in paths
        if _ext(p) in {".json", ".txt", ".model", ".jinja", ".yaml", ".yml"}
        and _ext(p) not in _JUNK_EXTS
        and not p.lower().endswith(".safetensors.index.json")
    ]
    always = [p for p in root if p.lower() in _ALWAYS_INCLUDE]

    def _finish(strategy: str, library: str, weights: list[str], indexes: list[str],
                attrs: dict[str, str], reason: str) -> RepoClassification:
        allow = sorted(set(weights) | set(indexes) | set(configs) | set(always))
        # Size gate on the SELECTED set, not the whole repo: only
        # allow_patterns are downloaded, and multi-quant GGUF repos
        # legitimately total 100s of GB while one quant is ~18GB.
        selected = sum(int(sizes.get(p, 0)) for p in allow)
        if selected > _SIZE_REFUSE_BYTES:
            raise RepoRefusal("too_large", files_seen=paths,
                              detail=f"{selected} bytes selected")
        return RepoClassification(
            strategy=strategy, runtime_library=library, allow_patterns=allow,
            attrs=attrs, detection_reason=reason,
        )

    # 1. sentence-transformers
    if "modules.json" in root_set or "config_sentence_transformers.json" in root_set:
        weights, dtype = _pick_weight_set(
            [p for p in paths if p.lower().endswith(".safetensors")], dtype_pref)
        if not weights:
            raise RepoRefusal("missing_safetensors", files_seen=paths)
        return _finish("sentence_transformers", "sentence-transformers", weights, [],
                       {"dtype": dtype or "fp32"}, "modules.json at root")

    # 2. PEFT adapter
    if "adapter_config.json" in root_set:
        weights = sorted(p for p in root if p.lower().endswith(".safetensors"))
        if not weights:
            raise RepoRefusal("missing_safetensors", files_seen=paths)
        return _finish("peft", "peft", weights, [], {}, "adapter_config.json at root")

    # 3. diffusers pipeline
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
                # Root-level safetensors next to model_index.json are
                # all-in-one duplicate checkpoints of the component tree
                # (e.g. SD1.5's v1-5-pruned*.safetensors, 12GB on top of a
                # 2.7GB fp16 component set) — never part of a
                # diffusers-layout ingest.
                continue
            comp_weights, comp_dtype = _pick_weight_set(group, dtype_pref)
            d_weights.extend(comp_weights)
            if comp and comp_dtype:
                resolved[comp] = comp_dtype
        # Sharded-set indexes for the picked weights only.
        picked_set = set(d_weights)
        for p in paths:
            if p.lower().endswith(".safetensors.index.json"):
                stem = p[: -len(".index.json")]
                tag = _variant_tag(stem)
                comp = p.split("/", 1)[0] if "/" in p else ""
                comp_pick = [w for w in picked_set if w.startswith(f"{comp}/")] if comp else list(picked_set)
                if any(_variant_tag(w) == tag for w in comp_pick):
                    d_indexes.append(p)
        if not d_weights:
            raise RepoRefusal("missing_safetensors", files_seen=paths)
        dtypes = sorted(set(resolved.values()))
        return _finish("diffusers", "diffusers", d_weights, d_indexes,
                       {"dtype": dtypes[0] if len(dtypes) == 1 else "",
                        "file_layout": "diffusers"},
                       "model_index.json at root")

    has_st = any(p.lower().endswith(".safetensors") for p in paths)
    has_st_index = any(p.lower().endswith(".safetensors.index.json") for p in paths)

    # 4. transformers
    if config_json is not None and "config.json" in root_set and (has_st or has_st_index):
        t_indexes = [p for p in root if p.lower().endswith(".safetensors.index.json")]
        t_weights, t_dtype = _pick_weight_set(
            [p for p in root if p.lower().endswith(".safetensors")], dtype_pref)
        attrs: dict[str, str] = {"file_layout": "singlefile"}
        if t_dtype:
            attrs["dtype"] = t_dtype
        arch = config_json.get("architectures")
        if isinstance(arch, list) and arch:
            attrs["architecture"] = str(arch[0])
        if config_json.get("quantization_config") is not None:
            attrs["subtype"] = "quantized"
        return _finish("transformers", "transformers", t_weights, t_indexes, attrs,
                       "config.json + safetensors")

    # 5. GGUF
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
                        "file_type": "gguf", "file_layout": "singlefile"},
                       f"{len(gguf_files)} *.gguf at root")

    # 6/7. root safetensors: native LoRA vs AIO singlefile
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
            attrs = {"file_layout": "singlefile"}
            for k in ("ss_base_model_version", "ss_sd_model_name", "ss_network_module"):
                if md.get(k):
                    attrs[f"kohya_{k}"] = str(md[k])
            return _finish("native_lora", "diffusers-lora", sorted(st_root), [], attrs,
                           "root safetensors + LoRA signature")
        if len(st_root) == 1:
            return _finish("aio_singlefile", "diffusers-single-file", st_root, [],
                           {"file_layout": "singlefile"},
                           "single safetensors at root, no structured signals")
        # Multiple untyped root safetensors: pick by dtype preference.
        weights, dtype = _pick_weight_set(st_root, dtype_pref)
        return _finish("aio_singlefile", "diffusers-single-file", weights, [],
                       {"file_layout": "singlefile", "dtype": dtype},
                       "root safetensors, dtype-variant pick")

    # Refusals — say what IS there.
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


__all__ = ["RepoClassification", "RepoRefusal", "classify_repo"]
