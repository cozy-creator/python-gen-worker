"""HuggingFace repo classifier and per-strategy file selectors.

Implements the design from e2e progress.json #67: classify a HuggingFace repo
by shape (diffusers / transformers / peft / native-LoRA / sentence-transformers
/ gguf / aio-singlefile), then run the strategy-specific selector that returns
the minimum file set needed to load the model.

Hard policy: pickle weight formats (.bin / .ckpt / .pt / .pth /
consolidated.*.pth) are NEVER downloaded. A repo whose only weights are pickle
is refused with `RepoPickleOnly`.

The classifier runs on a file listing + tiny config files (model_index.json,
config.json, adapter_config.json, modules.json, README.md) plus optional
safetensors `__metadata__` peeks for native-LoRA detection. No weight bytes
are downloaded during classification.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence


# ---------------------------------------------------------------------------
# Constants: pickle blocklist + junk-extension filter + always-include list
# ---------------------------------------------------------------------------

# Pickle weight extensions. NEVER downloaded — arbitrary code execution risk.
_PICKLE_EXTS: frozenset[str] = frozenset({".bin", ".ckpt", ".pt", ".pth"})

# Pickle filename patterns we explicitly reject regardless of extension match
# (e.g. Mistral's `consolidated.00.pth` which already has .pth — covered by
# _PICKLE_EXTS — but listed here for documentation).
_PICKLE_BASENAME_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^consolidated\.\d+\.pth$", re.IGNORECASE),
    re.compile(r"^pytorch_model.*\.bin$", re.IGNORECASE),
)

# Non-pickle, non-PyTorch sibling formats. Skipped at selection time so they're
# never downloaded (today these were filtered AFTER download — wasteful).
_JUNK_EXTS: frozenset[str] = frozenset({
    ".onnx", ".onnx_data",          # ONNX exports
    ".tflite",                      # TFLite
    ".h5",                          # TF/Keras
    ".msgpack",                     # Flax/JAX
    ".engine",                      # TensorRT
    ".mlmodel", ".mlpackage",       # CoreML
    ".gguf",                        # only allowed in GGUF strategy; junk elsewhere
})

# OpenVINO files have generic .xml/.bin extensions; match by basename.
_OPENVINO_BASENAMES: frozenset[str] = frozenset({
    "openvino_model.xml",
    "openvino_model.bin",
})

# Demo / preview media. Never weight files.
_DEMO_EXTS: frozenset[str] = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".bmp",
    ".mp4", ".mov", ".mkv", ".avi",
})

# Always-include allowlist (when present at root). Required for downstream
# attribution / license compliance / model card display.
_ALWAYS_INCLUDE_FILENAMES: frozenset[str] = frozenset({
    "license", "license.md", "license.txt",
    "notice", "notice.md", "notice.txt",
    "use_policy.md", "use_policy.txt",
    "model_card.md",
    "readme.md",
})

# Per-clone size budget thresholds.
_SIZE_WARN_BYTES = 25 * 1024 * 1024 * 1024
_SIZE_REFUSE_BYTES = 100 * 1024 * 1024 * 1024


# ---------------------------------------------------------------------------
# Refusal exception types
# ---------------------------------------------------------------------------

class RepoRefusal(RuntimeError):
    """Base class for refusals during classification or selection."""

    refusal_kind: str = "unknown"
    user_message: str = ""

    def __init__(self, *, files_seen: Sequence[str], extra: str = "") -> None:
        self.files_seen = list(files_seen)[:50]  # cap log spam
        self.extra = extra
        msg = self.user_message
        if extra:
            msg = f"{msg} ({extra})"
        super().__init__(f"{self.refusal_kind}: {msg}")


class RepoPickleOnly(RepoRefusal):
    refusal_kind = "pickle_only"
    user_message = (
        "this repo only ships pickle weights (.bin/.ckpt/.pt/.pth). "
        "Pickle is arbitrary code execution; we don't load it under any "
        "condition. Convert the source to safetensors upstream and re-upload, "
        "or pick a repo that already ships safetensors."
    )


class RepoTfOnly(RepoRefusal):
    refusal_kind = "tf_keras_only"
    user_message = (
        "this repo only ships TF/Keras weights (tf_model.h5). "
        "We don't run TensorFlow models — pick a PyTorch repo with safetensors."
    )


class RepoOnnxOnly(RepoRefusal):
    refusal_kind = "onnx_only"
    user_message = (
        "this repo only ships ONNX exports (*.onnx). "
        "We don't run ONNX in our worker stack — pick a PyTorch repo with safetensors."
    )


class RepoOpenVinoOnly(RepoRefusal):
    refusal_kind = "openvino_only"
    user_message = (
        "this repo only ships OpenVINO exports (openvino_model.xml/bin). "
        "We don't run OpenVINO — pick a PyTorch repo with safetensors."
    )


class RepoTfliteOnly(RepoRefusal):
    refusal_kind = "tflite_only"
    user_message = "this repo only ships TFLite weights (*.tflite). Not supported."


class RepoTensorRtOnly(RepoRefusal):
    refusal_kind = "tensorrt_only"
    user_message = (
        "this repo only ships TensorRT engine binaries (*.engine). "
        "Engines are GPU-specific compiled blobs — ingest the source repo instead."
    )


class RepoCoreMlOnly(RepoRefusal):
    refusal_kind = "coreml_only"
    user_message = (
        "this repo only ships CoreML packages (*.mlmodel / *.mlpackage). Not supported."
    )


class RepoFlaxOnly(RepoRefusal):
    refusal_kind = "flax_only"
    user_message = (
        "this repo only ships Flax/JAX weights (*.msgpack). "
        "Convert to PyTorch safetensors upstream or pick a different repo."
    )


class RepoNemoOnly(RepoRefusal):
    refusal_kind = "nemo_only"
    user_message = (
        "this repo only ships NVIDIA NeMo archives (*.nemo). Not supported."
    )


class RepoUnknownShape(RepoRefusal):
    refusal_kind = "unknown_shape"
    user_message = (
        "no recognized shape (no model_index.json / config.json / "
        "adapter_config.json / modules.json / *.safetensors / *.gguf at root)."
    )


class RepoTooLarge(RepoRefusal):
    refusal_kind = "too_large"
    user_message = (
        "selected files exceed 100 GB; re-run with --allow-large to override."
    )


class RepoMissingSafetensors(RepoRefusal):
    refusal_kind = "missing_safetensors"
    user_message = (
        "a required component has no .safetensors weight (only pickle/.bin/.ckpt). "
        "Convert upstream or pick a different repo."
    )


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RepoClassification:
    """Output of `classify_huggingface_repo`."""

    strategy: str  # diffusers | transformers | peft_canonical | native_lora
                   # | sentence_transformers | gguf | aio_singlefile
    runtime_library: str  # diffusers | transformers | peft | sentence-transformers
                          # | llama-cpp | diffusers-single-file | diffusers-lora
    subtype: str = ""  # vlm | audio | encoder | decoder | encoder_decoder |
                       # quantized | "" (none)
    refusal: Optional[RepoRefusal] = None  # None on success
    detection_reason: str = ""


@dataclass(frozen=True)
class SelectionResult:
    """Output of a strategy selector."""

    selected_paths: list[str]
    skipped_paths: list[str]
    attrs: dict[str, str]  # destination checkpoint attributes
    pickle_files_refused: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize(path: str) -> str:
    return str(path or "").strip().replace("\\", "/").lstrip("/")


def _basename(path: str) -> str:
    return path.rsplit("/", 1)[-1]


def _ext(path: str) -> str:
    base = _basename(path).lower()
    if "." not in base:
        return ""
    return "." + base.rsplit(".", 1)[-1]


def _is_pickle(path: str) -> bool:
    if _ext(path) in _PICKLE_EXTS:
        return True
    base = _basename(path).lower()
    return any(p.match(base) for p in _PICKLE_BASENAME_PATTERNS)


def _is_junk(path: str) -> bool:
    """Non-pickle, non-PyTorch siblings we never download (regardless of strategy).

    GGUF is in _JUNK_EXTS but the GGUF strategy unmasks it via a separate path.
    """
    return _ext(path) in _JUNK_EXTS or _basename(path).lower() in _OPENVINO_BASENAMES


def _is_demo(path: str) -> bool:
    return _ext(path) in _DEMO_EXTS


def _is_always_include(path: str) -> bool:
    """Root-level files we always pull (LICENSE, README, NOTICE)."""
    if "/" in path:
        return False
    return _basename(path).lower() in _ALWAYS_INCLUDE_FILENAMES


def _root_files(paths: Sequence[str]) -> list[str]:
    return [p for p in paths if "/" not in p]


def _component_files(paths: Sequence[str], component: str) -> list[str]:
    prefix = f"{component}/"
    return [p for p in paths if p.startswith(prefix)]


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

@dataclass
class ClassificationInputs:
    """All the cheap signals the classifier consults.

    Construct in the ingest layer (one HfApi.list_repo_tree + small file
    fetches) and pass in. `model_index_json` etc. are populated only when the
    listing showed those files exist.
    """

    file_paths: list[str]
    file_sizes: dict[str, int] = field(default_factory=dict)
    # Tiny config files (parsed JSON). None if not present in the listing.
    model_index_json: Optional[Mapping[str, object]] = None
    config_json: Optional[Mapping[str, object]] = None
    adapter_config_json: Optional[Mapping[str, object]] = None
    modules_json: Optional[Mapping[str, object]] = None
    config_sentence_transformers_json: Optional[Mapping[str, object]] = None
    # README YAML frontmatter (parsed). Empty dict if absent.
    readme_frontmatter: Mapping[str, object] = field(default_factory=dict)
    # safetensors `__metadata__` block for the largest root-level safetensors,
    # if any (used for native-LoRA detection — kohya signature).
    root_safetensors_metadata: Optional[Mapping[str, str]] = None
    root_safetensors_path: Optional[str] = None  # which file's metadata we read


def classify_huggingface_repo(inputs: ClassificationInputs) -> RepoClassification:
    """Classify a HF repo by shape using cheap signals only.

    First-match-wins per the algorithm in e2e progress.json #67. Returns a
    RepoClassification; refusals carry a `refusal` field set to the appropriate
    exception (caller decides whether to raise).
    """
    paths = [_normalize(p) for p in inputs.file_paths if _normalize(p)]
    root = _root_files(paths)

    # 1. Sentence-Transformers
    if inputs.modules_json is not None or inputs.config_sentence_transformers_json is not None:
        return RepoClassification(
            strategy="sentence_transformers",
            runtime_library="sentence-transformers",
            detection_reason="modules.json or config_sentence_transformers.json at root",
        )

    # 2. PEFT canonical
    if inputs.adapter_config_json is not None:
        return RepoClassification(
            strategy="peft_canonical",
            runtime_library="peft",
            detection_reason="adapter_config.json at root",
        )

    # 3. Diffusers
    if inputs.model_index_json is not None:
        return RepoClassification(
            strategy="diffusers",
            runtime_library="diffusers",
            detection_reason="model_index.json at root",
        )

    # 4. HF Transformers
    has_safetensors = any(p.lower().endswith(".safetensors") for p in paths)
    has_safetensors_index = any(p.lower().endswith(".safetensors.index.json") for p in paths)
    if inputs.config_json is not None and (has_safetensors or has_safetensors_index):
        cfg = dict(inputs.config_json)
        # Subtype detection (tag only).
        subtype = _detect_transformers_subtype(cfg, root)
        return RepoClassification(
            strategy="transformers",
            runtime_library="transformers",
            subtype=subtype,
            detection_reason="config.json + safetensors at root",
        )

    # 5. GGUF
    root_gguf = [p for p in root if p.lower().endswith(".gguf")]
    if root_gguf:
        return RepoClassification(
            strategy="gguf",
            runtime_library="llama-cpp",
            detection_reason=f"{len(root_gguf)} *.gguf at root",
        )

    # 6. Native LoRA (must come before AIO/Singlefile)
    root_safetensors = [p for p in root if p.lower().endswith(".safetensors")]
    if root_safetensors:
        if _looks_like_native_lora(root_safetensors, inputs):
            return RepoClassification(
                strategy="native_lora",
                runtime_library="diffusers-lora",
                detection_reason="safetensors at root + LoRA signature",
            )
        # 7. AIO/Singlefile (single full-model safetensors at root)
        if len(root_safetensors) == 1:
            return RepoClassification(
                strategy="aio_singlefile",
                runtime_library="diffusers-single-file",
                detection_reason="single safetensors at root, no structured signals",
            )

    # 8-10: Refusals based on what weights ARE present.
    return _classify_refusal(paths)


def _detect_transformers_subtype(config: Mapping[str, object], root: Sequence[str]) -> str:
    """Tag-only subtype for transformers: vlm | audio | quantized | encoder_decoder
    | encoder | decoder | (empty)."""
    root_set = {p.lower() for p in root}
    if "preprocessor_config.json" in root_set or "processor_config.json" in root_set or "image_processor_config.json" in root_set:
        return "vlm"
    if "feature_extractor_config.json" in root_set:
        return "audio"
    if config.get("quantization_config") is not None:
        return "quantized"
    if "quantize_config.json" in root_set or "awq_config.json" in root_set:
        return "quantized"
    architectures = config.get("architectures")
    if isinstance(architectures, list) and architectures:
        arch = str(architectures[0])
        arch_lower = arch.lower()
        if "encoderdecoder" in arch_lower or "seq2seq" in arch_lower or arch_lower.startswith(("t5", "bart", "marian", "mbart", "pegasus")):
            return "encoder_decoder"
        if arch_lower.endswith("forcausallm") or "causallm" in arch_lower:
            return "decoder"
        return "encoder"
    return ""


def _looks_like_native_lora(root_safetensors: Sequence[str], inputs: ClassificationInputs) -> bool:
    """Distinguish native LoRA from full-model AIO/singlefile.

    Strongest → weakest signal:
      1. safetensors __metadata__ contains `ss_network_module` or `ss_base_model_version`
         (kohya signature)
      2. README YAML frontmatter `tags:` includes `lora` (or `lycoris`)
      3. all root safetensors are below the LoRA size threshold (1 GB)
    """
    md = inputs.root_safetensors_metadata or {}
    if any(k.startswith("ss_") for k in md.keys()):
        return True
    fm_tags = inputs.readme_frontmatter.get("tags") if inputs.readme_frontmatter else None
    if isinstance(fm_tags, list):
        for t in fm_tags:
            ts = str(t or "").strip().lower()
            if ts in ("lora", "lycoris", "loha", "lokr", "adapter"):
                return True
    # Size heuristic — every root safetensors below 1 GB.
    sizes = inputs.file_sizes or {}
    threshold = 1 * 1024 * 1024 * 1024  # 1 GB
    if all(int(sizes.get(p, 0)) < threshold for p in root_safetensors):
        # AND there's no config.json that would route to transformers — already
        # excluded above.
        return True
    return False


def _classify_refusal(paths: Sequence[str]) -> RepoClassification:
    """Determine which refusal to surface based on the file extensions present."""
    pickle_files = [p for p in paths if _is_pickle(p)]
    onnx_files = [p for p in paths if _ext(p) in (".onnx", ".onnx_data")]
    openvino_files = [p for p in paths if _basename(p).lower() in _OPENVINO_BASENAMES]
    flax_files = [p for p in paths if _ext(p) == ".msgpack"]
    h5_files = [p for p in paths if _ext(p) == ".h5"]
    tflite_files = [p for p in paths if _ext(p) == ".tflite"]
    engine_files = [p for p in paths if _ext(p) == ".engine"]
    coreml_files = [p for p in paths if _ext(p) in (".mlmodel", ".mlpackage")]
    nemo_files = [p for p in paths if _ext(p) == ".nemo"]
    has_safetensors = any(p.lower().endswith(".safetensors") for p in paths)
    has_gguf = any(p.lower().endswith(".gguf") for p in paths)

    # If safetensors exist but we got here, the structured signals were missing
    # → unknown shape (not a "weight format" refusal).
    if has_safetensors or has_gguf:
        return RepoClassification(
            strategy="refuse",
            runtime_library="",
            detection_reason="safetensors/gguf present but no structured marker",
            refusal=RepoUnknownShape(files_seen=paths),
        )

    # Pickle-only: weight files exist but they're all pickle.
    if pickle_files and not (onnx_files or openvino_files or flax_files or h5_files or tflite_files or engine_files or coreml_files or nemo_files):
        return RepoClassification(
            strategy="refuse",
            runtime_library="",
            refusal=RepoPickleOnly(files_seen=pickle_files),
        )
    if onnx_files and not pickle_files:
        return RepoClassification(strategy="refuse", runtime_library="",
                                   refusal=RepoOnnxOnly(files_seen=onnx_files))
    if openvino_files and not pickle_files:
        return RepoClassification(strategy="refuse", runtime_library="",
                                   refusal=RepoOpenVinoOnly(files_seen=openvino_files))
    if h5_files and not pickle_files:
        return RepoClassification(strategy="refuse", runtime_library="",
                                   refusal=RepoTfOnly(files_seen=h5_files))
    if tflite_files and not pickle_files:
        return RepoClassification(strategy="refuse", runtime_library="",
                                   refusal=RepoTfliteOnly(files_seen=tflite_files))
    if engine_files and not pickle_files:
        return RepoClassification(strategy="refuse", runtime_library="",
                                   refusal=RepoTensorRtOnly(files_seen=engine_files))
    if coreml_files and not pickle_files:
        return RepoClassification(strategy="refuse", runtime_library="",
                                   refusal=RepoCoreMlOnly(files_seen=coreml_files))
    if flax_files and not pickle_files:
        return RepoClassification(strategy="refuse", runtime_library="",
                                   refusal=RepoFlaxOnly(files_seen=flax_files))
    if nemo_files:
        return RepoClassification(strategy="refuse", runtime_library="",
                                   refusal=RepoNemoOnly(files_seen=nemo_files))
    return RepoClassification(strategy="refuse", runtime_library="",
                               refusal=RepoUnknownShape(files_seen=paths))


# ---------------------------------------------------------------------------
# Per-strategy selectors
# ---------------------------------------------------------------------------

def select_for_classification(
    classification: RepoClassification,
    inputs: ClassificationInputs,
    *,
    dtype_pref: Sequence[str] = ("bf16", "fp16", "fp32"),
    gguf_quant: Optional[str] = None,
    weight_index_json_by_file: Optional[Mapping[str, Mapping[str, object]]] = None,
) -> SelectionResult:
    """Dispatch to the right per-strategy selector."""
    if classification.refusal is not None:
        raise classification.refusal
    strategy = classification.strategy
    if strategy == "diffusers":
        return _select_diffusers(inputs, dtype_pref, weight_index_json_by_file or {})
    if strategy == "transformers":
        return _select_transformers(inputs, dtype_pref, classification.subtype)
    if strategy == "peft_canonical":
        return _select_peft_canonical(inputs)
    if strategy == "native_lora":
        return _select_native_lora(inputs)
    if strategy == "sentence_transformers":
        return _select_sentence_transformers(inputs, dtype_pref)
    if strategy == "gguf":
        return _select_gguf(inputs, gguf_quant)
    if strategy == "aio_singlefile":
        return _select_aio_singlefile(inputs)
    raise ValueError(f"unknown classification strategy: {strategy!r}")


def select_for_classification_multi(
    classification: RepoClassification,
    inputs: ClassificationInputs,
    *,
    dtype_prefs: Sequence[str] = (),
    gguf_quants: Sequence[str] = (),
    weight_index_json_by_file: Optional[Mapping[str, Mapping[str, object]]] = None,
) -> list[SelectionResult]:
    """Multi-dtype variant of `select_for_classification` (e2e #72).

    Returns one `SelectionResult` per requested concrete dtype. Today
    multi-dtype is only meaningful for GGUF (one checkpoint per quant
    level under the same destination tag); transformers and diffusers
    return a single-element list per Phase-1 scope. Phase-2 detection
    of multi-variant transformers/diffusers (`model.bf16.safetensors`
    + `model.fp16.safetensors` side by side) is tracked separately.

    When `dtype_prefs`/`gguf_quants` is empty, returns a single-element
    list matching `select_for_classification` default behavior.

    Raises `DtypeUnavailable` when a requested concrete dtype isn't
    served by the source repo (with the available set listed for the
    user-facing 400).
    """
    if classification.refusal is not None:
        raise classification.refusal

    strategy = classification.strategy
    if strategy != "gguf":
        # Single-dtype fallback path. The dtype_pref tuple here is a
        # priority list (first-match-wins), not a multi-output list, so
        # we don't multiplex outputs for non-gguf today.
        eff_pref = tuple(dtype_prefs) or ("bf16", "fp16", "fp32")
        single = select_for_classification(
            classification,
            inputs,
            dtype_pref=eff_pref,
            gguf_quant=None,
            weight_index_json_by_file=weight_index_json_by_file,
        )
        return [single]

    # GGUF: one selection per requested quant.
    requested = [q.strip() for q in gguf_quants if q and q.strip()]
    if not requested:
        return [select_for_classification(classification, inputs, gguf_quant=None)]
    out: list[SelectionResult] = []
    for q in requested:
        out.append(_select_gguf(inputs, q))
    return out


# Apply the always-include allowlist (LICENSE/README/NOTICE) + defense-in-depth
# pickle/junk/demo filters.
def _finalize(
    selected: list[str],
    inputs: ClassificationInputs,
    attrs: dict[str, str],
) -> SelectionResult:
    selected_set = {_normalize(p) for p in selected}
    pickle_refused: list[str] = []
    final: list[str] = []
    for p in selected:
        np = _normalize(p)
        if _is_pickle(np):
            pickle_refused.append(np)
            continue
        if _is_junk(np) and not np.lower().endswith(".gguf"):
            # GGUF unmasked only inside the gguf strategy
            continue
        if _is_demo(np):
            continue
        final.append(np)

    # Always-include extras present at root.
    for p in inputs.file_paths:
        np = _normalize(p)
        if not np or np in selected_set:
            continue
        if _is_always_include(np):
            final.append(np)

    final_unique = sorted(set(final))
    skipped = sorted(set(_normalize(p) for p in inputs.file_paths) - set(final_unique))
    return SelectionResult(
        selected_paths=final_unique,
        skipped_paths=skipped,
        attrs=attrs,
        pickle_files_refused=pickle_refused,
    )


def _pick_dtype_safetensors(
    candidates: Sequence[str],
    dtype_pref: Sequence[str],
) -> list[str]:
    """Among multiple .safetensors variants for the same component+base, pick
    the best-matching dtype per the preference order.

    Logic:
      - filename suffix like `model.fp16.safetensors` / `model.bf16.safetensors`
        / `model.fp8.safetensors` declares the dtype.
      - unsuffixed `model.safetensors` is unknown dtype (could be bf16/fp32/etc.)
        — we'd need to peek the header to know. For selection purposes treat as
        the lowest-priority match (only picked when there's nothing else).
      - Sharded variants (`model-00001-of-00002.fp16.safetensors`) keep all shards.

    This function operates on a single component's candidates. The caller is
    responsible for grouping by component first.
    """
    if not candidates:
        return []
    # Group by base_name (strip dtype suffix and shard index).
    by_dtype: dict[str, list[str]] = {}

    def _classify(p: str) -> str:
        low = _basename(p).lower()
        for dtype in ("bf16", "fp16", "fp8", "fp32", "nvfp4"):
            if f".{dtype}." in low or low.endswith(f".{dtype}.safetensors"):
                return dtype
        return "default"

    for p in candidates:
        by_dtype.setdefault(_classify(p), []).append(p)

    # Pick by preference.
    for pref in list(dtype_pref) + ["default", "fp32"]:
        pref_l = pref.lower()
        if pref_l in by_dtype:
            return sorted(by_dtype[pref_l])
    # No match — return all, caller decides
    return sorted(candidates)


# --- Diffusers ---

# Diffusers components that are nice-to-have but not required for the model
# itself to load. SDXL family ships safety_checker as pickle-only — refusing
# the entire clone over an optional NSFW filter is wrong UX.
_DIFFUSERS_OPTIONAL_COMPONENTS: frozenset[str] = frozenset({
    "safety_checker",
    "feature_extractor",
    "image_encoder",  # Some pipelines ship this only for IP-Adapter use; skip if pickle-only
})


def _select_diffusers(
    inputs: ClassificationInputs,
    dtype_pref: Sequence[str],
    weight_index_json_by_file: Mapping[str, Mapping[str, object]],
) -> SelectionResult:
    mi = inputs.model_index_json or {}
    components = [k for k in mi.keys() if isinstance(k, str) and not k.startswith("_")]
    paths = [_normalize(p) for p in inputs.file_paths]
    selected: list[str] = ["model_index.json"]
    attrs: dict[str, str] = {
        "runtime_library": "diffusers",
        "file_layout": "diffusers",
        "file_type": "safetensors",
    }
    cls_name = mi.get("_class_name")
    if isinstance(cls_name, str) and cls_name.strip():
        attrs["pipeline_class"] = cls_name.strip()

    sizes = inputs.file_sizes or {}

    for component in components:
        comp_files = _component_files(paths, component)
        if not comp_files:
            continue  # component declared but no files (rare; diffusers will fail)

        # Configs / tokenizers / schedulers — keep all small JSON/TXT in the dir.
        for p in comp_files:
            base_low = _basename(p).lower()
            if (base_low.endswith(".json") or base_low.endswith(".txt")
                    or base_low.endswith(".jinja") or base_low.endswith(".model")):
                selected.append(p)

        # Weight selection: prefer .safetensors per dtype preference.
        weights = [p for p in comp_files if p.lower().endswith(".safetensors")
                   and not p.lower().endswith(".safetensors.index.json")]
        index_files = [p for p in comp_files if p.lower().endswith(".safetensors.index.json")]

        if not weights and not index_files:
            # Component has no safetensors at all.
            pickle_in_comp = [p for p in comp_files if _is_pickle(p)]
            if pickle_in_comp:
                # Optional components (safety_checker / feature_extractor /
                # image_encoder for IP-Adapter use) — skip rather than refuse
                # the whole clone. SDXL family ships safety_checker as pickle.
                if component in _DIFFUSERS_OPTIONAL_COMPONENTS:
                    continue
                raise RepoMissingSafetensors(
                    files_seen=pickle_in_comp,
                    extra=f"component={component} has only pickle weights",
                )
            # No weights at all → component has only configs (e.g. scheduler), fine.
            continue

        # Sharded path: pick the best dtype index, then expand to shards.
        if index_files:
            chosen_index = _pick_dtype_index(index_files, dtype_pref)
            if chosen_index:
                selected.append(chosen_index)
                # Try to expand shards from the index JSON if provided.
                idx_data = weight_index_json_by_file.get(chosen_index)
                if idx_data:
                    weight_map = idx_data.get("weight_map") or {}
                    if isinstance(weight_map, dict):
                        prefix = chosen_index.rsplit("/", 1)[0] + "/" if "/" in chosen_index else ""
                        for shard in {str(v) for v in weight_map.values() if isinstance(v, str)}:
                            full = shard if "/" in shard else (prefix + shard)
                            if full in paths:
                                selected.append(full)
                else:
                    # Index JSON contents not provided — fall back to picking
                    # all shards in the component matching the chosen dtype.
                    chosen_dtype = _index_dtype(chosen_index)
                    for w in weights:
                        if chosen_dtype == "default" or f".{chosen_dtype}." in _basename(w).lower():
                            selected.append(w)
        else:
            # Single-file path: pick one safetensors per dtype preference.
            picked = _pick_dtype_safetensors(weights, dtype_pref)
            # If multiple "shards" came back (e.g. model-00001-of-00002 + 00002),
            # keep all of them — they belong to the same chosen dtype.
            selected.extend(picked)

    return _finalize(selected, inputs, attrs)


def _pick_dtype_index(index_files: Sequence[str], dtype_pref: Sequence[str]) -> Optional[str]:
    """Pick one .safetensors.index.json per dtype preference."""
    if not index_files:
        return None
    by_dtype: dict[str, list[str]] = {}
    for p in index_files:
        by_dtype.setdefault(_index_dtype(p), []).append(p)
    for pref in list(dtype_pref) + ["default"]:
        if pref.lower() in by_dtype:
            return sorted(by_dtype[pref.lower()])[0]
    return sorted(index_files)[0]


def _index_dtype(index_path: str) -> str:
    low = _basename(index_path).lower()
    for dtype in ("bf16", "fp16", "fp8", "fp32", "nvfp4"):
        if f".{dtype}." in low:
            return dtype
    return "default"


# --- Transformers ---

def _select_transformers(
    inputs: ClassificationInputs,
    dtype_pref: Sequence[str],
    subtype: str,
) -> SelectionResult:
    paths = [_normalize(p) for p in inputs.file_paths]
    root = _root_files(paths)
    selected: list[str] = []
    attrs: dict[str, str] = {
        "runtime_library": "transformers",
        "file_layout": "transformers",
        "file_type": "safetensors",
    }
    cfg = inputs.config_json or {}
    archs = cfg.get("architectures")
    if isinstance(archs, list) and archs:
        attrs["architecture"] = str(archs[0])
    if subtype:
        attrs["subtype"] = subtype
    qcfg = cfg.get("quantization_config")
    if isinstance(qcfg, dict):
        scheme = qcfg.get("quant_method") or qcfg.get("scheme") or qcfg.get("quantization_method")
        if scheme:
            attrs["quant_scheme"] = str(scheme)

    # Keep every root .json / .txt / .jinja / .model (small files).
    for p in root:
        low = _basename(p).lower()
        if (low.endswith(".json") or low.endswith(".txt")
                or low.endswith(".jinja") or low.startswith("chat_template.")
                or low == "tokenizer.model" or low == "spiece.model"):
            selected.append(p)

    # Weight selection.
    root_safetensors = [p for p in root if p.lower().endswith(".safetensors")
                        and not p.lower().endswith(".safetensors.index.json")]
    root_indexes = [p for p in root if p.lower().endswith(".safetensors.index.json")]

    if root_indexes:
        # Sharded — pick one index by dtype, expand shards via index JSON.
        chosen_index = _pick_dtype_index(root_indexes, dtype_pref)
        if chosen_index:
            # Already in `selected` because it's a .json
            chosen_dtype = _index_dtype(chosen_index)
            for w in root_safetensors:
                wb = _basename(w).lower()
                # Match shards of the same dtype
                if chosen_dtype == "default":
                    # Plain `model.safetensors` shards
                    if not any(f".{d}." in wb for d in ("bf16", "fp16", "fp8", "fp32", "nvfp4")):
                        selected.append(w)
                elif f".{chosen_dtype}." in wb:
                    selected.append(w)
    elif root_safetensors:
        # Single-file or multiple-dtype variants without an index
        picked = _pick_dtype_safetensors(root_safetensors, dtype_pref)
        selected.extend(picked)

    return _finalize(selected, inputs, attrs)


# --- PEFT canonical ---

def _select_peft_canonical(inputs: ClassificationInputs) -> SelectionResult:
    paths = [_normalize(p) for p in inputs.file_paths]
    root = _root_files(paths)
    selected: list[str] = ["adapter_config.json"]
    attrs: dict[str, str] = {
        "runtime_library": "peft",
        "file_layout": "peft",
        "file_type": "safetensors",
    }
    cfg = inputs.adapter_config_json or {}
    if cfg.get("peft_type"):
        attrs["peft_type"] = str(cfg["peft_type"])
    if cfg.get("task_type"):
        attrs["task_type"] = str(cfg["task_type"])
    # e2e progress.json #71: structured base-model lineage.
    base_repo = str(cfg.get("base_model_name_or_path") or "").strip()
    if base_repo:
        # Strip optional @revision suffix for the repo id; keep revision separate.
        if "@" in base_repo:
            repo_part, rev_part = base_repo.split("@", 1)
            attrs["base_model_repo"] = repo_part.strip()
            if rev_part.strip():
                attrs["base_model_revision"] = rev_part.strip()
        else:
            attrs["base_model_repo"] = base_repo
        # Family lookup (best-effort)
        from .base_model_families import repo_to_family
        fam = repo_to_family(attrs.get("base_model_repo", base_repo))
        attrs["base_model_family"] = fam or "unknown"
        # Legacy single-string lineage field (back-compat)
        attrs["base_model_lineage"] = base_repo
        attrs["lineage_source"] = "adapter_config_json"
    if cfg.get("r") is not None:
        attrs["r"] = str(cfg["r"])
    if cfg.get("lora_alpha") is not None:
        attrs["lora_alpha"] = str(cfg["lora_alpha"])

    # Weights: the adapter safetensors at root.
    adapter_st = [p for p in root if p.lower() == "adapter_model.safetensors"]
    if not adapter_st:
        # Try sharded adapter (rare but possible)
        adapter_st = [p for p in root if p.lower().startswith("adapter_model")
                      and p.lower().endswith(".safetensors")]
    if not adapter_st:
        # Only adapter_model.bin exists
        pickle_adapters = [p for p in root if p.lower().startswith("adapter_model")
                           and _is_pickle(p)]
        if pickle_adapters:
            raise RepoMissingSafetensors(
                files_seen=pickle_adapters,
                extra="adapter_model.bin only — no safetensors variant",
            )
    selected.extend(adapter_st)

    # Optional tokenizer (when adapter trained with custom tokens).
    for fname in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                  "added_tokens.json", "tokenizer.model", "vocab.json", "merges.txt"):
        if fname in root:
            selected.append(fname)
    # chat_template.jinja*
    for p in root:
        if _basename(p).lower().startswith("chat_template."):
            selected.append(p)

    return _finalize(selected, inputs, attrs)


# --- Native LoRA ---

def _select_native_lora(inputs: ClassificationInputs) -> SelectionResult:
    paths = [_normalize(p) for p in inputs.file_paths]
    root = _root_files(paths)
    # Keep all root-level safetensors — a single repo often ships sd15 + sdxl
    # variants and we want both.
    selected: list[str] = [p for p in root if p.lower().endswith(".safetensors")]
    attrs: dict[str, str] = {
        "runtime_library": "diffusers-lora",
        "file_layout": "native_lora",
        "file_type": "safetensors",
    }

    # Lineage extraction (priority order, e2e progress.json #71):
    # 1. README YAML frontmatter `base_model:` (specific repo, highest fidelity)
    # 2. safetensors __metadata__ kohya signature: ss_base_model_version (family) +
    #    ss_sd_model_name (specific community fine-tune filename)
    # 3. README YAML `tags:` matched against known base markers (family only)
    # 4. unknown — anonymous community uploads, ingest anyway
    from .base_model_families import (
        kohya_to_family, repo_to_family, tags_to_family, is_canonical_family,
    )
    fm = inputs.readme_frontmatter or {}
    kohya = inputs.root_safetensors_metadata or {}

    base_repo: str = ""
    base_family: str = ""
    lineage_source: str = "none"
    specific_hint: str = ""

    bm = fm.get("base_model")
    bm_str = ""
    if isinstance(bm, str):
        bm_str = bm.strip()
    elif isinstance(bm, list) and bm:
        bm_str = str(bm[0]).strip()

    # Try README YAML frontmatter first (specific repo signal)
    if bm_str:
        # Strip optional @revision suffix
        if "@" in bm_str:
            repo_part, rev_part = bm_str.split("@", 1)
            base_repo = repo_part.strip()
            if rev_part.strip():
                attrs["base_model_revision"] = rev_part.strip()
        else:
            base_repo = bm_str
        fam = repo_to_family(base_repo)
        if fam:
            base_family = fam
        lineage_source = "yaml_frontmatter"

    # If we got a repo but no family, try kohya for the family fallback
    if not base_family:
        kbm = str(kohya.get("ss_base_model_version") or "").strip()
        kfam = kohya_to_family(kbm) if kbm else None
        if kfam:
            base_family = kfam
            if lineage_source == "none":
                lineage_source = "kohya_safetensors_metadata"

    # Specific hint from kohya filename — separate from `base_model_repo`
    # because it's often a community fine-tune like
    # `juggernautXL_v8Rundiffusion.safetensors` not resolvable to an HF repo.
    sd_model_name = str(kohya.get("ss_sd_model_name") or "").strip()
    if sd_model_name:
        specific_hint = sd_model_name

    # Last resort family fallback: HF repo tags
    if not base_family:
        tag_fam = tags_to_family(fm.get("tags"))
        if tag_fam:
            base_family = tag_fam
            if lineage_source == "none":
                lineage_source = "yaml_tags"

    if not base_family:
        base_family = "unknown"

    # Populate the structured attribute set
    if base_repo:
        attrs["base_model_repo"] = base_repo
    attrs["base_model_family"] = base_family
    attrs["lineage_source"] = lineage_source
    if specific_hint:
        attrs["base_model_specific_hint"] = specific_hint
    if bm_str and not base_repo:
        attrs["base_model_repo"] = bm_str

    # Legacy single-string `base_model_lineage` field, back-compat with
    # consumers that haven't migrated to the structured form yet. Priority
    # order matches the pre-#71 behavior: YAML frontmatter > kohya
    # ss_base_model_version > kohya ss_sd_model_name > family fallback.
    legacy_kohya_version = str(kohya.get("ss_base_model_version") or "").strip()
    if base_repo:
        attrs["base_model_lineage"] = base_repo
    elif legacy_kohya_version:
        attrs["base_model_lineage"] = legacy_kohya_version
    elif specific_hint:
        attrs["base_model_lineage"] = specific_hint
    else:
        attrs["base_model_lineage"] = base_family

    # Kohya network metadata (when present).
    for key in ("ss_network_module", "ss_network_dim", "ss_network_alpha"):
        v = kohya.get(key)
        if v:
            attrs[key.removeprefix("ss_")] = str(v)

    return _finalize(selected, inputs, attrs)


_TAG_TO_BASE = {
    "stable-diffusion-xl": "stabilityai/stable-diffusion-xl-base-1.0",
    "stable-diffusion-v1-5": "runwayml/stable-diffusion-v1-5",
    "stable-diffusion": "runwayml/stable-diffusion-v1-5",
    "flux": "black-forest-labs/FLUX.1-dev",
    "flux-1": "black-forest-labs/FLUX.1-dev",
    "flux-dev": "black-forest-labs/FLUX.1-dev",
    "flux-2": "black-forest-labs/FLUX.2-klein-4B",
    "wan2-1": "Wan-AI/Wan2.1-T2V-A14B",
    "wan2-2": "Wan-AI/Wan2.2-T2V-A14B",
}


def _infer_base_from_tags(tags: object) -> Optional[str]:
    if not isinstance(tags, list):
        return None
    for t in tags:
        ts = str(t or "").strip().lower()
        if ts in _TAG_TO_BASE:
            return _TAG_TO_BASE[ts]
    return None


# --- Sentence-Transformers ---

def _select_sentence_transformers(
    inputs: ClassificationInputs,
    dtype_pref: Sequence[str],
) -> SelectionResult:
    paths = [_normalize(p) for p in inputs.file_paths]
    root = _root_files(paths)
    selected: list[str] = []
    attrs: dict[str, str] = {
        "runtime_library": "sentence-transformers",
        "file_layout": "sentence_transformers",
        "file_type": "safetensors",
    }
    # Root files
    for p in root:
        low = _basename(p).lower()
        if low in ("modules.json", "config_sentence_transformers.json",
                   "sentence_bert_config.json", "config.json",
                   "tokenizer.json", "tokenizer_config.json",
                   "special_tokens_map.json", "added_tokens.json",
                   "vocab.json", "merges.txt", "tokenizer.model"):
            selected.append(p)

    # Modules from modules.json (if present). A module with `path: ""` (or
    # `.`) is at the repo root — e.g. `sentence-transformers/all-MiniLM-L6-v2`
    # has its 0_Transformer module at root: model.safetensors + config.json
    # + tokenizer.* live alongside modules.json itself, not in a 0_Transformer
    # subdir. We need to apply transformers-style selection to the root files
    # for those modules, otherwise we miss the weights entirely.
    modules = inputs.modules_json
    module_paths: list[str] = []
    has_root_module = False
    if isinstance(modules, list):
        for entry in modules:
            if isinstance(entry, dict):
                mp = str(entry.get("path") or "").strip()
                if mp and mp != ".":
                    module_paths.append(mp.strip("/"))
                else:
                    has_root_module = True
    if not module_paths and not has_root_module:
        # Fallback: enumerate directories that look like sentence-transformers modules.
        module_paths = sorted({p.split("/", 1)[0] for p in paths if "/" in p
                               and re.match(r"^\d+_", p.split("/", 1)[0])})

    # If the modules.json declared a root-level module (most all-MiniLM-style
    # repos do), apply transformers-style selection to root files too.
    if has_root_module:
        root_files = [p for p in paths if "/" not in p]
        # Pick chosen-dtype safetensors at root.
        root_safetensors = [p for p in root_files if p.lower().endswith(".safetensors")
                            and not p.lower().endswith(".safetensors.index.json")]
        root_indexes = [p for p in root_files if p.lower().endswith(".safetensors.index.json")]
        if root_indexes:
            chosen_index = _pick_dtype_index(root_indexes, dtype_pref)
            if chosen_index:
                # already in `selected` via the JSON branch above
                chosen_dtype = _index_dtype(chosen_index)
                for w in root_safetensors:
                    wb = _basename(w).lower()
                    if chosen_dtype == "default":
                        if not any(f".{d}." in wb for d in ("bf16", "fp16", "fp8", "fp32", "nvfp4")):
                            selected.append(w)
                    elif f".{chosen_dtype}." in wb:
                        selected.append(w)
        elif root_safetensors:
            picked = _pick_dtype_safetensors(root_safetensors, dtype_pref)
            selected.extend(picked)

    # For each named-subdir module, apply transformers-style selection
    for mp in module_paths:
        prefix = f"{mp}/"
        mod_files = [p for p in paths if p.startswith(prefix)]
        # Keep small configs/tokenizers
        for p in mod_files:
            low = _basename(p).lower()
            if (low.endswith(".json") or low.endswith(".txt")
                    or low.endswith(".jinja") or low == "tokenizer.model"):
                selected.append(p)
        # Pick one weight per dtype preference
        weights = [p for p in mod_files if p.lower().endswith(".safetensors")
                   and not p.lower().endswith(".safetensors.index.json")]
        indexes = [p for p in mod_files if p.lower().endswith(".safetensors.index.json")]
        if indexes:
            chosen_index = _pick_dtype_index(indexes, dtype_pref)
            if chosen_index:
                # Already in `selected` via the .json branch
                chosen_dtype = _index_dtype(chosen_index)
                for w in weights:
                    wb = _basename(w).lower()
                    if chosen_dtype == "default":
                        if not any(f".{d}." in wb for d in ("bf16", "fp16", "fp8", "fp32", "nvfp4")):
                            selected.append(w)
                    elif f".{chosen_dtype}." in wb:
                        selected.append(w)
        elif weights:
            picked = _pick_dtype_safetensors(weights, dtype_pref)
            selected.extend(picked)

    return _finalize(selected, inputs, attrs)


# --- GGUF ---

# Known quant levels in PRECISION-FIRST order. e2e progress.json #72 flipped
# the default from lossy-first (Q4_K_M) to high-fidelity-first (F16/BF16) so
# clones default to ingesting the source-of-truth precision when both lossy
# and high-precision variants exist upstream.
_GGUF_QUANT_PREFERENCE = (
    "F16", "BF16",
    "Q8_0",
    "Q6_K_L", "Q6_K",
    "Q5_K_L", "Q5_K_M", "Q5_K_S", "Q5_1", "Q5_0",
    "Q4_K_L", "Q4_K_M", "Q4_K_S", "Q4_1", "Q4_0",
    "Q3_K_XL", "Q3_K_L", "Q3_K_M", "Q3_K_S",
    "Q2_K",
    "F32", "FP32", "FP16",
)


def _select_gguf(
    inputs: ClassificationInputs,
    requested_quant: Optional[str],
) -> SelectionResult:
    paths = [_normalize(p) for p in inputs.file_paths]
    root = _root_files(paths)
    gguf_files = [p for p in root if p.lower().endswith(".gguf")]
    if not gguf_files:
        raise RepoUnknownShape(files_seen=paths, extra="gguf strategy with no *.gguf at root")

    # Group by quant level.
    by_quant: dict[str, list[str]] = {}
    for p in gguf_files:
        q = _gguf_quant_of(p)
        by_quant.setdefault(q, []).append(p)

    chosen_quant: Optional[str] = None
    if requested_quant:
        rq = requested_quant.strip()
        # e2e progress.json #72: accept fuzzy bit-width tokens (`4bit`, `8bit`,
        # `16bit`) and resolve to a concrete quant available in the repo.
        from .dtype_vocab import is_fuzzy_bitwidth, resolve_fuzzy_to_concrete
        if is_fuzzy_bitwidth(rq):
            available_lower = [k.lower() for k in by_quant]
            resolved = resolve_fuzzy_to_concrete(
                rq, source_kind="gguf", available_dtypes=available_lower,
            )
            if resolved:
                # by_quant is keyed by uppercase tokens; map back.
                for k in by_quant:
                    if k.lower() == resolved:
                        chosen_quant = k
                        break
            if not chosen_quant:
                available = ", ".join(sorted(by_quant.keys()))
                raise ValueError(
                    f"fuzzy gguf_quant={requested_quant!r} could not resolve to any quant in this repo "
                    f"(available: {available})"
                )
        else:
            # Concrete request — match case-insensitively
            for k in by_quant:
                if k.lower() == rq.lower():
                    chosen_quant = k
                    break
            if not chosen_quant:
                available = ", ".join(sorted(by_quant.keys()))
                raise ValueError(
                    f"requested gguf_quant={requested_quant!r} not present in repo "
                    f"(available: {available})"
                )
    else:
        # Default: pick the first preferred quant available. Skip unknown.
        # e2e #72 changed this to precision-first (F16/BF16 before Q*).
        for pref in _GGUF_QUANT_PREFERENCE:
            for k in by_quant:
                if k.lower() == pref.lower():
                    chosen_quant = k
                    break
            if chosen_quant:
                break
        if not chosen_quant:
            chosen_quant = sorted(by_quant.keys())[0]

    selected: list[str] = list(by_quant[chosen_quant])  # may be sharded

    # Tokenizer (rarely shipped alongside GGUF, but include if present)
    if "tokenizer.model" in root:
        selected.append("tokenizer.model")
    if "tokenizer.json" in root:
        selected.append("tokenizer.json")

    # e2e #72: dtype is the unified axis. The chosen GGUF quant token
    # (`q4_k_m`, `q8_0`, `f16`, `bf16`) is the dtype value; keep `quant_scheme`
    # as an alias for back-compat with consumers that haven't migrated.
    attrs: dict[str, str] = {
        "runtime_library": "llama-cpp",
        "file_layout": "gguf",
        "file_type": "gguf",
        "dtype": chosen_quant.lower(),
        "quant_scheme": chosen_quant,
    }

    # GGUF needs to be unmasked from the junk filter; do it manually.
    selected_set = {_normalize(p) for p in selected}
    final = list(selected_set)
    pickle_refused: list[str] = []
    for p in inputs.file_paths:
        np = _normalize(p)
        if not np or np in selected_set:
            continue
        if _is_always_include(np):
            final.append(np)
    final_unique = sorted(set(final))
    skipped = sorted(set(_normalize(p) for p in inputs.file_paths) - set(final_unique))
    return SelectionResult(
        selected_paths=final_unique,
        skipped_paths=skipped,
        attrs=attrs,
        pickle_files_refused=pickle_refused,
    )


def _gguf_quant_of(path: str) -> str:
    """Extract the quant level from a GGUF filename.

    Conventions vary widely; common ones:
      `model-Q4_K_M.gguf`, `model.Q4_K_M.gguf`,
      `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf`,
      `model-f16.gguf`, `model.bf16.gguf`,
      sharded: `model-Q4_K_M-00001-of-00009.gguf`
    """
    base = _basename(path)
    # Strip extension and any -NNNNN-of-NNNNN shard suffix
    stem = re.sub(r"\.gguf$", "", base, flags=re.IGNORECASE)
    stem = re.sub(r"-\d{5}-of-\d{5}$", "", stem)
    # Look for a known quant token at the end
    for pref in _GGUF_QUANT_PREFERENCE:
        if re.search(rf"(?:^|[-_.]){re.escape(pref)}$", stem, re.IGNORECASE):
            return pref.upper()
    # Last segment after - or .
    parts = re.split(r"[-_.]", stem)
    if parts:
        last = parts[-1].upper()
        if last:
            return last
    return "UNKNOWN"


# --- AIO / Singlefile ---

def _select_aio_singlefile(inputs: ClassificationInputs) -> SelectionResult:
    paths = [_normalize(p) for p in inputs.file_paths]
    root = _root_files(paths)
    safetensors = [p for p in root if p.lower().endswith(".safetensors")]
    if not safetensors:
        raise RepoUnknownShape(files_seen=paths, extra="aio_singlefile with no .safetensors at root")
    selected: list[str] = list(safetensors[:1])  # exactly one
    # Optional: a single .yaml config sibling (e.g. SD config)
    yamls = [p for p in root if p.lower().endswith((".yaml", ".yml"))]
    if len(yamls) == 1:
        selected.append(yamls[0])

    attrs: dict[str, str] = {
        "runtime_library": "diffusers-single-file",
        "file_layout": "singlefile",
        "file_type": "safetensors",
    }
    return _finalize(selected, inputs, attrs)


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# runtime_library → repo_kind mapping (e2e progress.json #70)
#
# Tensorhub's `repo_kind` enum is the file-layout shape (6 values). The
# classifier emits 7 `runtime_library` values (diffusers / diffusers-single-file
# / transformers / peft / sentence-transformers / llama-cpp / diffusers-lora).
# This map collapses the 7 onto the 6 by treating diffusers-single-file as a
# per-checkpoint variant of `repo_kind=diffusers` (carried via per-checkpoint
# `file_layout=singlefile`).
# ---------------------------------------------------------------------------

_RUNTIME_LIBRARY_TO_REPO_KIND: Mapping[str, str] = {
    "diffusers": "diffusers",
    "diffusers-single-file": "diffusers",
    "transformers": "transformers",
    "peft": "peft-adapter",
    "sentence-transformers": "sentence-transformers",
    "llama-cpp": "gguf",
    "diffusers-lora": "diffusers-lora",
}


def runtime_library_to_repo_kind(runtime_library: str) -> str:
    """Map a classifier `runtime_library` value to tensorhub's `repo_kind` enum.
    Returns "" when the input is empty or not recognized — caller should
    inherit the existing repo's kind (or fall through to tensorhub's default).
    """
    s = str(runtime_library or "").strip().lower()
    if s == "":
        return ""
    return _RUNTIME_LIBRARY_TO_REPO_KIND.get(s, "")


__all__ = [
    # Constants
    "_PICKLE_EXTS",
    "_JUNK_EXTS",
    "_DEMO_EXTS",
    "_ALWAYS_INCLUDE_FILENAMES",
    "_SIZE_WARN_BYTES",
    "_SIZE_REFUSE_BYTES",
    # Runtime-library → repo_kind map
    "runtime_library_to_repo_kind",
    # Refusals
    "RepoRefusal",
    "RepoPickleOnly",
    "RepoTfOnly",
    "RepoOnnxOnly",
    "RepoOpenVinoOnly",
    "RepoTfliteOnly",
    "RepoTensorRtOnly",
    "RepoCoreMlOnly",
    "RepoFlaxOnly",
    "RepoNemoOnly",
    "RepoUnknownShape",
    "RepoTooLarge",
    "RepoMissingSafetensors",
    # Result types
    "RepoClassification",
    "SelectionResult",
    "ClassificationInputs",
    # Functions
    "classify_huggingface_repo",
    "select_for_classification",
    "select_for_classification_multi",
]
