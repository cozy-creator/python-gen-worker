"""GGUF helpers: llama.cpp toolchain wrappers + header read via the ``gguf``
package (replaces the hand-rolled binary parser in gguf_utils.py)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

_SUPPORTED_ARCH_ALIASES: dict[str, str] = {
    "llama": "llama", "mistral": "llama", "gemma": "gemma",
    "qwen": "qwen", "qwen2": "qwen", "qwen2_moe": "qwen",
}
_SUPPORTED_ENCODINGS = {"f32", "f16", "bf16", "q8_0"}
_TOKENIZER_ASSET_NAMES = (
    "tokenizer.model", "tokenizer.json", "tokenizer_config.json", "vocab.json",
    "merges.txt", "special_tokens_map.json", "added_tokens.json",
)
_HF_SIDECAR_FILES = (
    "config.json", "generation_config.json", "preprocessor_config.json",
    "chat_template.jinja",
)


def read_gguf_metadata(path: Path) -> dict[str, Any]:
    """Header metadata of a GGUF file via the ``gguf`` package."""
    from gguf import GGUFReader

    reader = GGUFReader(str(path), "r")
    out: dict[str, Any] = {"tensor_count": len(reader.tensors)}
    for key, field in reader.fields.items():
        try:
            contents = field.contents()
        except Exception:
            continue
        out[str(key)] = contents
    return out


def normalize_gguf_encoding(value: str | None) -> str:
    encoding = str(value or "").strip().lower() or "f16"
    if encoding not in _SUPPORTED_ENCODINGS:
        raise ValueError(
            f"unsupported_gguf_encoding:{encoding}; supported={', '.join(sorted(_SUPPORTED_ENCODINGS))}")
    return encoding


def detect_supported_architecture(config_json_path: Path) -> str:
    if not config_json_path.exists():
        raise ValueError("gguf_missing_config_json")
    try:
        payload = json.loads(config_json_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError("gguf_invalid_config_json") from exc
    candidates: list[str] = []
    model_type = str(payload.get("model_type") or "").strip()
    if model_type:
        candidates.append(model_type)
    architectures = payload.get("architectures")
    if isinstance(architectures, list):
        candidates.extend(str(a or "").strip() for a in architectures if str(a or "").strip())
    for raw in candidates:
        cleaned = raw.strip().lower()
        if cleaned in _SUPPORTED_ARCH_ALIASES:
            return _SUPPORTED_ARCH_ALIASES[cleaned]
        for prefix, normalized in _SUPPORTED_ARCH_ALIASES.items():
            if cleaned.startswith(prefix):
                return normalized
    raise ValueError(f"unsupported_gguf_architecture:{candidates or ['<missing>']}")


def _copy_or_symlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    try:
        os.symlink(source.resolve(), destination)
    except Exception:
        shutil.copy2(source, destination)


def prepare_hf_source_tree_for_gguf(
    *,
    work_dir: Path,
    input_weights: Path,
    source_repo_dir: str | None,
) -> tuple[Path, str]:
    """Assemble an HF-model dir (weights + config + tokenizer) for
    ``convert_hf_to_gguf.py``."""
    explicit = str(source_repo_dir or "").strip()
    if explicit:
        source_dir = Path(explicit)
        if not source_dir.is_dir():
            raise ValueError("gguf_source_repo_dir_invalid")
    else:
        source_dir = input_weights.parent
        if not (source_dir / "config.json").exists() and (
                source_dir.parent / "config.json").exists():
            source_dir = source_dir.parent

    arch = detect_supported_architecture(source_dir / "config.json")
    if not any((source_dir / n).exists() for n in _TOKENIZER_ASSET_NAMES):
        raise ValueError("gguf_missing_tokenizer_assets")

    model_dir = work_dir / "hf-model"
    model_dir.mkdir(parents=True, exist_ok=True)
    for name in (*_HF_SIDECAR_FILES, *_TOKENIZER_ASSET_NAMES):
        src = source_dir / name
        if src.is_file():
            _copy_or_symlink(src, model_dir / name)
    _copy_or_symlink(input_weights, model_dir / "model.safetensors")
    return model_dir, arch


def resolve_gguf_convert_script() -> Path:
    discovered = shutil.which("convert_hf_to_gguf.py")
    if discovered:
        return Path(discovered)
    raise RuntimeError("gguf_tooling_missing:convert_hf_to_gguf.py")


def run_hf_to_gguf_conversion(
    *,
    script_path: Path,
    hf_model_dir: Path,
    output_path: Path,
    encoding: str,
) -> None:
    cmd = [sys.executable, str(script_path), str(hf_model_dir),
           "--outfile", str(output_path), "--outtype", normalize_gguf_encoding(encoding)]
    try:
        proc = subprocess.run(cmd, check=False, text=True, capture_output=True, timeout=7200.0)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("gguf_conversion_timeout") from exc
    except Exception as exc:
        raise RuntimeError("gguf_conversion_exec_failed") from exc
    if proc.returncode != 0:
        raise RuntimeError(f"gguf_conversion_failed:rc={proc.returncode}")
    if not output_path.exists() or output_path.stat().st_size <= 0:
        raise RuntimeError("gguf_conversion_failed:missing_output")


__all__ = [
    "detect_supported_architecture",
    "normalize_gguf_encoding",
    "prepare_hf_source_tree_for_gguf",
    "read_gguf_metadata",
    "resolve_gguf_convert_script",
    "run_hf_to_gguf_conversion",
]
