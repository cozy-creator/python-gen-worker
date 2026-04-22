from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any, BinaryIO

_SUPPORTED_ARCH_ALIASES: dict[str, str] = {
    "llama": "llama",
    "mistral": "llama",
    "gemma": "gemma",
    "qwen": "qwen",
    "qwen2": "qwen",
    "qwen2_moe": "qwen",
}
_SUPPORTED_ENCODINGS = {"f16", "bf16"}
_TOKENIZER_ASSET_NAMES = (
    "tokenizer.model",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
)
_HF_SIDECAR_FILES = (
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "chat_template.jinja",
)
_GGUF_FILE_TYPE_LABELS = {
    0: "all_f32",
    1: "mostly_f16",
    2: "mostly_q4_0",
    3: "mostly_q4_1",
    6: "mostly_q5_0",
    7: "mostly_q5_1",
    8: "mostly_q8_0",
    9: "mostly_q8_1",
    10: "mostly_q2_k",
    11: "mostly_q3_k_s",
    12: "mostly_q3_k_m",
    13: "mostly_q3_k_l",
    14: "mostly_q4_k_s",
    15: "mostly_q4_k_m",
    16: "mostly_q5_k_s",
    17: "mostly_q5_k_m",
    18: "mostly_q6_k",
}
_GGUF_KV_TYPE_UINT8 = 0
_GGUF_KV_TYPE_INT8 = 1
_GGUF_KV_TYPE_UINT16 = 2
_GGUF_KV_TYPE_INT16 = 3
_GGUF_KV_TYPE_UINT32 = 4
_GGUF_KV_TYPE_INT32 = 5
_GGUF_KV_TYPE_FLOAT32 = 6
_GGUF_KV_TYPE_BOOL = 7
_GGUF_KV_TYPE_STRING = 8
_GGUF_KV_TYPE_ARRAY = 9
_GGUF_KV_TYPE_UINT64 = 10
_GGUF_KV_TYPE_INT64 = 11
_GGUF_KV_TYPE_FLOAT64 = 12


def normalize_gguf_encoding(value: str | None) -> str:
    encoding = str(value or "").strip().lower() or "f16"
    if encoding not in _SUPPORTED_ENCODINGS:
        supported = ", ".join(sorted(_SUPPORTED_ENCODINGS))
        raise ValueError(f"unsupported_gguf_encoding:{encoding}; supported={supported}")
    return encoding


def normalize_gguf_architecture(value: str | None) -> str:
    raw = str(value or "").strip().lower()
    if raw == "":
        raise ValueError("gguf_architecture_missing")
    normalized = _SUPPORTED_ARCH_ALIASES.get(raw, "")
    if normalized == "":
        raise ValueError(f"unsupported_gguf_architecture:{raw}")
    return normalized


def _arch_from_candidates(candidates: list[str]) -> str:
    for raw in candidates:
        cleaned = str(raw or "").strip().lower()
        if cleaned == "":
            continue
        if cleaned in _SUPPORTED_ARCH_ALIASES:
            return _SUPPORTED_ARCH_ALIASES[cleaned]
        for prefix, normalized in _SUPPORTED_ARCH_ALIASES.items():
            if cleaned.startswith(prefix):
                return normalized
    return ""


def detect_supported_architecture(config_json_path: Path) -> str:
    if not config_json_path.exists():
        raise ValueError("gguf_missing_config_json")
    try:
        payload = json.loads(config_json_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError("gguf_invalid_config_json") from exc

    candidates: list[str] = []
    model_type = str(payload.get("model_type") or "").strip()
    if model_type != "":
        candidates.append(model_type)

    architectures = payload.get("architectures")
    if isinstance(architectures, list):
        for item in architectures:
            val = str(item or "").strip()
            if val != "":
                candidates.append(val)

    resolved = _arch_from_candidates(candidates)
    if resolved == "":
        raise ValueError("unsupported_gguf_architecture_from_config")
    return resolved


def _copy_or_symlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    try:
        os.symlink(source.resolve(), destination)
    except Exception:
        shutil.copy2(source, destination)


def _resolve_source_repo_dir(weights_path: Path, source_repo_dir: str | None) -> Path:
    explicit = str(source_repo_dir or "").strip()
    if explicit != "":
        source = Path(explicit)
        if not source.exists() or not source.is_dir():
            raise ValueError("gguf_source_repo_dir_invalid")
        return source

    probe_dirs = [weights_path.parent]
    if weights_path.parent.parent != weights_path.parent:
        probe_dirs.append(weights_path.parent.parent)

    for candidate in probe_dirs:
        if (candidate / "config.json").exists():
            return candidate

    return weights_path.parent


def prepare_hf_source_tree_for_gguf(
    *,
    work_dir: Path,
    input_weights: Path,
    source_repo_dir: str | None,
) -> tuple[Path, str]:
    source_dir = _resolve_source_repo_dir(input_weights, source_repo_dir)
    config_path = source_dir / "config.json"
    detected_architecture = detect_supported_architecture(config_path)

    has_tokenizer_asset = any((source_dir / name).exists() for name in _TOKENIZER_ASSET_NAMES)
    if not has_tokenizer_asset:
        raise ValueError("gguf_missing_tokenizer_assets")

    model_dir = work_dir / "hf-model"
    model_dir.mkdir(parents=True, exist_ok=True)
    for name in _HF_SIDECAR_FILES:
        src = source_dir / name
        if src.exists() and src.is_file():
            _copy_or_symlink(src, model_dir / name)
    for name in _TOKENIZER_ASSET_NAMES:
        src = source_dir / name
        if src.exists() and src.is_file():
            _copy_or_symlink(src, model_dir / name)

    _copy_or_symlink(input_weights, model_dir / "model.safetensors")
    return model_dir, detected_architecture


def resolve_gguf_convert_script() -> Path:
    env_candidates = [
        str(os.getenv("CONVERSION_GGUF_CONVERT_SCRIPT", "")).strip(),
        str(os.getenv("LLAMA_CPP_CONVERT_HF_TO_GGUF", "")).strip(),
    ]
    for raw in env_candidates:
        if raw == "":
            continue
        candidate = Path(raw)
        if candidate.exists() and candidate.is_file():
            return candidate

    discovered = shutil.which("convert_hf_to_gguf.py")
    if discovered:
        return Path(discovered)

    raise RuntimeError("gguf_tooling_missing:convert_hf_to_gguf.py")


def resolve_llama_cpp_commit(script_path: Path) -> str:
    from_env = str(os.getenv("LLAMA_CPP_COMMIT", "")).strip()
    if from_env != "":
        return from_env

    for parent in [script_path.parent, *script_path.parents]:
        git_dir = parent / ".git"
        if not git_dir.exists():
            continue
        try:
            proc = subprocess.run(
                ["git", "-C", str(parent), "rev-parse", "HEAD"],
                check=True,
                text=True,
                capture_output=True,
                timeout=5.0,
            )
            commit = str(proc.stdout or "").strip()
            if commit != "":
                return commit
        except Exception:
            continue
    return "unknown"


def run_hf_to_gguf_conversion(
    *,
    script_path: Path,
    hf_model_dir: Path,
    output_path: Path,
    encoding: str,
) -> None:
    python_bin = str(os.getenv("CONVERSION_GGUF_PYTHON", "")).strip() or sys.executable
    cmd = [
        python_bin,
        str(script_path),
        str(hf_model_dir),
        "--outfile",
        str(output_path),
        "--outtype",
        normalize_gguf_encoding(encoding),
    ]
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            text=True,
            capture_output=True,
            timeout=float(os.getenv("CONVERSION_GGUF_TIMEOUT_SECONDS", "7200")),
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("gguf_conversion_timeout") from exc
    except Exception as exc:
        raise RuntimeError("gguf_conversion_exec_failed") from exc

    if proc.returncode != 0:
        raise RuntimeError(f"gguf_conversion_failed:rc={proc.returncode}")
    if not output_path.exists() or output_path.stat().st_size <= 0:
        raise RuntimeError("gguf_conversion_failed:missing_output")


def _read_exact(reader: BinaryIO, size: int) -> bytes:
    data = reader.read(size)
    if data is None or len(data) != size:
        raise ValueError("gguf_truncated")
    return data


def _read_u32(reader: BinaryIO) -> int:
    return struct.unpack("<I", _read_exact(reader, 4))[0]


def _read_u64(reader: BinaryIO) -> int:
    return struct.unpack("<Q", _read_exact(reader, 8))[0]


def _read_i32(reader: BinaryIO) -> int:
    return struct.unpack("<i", _read_exact(reader, 4))[0]


def _read_i64(reader: BinaryIO) -> int:
    return struct.unpack("<q", _read_exact(reader, 8))[0]


def _read_f32(reader: BinaryIO) -> float:
    return struct.unpack("<f", _read_exact(reader, 4))[0]


def _read_f64(reader: BinaryIO) -> float:
    return struct.unpack("<d", _read_exact(reader, 8))[0]


def _read_gguf_string(reader: BinaryIO) -> str:
    size = _read_u64(reader)
    return _read_exact(reader, int(size)).decode("utf-8", errors="replace")


def _skip_kv_value(reader: BinaryIO, value_type: int) -> None:
    if value_type in {_GGUF_KV_TYPE_UINT8, _GGUF_KV_TYPE_INT8, _GGUF_KV_TYPE_BOOL}:
        _read_exact(reader, 1)
        return
    if value_type in {_GGUF_KV_TYPE_UINT16, _GGUF_KV_TYPE_INT16}:
        _read_exact(reader, 2)
        return
    if value_type in {_GGUF_KV_TYPE_UINT32, _GGUF_KV_TYPE_INT32, _GGUF_KV_TYPE_FLOAT32}:
        _read_exact(reader, 4)
        return
    if value_type in {_GGUF_KV_TYPE_UINT64, _GGUF_KV_TYPE_INT64, _GGUF_KV_TYPE_FLOAT64}:
        _read_exact(reader, 8)
        return
    if value_type == _GGUF_KV_TYPE_STRING:
        _ = _read_gguf_string(reader)
        return
    if value_type == _GGUF_KV_TYPE_ARRAY:
        elem_type = _read_u32(reader)
        count = _read_u64(reader)
        for _ in range(int(count)):
            _skip_kv_value(reader, int(elem_type))
        return
    raise ValueError("gguf_unknown_kv_type")


def _read_kv_value(reader: BinaryIO, value_type: int) -> Any:
    if value_type == _GGUF_KV_TYPE_UINT8:
        return _read_exact(reader, 1)[0]
    if value_type == _GGUF_KV_TYPE_INT8:
        return struct.unpack("<b", _read_exact(reader, 1))[0]
    if value_type == _GGUF_KV_TYPE_UINT16:
        return struct.unpack("<H", _read_exact(reader, 2))[0]
    if value_type == _GGUF_KV_TYPE_INT16:
        return struct.unpack("<h", _read_exact(reader, 2))[0]
    if value_type == _GGUF_KV_TYPE_UINT32:
        return _read_u32(reader)
    if value_type == _GGUF_KV_TYPE_INT32:
        return _read_i32(reader)
    if value_type == _GGUF_KV_TYPE_FLOAT32:
        return _read_f32(reader)
    if value_type == _GGUF_KV_TYPE_BOOL:
        return _read_exact(reader, 1) != b"\x00"
    if value_type == _GGUF_KV_TYPE_STRING:
        return _read_gguf_string(reader)
    if value_type == _GGUF_KV_TYPE_ARRAY:
        elem_type = _read_u32(reader)
        count = _read_u64(reader)
        return [_read_kv_value(reader, int(elem_type)) for _ in range(int(count))]
    if value_type == _GGUF_KV_TYPE_UINT64:
        return _read_u64(reader)
    if value_type == _GGUF_KV_TYPE_INT64:
        return _read_i64(reader)
    if value_type == _GGUF_KV_TYPE_FLOAT64:
        return _read_f64(reader)
    raise ValueError("gguf_unknown_kv_type")


def parse_gguf_header(path: Path, *, max_kv_pairs: int = 256) -> dict[str, Any]:
    with path.open("rb") as reader:
        magic = _read_exact(reader, 4)
        if magic != b"GGUF":
            raise ValueError("gguf_invalid_magic")

        version = _read_u32(reader)
        tensor_count = _read_u64(reader)
        kv_count = _read_u64(reader)
        kv_pairs: dict[str, Any] = {}
        for idx in range(int(kv_count)):
            key = _read_gguf_string(reader)
            value_type = _read_u32(reader)
            if idx < max_kv_pairs:
                kv_pairs[key] = _read_kv_value(reader, int(value_type))
            else:
                _skip_kv_value(reader, int(value_type))

    file_type_raw = kv_pairs.get("general.file_type")
    file_type_value = int(file_type_raw) if isinstance(file_type_raw, int) else None
    return {
        "version": int(version),
        "tensor_count": int(tensor_count),
        "kv_count": int(kv_count),
        "architecture": str(kv_pairs.get("general.architecture") or "").strip(),
        "name": str(kv_pairs.get("general.name") or "").strip(),
        "quantization_version": (
            int(kv_pairs.get("general.quantization_version"))
            if isinstance(kv_pairs.get("general.quantization_version"), int)
            else None
        ),
        "file_type": file_type_value,
        "file_type_label": _GGUF_FILE_TYPE_LABELS.get(file_type_value or -1, "unknown"),
        "kv": kv_pairs,
    }


__all__ = [
    "detect_supported_architecture",
    "normalize_gguf_architecture",
    "normalize_gguf_encoding",
    "parse_gguf_header",
    "prepare_hf_source_tree_for_gguf",
    "resolve_gguf_convert_script",
    "resolve_llama_cpp_commit",
    "run_hf_to_gguf_conversion",
]

