from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence, Set


_OPTIONAL_COMPONENTS = {"safety_checker", "feature_extractor"}
_SMALL_TREE_COMPONENTS = {"tokenizer", "tokenizer_2", "scheduler"}


@dataclass(frozen=True)
class HFSelectionPolicy:
    """
    Policy knobs for selecting a minimal diffusers file set from a HF repo.

    Notes:
    - `variant` is the preferred diffusers variant (e.g. "fp16" / "bf16") when present.
    - `weight_precisions` is the ordered list of accepted precisions to fall back across.
    """

    components_override: Optional[Sequence[str]] = None
    include_optional_components: bool = False
    # Escape hatch: allow fp32 safetensors only when explicitly included here.
    weight_precisions: Sequence[str] = ("fp16", "bf16")
    allow_root_json: bool = False


@dataclass(frozen=True)
class HFSelectionPlan:
    selected_files: Set[str]
    required_weight_index_files: Set[str]


def _norm_component(c: str) -> str:
    return (c or "").strip().strip("/")


def _derive_components(model_index: Mapping[str, object], policy: HFSelectionPolicy) -> list[str]:
    if policy.components_override is not None:
        comps = [_norm_component(c) for c in policy.components_override]
        return [c for c in comps if c]

    keys = [k for k in model_index.keys() if isinstance(k, str) and not k.startswith("_")]
    if not policy.include_optional_components:
        keys = [k for k in keys if k not in _OPTIONAL_COMPONENTS]
    return [_norm_component(k) for k in keys if _norm_component(k)]


def _repo_has_component(repo_files: Sequence[str], component: str) -> bool:
    prefix = f"{component}/"
    return any(p.startswith(prefix) for p in repo_files)


def _get_root_files(repo_files: Sequence[str]) -> set[str]:
    return {p for p in repo_files if "/" not in p}


def _is_weight_file(path: str) -> bool:
    p = path.lower()
    return p.endswith(".safetensors") or p.endswith(".bin") or p.endswith(".ckpt")


def _is_index_file(path: str) -> bool:
    return path.lower().endswith(".safetensors.index.json")


def _precision_order(policy: HFSelectionPolicy) -> list[str]:
    """
    Hardcoded preferred order for diffusers variants (legacy behavior):
      bf16 → fp8 → fp16 → default

    `fp32` is only accepted if explicitly enabled via policy.weight_precisions,
    and is tried last.
    """
    order = ["bf16", "fp8", "fp16", "default"]
    allow_fp32 = any(p.strip().lower() == "fp32" for p in policy.weight_precisions)
    if allow_fp32:
        order.append("fp32")
    return order


def _matches_precision(path: str, precision: str) -> bool:
    p = path.lower()
    prec = precision.lower()
    if prec == "default":
        return not any(x in p for x in ("bf16", "fp8", "fp16", "fp32"))
    return prec in p


def _select_weight_index_file(component: str, repo_files: Sequence[str], precision: str) -> Optional[str]:
    prefix = f"{component}/"
    candidates = [
        p
        for p in repo_files
        if p.startswith(prefix) and _is_index_file(p) and _matches_precision(p, precision)
    ]
    if not candidates:
        return None
    # Prefer a deterministic pick if multiple exist: stable sort.
    return sorted(candidates)[0]


def _select_weight_files(component: str, repo_files: Sequence[str], precision: str) -> set[str]:
    """
    Select non-sharded weight files (or shards, if they match) for a given precision.

    This intentionally selects *all* matching safetensors files in the component dir.
    If the repo is sharded, we expect `finalize_diffusers_download()` to narrow it to
    the exact shards from the index JSON when available.
    """
    prefix = f"{component}/"
    out: set[str] = set()
    for p in repo_files:
        if not p.startswith(prefix):
            continue
        if not p.lower().endswith(".safetensors"):
            continue
        if _matches_precision(p, precision):
            out.add(p)
    return out


def plan_diffusers_download(
    *,
    model_index: Mapping[str, object],
    repo_files: Sequence[str],
    policy: HFSelectionPolicy,
) -> HFSelectionPlan:
    components = _derive_components(model_index, policy)

    present_components = [c for c in components if _repo_has_component(repo_files, c)]
    if not present_components:
        raise RuntimeError(
            "repo does not look like a diffusers pipeline (no component folders found); "
            "use full-repo download override if this is intentional"
        )

    selected: set[str] = {"model_index.json"}
    required_indexes: set[str] = set()

    if policy.allow_root_json:
        for p in _get_root_files(repo_files):
            if p.lower().endswith(".json"):
                selected.add(p)

    precisions = _precision_order(policy)

    for c in present_components:
        # For tokenizer/scheduler trees, include the whole directory explicitly.
        if c in _SMALL_TREE_COMPONENTS:
            selected |= {p for p in repo_files if p.startswith(f"{c}/")}
            continue

        # Common config file.
        if f"{c}/config.json" in repo_files:
            selected.add(f"{c}/config.json")

        # Prefer safetensors for diffusers repos. Select either a sharded index (preferred)
        # or the matching safetensors files.
        chosen: set[str] = set()
        chosen_index: Optional[str] = None
        for prec in precisions:
            idx = _select_weight_index_file(c, repo_files, prec)
            if idx:
                chosen_index = idx
                break
            files = _select_weight_files(c, repo_files, prec)
            if files:
                chosen = files
                break

        if chosen_index:
            selected.add(chosen_index)
            required_indexes.add(chosen_index)
            continue

        if chosen:
            selected |= chosen
            continue

        raise RuntimeError(
            f"missing required reduced-precision safetensors for component '{c}'. "
            "Override with COZY_HF_WEIGHT_PRECISIONS=fp16,bf16,fp32, or set COZY_HF_FULL_REPO_DOWNLOAD=1."
        )

    return HFSelectionPlan(selected_files=selected, required_weight_index_files=required_indexes)


def finalize_diffusers_download(
    *,
    plan: HFSelectionPlan,
    repo_files: Sequence[str],
    weight_index_json_by_file: Mapping[str, Mapping[str, object]],
) -> Set[str]:
    """
    Expand a selection plan by resolving sharded weights via `*.safetensors.index.json`.

    The index JSON is expected to be in the standard HF format:
      {"weight_map": {"key": "shard-filename.safetensors", ...}, ...}
    """
    selected = set(plan.selected_files)

    for idx_path in plan.required_weight_index_files:
        idx_json = weight_index_json_by_file.get(idx_path)
        if not idx_json:
            # If the caller didn't provide the index, keep the index file itself and rely on
            # whatever other safetensors were already selected (best-effort).
            continue

        weight_map = idx_json.get("weight_map")
        if not isinstance(weight_map, dict):
            continue

        component_prefix = idx_path.split("/", 1)[0] + "/"
        for v in weight_map.values():
            if not isinstance(v, str) or not v:
                continue
            shard = v
            if "/" not in shard:
                shard = component_prefix + shard

            if shard not in repo_files:
                raise RuntimeError(f"weight shard referenced by {idx_path} not found in repo: {shard}")
            selected.add(shard)

    return selected


_SHARD_RE = re.compile(r"-\\d{5}-of-\\d{5}\\.safetensors$", re.IGNORECASE)


def is_probably_shard_file(path: str) -> bool:
    return bool(_SHARD_RE.search(path))
