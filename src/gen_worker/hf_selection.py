from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Optional, Sequence, Set


_OPTIONAL_COMPONENTS = {"safety_checker", "feature_extractor"}
_SMALL_TREE_COMPONENTS = {"tokenizer", "tokenizer_2", "scheduler"}  # legacy fallback only


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


def _derive_component_types(model_index: Mapping[str, object]) -> dict[str, tuple[str, str]]:
    out: dict[str, tuple[str, str]] = {}
    for k, v in model_index.items():
        if not isinstance(k, str) or k.startswith("_"):
            continue
        if not isinstance(v, list) or len(v) != 2:
            continue
        lib, cls = v
        if not isinstance(lib, str) or not isinstance(cls, str):
            continue
        lib = lib.strip()
        cls = cls.strip()
        if not lib or not cls:
            continue
        out[_norm_component(k)] = (lib, cls)
    return out


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
    # Legacy filename-based fallback order (only used when dtype probing is unavailable).
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


def _is_tokenizer_component(component_type: Optional[tuple[str, str]]) -> bool:
    if not component_type:
        return False
    lib, cls = component_type
    return lib == "transformers" and "Tokenizer" in cls


def _is_scheduler_component(component_type: Optional[tuple[str, str]]) -> bool:
    if not component_type:
        return False
    lib, cls = component_type
    return lib == "diffusers" and "Scheduler" in cls


def _files_in_component(repo_files: Sequence[str], component: str) -> list[str]:
    prefix = f"{component}/"
    return [p for p in repo_files if p.startswith(prefix)]


def _support_files_for_component(repo_files: Sequence[str], component: str) -> set[str]:
    out: set[str] = set()
    for p in _files_in_component(repo_files, component):
        low = p.lower()
        if low.endswith(".bin") or low.endswith(".ckpt"):
            continue
        if low.endswith(".safetensors") or low.endswith(".safetensors.index.json"):
            continue
        out.add(p)
    return out


def _dtype_score(dtypes: Optional[Set[str]], policy: HFSelectionPolicy) -> int:
    """
    Score a safetensors dtype set.

    Preference is hardcoded (same as cozy-hub): F16 first, then BF16.
    `fp32` is only considered when explicitly enabled in policy.weight_precisions.
    """
    if not dtypes:
        return 0

    allow_fp32 = any(p.strip().lower() == "fp32" for p in policy.weight_precisions)
    order = ["F16", "BF16"]
    if allow_fp32:
        order.append("F32")

    for i, want in enumerate(order):
        has = want in dtypes
        if not has:
            continue
        # Prefer candidates that are "pure" dtype (all tensors same float dtype).
        if all(dt == want for dt in dtypes):
            return 100 - i
        return 50 - i
    return 1


def _select_component_weights(
    *,
    component: str,
    repo_files: Sequence[str],
    repo_file_sizes: Mapping[str, int],
    weight_index_json_by_file: Mapping[str, Mapping[str, object]],
    policy: HFSelectionPolicy,
    probe_safetensors_dtypes: Optional[Callable[[str], Optional[Set[str]]]],
) -> tuple[set[str], set[str]]:
    """
    Select a single weight set (single-file or sharded) for a component.

    Returns:
      - selected_files (includes index JSON when sharded)
      - required_weight_index_files (index JSONs that must be downloaded to expand shards)
    """
    prefix = f"{component}/"
    # Candidates are:
    #  - each `*.safetensors.index.json` (sharded set)
    #  - each standalone `*.safetensors` not referenced by any index
    index_files = sorted(p for p in repo_files if p.startswith(prefix) and _is_index_file(p))
    has_any_weight_file = any(p.startswith(prefix) and _is_weight_file(p) for p in repo_files)

    # If we have sharded index files but no index contents yet, pick a deterministic index
    # using the legacy filename-based variant order and let finalize() expand shards later.
    if index_files and not any(idx in weight_index_json_by_file for idx in index_files):
        for prec in _precision_order(policy):
            idx = _select_weight_index_file(component, repo_files, prec)
            if idx:
                return {idx}, {idx}

    shard_files: set[str] = set()
    sharded_candidates: list[tuple[str, set[str]]] = []
    for idx in index_files:
        idx_json = weight_index_json_by_file.get(idx)
        if not idx_json:
            # We can't score this index without its contents; keep it as a possible fallback.
            sharded_candidates.append((idx, set()))
            continue
        weight_map = idx_json.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            continue
        shards: set[str] = set()
        component_prefix = idx.split("/", 1)[0] + "/"
        for v in weight_map.values():
            if not isinstance(v, str) or not v:
                continue
            shard = v
            if "/" not in shard:
                shard = component_prefix + shard
            if shard in repo_files:
                shards.add(shard)
        if shards:
            shard_files |= shards
            sharded_candidates.append((idx, shards))

    single_candidates = sorted(
        p for p in repo_files if p.startswith(prefix) and p.lower().endswith(".safetensors") and p not in shard_files
    )

    # If there are no weight candidates at all, treat as config-only component.
    if not sharded_candidates and not single_candidates:
        if has_any_weight_file:
            raise RuntimeError(
                f"missing required reduced-precision safetensors for component '{component}'. "
                "Override with COZY_HF_WEIGHT_PRECISIONS=fp16,bf16,fp32, or set COZY_HF_FULL_REPO_DOWNLOAD=1."
            )
        return set(), set()

    best_files: set[str] = set()
    best_required_idx: set[str] = set()
    best_score = -1
    best_total = -1

    def consider(*, probe_file: str, selected: set[str], required_idx: set[str]) -> None:
        nonlocal best_files, best_required_idx, best_score, best_total
        dtypes = None
        if probe_safetensors_dtypes is not None:
            try:
                dtypes = probe_safetensors_dtypes(probe_file)
            except Exception:
                dtypes = None

        score = _dtype_score(dtypes, policy)
        total = sum(int(repo_file_sizes.get(p, 0) or 0) for p in selected)
        if score > best_score or (score == best_score and total > best_total):
            best_score = score
            best_total = total
            best_files = selected
            best_required_idx = required_idx

    # Consider sharded candidates first: if present, they often represent the canonical weights.
    for idx, shards in sharded_candidates:
        selected = {idx} | set(shards)
        if not shards:
            # can't probe without knowing shards; pick idx as placeholder and rely on finalize()
            continue
        probe = sorted(shards)[0]
        consider(probe_file=probe, selected=selected, required_idx={idx})

    for p in single_candidates:
        consider(probe_file=p, selected={p}, required_idx=set())

    # If we couldn't probe/score anything, fall back to legacy filename-based precision selection.
    if best_score < 0:
        precisions = _precision_order(policy)
        chosen: set[str] = set()
        chosen_index: Optional[str] = None
        for prec in precisions:
            idx = _select_weight_index_file(component, repo_files, prec)
            if idx:
                chosen_index = idx
                break
            files = _select_weight_files(component, repo_files, prec)
            if files:
                chosen = files
                break

        if chosen_index:
            return {chosen_index}, {chosen_index}
        if chosen:
            return chosen, set()

        raise RuntimeError(
            f"missing required reduced-precision safetensors for component '{component}'. "
            "Override with COZY_HF_WEIGHT_PRECISIONS=fp16,bf16,fp32, or set COZY_HF_FULL_REPO_DOWNLOAD=1."
        )

    if best_files:
        return best_files, best_required_idx

    # We had candidates but couldn't select; be strict.
    raise RuntimeError(
        f"missing required reduced-precision safetensors for component '{component}'. "
        "Override with COZY_HF_WEIGHT_PRECISIONS=fp16,bf16,fp32, or set COZY_HF_FULL_REPO_DOWNLOAD=1."
    )


def plan_diffusers_download(
    *,
    model_index: Mapping[str, object],
    repo_files: Sequence[str],
    policy: HFSelectionPolicy,
    weight_index_json_by_file: Optional[Mapping[str, Mapping[str, object]]] = None,
    repo_file_sizes: Optional[Mapping[str, int]] = None,
    probe_safetensors_dtypes: Optional[Callable[[str], Optional[Set[str]]]] = None,
) -> HFSelectionPlan:
    components = _derive_components(model_index, policy)
    component_types = _derive_component_types(model_index)

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
    weight_index_json_by_file = weight_index_json_by_file or {}
    repo_file_sizes = repo_file_sizes or {}

    for c in present_components:
        ctype = component_types.get(c)

        selected |= _support_files_for_component(repo_files, c)

        # For tokenizer/scheduler trees, include the whole directory explicitly.
        if _is_tokenizer_component(ctype) or _is_scheduler_component(ctype) or c in _SMALL_TREE_COMPONENTS:
            selected |= {p for p in repo_files if p.startswith(f"{c}/") and not _is_weight_file(p)}
            # Tokenizers still need at least one core file.
            if _is_tokenizer_component(ctype) or c.startswith("tokenizer"):
                if not any(p in repo_files for p in (f"{c}/tokenizer.json", f"{c}/vocab.json", f"{c}/tokenizer.model", f"{c}/spiece.model")):
                    raise RuntimeError(f"tokenizer component '{c}' missing required tokenizer files")
            continue

        chosen, required = _select_component_weights(
            component=c,
            repo_files=repo_files,
            repo_file_sizes=repo_file_sizes,
            weight_index_json_by_file=weight_index_json_by_file,
            policy=policy,
            probe_safetensors_dtypes=probe_safetensors_dtypes,
        )
        selected |= chosen
        required_indexes |= required

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
