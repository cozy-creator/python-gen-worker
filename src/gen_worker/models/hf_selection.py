from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Sequence, Set


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

    # Every non-private key in model_index.json is a pipeline component that
    # `DiffusionPipeline.from_pretrained` WILL try to instantiate — including
    # "optional" ones (safety_checker / feature_extractor). Dropping them from
    # the download leaves their folder empty, and diffusers then raises
    # (e.g. "Can't load image processor ... preprocessor_config.json"), which
    # cascades into the no-variant fallback and the misleading "no file named
    # diffusion_pytorch_model.bin" error. So we keep ALL components by
    # default; their weights still go through reduced-precision selection.
    # `include_optional_components` is retained for callers that truly want to
    # prune them, but the default must yield a *loadable* snapshot.
    keys = [k for k in model_index.keys() if isinstance(k, str) and not k.startswith("_")]
    if not policy.include_optional_components:
        # Drop only components that are explicitly null-typed in the index
        # (``"comp": [null, null]`` — diffusers treats those as absent).
        pruned: list[str] = []
        for k in keys:
            v = model_index.get(k)
            if isinstance(v, list) and len(v) == 2 and all(x is None for x in v):
                continue
            pruned.append(k)
        keys = pruned
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


def _allow_fp32(policy: HFSelectionPolicy) -> bool:
    return any(p.strip().lower() == "fp32" for p in policy.weight_precisions)


def _precision_order(policy: HFSelectionPolicy) -> list[str]:
    """Filename-based precision preference order.

    16-bit first (bf16 then fp16 — both are "16-bit"; whichever the repo has),
    then fp8, then the non-variant (``default``) filename which on most
    diffusers repos is the fp32 master. ``fp32`` (an explicit ``.fp32.``
    filename token) is appended last and only used if the repo literally
    publishes one, but ALWAYS comes after ``default`` so we never prefer an
    explicit fp32 over anything smaller.

    Crucially this list is non-empty and ALWAYS includes ``default`` so a repo
    whose only weight is the plain ``model.safetensors`` (no precision token in
    the name — the sd1.5 / fp32-master shape) still selects a weight file
    rather than nothing. fp32 is wasteful but a missing weight is fatal.
    """
    order = ["bf16", "fp16", "fp8", "default"]
    if _allow_fp32(policy):
        order.append("fp32")
    return order


def _matches_precision(path: str, precision: str) -> bool:
    p = path.lower()
    prec = precision.lower()
    if prec == "default":
        return not any(x in p for x in ("bf16", "fp8", "fp16", "fp32"))
    return prec in p


def _filename_precision_score(path: str, policy: HFSelectionPolicy) -> int:
    """Rank a weight file by the precision token in its *filename*.

    Used as the fallback signal when byte-range dtype probing is unavailable
    (no network / redirect / private-repo header miss). Mirrors
    ``_precision_order``: 16-bit first, then fp8, then the non-variant
    ``default`` name (usually the fp32 master), then an explicit ``.fp32.``.

    ALWAYS returns a positive score for any weight file (``default`` matches
    everything without a precision token), so a probe-less selection can never
    fall through to "no weights". A ``.safetensors`` outranks a ``.bin`` of the
    same precision (we prefer the safe format for our runtime).
    """
    order = _precision_order(policy)
    fmt_bonus = 1 if path.lower().endswith(".safetensors") else 0
    for i, prec in enumerate(order):
        if _matches_precision(path, prec):
            return (len(order) - i) * 2 + fmt_bonus
    return fmt_bonus  # unknown token: still selectable, lowest rank


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
        # Skip alternate export formats that are often very large and not needed for
        # our PyTorch diffusers runtime (onnx/flax/openvino/etc).
        if low.endswith(".onnx") or low.endswith(".onnx_data") or low.endswith(".msgpack"):
            continue
        if "openvino" in low:
            continue
        if low.endswith(".safetensors") or low.endswith(".safetensors.index.json"):
            continue
        out.add(p)
    return out


# Returned by `_dtype_score` when probing did not yield a dtype for a file.
# The caller falls back to filename-based precision scoring in that case
# instead of treating "unknown" as a real (and falsely comparable) score.
_DTYPE_SCORE_UNKNOWN = -1


def _dtype_score(dtypes: Optional[Set[str]], policy: HFSelectionPolicy) -> int:
    """
    Score a safetensors dtype set.

    Preference: BF16 / F16 (both "16-bit") rank highest, then F8 variants,
    then F32 (only when ``fp32`` is explicitly allowed; otherwise F32 is the
    lowest *real* score so it is chosen only when nothing smaller exists —
    we never return zero/empty for a file that genuinely has weights).

    Returns ``_DTYPE_SCORE_UNKNOWN`` (a negative sentinel) when the dtype
    could not be probed, so the caller can fall back to filename-based
    precision ordering rather than comparing an unknown as if it were 0.
    """
    if not dtypes:
        return _DTYPE_SCORE_UNKNOWN

    # 16-bit family first (bf16/fp16 are equivalent "16-bit"), then fp8, then
    # fp32. F32 always scores below every smaller precision so a repo that has
    # BOTH fp16 and fp32 weights picks fp16.
    order = ["BF16", "F16", "F8_E4M3", "F8_E5M2", "F8", "F32"]

    if not _allow_fp32(policy) and dtypes == {"F32"}:
        # fp32 not allowed and this file is *purely* fp32: rank it below any
        # reduced-precision candidate but still > 0 so it remains selectable
        # as a last resort (never select nothing).
        return 2

    for i, want in enumerate(order):
        if want not in dtypes:
            continue
        # Prefer candidates that are "pure" dtype (all tensors same float dtype).
        if all(dt == want for dt in dtypes):
            return 1000 - i
        return 500 - i
    # Some other/unknown reduced precision (e.g. int8/uint8 quant): still a
    # real, selectable candidate — better than nothing.
    return 10


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

    single_safetensors = sorted(
        p for p in repo_files if p.startswith(prefix) and p.lower().endswith(".safetensors") and p not in shard_files
    )
    # Non-safetensors single-file weights (.bin / .ckpt) are a LAST-RESORT
    # fallback: many older diffusers repos (sd1.5 has them too) ship .bin
    # weights, and a valid repo with only .bin must still come down with a
    # usable weight rather than nothing. We prefer safetensors when present.
    single_other = sorted(
        p
        for p in repo_files
        if p.startswith(prefix)
        and p not in shard_files
        and (p.lower().endswith(".bin") or p.lower().endswith(".ckpt"))
    )

    # If there are no weight candidates at all, this is a genuinely
    # config-only component (e.g. a scheduler) — return nothing, no error.
    if not sharded_candidates and not single_safetensors and not single_other:
        return set(), set()

    best_files: set[str] = set()
    best_required_idx: set[str] = set()
    # Track (dtype_score, filename_score) lexicographically; fall back to
    # filename score whenever the dtype probe is unavailable so a probe-less
    # run still prefers fp16/bf16 over fp32/non-variant. `best_score` starts
    # below any real candidate so we ALWAYS pick something when weights exist.
    best_dscore = _DTYPE_SCORE_UNKNOWN - 1
    best_fscore = -1
    best_total = -1
    # Format rank: safetensors (1) outranks .bin/.ckpt (0). Used so a .bin is
    # only chosen when no safetensors candidate scored at all.
    best_fmt = -1

    def consider(*, probe_file: str, selected: set[str], required_idx: set[str], fmt_rank: int) -> None:
        nonlocal best_files, best_required_idx, best_dscore, best_fscore, best_total, best_fmt
        dtypes = None
        if probe_safetensors_dtypes is not None:
            try:
                dtypes = probe_safetensors_dtypes(probe_file)
            except Exception:
                dtypes = None

        dscore = _dtype_score(dtypes, policy)
        fscore = _filename_precision_score(probe_file, policy)
        total = sum(int(repo_file_sizes.get(p, 0) or 0) for p in selected)

        # Compare on (format, dtype-score, filename-score, -size). Safetensors
        # beats .bin; then a confirmed reduced-precision dtype; then the
        # filename precision token; then the smaller download.
        cand = (fmt_rank, dscore, fscore, -total if total >= 0 else 0)
        best = (best_fmt, best_dscore, best_fscore, -best_total if best_total >= 0 else 0)
        if best_files and cand <= best:
            return
        best_fmt = fmt_rank
        best_dscore = dscore
        best_fscore = fscore
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
        consider(probe_file=probe, selected=selected, required_idx={idx}, fmt_rank=1)

    for p in single_safetensors:
        consider(probe_file=p, selected={p}, required_idx=set(), fmt_rank=1)

    # Only fall back to .bin/.ckpt if NO safetensors candidate was selectable.
    if not best_files:
        for p in single_other:
            consider(probe_file=p, selected={p}, required_idx=set(), fmt_rank=0)

    if best_files:
        return best_files, best_required_idx

    # Defensive: we had candidates but somehow selected nothing. Rather than
    # raise (which produced the zero-weights failure), pick a deterministic
    # weight by filename precision so a valid repo never comes down empty.
    for prec in _precision_order(policy):
        idx = _select_weight_index_file(component, repo_files, prec)
        if idx:
            return {idx}, {idx}
        files = _select_weight_files(component, repo_files, prec)
        if files:
            return files, set()
    # Absolute last resort: any single weight file we know about.
    fallback = single_safetensors or single_other
    if fallback:
        return {fallback[0]}, set()
    return set(), set()


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


