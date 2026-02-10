from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

from .model_refs import parse_model_ref


def _rfc3339_now() -> str:
    # RFC3339 / ISO8601 UTC with Z suffix.
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _blob_path(blobs_root: Path, digest: str) -> Path:
    digest = (digest or "").strip().lower()
    if len(digest) < 4:
        raise ValueError("invalid blake3 digest")
    return blobs_root / "blake3" / digest[:2] / digest[2:4] / digest


def _cozy_snapshot_dir(base_dir: Path, snapshot_digest: str) -> Path:
    return base_dir / "cozy" / "snapshots" / snapshot_digest


def _cozy_blobs_root(base_dir: Path) -> Path:
    return base_dir / "cozy" / "blobs"


def _extract_snapshot_digest(resolved_entry: Any) -> Optional[str]:
    if resolved_entry is None:
        return None
    snap = getattr(resolved_entry, "snapshot_digest", None)
    if not snap:
        snap = getattr(resolved_entry, "snapshotDigest", None)
    snap = str(snap or "").strip()
    return snap or None


def _extract_resolved_files(resolved_entry: Any) -> List[Any]:
    if resolved_entry is None:
        return []
    raw = getattr(resolved_entry, "files", None)
    if raw is None:
        raw = getattr(resolved_entry, "Files", None)
    return list(raw or [])


def _extract_file_blake3(ent: Any) -> str:
    b = getattr(ent, "blake3", None)
    if not b:
        b = getattr(ent, "BLAKE3", None)
    return str(b or "").strip().lower()


def _extract_file_size(ent: Any) -> int:
    s = getattr(ent, "size_bytes", None)
    if s is None:
        s = getattr(ent, "sizeBytes", None)
    try:
        return int(s or 0)
    except Exception:
        return 0


def _missing_bytes_for_resolved_model(base_dir: Path, resolved_entry: Any) -> Optional[int]:
    try:
        files = _extract_resolved_files(resolved_entry)
        if not files:
            return None
        blobs_root = _cozy_blobs_root(base_dir)
        missing = 0
        for ent in files:
            d = _extract_file_blake3(ent)
            if not d:
                continue
            p = _blob_path(blobs_root, d)
            if not p.exists():
                missing += _extract_file_size(ent)
        return int(missing)
    except Exception:
        return None


def _cache_dir() -> Path:
    return Path(os.getenv("WORKER_MODEL_CACHE_DIR", "/tmp/cozy/models"))


@dataclass
class ModelMetricsV1:
    model_id: str
    variant_label: Optional[str] = None
    snapshot_digest: Optional[str] = None
    cache_state: Optional[str] = None  # hot_vram|warm_disk|cold_remote
    bytes_downloaded: Optional[int] = None
    download_ms: Optional[int] = None
    bytes_read_disk: Optional[int] = None
    disk_fstype: Optional[str] = None
    disk_backend: Optional[str] = None  # local|nfs
    localized: Optional[bool] = None
    nfs_to_local_copy_ms: Optional[int] = None
    bytes_copied: Optional[int] = None

    def to_json(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"model_id": self.model_id}
        if self.variant_label is not None:
            out["variant_label"] = self.variant_label
        if self.snapshot_digest is not None:
            out["snapshot_digest"] = self.snapshot_digest
        if self.cache_state is not None:
            out["cache_state"] = self.cache_state
        if self.bytes_downloaded is not None:
            out["bytes_downloaded"] = int(self.bytes_downloaded)
        if self.download_ms is not None:
            out["download_ms"] = int(self.download_ms)
        if self.bytes_read_disk is not None:
            out["bytes_read_disk"] = int(self.bytes_read_disk)
        if self.disk_fstype is not None:
            out["disk_fstype"] = str(self.disk_fstype)
        if self.disk_backend is not None:
            out["disk_backend"] = str(self.disk_backend)
        if self.localized is not None:
            out["localized"] = bool(self.localized)
        if self.nfs_to_local_copy_ms is not None:
            out["nfs_to_local_copy_ms"] = int(self.nfs_to_local_copy_ms)
        if self.bytes_copied is not None:
            out["bytes_copied"] = int(self.bytes_copied)
        return out


@dataclass
class RunMetricsV1:
    run_id: str
    function_name: str
    required_models: List[str] = field(default_factory=list)
    resolved_cozy_models_by_id: Optional[Mapping[str, Any]] = None

    # Times (ms)
    fetch_ms: Optional[int] = None
    pipeline_init_ms: Optional[int] = None
    gpu_load_ms: Optional[int] = None
    warmup_ms: Optional[int] = None
    inference_ms: Optional[int] = None
    png_encode_ms: Optional[int] = None
    upload_ms: Optional[int] = None

    # Diffusion-ish (optional)
    steps: Optional[int] = None
    iters_per_s: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    guidance: Optional[float] = None

    # Resources (best-effort)
    peak_vram_bytes: Optional[int] = None
    peak_ram_bytes: Optional[int] = None

    # Timestamps
    compute_started_at: Optional[str] = None
    compute_completed_at: Optional[str] = None

    models: Dict[str, ModelMetricsV1] = field(default_factory=dict)  # canonical model_id -> metrics

    _t0_monotonic: float = field(default_factory=time.monotonic, repr=False)
    _upload_ms_accum: int = field(default=0, repr=False)
    _fetch_ms_accum: int = field(default=0, repr=False)
    _pipeline_init_ms_accum: int = field(default=0, repr=False)
    _gpu_load_ms_accum: int = field(default=0, repr=False)

    def mark_compute_started(self) -> None:
        self.compute_started_at = _rfc3339_now()

    def mark_compute_completed(self) -> None:
        self.compute_completed_at = _rfc3339_now()

    def _get_model_entry(self, model_id: str) -> ModelMetricsV1:
        key = str(model_id or "").strip()
        ent = self.models.get(key)
        if ent is None:
            ent = ModelMetricsV1(model_id=key)
            self.models[key] = ent
        return ent

    def set_initial_model_state(self, model_id: str, cache_state: Optional[str], snapshot_digest: Optional[str]) -> None:
        ent = self._get_model_entry(model_id)
        if cache_state:
            ent.cache_state = cache_state
        if snapshot_digest:
            ent.snapshot_digest = snapshot_digest

    def add_fetch_time(self, model_id: str, ms: int, bytes_downloaded: Optional[int] = None) -> None:
        try:
            ms_i = int(ms)
        except Exception:
            return
        if ms_i < 0:
            return
        self._fetch_ms_accum += ms_i
        ent = self._get_model_entry(model_id)
        # 0 is meaningful for warm disk hits.
        ent.download_ms = ms_i
        if bytes_downloaded is not None:
            ent.bytes_downloaded = int(bytes_downloaded)

    def set_model_disk_backend(
        self,
        model_id: str,
        *,
        disk_fstype: Optional[str] = None,
        disk_backend: Optional[str] = None,
        localized: Optional[bool] = None,
        nfs_to_local_copy_ms: Optional[int] = None,
        bytes_copied: Optional[int] = None,
    ) -> None:
        ent = self._get_model_entry(model_id)
        if disk_fstype is not None:
            ent.disk_fstype = str(disk_fstype)
        if disk_backend is not None:
            ent.disk_backend = str(disk_backend)
        if localized is not None:
            ent.localized = bool(localized)
        if nfs_to_local_copy_ms is not None:
            try:
                ent.nfs_to_local_copy_ms = int(nfs_to_local_copy_ms)
            except Exception:
                pass
        if bytes_copied is not None:
            try:
                ent.bytes_copied = int(bytes_copied)
            except Exception:
                pass

    def add_pipeline_init_time(self, ms: int) -> None:
        try:
            ms_i = int(ms)
        except Exception:
            return
        if ms_i < 0:
            return
        self._pipeline_init_ms_accum += ms_i

    def add_gpu_load_time(self, ms: int) -> None:
        try:
            ms_i = int(ms)
        except Exception:
            return
        if ms_i < 0:
            return
        self._gpu_load_ms_accum += ms_i

    def add_upload_time(self, ms: int) -> None:
        try:
            ms_i = int(ms)
        except Exception:
            return
        if ms_i < 0:
            return
        self._upload_ms_accum += ms_i

    def finalize(self) -> None:
        # Avoid emitting misleading zeros: only set *_ms fields when we observed
        # the corresponding activity (or models exist for fetch_ms).
        if self.fetch_ms is None:
            if self.models or self.required_models:
                self.fetch_ms = int(self._fetch_ms_accum)
        if self.pipeline_init_ms is None and self._pipeline_init_ms_accum > 0:
            self.pipeline_init_ms = int(self._pipeline_init_ms_accum)
        if self.gpu_load_ms is None and self._gpu_load_ms_accum > 0:
            self.gpu_load_ms = int(self._gpu_load_ms_accum)
        if self.upload_ms is None and self._upload_ms_accum > 0:
            self.upload_ms = int(self._upload_ms_accum)

        if self.inference_ms is not None and self.steps is not None and self.inference_ms > 0:
            try:
                self.iters_per_s = float(self.steps) / (float(self.inference_ms) / 1000.0)
            except Exception:
                pass

    def overall_cache_state(self) -> Optional[str]:
        # cold_remote if any required model is cold, else warm_disk if any is warm, else hot_vram.
        states = [m.cache_state for m in self.models.values() if m.cache_state]
        if not states:
            return None
        if "cold_remote" in states:
            return "cold_remote"
        if "warm_disk" in states:
            return "warm_disk"
        if "hot_vram" in states:
            return "hot_vram"
        return None

    def to_metrics_run_payload(self) -> Dict[str, Any]:
        self.finalize()
        out: Dict[str, Any] = {
            "schema_version": 1,
            "function_name": self.function_name,
        }
        cs = self.overall_cache_state()
        if cs is not None:
            out["cache_state"] = cs

        if self.models:
            out["models"] = [m.to_json() for m in self.models.values()]

        for k, v in (
            ("pipeline_init_ms", self.pipeline_init_ms),
            ("gpu_load_ms", self.gpu_load_ms),
            ("warmup_ms", self.warmup_ms),
            ("inference_ms", self.inference_ms),
            ("steps", self.steps),
            ("iters_per_s", self.iters_per_s),
            ("width", self.width),
            ("height", self.height),
            ("guidance", self.guidance),
            ("png_encode_ms", self.png_encode_ms),
            ("upload_ms", self.upload_ms),
            ("peak_vram_bytes", self.peak_vram_bytes),
            ("peak_ram_bytes", self.peak_ram_bytes),
        ):
            if v is None:
                continue
            out[k] = v
        return out

    def canonical_events(self) -> List[tuple[str, Dict[str, Any]]]:
        # Only include canonical events when values exist; fetch_ms may be 0 for warm-disk hits.
        self.finalize()
        evs: List[tuple[str, Dict[str, Any]]] = []
        if self.compute_started_at:
            evs.append(("metrics.compute.started", {"at": self.compute_started_at}))
        if self.compute_completed_at:
            evs.append(("metrics.compute.completed", {"at": self.compute_completed_at}))
        if self.fetch_ms is not None:
            evs.append(("metrics.fetch", {"ms": int(self.fetch_ms)}))
        if self.gpu_load_ms is not None:
            evs.append(("metrics.gpu_load", {"ms": int(self.gpu_load_ms)}))
        if self.inference_ms is not None:
            evs.append(("metrics.inference", {"ms": int(self.inference_ms)}))
        return evs


def best_effort_init_model_metrics(
    rm: RunMetricsV1,
    model_ids: Iterable[str],
    *,
    vram_models: Optional[Iterable[str]] = None,
    disk_models: Optional[Iterable[str]] = None,
    cache_dir: Optional[Path] = None,
) -> None:
    """
    Initialize rm.models entries with snapshot digest + cache_state best-effort.

    - vram_models/disk_models are the worker-reported canonical lists.
    - For Cozy refs with resolved snapshot digests, we can also check snapshot dirs.
    """
    vram_set = set(str(x) for x in (vram_models or []))
    disk_set = set(str(x) for x in (disk_models or []))
    base = cache_dir or _cache_dir()

    resolved = rm.resolved_cozy_models_by_id or {}
    for raw in model_ids:
        mid = str(raw or "").strip()
        if not mid:
            continue
        canon = mid
        snap: Optional[str] = None
        cache_state: Optional[str] = None
        try:
            parsed = parse_model_ref(mid)
            if parsed.scheme == "cozy" and parsed.cozy is not None:
                canon = parsed.cozy.canonical()
        except Exception:
            pass

        resolved_entry = resolved.get(canon) if isinstance(resolved, Mapping) else None
        snap = _extract_snapshot_digest(resolved_entry)

        if canon in vram_set:
            cache_state = "hot_vram"
        elif canon in disk_set:
            cache_state = "warm_disk"
        else:
            # Check snapshot dir for cozy refs.
            if snap:
                try:
                    if _cozy_snapshot_dir(base, snap).exists():
                        cache_state = "warm_disk"
                except Exception:
                    pass
        if cache_state is None:
            cache_state = "cold_remote"

        rm.set_initial_model_state(canon, cache_state, snap)


def best_effort_bytes_downloaded(base_dir: Path, resolved_entry: Any) -> Optional[int]:
    return _missing_bytes_for_resolved_model(base_dir, resolved_entry)


def safe_json_bytes(payload: Dict[str, Any]) -> bytes:
    return json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")


def emit_best_effort(
    emitter: Callable[[str, bytes], None],
    *,
    event_type: str,
    payload: Dict[str, Any],
) -> None:
    try:
        emitter(event_type, safe_json_bytes(payload))
    except Exception:
        # must never fail a run
        return
