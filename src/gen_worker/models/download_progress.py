from __future__ import annotations

import time
from typing import Any, Callable, Optional


class DownloadProgressReporter:
    """Shared byte-progress accounting for provider model downloads.

    Provider downloaders only call ``update(bytes_downloaded, bytes_total)``.
    This object owns event throttling, throughput, ETA, and terminal payloads
    so Tensorhub/HF/Civitai do not grow provider-specific event math.
    """

    def __init__(
        self,
        *,
        model_id: str,
        provider: str,
        source: str,
        emit: Callable[[str, dict[str, Any]], None],
        request_id: str = "",
        total_bytes: Optional[int] = None,
        min_interval_s: float = 1.0,
        max_interval_s: float = 5.0,
        pct_step: float = 10.0,
        hard_cap: int = 25,
    ) -> None:
        self.model_id = str(model_id or "")
        self.provider = str(provider or "tensorhub")
        self.source = str(source or self.provider)
        self.request_id = str(request_id or "")
        self._emit = emit
        self._total_bytes = int(total_bytes) if isinstance(total_bytes, int) and total_bytes > 0 else None
        self._min_interval_s = max(0.0, float(min_interval_s))
        self._max_interval_s = max(self._min_interval_s, float(max_interval_s))
        self._pct_step = max(0.0, float(pct_step))
        self._hard_cap = max(1, int(hard_cap))
        self._started_at = 0.0
        self._last_emit_t = 0.0
        self._last_emit_pct = -100.0
        self._emitted = 0
        self._last_bytes = 0
        self._last_sample_t = 0.0
        self._throughput_ewma = 0.0

    @property
    def total_bytes(self) -> Optional[int]:
        return self._total_bytes

    def start(self) -> None:
        self._started_at = time.monotonic()
        self._last_sample_t = self._started_at
        self._emit(
            "started",
            self._base_payload(
                {
                    "estimated_total_bytes": int(self._total_bytes or 0),
                    "estimated_eta_seconds": -1,
                }
            ),
        )

    def update(
        self,
        bytes_downloaded: int,
        bytes_total: Optional[int] = None,
        *,
        force: bool = False,
    ) -> None:
        if isinstance(bytes_total, int) and bytes_total > 0:
            self._total_bytes = int(bytes_total)
        total = int(self._total_bytes or 0)
        downloaded = max(0, int(bytes_downloaded or 0))
        if total > 0 and downloaded > total:
            downloaded = total

        now = time.monotonic()
        if self._last_sample_t <= 0:
            self._last_sample_t = now
        dt = now - self._last_sample_t
        dbytes = downloaded - int(self._last_bytes)
        if dt > 0 and dbytes >= 0:
            inst = float(dbytes) / dt
            if self._throughput_ewma <= 0:
                self._throughput_ewma = inst
            else:
                self._throughput_ewma = 0.3 * inst + 0.7 * float(self._throughput_ewma)
        self._last_sample_t = now
        self._last_bytes = downloaded

        if total <= 0:
            return
        pct = (float(downloaded) / float(total)) * 100.0
        since_emit = now - float(self._last_emit_t)
        pct_delta = pct - float(self._last_emit_pct)
        should = force or (
            since_emit >= self._min_interval_s
            and (pct_delta >= self._pct_step or since_emit >= self._max_interval_s)
        )
        if not should:
            return
        if self._emitted >= self._hard_cap and not force:
            return

        tput = float(self._throughput_ewma)
        remaining = max(total - downloaded, 0)
        eta = int(remaining / tput) if tput > 0 else -1
        self._emit(
            "progress",
            self._base_payload(
                {
                    "bytes_downloaded": int(downloaded),
                    "bytes_total": int(total),
                    "percent_complete": round(pct, 1),
                    "eta_remaining_seconds": eta,
                    "throughput_bytes_per_sec": int(tput),
                }
            ),
        )
        self._last_emit_t = now
        self._last_emit_pct = pct
        self._emitted += 1

    def complete(self, bytes_total: Optional[int] = None) -> None:
        if isinstance(bytes_total, int) and bytes_total > 0:
            self._total_bytes = int(bytes_total)
        total = int(self._total_bytes or 0)
        if total > 0:
            self.update(total, total, force=True)
        self._emit(
            "completed",
            self._base_payload(
                {
                    "bytes_total": total,
                    "duration_ms": self._duration_ms(),
                }
            ),
        )

    def fail(self, exc: BaseException) -> None:
        self._emit(
            "failed",
            self._base_payload(
                {
                    "error_type": type(exc).__name__,
                    "duration_ms": self._duration_ms(),
                }
            ),
        )

    def sink(self) -> Callable[[int, Optional[int]], None]:
        def _update(bytes_downloaded: int, bytes_total: Optional[int] = None) -> None:
            self.update(bytes_downloaded, bytes_total)

        return _update

    def _base_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {
            "model_id": self.model_id,
            "provider": self.provider,
            "source": self.source,
        }
        if self.request_id:
            out["request_id"] = self.request_id
        out.update(payload)
        return out

    def _duration_ms(self) -> int:
        if self._started_at <= 0:
            return 0
        return int((time.monotonic() - self._started_at) * 1000)
