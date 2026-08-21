"""What previous loads and requests ACTUALLY cost — measured, not derived."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_GIB = 1 << 30

WINDOW = 64

MIN_SAMPLES_FOR_PERCENTILE = 8


def _ledger_root() -> Path:
    env = os.environ.get("COZY_RESIDENCY_LEDGER_DIR")
    if env:
        return Path(env)
    return Path.home() / ".cache/cozy/residency-ledger"


REGIMES = ("eager", "compiled")


def shape_key(
    *,
    width: Optional[int] = None,
    height: Optional[int] = None,
    batch: Optional[int] = None,
    extras: str = "none",
    regime: str = "eager",
) -> str:
    """The shape-class half of the key."""
    w = str(width) if width else "?"
    h = str(height) if height else "?"
    b = str(batch) if batch else "?"
    tag = extras if extras in ("none", "adapters", "controlnet") else "none"
    reg = regime if regime in REGIMES else f"unknown({regime or '?'})"
    return f"{w}x{h}x{b}:{tag}:{reg}"


@dataclass
class KeyStats:
    """Samples for one endpoint x checkpoint x shape-class."""

    activation_bytes: List[int] = field(default_factory=list)
    retry_counts: List[int] = field(default_factory=list)
    placement_bytes: Optional[Dict[str, int]] = None
    requests_per_boot: List[int] = field(default_factory=list)

    def observe_request(self, activation: int, retries: int = 0) -> None:
        self.activation_bytes.append(int(activation))
        self.retry_counts.append(int(retries))
        del self.activation_bytes[:-WINDOW]
        del self.retry_counts[:-WINDOW]

    def activation_percentile(self, pct: float = 0.99) -> Optional[int]:
        """The windowed percentile, or **None when the window is too small to mean anything.** None is the honest answer and the caller must use its own floor — see :data:`MIN_SAMPLES_FOR_PERCENTILE`."""
        n = len(self.activation_bytes)
        if n < MIN_SAMPLES_FOR_PERCENTILE:
            return None
        ordered = sorted(self.activation_bytes)
        idx = min(n - 1, max(0, int(round(pct * (n - 1)))))
        return ordered[idx]

    def to_json(self) -> Dict[str, Any]:
        return {
            "activation_bytes": list(self.activation_bytes),
            "retry_counts": list(self.retry_counts),
            "placement_bytes": self.placement_bytes,
            "requests_per_boot": list(self.requests_per_boot),
        }

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "KeyStats":
        s = cls()
        s.activation_bytes = [int(x) for x in d.get("activation_bytes", [])][-WINDOW:]
        s.retry_counts = [int(x) for x in d.get("retry_counts", [])][-WINDOW:]
        pb = d.get("placement_bytes")
        s.placement_bytes = (
            {str(k): int(v) for k, v in pb.items()} if isinstance(pb, dict) else None
        )
        s.requests_per_boot = [
            int(x) for x in d.get("requests_per_boot", [])
        ][-WINDOW:]
        return s


class ResidencyLedger:
    """One endpoint's measured history."""

    def __init__(self, endpoint: str, checkpoint: str, root: Optional[Path] = None):
        self.endpoint = endpoint or "unknown"
        self.checkpoint = checkpoint or "unknown"
        self._root = root or _ledger_root()
        self._lock = threading.Lock()
        self._keys: Dict[str, KeyStats] = {}
        self._boot_requests = 0
        self._dirty = False
        self._load()

    @property
    def path(self) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in self.endpoint)
        ck = "".join(c if c.isalnum() else "_" for c in self.checkpoint)[:32]
        return self._root / f"{safe}.{ck}.json"

    def _load(self) -> None:
        try:
            raw = json.loads(self.path.read_text())
        except Exception:  # noqa: BLE001 - a missing or corrupt ledger is a COLD ledger
            return
        for k, v in (raw.get("keys") or {}).items():
            try:
                self._keys[str(k)] = KeyStats.from_json(v)
            except Exception:  # noqa: BLE001
                continue

    def stats(self, key: str) -> KeyStats:
        with self._lock:
            return self._keys.setdefault(key, KeyStats())

    def observe_request(
        self, key: str, *, activation_bytes: int, retries: int = 0
    ) -> None:
        """One request's activation peak."""
        with self._lock:
            self._keys.setdefault(key, KeyStats()).observe_request(
                activation_bytes, retries
            )
            self._boot_requests += 1
            self._dirty = True

    def observe_placement(self, key: str, attribution: Dict[str, int]) -> None:
        with self._lock:
            st = self._keys.setdefault(key, KeyStats())
            st.placement_bytes = {str(k): int(v) for k, v in attribution.items()}
            self._dirty = True

    def close_boot(self) -> None:
        """Bank this boot's request count."""
        with self._lock:
            if self._boot_requests <= 0:
                return
            for st in self._keys.values():
                st.requests_per_boot.append(self._boot_requests)
                del st.requests_per_boot[:-WINDOW]
            self._boot_requests = 0
            self._dirty = True

    def flush(self) -> bool:
        """Write atomically."""
        with self._lock:
            if not self._dirty:
                return False
            payload = {
                "endpoint": self.endpoint,
                "checkpoint": self.checkpoint,
                "keys": {k: v.to_json() for k, v in self._keys.items()},
            }
            try:
                self._root.mkdir(parents=True, exist_ok=True)
                fd, tmp = tempfile.mkstemp(dir=str(self._root), suffix=".tmp")
                with os.fdopen(fd, "w") as fh:
                    json.dump(payload, fh)
                os.replace(tmp, self.path)
            except Exception as exc:  # noqa: BLE001
                logger.info(
                    "residency ledger: could not write %s (%s: %s); next load "
                    "starts cold", self.path, type(exc).__name__, exc,
                )
                return False
            self._dirty = False
            return True

    def summary(self, key: str) -> str:
        st = self._keys.get(key)
        if st is None or not st.activation_bytes:
            return f"ledger[{key}]=cold"
        p99 = st.activation_percentile()
        n = len(st.activation_bytes)
        act = (
            f"{p99 / _GIB:.2f}GiB(p99,n={n})" if p99 is not None
            else f"n={n}<{MIN_SAMPLES_FOR_PERCENTILE}, NOT TRUSTED"
        )
        rpb = st.requests_per_boot
        return (
            f"ledger[{key}] activation={act} "
            f"retries_max={max(st.retry_counts) if st.retry_counts else 0} "
            f"requests_per_boot={rpb[-1] if rpb else '?'}"
        )


__all__ = [
    "MIN_SAMPLES_FOR_PERCENTILE",
    "WINDOW",
    "KeyStats",
    "REGIMES",
    "ResidencyLedger",
    "shape_key",
]
