"""What previous loads and requests ACTUALLY cost — measured, not derived.

pgw#1586 phase 1. **This module records and changes nothing.** No caller reads
it to make a decision yet; that is phases 2-4, and phases 2+ wait on a ruling.
Landing the measurement first is deliberate: it answers, from the field, four
questions the residency ladder is currently guessing at, and it cannot regress
a placement because no placement consults it.

THE DEFECT IT EXISTS TO END, in one line: *the ladder decides from DERIVED
numbers when MEASURED ones exist or could exist.* Five sightings in one day —
a ceiling constant that could not scale with the card, a rung selected without
knowing it costs ~11 s of cold start to save ~4 s a request, a reserve read
from a declaration no endpoint makes, a reserve measured on a DIFFERENT
residency configuration and 3.6x too small on this one, and a probe that read
driver-free while the bytes it wanted sat reusable in the allocator's pool.

## The four facts

``activation_bytes``  ``max_memory_allocated() - baseline``, per request. It is
    the activation peak ALONE and therefore **residency-independent**, which is
    exactly the property pgw#1595's constant lacked: 1.25 GiB was true of the
    ``model_offload`` rung and false by 3.6x of ``partial_resident``.
``placement_bytes``   the ``_placement_attribution`` split at arm time. The
    resident set is sized in WEIGHT bytes; the process costs more than its
    weights.
``requests_per_boot`` a counter. The only input that can answer "is this
    endpoint worth ~11 s of cold start". **Per BOOT, not per scale-to-zero
    cycle** — cadence is hub knowledge and the worker cannot honestly see it
    (coordinator ruling; a hub cadence hint is a possible phase-5 seam, and is
    deliberately NOT plumbed here).
``retry_count``       allocator-OOM retries per request. **The softness signal
    for cached memory**: 1.56 GiB of cached blocks is not 1.56 GiB of
    contiguous space, and retries are that discount showing up at runtime.

## The key

``endpoint x checkpoint-identity x {w}x{h}x{batch} x extras x REGIME``

*Checkpoint identity* makes staleness free — weights change, the key changes,
the ledger is cold, and the safe default applies with no invalidation machinery.
*Shape* is width/height/batch and **not steps**: activations are re-allocated
each step, so a 28-step job shares a class with a 20-step one — which pgw#1595
demonstrated the hard way. *Extras* is one coarse tag (``none``/``adapters``/
``controlnet``) because the activation peak differs by WHETHER extra towers run,
much less by which. *Regime* separates ``eager`` from ``compiled``: pgw#1548
measured the same SDXL shape needing **>1198 MiB of request-time headroom
compiled against 764 eager**, with identical boot VRAM — so a ledger that pooled
them would hand an eager number to a compiled serve and kill the daemon with a
figure that looked measured.

**And on the compiled path there is no second chance:** a mid-graph OOM inside a
compiled artifact is NOT catchable — it is process death (pgw#1255 leg 2, and
``oom_ladder`` says so in its own docstring). Degrade-never-OOM therefore cannot
be upheld reactively there; ADMISSION is the only lever, which is exactly why
this ledger's numbers have to be right per regime.

## The rule that keeps it out of pgw#1560's way

**Read at LOAD boundaries. Written in memory per request, flushed on shutdown.
Never consulted mid-serve.** A sample recorded during request N can only change
the plan at the NEXT load.
"""

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

#: Samples kept per key. Bounded so one anomalous request cannot inflate the
#: reserve forever — pgw#1586 revised the original `max(samples)` to a windowed
#: p99 for exactly that reason.
WINDOW = 64

#: **AT SMALL n, p99 IS MAX.** Below this many samples the ledger REFUSES to
#: offer a percentile, and the caller must fall back to its default floor.
#: Without this guard a single-sample "p99" masquerades as a distribution fact,
#: which is the same derived-dressed-as-measured error the ledger exists to end.
MIN_SAMPLES_FOR_PERCENTILE = 8


def _ledger_root() -> Path:
    env = os.environ.get("COZY_RESIDENCY_LEDGER_DIR")
    if env:
        return Path(env)
    return Path.home() / ".cache/cozy/residency-ledger"


#: Execution regimes whose activation peaks are NOT interchangeable.
REGIMES = ("eager", "compiled")


def shape_key(
    *,
    width: Optional[int] = None,
    height: Optional[int] = None,
    batch: Optional[int] = None,
    extras: str = "none",
    regime: str = "eager",
) -> str:
    """The shape-class half of the key. Unknown dimensions become ``?`` rather
    than a guess: a key that lies is worse than a key that admits ignorance,
    because the samples under it would be pooled across shapes.

    **``regime`` IS LOAD-BEARING AND WAS ADDED BEFORE THIS MODULE SHIPPED, on
    evidence from the pgw#1548 benchmark lane.** SDXL compiled batch-2 1024² on
    the 8 GiB card needs **>1198 MiB of request-time headroom where eager needs
    764** — the same shape, the same weights, the same boot VRAM (6990 MiB
    either way, so arming is free), and the compiled arm KILLS THE DAEMON while
    the eager arm serves at a 7754 MiB peak. A ledger keyed on shape alone would
    hand an eager measurement to a compiled serve and **reproduce that death
    with a number that looked measured.** That is precisely the failure this
    ledger exists to end, so the regime is in the key from the first commit
    rather than added after it bites.
    """
    w = str(width) if width else "?"
    h = str(height) if height else "?"
    b = str(batch) if batch else "?"
    tag = extras if extras in ("none", "adapters", "controlnet") else "none"
    # AN UNRECOGNISED REGIME MUST NOT COLLAPSE ONTO A KNOWN ONE. Folding it to
    # "eager" would hand a compiled serve the eager samples — which IS the
    # pgw#1548 daemon death, delivered by a number that looks measured. A
    # distinct label pools with nothing, stays cold, and returns the caller to
    # its default floor. Cold is safe; wrong is not.
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
        """The windowed percentile, or **None when the window is too small to
        mean anything.** None is the honest answer and the caller must use its
        own floor — see :data:`MIN_SAMPLES_FOR_PERCENTILE`."""
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
    """One endpoint's measured history. **Phase 1 records; nothing reads it to
    decide.**"""

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

    # -- recording -------------------------------------------------------

    def observe_request(
        self, key: str, *, activation_bytes: int, retries: int = 0
    ) -> None:
        """One request's activation peak. ``activation_bytes`` must be
        ``max_memory_allocated() - baseline`` — the peak ALONE, never a total,
        or the sample stops being residency-independent and inherits exactly the
        bug pgw#1595 found."""
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
        """Bank this boot's request count. Called at shutdown; the number is
        meaningless until the process ends, which is why it is not recorded
        incrementally."""
        with self._lock:
            if self._boot_requests <= 0:
                return
            for st in self._keys.values():
                st.requests_per_boot.append(self._boot_requests)
                del st.requests_per_boot[:-WINDOW]
            self._boot_requests = 0
            self._dirty = True

    # -- persistence -----------------------------------------------------

    def flush(self) -> bool:
        """Write atomically. Returns whether anything was written. A ledger that
        cannot be written is not an error — it is a cold ledger next time."""
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

    # -- reading (phase 2+; unused by any decision today) -----------------

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
