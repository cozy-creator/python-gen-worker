"""Host CPU ISA portability for compiled artifacts.

An AOTI ``.pt2`` ships host-side machine code (the wrapper ``.so`` plus any
CPU kernels). torch compiles that code ``-march=native`` when
``inductor.config.cpp.march`` is None and vectorizes CPU kernels with
``cpu_vec_isa.pick_vec_isa()`` — both resolve to the MINT host's CPU. A compiled graph
minted on an AVX-512 host therefore carries EVEX-encoded instructions that
SIGILL any serving host without AVX-512 (exit 132 inside ``aoti_load_package``,
in a crash loop). GPU compatibility is keyed (``sm``); host CPU compatibility is
not keyed by anything upstream — ``cpp.march=None`` hashes identically into the
env seal on every host while the emitted code differs per host, so the key
cannot see the difference.

So: at boot (``env_seal.establish``) the effective codegen target is clamped
to ``min(host level, BASELINE)`` — psABI micro-architecture levels, baseline
``x86-64-v3`` (AVX2/FMA/BMI2; every GPU host SKU family we rent is >= v3).
Measured on the live artifact this costs nothing: the ``-march=native``
wrapper contains ZERO ymm/zmm vector instructions (inductor passes
``-fno-tree-loop-vectorize``); the only above-baseline code is a handful of
incidental EVEX scalar encodings with exact SSE2 equivalents. Because
``cpp.march``/``cpp.simdlen`` are part of ``save_config_portable`` the clamp
is env_seal- and therefore compiled graph-key-visible: hosts below baseline mint and
serve their own honestly-keyed cohort instead of sharing a lying key.

TCG owns artifact admission and runner loading. The worker's responsibility is
therefore only the process-wide compiler clamp, which is exercised against a
real TCG AOTI package below this module's tests.
"""

from __future__ import annotations

import logging
import platform
from functools import lru_cache
from typing import Dict, FrozenSet, Optional, Tuple

from . import torch_capability

logger = logging.getLogger(__name__)

#: The fleet-wide mint target on x86-64 hosts at or above it (psABI level).
BASELINE = "x86-64-v3"

#: ``cpp.simdlen`` companion per effective march level: AVX2-wide CPU-kernel
#: vectorization at >= v3 (within the v3 envelope), 128-bit below (resolves
#: to scalar on x86 torch builds — nothing above the host's own level).
_SIMDLEN_V3 = 256
_SIMDLEN_BELOW_V3 = 128

#: psABI micro-architecture levels as CUMULATIVE /proc/cpuinfo flag sets.
_LEVELS: Tuple[Tuple[str, FrozenSet[str]], ...] = (
    ("x86-64", frozenset()),
    ("x86-64-v2", frozenset(
        {"cx16", "lahf_lm", "popcnt", "sse4_1", "sse4_2", "ssse3"})),
    ("x86-64-v3", frozenset(
        {"abm", "avx", "avx2", "bmi1", "bmi2", "f16c", "fma", "movbe",
         "xsave"})),
    ("x86-64-v4", frozenset(
        {"avx512f", "avx512bw", "avx512cd", "avx512dq", "avx512vl"})),
)

_RANK: Dict[str, int] = {name: i for i, (name, _) in enumerate(_LEVELS)}


class HostIsaError(RuntimeError):
    """The ISA clamp could not be imposed or read back."""


def machine() -> str:
    return platform.machine()


@lru_cache(maxsize=1)
def host_flags() -> FrozenSet[str]:
    """The host CPU feature flags (``/proc/cpuinfo``), lowercased."""
    try:
        with open("/proc/cpuinfo", encoding="ascii", errors="replace") as f:
            for line in f:
                if line.lower().startswith("flags"):
                    _, _, rest = line.partition(":")
                    return frozenset(rest.lower().split())
    except OSError as exc:
        logger.warning("host-isa: /proc/cpuinfo unreadable: %s", exc)
    return frozenset()


def _required_flags(level: str) -> FrozenSet[str]:
    """Cumulative flag set a host must have to EXECUTE code built for
    ``level``. Unknown level names are conservatively treated as the
    host-native worst case (require the highest defined level)."""
    rank = _RANK.get(level)
    if rank is None:
        rank = len(_LEVELS) - 1
    out: set[str] = set()
    for _, flags in _LEVELS[: rank + 1]:
        out |= flags
    return frozenset(out)


def host_level() -> str:
    """The highest psABI level this host fully supports."""
    flags = host_flags()
    level = _LEVELS[0][0]
    for name, _ in _LEVELS[1:]:
        if _required_flags(name) <= flags:
            level = name
        else:
            break
    return level


def mint_march() -> Optional[str]:
    """The march value mints must build with: ``min(host, BASELINE)``.

    None on non-x86 machines (no clamp; torch's own default applies and the
    stamp records the machine so a cross-machine arm still refuses by name).
    """
    if machine() != "x86_64":
        return None
    host = host_level()
    if _RANK[host] >= _RANK[BASELINE]:
        return BASELINE
    return host


def mint_simdlen(march: Optional[str]) -> Optional[int]:
    if march is None:
        return None
    if _RANK.get(march, 0) >= _RANK[BASELINE]:
        return _SIMDLEN_V3
    return _SIMDLEN_BELOW_V3


def _impose_default(inductor_config: object, key: str, value: object) -> None:
    """Process-wide fallback write, via the ONE shared mechanism
    (``settings_authority.impose_config_default``), wrapped so this module's
    callers keep their typed :class:`HostIsaError`."""
    from . import settings_authority

    try:
        settings_authority.impose_config_default(inductor_config, key, value)
    except settings_authority.SettingsImpositionError as exc:
        raise HostIsaError(
            f"isa clamp cannot reach a process-wide target: {exc}") from exc


def _read_in_fresh_thread(fn: object) -> object:
    from . import settings_authority

    return settings_authority.read_in_fresh_thread(fn)  # type: ignore[arg-type]


def impose() -> Dict[str, str]:
    """Clamp torch's inductor codegen target to the portable mint target and
    verify the read-back ON A FOREIGN THREAD. Called from
    ``env_seal.establish`` at boot, before any compile. No-op (empty dict) on
    non-x86 machines, and on a torchless worker — there is no inductor
    codegen to clamp.

    The foreign-thread read-back is the whole point, not belt-and-braces.
    torch's ``user_override`` layer — the one a plain attribute assignment
    writes — is a ``ContextVar``, i.e. THREAD-LOCAL by torch's own
    documentation. Boot imposes on the boot thread; every host compile that
    happens anywhere else does not inherit it. Those are not hypothetical
    threads: ``hot_swap``'s process-global background shape-warm/heal worker
    and the K-way ``run_impl`` splitter pool both host-compile off the boot
    thread, and an unclamped ``-march=native`` object there is the SIGILL
    class this exists to prevent. A same-thread read-back could never see it.
    """
    if not torch_capability.present():
        return {}
    march = mint_march()
    if march is None:
        return {}
    simdlen = mint_simdlen(march)
    import torch._inductor.config as inductor_config

    # Process-wide first: the fallback every non-imposing thread reads.
    _impose_default(inductor_config, "cpp.march", march)
    _impose_default(inductor_config, "cpp.simdlen", simdlen)
    # Then this thread's own override, so a caller that has one already (a
    # test's monkeypatch, a torch `config.patch`) still ends up clamped.
    inductor_config.cpp.march = march
    inductor_config.cpp.simdlen = simdlen
    got_march = inductor_config.cpp.march
    got_simdlen = inductor_config.cpp.simdlen
    if got_march != march or got_simdlen != simdlen:
        raise HostIsaError(
            f"isa clamp did not take effect: imposed march={march!r} "
            f"simdlen={simdlen!r}, effective march={got_march!r} "
            f"simdlen={got_simdlen!r}")
    foreign = _read_in_fresh_thread(
        lambda: (inductor_config.cpp.march, inductor_config.cpp.simdlen))
    if foreign != (march, simdlen):
        raise HostIsaError(
            f"isa clamp is thread-local only: this thread reads "
            f"march={got_march!r} simdlen={got_simdlen!r} but a fresh thread "
            f"reads {foreign!r}. Any host compile off the boot thread would "
            f"be built -march=native (pgw#754).")
    logger.info(
        "host-isa: codegen clamped to march=%s simdlen=%s (host level %s), "
        "process-wide", march, simdlen, host_level())
    return {"cpp_march": march, "cpp_simdlen": str(simdlen)}


def effective() -> Dict[str, str]:
    """Read-back of the live codegen target (seal fact; never assumed).
    Empty on a torchless worker: no codegen target exists to read."""
    if not torch_capability.present():
        return {}
    import torch._inductor.config as inductor_config

    return {
        "cpp_march": str(inductor_config.cpp.march or ""),
        "cpp_simdlen": str(inductor_config.cpp.simdlen or ""),
    }


__all__ = [
    "BASELINE",
    "HostIsaError",
    "effective",
    "host_flags",
    "host_level",
    "impose",
    "machine",
    "mint_march",
    "mint_simdlen",
]
