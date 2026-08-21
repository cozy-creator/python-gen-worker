"""Host CPU ISA portability for compiled artifacts."""

from __future__ import annotations

import logging
import platform
from functools import lru_cache
from typing import Dict, FrozenSet, Optional, Tuple

from . import torch_capability

logger = logging.getLogger(__name__)

BASELINE = "x86-64-v3"

_SIMDLEN_V3 = 256
_SIMDLEN_BELOW_V3 = 128

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
    """The march value mints must build with: ``min(host, BASELINE)``."""
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
    """Clamp torch's inductor codegen target to the portable mint target and verify the read-back ON A FOREIGN THREAD."""
    if not torch_capability.present():
        return {}
    march = mint_march()
    if march is None:
        return {}
    simdlen = mint_simdlen(march)
    import torch._inductor.config as inductor_config

    _impose_default(inductor_config, "cpp.march", march)
    _impose_default(inductor_config, "cpp.simdlen", simdlen)
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
    """Read-back of the live codegen target (seal fact; never assumed)."""
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
