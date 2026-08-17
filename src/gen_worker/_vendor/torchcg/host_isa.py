"""Fail-closed host-code policy for every AOTInductor target."""

from __future__ import annotations

import platform
import shlex
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

_X86_64 = "x86_64"
_X86_64_V3 = "x86-64-v3"
_LEVELS: tuple[tuple[str, frozenset[str]], ...] = (
    ("x86-64", frozenset()),
    ("x86-64-v2", frozenset({"cx16", "lahf_lm", "popcnt", "sse4_1", "sse4_2", "ssse3"})),
    (
        _X86_64_V3,
        frozenset({"abm", "avx", "avx2", "bmi1", "bmi2", "f16c", "fma", "movbe", "xsave"}),
    ),
    (
        "x86-64-v4",
        frozenset({"avx512f", "avx512bw", "avx512cd", "avx512dq", "avx512vl"}),
    ),
)
_RANK = {name: index for index, (name, _) in enumerate(_LEVELS)}
_X86_FEATURE_SWITCHES = {
    "abm": "-mabm",
    "avx": "-mavx",
    "avx2": "-mavx2",
    "bmi1": "-mbmi",
    "bmi2": "-mbmi2",
    "cx16": "-mcx16",
    "f16c": "-mf16c",
    "fma": "-mfma",
    "lahf_lm": "-msahf",
    "movbe": "-mmovbe",
    "popcnt": "-mpopcnt",
    "sse4_1": "-msse4.1",
    "sse4_2": "-msse4.2",
    "ssse3": "-mssse3",
    "xsave": "-mxsave",
}
_HOST_FACTS = frozenset(
    {"machine", "host_isa_level", "host_isa_features", "cpp_march", "cpp_simdlen"}
)
_POLICY_LOCK = threading.RLock()


class HostISAError(RuntimeError):
    """The host cannot state, impose, or satisfy its AOTI ISA contract."""


def _required_flags(level: str) -> frozenset[str]:
    rank = _RANK.get(level)
    if rank is None:
        raise HostISAError(f"unknown x86-64 ISA level {level!r}")
    required: set[str] = set()
    for _, flags in _LEVELS[: rank + 1]:
        required.update(flags)
    return frozenset(required)


def _cpu_features() -> frozenset[str]:
    """Return features shared by every processor described by Linux."""

    try:
        rows = Path("/proc/cpuinfo").read_text(encoding="ascii", errors="replace").splitlines()
    except OSError as exc:
        raise HostISAError(f"cannot read /proc/cpuinfo: {exc}") from exc
    feature_sets = []
    for row in rows:
        name, separator, value = row.partition(":")
        if separator and name.strip().lower() in {"flags", "features"}:
            feature_sets.append(set(value.lower().split()))
    if not feature_sets:
        raise HostISAError("/proc/cpuinfo states no CPU flags or features")
    common = feature_sets[0]
    for features in feature_sets[1:]:
        common.intersection_update(features)
    return frozenset(common)


def _encoded_features(features: frozenset[str]) -> str:
    return ",".join(sorted(features)) or "none"


def _decoded_features(value: str) -> frozenset[str]:
    if value == "none":
        return frozenset()
    rows = value.split(",")
    if (
        not rows
        or rows != sorted(set(rows))
        or any(not row or row != row.strip() or row.lower() != row for row in rows)
    ):
        raise HostISAError("host_isa_features must be sorted, unique lowercase names")
    return frozenset(rows)


@dataclass(frozen=True, slots=True)
class _Requirement:
    machine: str
    level: str
    features: frozenset[str]
    march: str
    simdlen: str

    def facts(self) -> dict[str, str]:
        return {
            "machine": self.machine,
            "host_isa_level": self.level,
            "host_isa_features": _encoded_features(self.features),
            "cpp_march": self.march,
            "cpp_simdlen": self.simdlen,
        }


def _host_requirement() -> _Requirement:
    machine = platform.machine().strip().lower()
    if not machine:
        raise HostISAError("platform states no host machine")
    features = _cpu_features()
    if machine != _X86_64:
        if not features:
            raise HostISAError(f"cannot state a native ISA requirement for {machine}")
        return _Requirement(machine, "native", features, "native", "native")

    level = _LEVELS[0][0]
    for candidate, _ in _LEVELS[1:]:
        if _required_flags(candidate) <= features:
            level = candidate
        else:
            break
    march = _X86_64_V3 if _RANK[level] >= _RANK[_X86_64_V3] else level
    return _Requirement(
        machine, march, _required_flags(march), march, "256" if march == _X86_64_V3 else "128"
    )


def _fresh_thread_value(function: Any) -> object:
    value: list[object] = []

    def read() -> None:
        value.append(function())

    thread = threading.Thread(target=read, name="torch-compiled-graphs-isa-readback", daemon=True)
    thread.start()
    thread.join()
    if not value:
        raise HostISAError("fresh-thread ISA readback did not complete")
    return value[0]


def _impose_host_policy() -> dict[str, str]:
    """Impose the closed host-code policy process-wide and return its key facts."""

    with _POLICY_LOCK:
        requirement = _host_requirement()
        try:
            import torch._inductor.config as config
        except ImportError as exc:
            raise HostISAError("PyTorch AOTInductor is required to impose host ISA") from exc

        x86 = requirement.machine == _X86_64
        march = requirement.march if x86 else None
        simdlen = int(requirement.simdlen) if x86 else None
        expected: dict[str, object] = {"cpp.march": march, "cpp.simdlen": simdlen}
        entries = cast(Mapping[str, Any], getattr(config, "_config", {}))
        for name, value in expected.items():
            entry = entries.get(name)
            if entry is None:
                raise HostISAError(f"PyTorch config has no process-wide {name!r} default")
            entry.default = value
        config.cpp.march = march
        config.cpp.simdlen = simdlen
        current = (config.cpp.march, config.cpp.simdlen)
        wanted = (march, simdlen)
        if current != wanted:
            raise HostISAError(f"host ISA clamp reads {current!r}, expected {wanted!r}")
        foreign = _fresh_thread_value(lambda: (config.cpp.march, config.cpp.simdlen))
        if foreign != wanted:
            raise HostISAError(f"host ISA clamp is thread-local: fresh thread reads {foreign!r}")
        return requirement.facts()


def _assert_command_is_clamped(command: str) -> None:
    """Refuse an x86 host compile that escaped the process-wide ISA clamp."""

    if platform.machine().strip().lower() != _X86_64:
        return
    try:
        arguments = shlex.split(command)
    except ValueError as exc:
        raise HostISAError(f"cannot parse host compiler command: {exc}") from exc
    requirement = _host_requirement()
    expected_march = f"-march={requirement.march}"
    marches = [argument for argument in arguments if argument.startswith("-march=")]
    if marches != [expected_march]:
        raise HostISAError(
            "host compiler command violates the ISA clamp: "
            f"expected exactly {expected_march!r}, found {marches!r}"
        )

    allowed_features = {
        _X86_FEATURE_SWITCHES[feature]
        for feature in requirement.features
        if feature in _X86_FEATURE_SWITCHES
    }
    for argument in arguments:
        if argument == expected_march or argument in allowed_features:
            continue
        if argument == "-m64" or argument.startswith(("-mno-", "-mtune=")):
            continue
        if argument.startswith("-m") or argument.startswith("--param=march="):
            raise HostISAError(f"host compiler command violates the ISA clamp with {argument!r}")


def _validate_host_facts(toolchain: Mapping[str, str]) -> _Requirement:
    missing = sorted(_HOST_FACTS - set(toolchain))
    if missing:
        raise HostISAError(f"toolchain is missing host ISA facts {missing!r}")
    machine = toolchain["machine"]
    level = toolchain["host_isa_level"]
    march = toolchain["cpp_march"]
    simdlen = toolchain["cpp_simdlen"]
    features = _decoded_features(toolchain["host_isa_features"])
    if not machine or machine != machine.strip().lower():
        raise HostISAError("machine must be one canonical lowercase name")
    if machine == _X86_64:
        if level not in _RANK or march != level:
            raise HostISAError("x86-64 host ISA level and cpp_march must match a known level")
        expected_simdlen = "256" if level == _X86_64_V3 else "128"
        if level == "x86-64-v4" or simdlen != expected_simdlen:
            raise HostISAError("x86-64 host ISA exceeds the v3 cap or has an invalid simdlen")
        if features != _required_flags(level):
            raise HostISAError("x86-64 host ISA features do not restate its level")
    elif level != "native" or march != "native" or simdlen != "native" or not features:
        raise HostISAError("non-x86 host ISA must carry a complete native feature requirement")
    return _Requirement(machine, level, features, march, simdlen)


def _admit_host(toolchain: Mapping[str, str]) -> None:
    requirement = _validate_host_facts(toolchain)
    machine = platform.machine().strip().lower()
    if requirement.machine != machine:
        raise HostISAError(
            f"artifact host code is {requirement.machine}, this host is {machine or 'unknown'}"
        )
    missing = sorted(requirement.features - _cpu_features())
    if missing:
        raise HostISAError(
            f"artifact requires {requirement.level} but this host lacks {','.join(missing)}"
        )
