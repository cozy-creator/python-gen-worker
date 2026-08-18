"""Fleet-line preflight for measurement rigs.

A rig measures a rig. When the rig's torch/CUDA line drifts off the fleet's, its
verdict is about software nobody runs — and it reads as a verdict about
production. So the rule is mechanical: **every rig calls
:func:`assert_fleet_line` as its first act, and a rig that has not asserted the
fleet line has produced no evidence.**

THE EXPECTED VALUES ARE READ, NEVER HARDCODED. A constant copied into this module
is the same failure one layer down — it would go stale the next time the fleet
moves and nobody would notice. Authorities, all of them files production itself
reads, best-first:

1. ``endpoint.toml`` ``[[build.profiles]]`` — ``torch`` + ``cuda``. This is the
   exact declaration tensorhub's builder resolves the managed base image from
   (``internal/builder/baseimage.go``), so it is what the endpoint's production
   image literally runs. It ships inside the endpoint sdist, so a rig that has the
   endpoint source — which every rig must, since the rig imports the endpoint's
   own code — has this file. The only authority that declares CUDA.
2. ``fleet-floors.toml`` ``[floors] torch`` — the fleet-wide floor CI enforces
   against every endpoint's lock. Fleet-scoped, torch only.
3. Installed distribution metadata — the ENDPOINT dist's own ``torch>=…``
   requirement. Torch only, and never this SDK's own: ``gen-worker`` pins the
   very torch the rig is being asked to prove, so consulting it made the SDK
   its own last-resort authority and no rig could ever fail to find one.

Every authority found is compared and the STRICTEST floor wins, so a rig sitting
between two of them fails rather than picking the lenient one. **No authority
found is also an abort**: nothing to compare against is not permission to
measure.

There is deliberately no override and no "warn" mode. A missing wheel for the
fleet line is a FINDING to report, never permission to measure on an old one.

VERSION STRINGS ARE NOT USABILITY. Matching ``torch.version.cuda``
proves the WHEEL is right and says nothing about the HOST. RunPod's driver is
per-host: on ``570.211.01`` (CUDA 12.8) a cu130 torch imports fine, prints every
version correctly, and then cannot allocate — a failure that surfaces ~20 minutes
in, after the weight fetch, and reads as a torch bug. So the assertion also
performs a REAL tiny allocation and raises :class:`CudaUnusable` when a card is
present and it fails. Repair it at bring-up with
``python3 -m gen_worker.rigboot`` (forward-compat libcuda, or a re-roll verdict)
BEFORE anything is downloaded.
"""

from __future__ import annotations

import os
import re
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence
from .hostfacts import cuda_ready

__all__ = [
    "CudaUnusable",
    "FleetLine",
    "FleetLineMismatch",
    "FleetLineUnknown",
    "assert_fleet_line",
    "format_report",
    "resolve_environment",
    "resolve_fleet_line",
]

#: Config value (a path), not a logic flag: point the resolver at the authority
#: when the rig's layout hides it. It changes WHERE the expectation is read from,
#: never WHETHER the assertion runs.
FLEET_LINE_FILE_ENV = "GEN_WORKER_FLEET_LINE_FILE"


#: Wheels whose version decides whether a kernel/quantization result is about the
#: fleet's stack or about the rig's. Reported on every run; absence is reported,
#: not raised — an unbuildable wheel is the finding.
REPORTED_PACKAGES = (
    "gen-worker",
    "torch",
    "torchvision",
    "torchaudio",
    "triton",
    "transformers",
    "diffusers",
    "accelerate",
    "torchao",
    "sageattention",
    "flash-attn",
    "flashinfer-python",
    "nvidia-cudnn-cu13",
    "nvidia-cudnn-cu12",
)

_MAX_WALK_UP = 8


class FleetLineMismatch(RuntimeError):
    """The rig is not on the fleet's torch/CUDA line. Measuring is forbidden."""


class FleetLineUnknown(RuntimeError):
    """No authority declared a fleet line. Nothing to assert against."""


class CudaUnusable(FleetLineMismatch):
    """The wheel is on the fleet line but the HOST cannot run it.

    A subclass of :class:`FleetLineMismatch` so every existing rig handler still
    aborts, and a distinct type so the caller can tell "rebuild the environment"
    (wrong wheel) from "repair or re-roll the host" (wrong driver).
    """


@dataclass(frozen=True)
class Authority:
    """One declaration of the fleet line, and where it was read from."""

    torch: Optional[tuple[int, ...]]
    cuda: Optional[tuple[int, ...]]
    source: str
    origin: str


@dataclass(frozen=True)
class FleetLine:
    """The strictest floor across every authority that could be read."""

    torch: tuple[int, ...]
    cuda: Optional[tuple[int, ...]]
    authorities: tuple[Authority, ...] = field(default=())

    @property
    def torch_text(self) -> str:
        return ".".join(str(p) for p in self.torch)

    @property
    def cuda_text(self) -> str:
        return ".".join(str(p) for p in self.cuda) if self.cuda else "undeclared"


# --------------------------------------------------------------------------- #
# version parsing
# --------------------------------------------------------------------------- #


def _version(text: object, parts: int = 3) -> Optional[tuple[int, ...]]:
    """``'2.13.0+cu130'`` -> ``(2, 13, 0)``. None when nothing numeric is there."""
    if text is None:
        return None
    head = str(text).strip().split("+")[0].split(" ")[0]
    out: list[int] = []
    for token in head.split("."):
        digits = re.match(r"\d+", token)
        if not digits:
            break
        out.append(int(digits.group()))
        if len(out) == parts:
            break
    if not out:
        return None
    return tuple(out)


def _pad(v: Sequence[int], n: int) -> tuple[int, ...]:
    return tuple(list(v)[:n] + [0] * (n - len(v)))


def _below(found: Optional[Sequence[int]], floor: Sequence[int]) -> bool:
    if found is None:
        return True
    n = max(len(found), len(floor))
    return _pad(found, n) < _pad(floor, n)


def _requirement_floor(requirement: str, name: str) -> Optional[tuple[int, ...]]:
    """Lowest version ``name`` may take under one PEP 508 requirement string."""
    req = requirement.split(";")[0].strip()
    head = re.match(r"^([A-Za-z0-9._-]+)", req)
    if not head or head.group(1).lower().replace("_", "-") != name:
        return None
    best: Optional[tuple[int, ...]] = None
    for op, ver in re.findall(r"(>=|==|~=|>)\s*([0-9][0-9A-Za-z.*+!-]*)", req):
        parsed = _version(ver.replace("*", "0"))
        if parsed and (best is None or parsed > best):
            best = parsed
    return best


# --------------------------------------------------------------------------- #
# authorities
# --------------------------------------------------------------------------- #


def _walk_up(start: Path, name: str) -> Iterable[Path]:
    here = start if start.is_dir() else start.parent
    for _ in range(_MAX_WALK_UP):
        candidate = here / name
        if candidate.is_file():
            yield candidate
        if here.parent == here:
            break
        here = here.parent


def _search_roots(start: Optional[Path]) -> list[Path]:
    roots: list[Path] = []
    for p in (start, Path.cwd(), Path("/src"), Path("/rig")):
        if p and p.exists() and p not in roots:
            roots.append(p)
    return roots


def _candidate_files(name: str, start: Optional[Path]) -> list[Path]:
    found: list[Path] = []
    for root in _search_roots(start):
        for hit in _walk_up(root, name):
            if hit not in found:
                found.append(hit)
        if root.is_dir():
            # One level down covers the pod layout `/src/<endpoint>/endpoint.toml`.
            for child in sorted(root.iterdir()):
                hit = child / name
                if child.is_dir() and hit.is_file() and hit not in found:
                    found.append(hit)
    return found


def _endpoint_toml_authority(path: Path) -> Optional[Authority]:
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return None
    profiles = (data.get("build") or {}).get("profiles") or []
    if not isinstance(profiles, list):
        return None
    chosen: Optional[Mapping[str, Any]] = None
    for profile in profiles:
        if not isinstance(profile, Mapping):
            continue
        if str(profile.get("accelerator", "")).lower() == "cuda":
            chosen = profile
            break
        chosen = chosen or profile
    if chosen is None:
        return None
    torch = _version(chosen.get("torch"))
    cuda = _version(chosen.get("cuda"), parts=2)
    if torch is None and cuda is None:
        return None
    return Authority(
        torch=torch,
        cuda=cuda,
        source="endpoint.toml [[build.profiles]]",
        origin=f"{path} (profile {chosen.get('name', '?')})",
    )


def _fleet_floors_authority(path: Path) -> Optional[Authority]:
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return None
    torch = _version((data.get("floors") or {}).get("torch"))
    if torch is None:
        return None
    return Authority(
        torch=torch,
        cuda=None,
        source="fleet-floors.toml [floors]",
        origin=str(path),
    )


def _metadata_authority(dist: str, extra_hint: str = "") -> Optional[Authority]:
    from importlib import metadata

    try:
        requirements = metadata.requires(dist) or []
    except metadata.PackageNotFoundError:
        return None
    best: Optional[tuple[int, ...]] = None
    for requirement in requirements:
        floor = _requirement_floor(requirement, "torch")
        if floor and (best is None or floor > best):
            best = floor
    if best is None:
        return None
    return Authority(
        torch=best,
        cuda=None,
        source=f"{dist} distribution metadata{extra_hint}",
        origin=f"importlib.metadata.requires({dist!r}) -> torch>={'.'.join(map(str, best))}",
    )


def _collect_authorities(
    start: Optional[Path], endpoint_dists: Sequence[str]
) -> list[Authority]:
    explicit = os.environ.get(FLEET_LINE_FILE_ENV, "").strip()
    if explicit:
        path = Path(explicit)
        found = _endpoint_toml_authority(path) or _fleet_floors_authority(path)
        if found is None:
            raise FleetLineUnknown(
                f"{FLEET_LINE_FILE_ENV}={explicit} declares no fleet line "
                "(expected an endpoint.toml [[build.profiles]] or a "
                "fleet-floors.toml [floors] torch)"
            )
        return [found]

    authorities: list[Authority] = []
    for path in _candidate_files("endpoint.toml", start):
        found = _endpoint_toml_authority(path)
        if found:
            authorities.append(found)
    for path in _candidate_files("fleet-floors.toml", start):
        found = _fleet_floors_authority(path)
        if found:
            authorities.append(found)
    # ENDPOINT dists only — never `gen-worker` itself. The SDK certifying its own
    # floor means every rig passes and `FleetLineUnknown`, the finding this
    # function exists to produce, is unreachable on any machine that can run one.
    for dist in endpoint_dists:
        found = _metadata_authority(dist)
        if found:
            authorities.append(found)
    return authorities


def resolve_fleet_line(
    *,
    start: Optional[os.PathLike[str] | str] = None,
    endpoint_dists: Sequence[str] = (),
) -> FleetLine:
    """Read every reachable authority and return the STRICTEST floor.

    Raises :class:`FleetLineUnknown` when nothing declares a line.
    """
    root = Path(start).resolve() if start else None
    authorities = _collect_authorities(root, endpoint_dists)
    torch_floors = [a.torch for a in authorities if a.torch]
    if not torch_floors:
        raise FleetLineUnknown(
            "no fleet-line authority is reachable from "
            f"{[str(p) for p in _search_roots(root)]}. A rig with nothing to "
            "compare against has produced no evidence: ship the endpoint's "
            f"endpoint.toml next to the rig, or set {FLEET_LINE_FILE_ENV} to it."
        )
    cuda_floors = [a.cuda for a in authorities if a.cuda]
    return FleetLine(
        torch=max(torch_floors),
        cuda=max(cuda_floors) if cuda_floors else None,
        authorities=tuple(authorities),
    )


# --------------------------------------------------------------------------- #
# resolved environment
# --------------------------------------------------------------------------- #


def _driver_version() -> Optional[str]:
    import shutil
    import subprocess

    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    try:
        out = subprocess.run(
            [smi, "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    line = out.stdout.strip().splitlines()
    return line[0].strip() if line else None


def _package_versions() -> dict[str, str]:
    from importlib import metadata

    out: dict[str, str] = {}
    for name in REPORTED_PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return out


def resolve_environment() -> dict[str, Any]:
    """torch / CUDA / driver / device / wheel versions, as actually installed."""
    env: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "packages": _package_versions(),
    }
    try:
        import torch  # noqa: PLC0415 — heavy, and torch is a capability
    except Exception as exc:  # noqa: BLE001 — an unimportable torch IS the report
        env["torch"] = None
        env["torch_import_error"] = f"{type(exc).__name__}: {exc}"[:400]
        env["cuda"] = None
        return env

    env["torch"] = getattr(torch, "__version__", None)
    env["cuda"] = getattr(getattr(torch, "version", None), "cuda", None)
    try:
        env["cudnn"] = torch.backends.cudnn.version()
    except Exception:  # noqa: BLE001
        env["cudnn"] = None
    try:
        env["cuda_available"] = bool(cuda_ready())
    except Exception:  # noqa: BLE001
        env["cuda_available"] = False
    if env["cuda_available"]:
        try:
            props = torch.cuda.get_device_properties(0)
            env["device"] = props.name
            env["sm"] = f"{props.major}.{props.minor}"
            env["vram_gib"] = round(props.total_memory / 2**30, 2)
        except Exception:  # noqa: BLE001
            pass
    env["driver"] = _driver_version()
    env.update(_usability(env))
    return env


def _usability(env: Mapping[str, Any]) -> dict[str, Any]:
    """Can this interpreter actually ALLOCATE on the card?

    ``torch.version.cuda`` matching the fleet line proves the WHEEL is right and
    nothing about the HOST. RunPod's driver is per-host: on a 570.211.01 host
    (CUDA 12.8) a cu130 torch imports fine, reports every version string
    correctly, and then dies on the first allocation ~20 minutes later, which
    reads as a torch bug. So the check is a real allocation, and the result is
    part of the environment table.
    """
    from gen_worker.cuda_probe import classify_probe_failure, probe_cuda

    out: dict[str, Any] = {}
    if env.get("torch") is None:
        return out
    result = probe_cuda()
    out["cuda_usable"] = result.ok
    if not result.ok:
        out["cuda_unusable_reason"] = result.reason
        out["cuda_unusable_class"] = classify_probe_failure(result.reason)
    return out


def _usability_text(env: Mapping[str, Any]) -> str:
    """One line: a real allocation either worked, or why it did not."""
    usable = env.get("cuda_usable")
    if usable is None:
        return "not probed"
    if usable:
        return "yes (real allocation)"
    return (
        f"NO — {env.get('cuda_unusable_class', 'unknown')}: "
        f"{env.get('cuda_unusable_reason', '')}"
    )


def format_report(env: Mapping[str, Any], line: FleetLine) -> str:
    """The environment table every rig report must carry."""
    rows: list[tuple[str, str]] = [
        ("torch", f"{env.get('torch')}  (fleet floor {line.torch_text})"),
        ("torch CUDA", f"{env.get('cuda')}  (fleet floor {line.cuda_text})"),
        ("cudnn", str(env.get("cudnn"))),
        ("driver", str(env.get("driver"))),
        ("cuda usable", _usability_text(env)),
        ("device", f"{env.get('device', 'none')}  sm{env.get('sm', '-')}  "
                   f"{env.get('vram_gib', '-')} GiB"),
        ("python", str(env.get("python"))),
    ]
    for name, version in (env.get("packages") or {}).items():
        if name != "torch":
            rows.append((name, version))
    width = max(len(k) for k, _ in rows)
    body = "\n".join(f"  {k.ljust(width)}  {v}" for k, v in rows)
    provenance = "\n".join(
        f"  - {a.source}: torch>={'.'.join(map(str, a.torch)) if a.torch else '-'}"
        f"{', cuda>=' + '.'.join(map(str, a.cuda)) if a.cuda else ''}\n"
        f"      {a.origin}"
        for a in line.authorities
    )
    return (
        "FLEET LINE — resolved environment\n"
        f"{body}\n"
        "read from:\n"
        f"{provenance}"
    )


def assert_fleet_line(
    what: str = "rig",
    *,
    start: Optional[os.PathLike[str] | str] = None,
    endpoint_dists: Sequence[str] = (),
    stream: Any = None,
) -> dict[str, Any]:
    """Abort unless this interpreter is on the fleet's torch/CUDA line.

    Call it as the FIRST act of any rig, harness, probe or scoring script. On
    success it prints the resolved environment table (so every report carries
    it) and returns that environment as a dict. On mismatch it raises
    :class:`FleetLineMismatch`; when no authority is reachable it raises
    :class:`FleetLineUnknown`. Both are fatal by design — there is no override.
    """
    out = stream if stream is not None else sys.stderr
    line = resolve_fleet_line(start=start, endpoint_dists=endpoint_dists)
    env = resolve_environment()
    report = format_report(env, line)

    problems: list[str] = []
    if env.get("torch") is None:
        problems.append(
            f"torch does not import: {env.get('torch_import_error', 'unknown')}"
        )
    elif _below(_version(env["torch"]), line.torch):
        problems.append(
            f"torch {env['torch']} is below the fleet line {line.torch_text}"
        )
    if line.cuda is not None:
        found = _version(env.get("cuda"), parts=2)
        if _below(found, line.cuda):
            problems.append(
                f"torch CUDA build {env.get('cuda')} is below the fleet line "
                f"{line.cuda_text}"
            )

    # A HOST fault is reported before a WHEEL fault even when both are present:
    # on a too-old driver every version string above can be perfect while nothing
    # allocates, and "rebuild the environment" is then the wrong instruction.
    # NOT gated on `driver` any more: reading a driver version means
    # `nvidia-smi` RAN, and a host broken enough that it does not is exactly the
    # host this refusal is for — the check was skipping the machines it was
    # written for. The carve-out it was carrying ("no card at all is a different
    # mistake") survives, on a signal a broken diagnostic cannot fake: the probe
    # says WHICH failure this is, and only a host that has a driver to fail can
    # answer `driver_too_old`/`cuda_error`.
    from .cuda_probe import CARDLESS_PROBE_CLASSES

    if (env.get("cuda_usable") is False
            and str(env.get("cuda_unusable_class") or "")
            not in CARDLESS_PROBE_CLASSES):
        raise CudaUnusable(
            f"{what}: REFUSING TO MEASURE — the wheel is on the fleet line but "
            f"this HOST cannot run it.\n"
            f"  * driver {env.get('driver') or '<nvidia-smi unreadable>'} / "
            f"torch CUDA {env.get('cuda')}: "
            f"{env.get('cuda_unusable_class')} — {env.get('cuda_unusable_reason')}\n\n"
            + report
            + "\n\nThis is a DRIVER fact, not a torch defect (pgw#1120). RunPod's "
            "driver version is PER-HOST: 570.211.01 is CUDA 12.8 and cannot run a "
            "cu130 build, while other hosts of the same GPU type draw 580.x.\n"
            "Fix it before the fetch, not after: `python3 -m gen_worker.rigboot "
            "--cuda <line>` installs NVIDIA's forward-compat libcuda "
            "(cuda-compat-13-0) and re-verifies with a real allocation, and exits "
            "91 when the host must be re-rolled instead. See research/RIG-ENV.md "
            "§3c."
        )

    if problems:
        raise FleetLineMismatch(
            f"{what}: REFUSING TO MEASURE — this environment is not the fleet's.\n"
            + "\n".join(f"  * {p}" for p in problems)
            + "\n\n"
            + report
            + "\n\nA rig off the fleet line produces no evidence (pgw#1114). Rebuild "
            "the environment per RIG-ENV.md; a missing wheel for the fleet line is "
            "the finding to report, not permission to measure on an old one."
        )

    print(f"{what}: on the fleet line.\n{report}", file=out, flush=True)
    return env


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``python -m gen_worker.rigcheck`` — 0 on the line, 90 off it, 91 when the
    line is installed but the HOST cannot run it (repair or re-roll)."""
    args = list(sys.argv[1:] if argv is None else argv)
    start = args[0] if args else None
    try:
        assert_fleet_line("rigcheck", start=start, stream=sys.stdout)
    except CudaUnusable as exc:
        print(str(exc), file=sys.stderr, flush=True)
        return 91
    except (FleetLineMismatch, FleetLineUnknown) as exc:
        print(str(exc), file=sys.stderr, flush=True)
        return 90
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
