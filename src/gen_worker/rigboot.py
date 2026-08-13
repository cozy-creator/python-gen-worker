"""Host-driver preflight for measurement rigs.

``rigcheck`` answers "is the SOFTWARE on the fleet line". This module answers
**can this HOST run it at all.** RunPod's driver version is per-host, not
per-datacenter and not per-GPU type — one host draws ``580.159.04`` while an
H100 elsewhere draws ``570.211.01``, which is CUDA 12.8 and cannot run a cu130
build. torch 2.13+cu130 *imports* perfectly there and then fails on the first
allocation with
``CUDA initialization: driver too old``, so the failure looks like a torch bug and
arrives ~20 minutes in, after the multi-GB weight fetch.

Two consequences, both mechanical here:

1. **Probe on the FIRST ssh, before any fetch.** :func:`probe_host` shells out to
   ``nvidia-smi`` only; it needs no torch, no venv and no gen-worker install, so
   it runs as the first command on a bare pod.
2. **A too-old driver is repairable, not fatal.** NVIDIA ships a forward-compat
   libcuda (``cuda-compat-13-0``) for data-center GPUs; installing it and putting
   it ahead of the host's libcuda makes a cu130 build run against a 570 host
   driver. :func:`ensure_cuda_line` installs it and re-verifies; only when THAT
   fails is the verdict "re-roll this host".

The module always says which of the three paths it took — ``native``,
``compat`` or ``reroll`` — because a wall measured through forward-compat libcuda
is a fact about the report, not a detail.

CLI (the bringup call, exit codes are the contract)::

    python3 -m gen_worker.rigboot --cuda 13.0        # probe, repair, verify
    #   0  usable (see JSON `path`: native | compat)
    #  91  RE-ROLL THIS HOST — no usable path to the required CUDA line
    #  92  no NVIDIA driver at all (not a GPU host)

It is deliberately NOT gated on :func:`gen_worker.rigcheck.assert_fleet_line`: it
runs BEFORE the environment is correct, and its whole job is to make it correct.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Optional, Sequence, TextIO

__all__ = [
    "DriverTooOld",
    "HostProbe",
    "NoDriver",
    "assert_cuda_usable",
    "compat_package",
    "driver_supports_cuda",
    "ensure_cuda_line",
    "min_driver_for_cuda",
    "parse_smi",
    "probe_host",
    "verify_allocation",
]

#: Where NVIDIA's forward-compat libcuda lands, and what has to precede the
#: host's copy on the loader path for a cu13x build to see it.
COMPAT_LIB_DIRS = ("/usr/local/cuda/compat", "/usr/local/cuda-{major}.{minor}/compat")

#: Written by :func:`ensure_cuda_line` so every later step of the bring-up (and
#: the detached pod-side driver, which does not inherit our shell) picks up the
#: loader path. Sourced, not exported — LD_LIBRARY_PATH only takes effect at
#: process start, so a repair applied inside a running interpreter is invisible
#: to it and must be re-exec'd from a fresh one.
ENV_FILE = "/etc/profile.d/zz-cuda-compat.sh"

#: Linux minimum driver per CUDA major, from NVIDIA's CUDA compatibility table.
#: Keyed by MAJOR only: within a major, minor-version compatibility means any
#: driver from that major's branch runs any minor of it. Off-table majors are
#: treated as unknown and the probe reports rather than guesses.
#:
#: TUPLES, not floats. ``580.159.04`` is NEWER than the ``580.65.06`` floor, and
#: as floats 580.159 < 580.65 — a float comparison rejects hosts that run cu130
#: natively.
_MIN_DRIVER_BY_CUDA_MAJOR: dict[int, tuple[int, ...]] = {
    11: (450, 80, 2),
    12: (525, 60, 13),
    13: (580, 65, 6),
}


class NoDriver(RuntimeError):
    """No NVIDIA driver on this host at all — nothing to make usable."""


class DriverTooOld(RuntimeError):
    """The host driver cannot run the required CUDA line, and no repair worked.

    Carries the full :func:`ensure_cuda_line` record on ``.record`` so a caller
    that wants to bank the evidence does not have to re-run the probe.
    """

    def __init__(self, message: str, record: Optional[dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.record: dict[str, Any] = record or {}


class CudaUnusable(RuntimeError):
    """A CUDA device is present but a real allocation on it fails."""


@dataclass(frozen=True)
class HostProbe:
    """What ``nvidia-smi`` says, before anything heavy has been downloaded."""

    driver: Optional[str]
    driver_cuda: Optional[str]
    gpus: tuple[str, ...]
    raw: str = ""

    @property
    def present(self) -> bool:
        return self.driver is not None


# --------------------------------------------------------------------------- #
# probe
# --------------------------------------------------------------------------- #


def _version_tuple(text: Optional[str]) -> Optional[tuple[int, ...]]:
    """``'570.211.01'`` -> ``(570, 211, 1)``; ``'13.0'`` -> ``(13, 0)``."""
    if not text:
        return None
    parts = re.findall(r"\d+", str(text).strip())
    if not parts:
        return None
    return tuple(int(p) for p in parts[:3])


def _pad(v: Sequence[int], n: int) -> tuple[int, ...]:
    return tuple(list(v)[:n] + [0] * (n - len(v)))


def parse_smi(query_csv: str, banner: str = "") -> HostProbe:
    """Parse ``nvidia-smi`` output into a :class:`HostProbe`.

    ``query_csv`` is the body of ``--query-gpu=driver_version,name
    --format=csv,noheader``; ``banner`` is plain ``nvidia-smi`` output, whose
    header carries the *driver's* max supported CUDA version (the number that
    decides whether a cu130 build can run, and the one that is NOT the CUDA
    toolkit version in the image).
    """
    drivers: list[str] = []
    gpus: list[str] = []
    for line in (query_csv or "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if not parts or not parts[0] or not re.match(r"^\d", parts[0]):
            continue
        drivers.append(parts[0])
        if len(parts) > 1 and parts[1]:
            gpus.append(parts[1])
    cuda = None
    found = re.search(r"CUDA Version:\s*([0-9]+\.[0-9]+)", banner or "")
    if found:
        cuda = found.group(1)
    return HostProbe(
        driver=drivers[0] if drivers else None,
        driver_cuda=cuda,
        gpus=tuple(gpus),
        raw=(query_csv or "").strip(),
    )


def _run(cmd: Sequence[str], timeout: float = 60.0) -> tuple[int, str]:
    try:
        out = subprocess.run(
            list(cmd), capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return 127, f"{type(exc).__name__}: {exc}"
    return out.returncode, (out.stdout or "") + (out.stderr or "")


def probe_host() -> HostProbe:
    """Ask the host's driver what it is. No torch, no venv, no fetch."""
    smi = shutil.which("nvidia-smi")
    if not smi:
        return HostProbe(driver=None, driver_cuda=None, gpus=())
    _, query = _run([smi, "--query-gpu=driver_version,name", "--format=csv,noheader"])
    _, banner = _run([smi])
    return parse_smi(query, banner)


# --------------------------------------------------------------------------- #
# the decision
# --------------------------------------------------------------------------- #


def min_driver_for_cuda(cuda: str) -> Optional[tuple[int, ...]]:
    """Lowest Linux driver that runs ``cuda`` natively, or None if off-table."""
    parsed = _version_tuple(cuda)
    if parsed is None:
        return None
    return _MIN_DRIVER_BY_CUDA_MAJOR.get(parsed[0])


def driver_supports_cuda(driver: Optional[str], cuda: str) -> Optional[bool]:
    """Can ``driver`` run a build of ``cuda``? None when it cannot be decided.

    Decided from the driver version against NVIDIA's table rather than from
    ``nvidia-smi``'s "CUDA Version:" banner alone, because the banner is absent
    on some container images while the driver version never is.
    """
    floor = min_driver_for_cuda(cuda)
    found = _version_tuple(driver)
    if floor is None or found is None:
        return None
    n = max(len(found), len(floor))
    return _pad(found, n) >= _pad(floor, n)


def _cuda_major_minor(cuda: str) -> Optional[tuple[int, int]]:
    parsed = _version_tuple(cuda)
    if parsed is None:
        return None
    return parsed[0], (parsed[1] if len(parsed) > 1 else 0)


def compat_package(cuda: str) -> Optional[str]:
    """``'13.0'`` -> ``'cuda-compat-13-0'``; None when the line is unparsable."""
    pair = _cuda_major_minor(cuda)
    if pair is None:
        return None
    return f"cuda-compat-{pair[0]}-{pair[1]}"


def _compat_dirs(cuda: str) -> list[str]:
    major, minor = _cuda_major_minor(cuda) or (0, 0)
    return [d.format(major=major, minor=minor) for d in COMPAT_LIB_DIRS]


def _existing_compat_dir(cuda: str) -> Optional[str]:
    for directory in _compat_dirs(cuda):
        if os.path.isdir(directory) and any(
            name.startswith("libcuda.so") for name in os.listdir(directory)
        ):
            return directory
    return None


def _install_compat(cuda: str, log: TextIO) -> Optional[str]:
    """Install NVIDIA's forward-compat libcuda. Returns its directory, or None.

    The fleet base image (``pytorch/pytorch:*-cuda13.0-*``) descends from
    ``nvidia/cuda``, so the CUDA apt repo is usually already configured; the
    keyring install is the fallback for images where it is not.
    """
    package = compat_package(cuda)
    if not package:
        return None
    env = dict(os.environ, DEBIAN_FRONTEND="noninteractive")

    def apt(*args: str) -> tuple[int, str]:
        try:
            proc = subprocess.run(
                ["apt-get", *args],
                capture_output=True,
                text=True,
                timeout=900,
                check=False,
                env=env,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return 127, f"{type(exc).__name__}: {exc}"
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")

    apt("update", "-qq")
    code, out = apt("install", "-y", "-qq", package)
    if code != 0:
        # The fleet base (`pytorch/pytorch:*-runtime`) does NOT carry NVIDIA's
        # apt repo, and it has no curl either. So the deb is fetched with the
        # stdlib and dpkg'd directly: no keyring, no repo, no extra package
        # needed to install a package.
        print(f"[rigboot] apt has no {package} ({out.strip()[-200:]}); "
              "fetching the .deb straight from NVIDIA", file=log, flush=True)
        deb = _download_compat_deb(package, log)
        if not deb:
            return None
        code, out = _run(["dpkg", "-i", deb], timeout=600)
        if code != 0:
            print(f"[rigboot] dpkg rc={code}: {out[-400:]}", file=log)
            apt("-f", "install", "-y", "-qq")
            code, out = _run(["dpkg", "-i", deb], timeout=600)
        if code != 0:
            print(f"[rigboot] {package} install FAILED: {out[-600:]}", file=log)
            return None
    return _existing_compat_dir(cuda)


def _download_compat_deb(package: str, log: TextIO) -> Optional[str]:
    """Newest ``<package>_*.deb`` from NVIDIA's repo, fetched with the stdlib."""
    import urllib.request

    arch = {"x86_64": "x86_64", "aarch64": "sbsa"}.get(os.uname().machine, "x86_64")
    deb_arch = {"x86_64": "amd64", "sbsa": "arm64"}[arch]
    base = (
        "https://developer.download.nvidia.com/compute/cuda/repos/"
        f"{_distro_tag()}/{arch}/"
    )
    try:
        request = urllib.request.Request(base, headers={"User-Agent": "curl/8.5.0"})
        with urllib.request.urlopen(request, timeout=120) as response:
            index = response.read().decode("utf-8", "ignore")
    except Exception as exc:  # noqa: BLE001 — the network is the failure mode here
        print(f"[rigboot] cannot read {base}: {exc}", file=log)
        return None
    names = set(
        re.findall(rf"{re.escape(package)}_[^\"'<>]*_{deb_arch}\.deb", index)
    )
    if not names:
        print(f"[rigboot] no {package} build for {deb_arch} at {base}", file=log)
        return None
    newest = max(names, key=lambda n: _version_tuple(n.split("_")[1]) or (0,))
    target = os.path.join("/tmp", newest)
    try:
        request = urllib.request.Request(
            base + newest, headers={"User-Agent": "curl/8.5.0"}
        )
        with urllib.request.urlopen(request, timeout=600) as response, open(
            target, "wb"
        ) as handle:
            handle.write(response.read())
    except Exception as exc:  # noqa: BLE001
        print(f"[rigboot] download of {newest} failed: {exc}", file=log)
        return None
    print(f"[rigboot] fetched {newest} ({os.path.getsize(target)} bytes)", file=log,
          flush=True)
    return target


def _distro_tag() -> str:
    """``ubuntu2404`` etc., for NVIDIA's repo path. Defaults to the fleet base's."""
    try:
        with open("/etc/os-release", encoding="utf-8") as handle:
            data = dict(
                line.strip().split("=", 1)
                for line in handle
                if "=" in line and not line.startswith("#")
            )
    except OSError:
        return "ubuntu2404"
    name = data.get("ID", "ubuntu").strip('"')
    version = data.get("VERSION_ID", "24.04").strip('"').replace(".", "")
    return f"{name}{version}"


def _persist_ld_path(directory: str, log: TextIO) -> None:
    """Make the compat libcuda win for every LATER process on this pod."""
    line = f'export LD_LIBRARY_PATH="{directory}:${{LD_LIBRARY_PATH:-}}"\n'
    try:
        os.makedirs(os.path.dirname(ENV_FILE), exist_ok=True)
        with open(ENV_FILE, "w", encoding="utf-8") as handle:
            handle.write(line)
        os.chmod(ENV_FILE, 0o644)
    except OSError as exc:  # noqa: BLE001 — a rig may run unprivileged
        print(f"[rigboot] could not write {ENV_FILE}: {exc}", file=log)
    for profile in ("/root/.bashrc", "/root/.profile"):
        try:
            with open(profile, "a", encoding="utf-8") as handle:
                handle.write(line)
        except OSError:
            pass
    os.environ["LD_LIBRARY_PATH"] = (
        f"{directory}:{os.environ.get('LD_LIBRARY_PATH', '')}".rstrip(":")
    )


_ALLOC_SNIPPET = (
    "import json,sys\n"
    "out={}\n"
    "try:\n"
    "    import torch\n"
    "    out['torch']=torch.__version__; out['cuda']=torch.version.cuda\n"
    "    torch.cuda.init()\n"
    "    t=torch.ones(8,device='cuda'); t=t+1; torch.cuda.synchronize()\n"
    "    p=torch.cuda.get_device_properties(0)\n"
    "    out.update(ok=True,device=p.name,sm=f'{p.major}.{p.minor}',\n"
    "               vram_gib=round(p.total_memory/2**30,2))\n"
    "except Exception as exc:\n"
    "    out.update(ok=False,error=f'{type(exc).__name__}: {exc}'[:400])\n"
    "print(json.dumps(out))\n"
)


def verify_allocation() -> dict[str, Any]:
    """Run a REAL tiny allocation in a FRESH interpreter and return the verdict.

    Fresh, because a loader path repaired inside this process cannot change the
    libcuda already mapped into it; real, because the whole trap is that version
    strings look right while allocation fails.
    """
    code, out = _run([sys.executable, "-c", _ALLOC_SNIPPET], timeout=300)
    for line in reversed(out.strip().splitlines()):
        try:
            return json.loads(line)
        except ValueError:
            continue
    return {"ok": False, "error": f"probe produced no JSON (rc={code}): {out[-300:]}"}


def ensure_cuda_line(
    cuda: str, *, log: Optional[TextIO] = None, allow_compat: bool = True
) -> dict[str, Any]:
    """Make this host able to run ``cuda``, or say it must be re-rolled.

    Returns a record carrying ``path`` — ``native`` (host driver is new enough),
    ``compat`` (forward-compat libcuda installed and verified) or ``reroll``
    (nothing worked) — plus the probe and the allocation verdict. Raises nothing
    for the reroll case: the CLI turns it into exit 91 and the caller kills the
    pod. :class:`NoDriver` is raised only when there is no GPU at all, which is a
    different mistake (wrong pod type) than an old driver.
    """
    log = log or sys.stderr
    probe = probe_host()
    if not probe.present:
        raise NoDriver(
            "nvidia-smi reports no driver: this is not a GPU host. Nothing to "
            "repair — check the pod type before spending on a fetch."
        )
    supported = driver_supports_cuda(probe.driver, cuda)
    floor_t = min_driver_for_cuda(cuda)
    floor = ".".join(str(x) for x in floor_t) if floor_t else "unknown"
    print(
        f"[rigboot] host driver {probe.driver} (max CUDA {probe.driver_cuda}) "
        f"vs CUDA {cuda} floor {floor}: "
        f"{'native' if supported else 'TOO OLD' if supported is False else 'undecidable'}"
        f" — gpus={list(probe.gpus)}",
        file=log,
        flush=True,
    )

    record: dict[str, Any] = {
        "cuda_line": cuda,
        "driver": probe.driver,
        "driver_cuda": probe.driver_cuda,
        "driver_floor": floor,
        "gpus": list(probe.gpus),
        "native_ok": supported,
        # The FLEET question, answered per host in a stable field: a serving pod
        # on this host would need the same forward-compat libcuda. `hardware_report`
        # already ships `driver_version` for every worker, so the same predicate
        # (`driver_supports_cuda`) classifies the whole deployed fleet off-pod.
        "needs_compat": supported is False,
    }

    existing = _existing_compat_dir(cuda)
    if supported is False and existing:
        # A previous run (or the image) already staged it; use it rather than
        # reinstalling, but still prove it works.
        _persist_ld_path(existing, log)
        record["compat_dir"] = existing

    verdict = verify_allocation()
    if verdict.get("ok"):
        record["path"] = "compat" if record.get("compat_dir") else "native"
        record["allocation"] = verdict
        print(f"[rigboot] CUDA USABLE via {record['path']}: {verdict}", file=log,
              flush=True)
        return record

    record["allocation_first"] = verdict
    if not allow_compat:
        record["path"] = "reroll"
        return record

    print(
        f"[rigboot] allocation FAILED on driver {probe.driver}: "
        f"{verdict.get('error')} — installing forward-compat libcuda "
        f"({compat_package(cuda)}). This is a DRIVER fact, not a torch bug.",
        file=log,
        flush=True,
    )
    directory = _existing_compat_dir(cuda) or _install_compat(cuda, log)
    if not directory:
        record["path"] = "reroll"
        record["reason"] = (
            f"{compat_package(cuda)} could not be installed on this host"
        )
        return record

    _persist_ld_path(directory, log)
    record["compat_dir"] = directory
    after = verify_allocation()
    record["allocation"] = after
    if after.get("ok"):
        record["path"] = "compat"
        print(f"[rigboot] CUDA USABLE via compat ({directory}): {after}", file=log,
              flush=True)
        return record

    record["path"] = "reroll"
    record["reason"] = (
        f"forward-compat libcuda installed at {directory} and allocation still "
        f"fails: {after.get('error')}"
    )
    print(f"[rigboot] RE-ROLL THIS HOST: {record['reason']}", file=log, flush=True)
    return record


def assert_cuda_usable(cuda: str, *, log: Optional[TextIO] = None) -> dict[str, Any]:
    """:func:`ensure_cuda_line`, but a re-roll verdict is a typed exception.

    For rigs that want the abort rather than the record. :class:`DriverTooOld`
    carries the whole diagnosis, because the failure it replaces (a cu130
    allocation error 20 minutes into a run) reads as a torch bug.
    """
    record = ensure_cuda_line(cuda, log=log)
    if record.get("path") == "reroll":
        raise DriverTooOld(
            f"host driver {record.get('driver')} cannot run CUDA {cuda} "
            f"(floor {record.get('driver_floor')}) and the forward-compat "
            f"libcuda did not repair it: {record.get('reason')}\n"
            "RE-ROLL THE HOST. RunPod's driver is per-host; creating another pod "
            "of the same GPU type in the same datacenter can and does draw a "
            "different one. This is NOT a torch defect (research/RIG-ENV.md §3c).",
            record,
        )
    return record


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m gen_worker.rigboot",
        description="Probe the host driver and make the fleet CUDA line usable.",
    )
    parser.add_argument(
        "--cuda",
        default=None,
        help="required CUDA line, e.g. 13.0. Default: read from the fleet-line "
             "authorities (endpoint.toml [[build.profiles]] cuda).",
    )
    parser.add_argument("--json", default=None, help="write the record here too")
    parser.add_argument(
        "--no-compat",
        action="store_true",
        help="do not attempt the forward-compat install; probe and report only",
    )
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))

    cuda = args.cuda
    if not cuda:
        try:
            from gen_worker.rigcheck import resolve_fleet_line

            line = resolve_fleet_line()
            cuda = line.cuda_text if line.cuda else None
        except Exception as exc:  # noqa: BLE001 — bringup runs before the env is right
            print(f"[rigboot] no CUDA authority reachable ({exc})", file=sys.stderr)
            cuda = None
    if not cuda or cuda == "undeclared":
        print(
            "[rigboot] no CUDA line declared and none given: pass --cuda 13.0 or "
            "ship the endpoint.toml next to the rig.",
            file=sys.stderr,
        )
        return 2

    reroll = ""
    try:
        record = assert_cuda_usable(cuda)
    except NoDriver as exc:
        print(json.dumps({"path": "no_driver", "error": str(exc)}), flush=True)
        print(str(exc), file=sys.stderr)
        return 92
    except DriverTooOld as exc:
        # The CLI and the library agree by construction: one decision path, the
        # exception carries the diagnosis, the exit code carries the verdict.
        reroll = str(exc)
        record = exc.record

    print(json.dumps(record, indent=1, default=str), flush=True)
    if args.json:
        try:
            with open(args.json, "w", encoding="utf-8") as handle:
                json.dump(record, handle, indent=1, default=str)
        except OSError as exc:  # noqa: BLE001
            print(f"[rigboot] could not write {args.json}: {exc}", file=sys.stderr)

    if reroll:
        print(f"RIGBOOT_REROLL — {reroll}", file=sys.stderr, flush=True)
        return 91
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
