"""A stand-in mint child for pgw#784's supervisor tests.

Substituted for ``gen_worker.mint_child`` by monkeypatching
``mint_process.MINT_CHILD_MODULE``. It exercises the PROTOCOL and the
supervisor's classification — frames, report, exit codes, signal deaths,
CPU-burning liveness, silence — without needing a GPU or a real compile.

Behaviour is chosen with ``MINT_STUB_MODE``:

``minted``      frames, write an artifact + report, exit 0
``no_artifact`` report says minted, but no file exists (exit 0)
``refused``     write a refusal report, exit 2
``resource``    write a resource report, exit 3
``bad_request`` exit 4
``crash``       exit 1 with stderr noise and no report (UNCLASSIFIED death)
``failed``      exit 1 having written a ``status="failed"`` report (classified)
``sigkill``     SIGKILL itself
``busy``        burn CPU for MINT_STUB_SECONDS, then mint
``silent``      sleep for MINT_STUB_SECONDS doing nothing at all
``grow``        write work-root bytes slowly (CPU-quiet but progressing)
"""

from __future__ import annotations

import os
import signal
import sys
import time
from pathlib import Path

import msgspec

from gen_worker.child_contract import frame_line  # type: ignore
from gen_worker.mint_process import MintReport, MintRequest  # type: ignore


def _seconds() -> float:
    return float(os.environ.get("MINT_STUB_SECONDS", "1") or 1)


def _report(request: MintRequest, report: MintReport) -> None:
    Path(request.report).parent.mkdir(parents=True, exist_ok=True)
    Path(request.report).write_bytes(msgspec.json.encode(report))


def _mint(request: MintRequest) -> None:
    target = Path(request.target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"stub-cell-bytes")
    _report(request, MintReport(
        status="minted",
        # pgw#1176: a mint reports its ENTRY SET. A double that
        # names one artifact models a product the child no
        # longer makes, and `mint_process` reads `entries`.
        entries=((str(request.arm_token), str(target), "blake3:stub"),),
        cell_key=request.arm_token, phase="finalize",
        peak_vram_bytes=int(os.environ.get("MINT_STUB_PEAK", "0") or 0),
        detail="stub mint"))


def main() -> int:
    request = msgspec.json.decode(Path(sys.argv[1]).read_bytes(), type=MintRequest)
    mode = os.environ.get("MINT_STUB_MODE", "minted")
    sys.stdout.write(frame_line(phase="load", note=f"stub {mode}"))
    sys.stdout.flush()

    if mode == "sigkill":
        os.kill(os.getpid(), signal.SIGKILL)
    if mode == "bad_request":
        return 4
    if mode == "crash":
        sys.stderr.write("stub child exploded\n")
        return 1
    if mode == "failed":
        # pgw#816: what the REAL child does when its own mint raises — it
        # catches, classifies, names the exception and exits 1. Distinct from
        # `crash` (which dies without a report) because the retry policy turns
        # on exactly that distinction.
        sys.stderr.write("stub child: classified failure\n")
        _report(request, MintReport(
            status="failed", phase="load",
            detail="OSError: Error no file named config.json found in "
                   "directory /tmp/tensorhub-cache/cas/snapshots/sha256:"
                   "32fa2ba6__x76b2ae62d32f"))
        return 1
    if mode == "refused":
        _report(request, MintReport(
            status="refused", detail="stub refusal: named and deterministic",
            phase="load"))
        return 2
    if mode == "resource":
        _report(request, MintReport(
            status="resource", detail="stub CUDA OOM", phase="warmup_forward"))
        return 3
    if mode == "no_artifact":
        _report(request, MintReport(
            status="minted", artifact=str(Path(request.target)),
            phase="finalize"))
        return 0
    if mode == "silent":
        time.sleep(_seconds())
        _mint(request)
        return 0
    if mode == "grow":
        capture = Path(request.work_root)
        capture.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + _seconds()
        n = 0
        while time.monotonic() < deadline:
            (capture / f"chunk-{n}").write_bytes(b"x" * (2 << 20))
            n += 1
            time.sleep(0.2)
        _mint(request)
        return 0
    if mode == "busy":
        sys.stdout.write(frame_line(phase="inductor_compile", note="burning"))
        sys.stdout.flush()
        deadline = time.monotonic() + _seconds()
        acc = 0
        while time.monotonic() < deadline:
            for i in range(200000):
                acc += i * i
        sys.stdout.write(frame_line(phase="seal_publish", note=str(acc)[:8]))
        sys.stdout.flush()
        _mint(request)
        return 0

    _mint(request)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
