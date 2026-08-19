"""Engine-hosted runtimes: boot / health-wait / abort / stop for an EXTERNAL
engine binary (``llama-server``, ``vllm serve``).

pgw#1421 — the v2 successor to the deleted ``gen_worker.runtimes.server``.
The #1373 hardcut removed the boot half of this tier and left the teardown
half standing (``Model.unload``'s docstring names *"external server
processes"* as its reason to exist), while Paul's 2026-08-19 ruling narrowed
the #1303 ladder's tier 3 to exactly this class and made it PERMANENT
("the fill rung narrows to external binaries"). So this is not a restoration
of v1 machinery: it is the boot half of a ratified, permanent tier, rewritten
on the post-hardcut surface.

**The one spelling**, and it is the sibling of ``ctx.load``::

    class QwenMtpModel(Model[Qwen36MtpGguf], lanes=()):
        def load(self, ctx: LoadContext[Qwen36MtpGguf]) -> None:
            self.engine = ctx.engine(LlamaServer(extra_args=["-ngl", "99"]))

        async def chat(...):
            ... POST f"{self.engine.base_url}/v1/chat/completions" ...

An :class:`EngineSpec` is a DECLARATION — no path, no process, no port. It is
frozen, constructible at import, and carries only what the author knows.
``ctx.engine(spec)`` is what turns it into a running process: it supplies the
checkpoint tree from the deploy binding, plans the boot ladder, starts the
process, waits on real liveness, and REGISTERS the handle so teardown is
structural rather than remembered (``EndpointHost.evict`` stops every engine
after the author's ``unload``, whether or not ``unload`` was written).

Two properties this module refuses to compromise on:

* **A boot is bounded by SILENCE, never by a clock.** A flat deadline whose
  only liveness check is ``proc.poll()`` says nothing about progress, and a
  35B cold load routinely outlives ten minutes — raising the number is the
  same mistake made bigger. Every line the engine prints is proof it is still
  loading (pgw#1243's doctrine, and :class:`gen_worker.stall.SilenceWindow`
  is the one implementation of it). There is deliberately no ``timeout=``
  argument anywhere in this file.
* **Degrade, never OOM.** A tight card gets fewer GPU layers, not a dead
  boot, and the degraded rung is a COUNTABLE typed event rather than a log
  line an operator has to go find.
"""

from __future__ import annotations

import logging
import os
import shlex
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Mapping, Optional, Sequence

import msgspec

from .. import activity as activity_mod
from ..subproc import LineTail

logger = logging.getLogger(__name__)

#: Silence window over the engine's own output. NOT a boot budget: an engine
#: that keeps talking boots for as long as it needs, and one that says nothing
#: for 15 minutes is wedged. Orders of magnitude past any real silent phase
#: (CUDA graph capture, torch.compile) and matches the window the llama.cpp
#: toolchain gets.
BOOT_STALL_WINDOW_S = 900.0
TERM_GRACE_S = 10.0

__all__ = [
    "BOOT_STALL_WINDOW_S",
    "ENGINE_SPEC_RUNTIMES",
    "EngineBootError",
    "EngineCommand",
    "EngineHandle",
    "EngineSpec",
    "LlamaServer",
    "TERM_GRACE_S",
    "VllmServer",
    "boot_engine",
    "free_port",
    "process_vram_bytes",
]


class EngineBootError(RuntimeError):
    """The engine exited, or went silent, during boot.

    The degrade ladder keys on this — so it is raised only on real evidence
    (the process died, or it produced no output for its stall window), never
    because a stopwatch expired.
    """


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@dataclass(frozen=True)
class EngineCommand:
    """One rung of a boot ladder: the argv to run and the port it serves on.

    ``label`` is what a typed event calls this rung — "planned", "half-gpu",
    "cpu-only" — so a degrade is readable without diffing two argv lists.
    """

    argv: tuple[str, ...]
    port: int
    label: str = "planned"


@dataclass
class EngineHandle:
    """A running engine: base URL + process control.

    ``base_url`` is what an entrypoint POSTs to. ``stop`` is idempotent, so
    an author who stops the engine in ``unload`` and the host's structural
    stop cannot double-kill.
    """

    runtime: str
    base_url: str
    process: "subprocess.Popen[str]" = field(repr=False)
    rung: str = "planned"

    @property
    def alive(self) -> bool:
        return self.process.poll() is None

    def stop(self, *, grace_s: float = TERM_GRACE_S) -> None:
        """SIGTERM the process GROUP, wait ``grace_s``, then SIGKILL.

        The group, not the process: vLLM forks worker processes that survive
        a bare SIGTERM to the parent and keep the card's VRAM pinned, which
        is a leak the next residency admit pays for.
        """
        proc = self.process
        if proc.poll() is not None:
            return
        pgid: Optional[int] = None
        try:
            pgid = os.getpgid(proc.pid)
        except (ProcessLookupError, PermissionError, OSError):
            pgid = None
        # HOW it stopped is the interesting half. An engine that had to be
        # SIGKILLed is the one an operator wants to know about — it is the
        # shape that strands VRAM when anything upstream is less careful than
        # this — so the manner is recorded and the row is emitted on EVERY
        # path, not only the clean one.
        manner = "term"
        try:
            if pgid is not None:
                os.killpg(pgid, signal.SIGTERM)
            else:
                proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=grace_s)
        except subprocess.TimeoutExpired:
            manner = "kill"
            logger.warning("%s ignored SIGTERM for %.0fs; killing",
                           self.runtime, grace_s)
            try:
                if pgid is not None:
                    os.killpg(pgid, signal.SIGKILL)
                else:
                    proc.kill()
            except (ProcessLookupError, PermissionError, OSError):
                pass
            try:
                proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:  # pragma: no cover - kernel wedge
                manner = "survived_kill"
                logger.error("%s survived SIGKILL", self.runtime)
        except Exception:  # pragma: no cover - best-effort tidiness
            manner = "error"
            logger.exception("%s stop failed", self.runtime)
        activity_mod.emit_event(
            activity_mod.KIND_ENGINE_BOOT,
            f"runtime={self.runtime} base_url={self.base_url} "
            f"rung={self.rung} manner={manner}",
            phase=PHASE_STOPPED,
        )


# -- the typed phase vocabulary -------------------------------------------
#
# CLOSED, and every phase of a boot lands on exactly one of them, so
# "where did this engine's boot go" is answerable by grouping on
# (kind, phase) with no join and no regex over a sentence.
PHASE_PLANNED = "engine_planned"
PHASE_STARTED = "engine_started"
PHASE_HEALTHY = "engine_healthy"
PHASE_DEGRADED = "engine_degraded"
PHASE_FAILED = "engine_boot_failed"
PHASE_STOPPED = "engine_stopped"

ENGINE_PHASES: tuple[str, ...] = (
    PHASE_PLANNED, PHASE_STARTED, PHASE_HEALTHY,
    PHASE_DEGRADED, PHASE_FAILED, PHASE_STOPPED,
)


class EngineSpec(msgspec.Struct, frozen=True, kw_only=True):
    """What an author DECLARES about the engine that serves this model.

    No checkpoint path and no port: both are the platform's to supply, and an
    author who could write them would be writing deploy config into code.
    Subclasses state their binary and their health route as class constants
    and build the boot ladder in :meth:`ladder`.
    """

    #: Extra engine flags, verbatim. The author's tuning surface.
    extra_args: Sequence[str] = ()
    #: Pin the port. Default: allocate a free one.
    port: Optional[int] = None
    #: Environment overlay for the child.
    env: Optional[Mapping[str, str]] = None
    #: Silence window for THIS engine's boot. Raise it only for an engine
    #: with a known-silent phase longer than the default; never as a fix for
    #: a boot that failed.
    stall_window_s: float = BOOT_STALL_WINDOW_S

    #: The engine's stable runtime handle — the word discovery puts in
    #: endpoint.lock and an operator reads in an event.
    runtime: ClassVar[str] = ""
    #: Path appended to the base URL for the readiness probe.
    health_path: ClassVar[str] = "/health"

    def ladder(self, checkpoint_dir: Path) -> list[EngineCommand]:
        """The boot ladder for this checkpoint, best rung FIRST.

        A single-element list is a boot with no degrade path, which is the
        honest answer for an engine that sizes itself.
        """
        raise NotImplementedError

    # -- helpers subclasses share ------------------------------------------

    def _args(self) -> list[str]:
        return [str(a) for a in self.extra_args]

    def _port(self) -> int:
        return int(self.port) if self.port else free_port()


class VllmServer(EngineSpec, frozen=True, kw_only=True):
    """``vllm serve <checkpoint>`` — OpenAI-compatible API + ``/health``.

    vLLM sizes itself from ``--gpu-memory-utilization`` and refuses rather
    than degrades, so there is ONE rung. That is stated here rather than
    faked with a ladder whose lower rungs cannot work: a degrade path that
    cannot fire reads as resilience the endpoint does not have.
    """

    runtime: ClassVar[str] = "vllm"

    def ladder(self, checkpoint_dir: Path) -> list[EngineCommand]:
        port = self._port()
        return [EngineCommand(
            argv=("vllm", "serve", str(checkpoint_dir),
                  "--host", "127.0.0.1", "--port", str(port), *self._args()),
            port=port,
        )]


#: Flags whose presence in ``extra_args`` means the AUTHOR sized the fit and
#: this module must not second-guess it.
_NGL_FLAGS = frozenset({"-ngl", "--n-gpu-layers", "--gpu-layers"})
_CTX_FLAGS = frozenset({"-c", "--ctx-size"})


class LlamaServer(EngineSpec, frozen=True, kw_only=True):
    """``llama-server -m <gguf>`` — llama.cpp's built-in ``/health``.

    The checkpoint tree resolves to its single logical GGUF model (split
    shards count as one). Unless the author pins ``-ngl``/``-c``, the GGUF
    header sizes ``-ngl`` and the context to the free-VRAM budget and the
    ladder degrades through fewer GPU layers — down to CPU-only — rather
    than failing the boot.
    """

    runtime: ClassVar[str] = "llama-server"

    #: Serving context. ``None`` = the header's training context.
    n_ctx: Optional[int] = None
    #: Override the measured free-VRAM budget (a test seam, and the escape
    #: hatch for a card shared with something this process cannot see).
    vram_budget_gb: Optional[float] = None

    def ladder(self, checkpoint_dir: Path) -> list[EngineCommand]:
        from .gguf_fit import plan_for, resolve_gguf

        gguf_path = str(resolve_gguf(checkpoint_dir))
        args = self._args()

        def rung(fit_args: Sequence[str], label: str) -> EngineCommand:
            port = self._port()
            return EngineCommand(
                argv=("llama-server", "-m", gguf_path,
                      "--host", "127.0.0.1", "--port", str(port),
                      *fit_args, *args),
                port=port,
                label=label,
            )

        if any(a in _NGL_FLAGS for a in args):
            return [rung((), "author-pinned")]
        plan = plan_for(
            gguf_path, vram_budget_gb=self.vram_budget_gb, n_ctx=self.n_ctx
        )
        if plan is None:
            # Header unreadable: llama.cpp's own defaults beat a guess.
            return [rung((), "engine-default")]
        ctx_args = (
            [] if any(a in _CTX_FLAGS for a in args)
            else ["-c", str(plan.n_ctx)]
        )
        rungs: list[tuple[int, str]] = [(plan.n_gpu_layers, "planned")]
        if plan.n_gpu_layers > 1:
            rungs.append((plan.n_gpu_layers // 2, "half-gpu"))
        if plan.n_gpu_layers > 0:
            rungs.append((0, "cpu-only"))
        return [
            rung(["-ngl", str(n), *ctx_args], label) for n, label in rungs
        ]


#: Spec CLASS NAME -> runtime handle. Discovery reads an author's
#: ``ctx.engine(LlamaServer(...))`` STATICALLY off the AST and needs the
#: handle without importing anything the author wrote, so the mapping is a
#: plain table rather than an attribute lookup on a resolved class.
ENGINE_SPEC_RUNTIMES: dict[str, str] = {
    "VllmServer": VllmServer.runtime,
    "LlamaServer": LlamaServer.runtime,
}


# ---------------------------------------------------------------------------
# The boot itself
# ---------------------------------------------------------------------------


def boot_engine(spec: EngineSpec, checkpoint_dir: Path) -> EngineHandle:
    """Walk ``spec``'s ladder; the first rung that becomes healthy wins.

    Raises :class:`EngineBootError` with the LAST rung's failure when every
    rung fails — the boot is over, and a caller that swallowed it would serve
    requests against a dead base URL.
    """
    ladder = spec.ladder(Path(checkpoint_dir))
    if not ladder:  # pragma: no cover - a spec with no rungs is a bug
        raise EngineBootError(
            f"{spec.runtime}: the engine spec produced no boot command"
        )
    activity_mod.emit_event(
        activity_mod.KIND_ENGINE_BOOT,
        f"runtime={spec.runtime} rungs={len(ladder)} "
        f"ladder={','.join(c.label for c in ladder)} "
        f"checkpoint={Path(checkpoint_dir).name}",
        phase=PHASE_PLANNED,
        step=0, total_steps=len(ladder),
    )

    last: Optional[EngineBootError] = None
    for index, command in enumerate(ladder):
        try:
            handle = _start(spec, command, rung_index=index, rungs=len(ladder))
        except EngineBootError as exc:
            last = exc
            logger.warning(
                "%s boot failed on rung %d/%d (%s): %s",
                spec.runtime, index + 1, len(ladder), command.label, exc,
            )
            activity_mod.emit_event(
                activity_mod.KIND_ENGINE_BOOT,
                f"runtime={spec.runtime} rung={command.label} error={exc}",
                phase=PHASE_FAILED,
                step=index + 1, total_steps=len(ladder),
            )
            continue
        if index > 0:
            # Serving on a degraded rung (fewer GPU layers / CPU-only) is a
            # QUALITY decision the planned rung's failure would otherwise
            # hide. Two channels on purpose: this countable operator row, and
            # the boot log the warning above wrote.
            activity_mod.emit_event(
                activity_mod.KIND_SERVE_DEGRADE,
                f"runtime={spec.runtime} rung={command.label} "
                f"({index} of {len(ladder) - 1} degrade steps): engine booted "
                f"on a degraded rung after: {last}",
                phase=PHASE_DEGRADED,
                step=index, total_steps=len(ladder) - 1,
            )
        return handle
    assert last is not None
    raise last


def _start(
    spec: EngineSpec, command: EngineCommand, *, rung_index: int, rungs: int
) -> EngineHandle:
    argv = list(command.argv)
    base_url = f"http://127.0.0.1:{command.port}"
    logger.info("booting %s: %s", spec.runtime, shlex.join(argv))
    env = dict(os.environ)
    if spec.env:
        env.update({str(k): str(v) for k, v in spec.env.items()})
    started = time.monotonic()
    activity_mod.emit_event(
        activity_mod.KIND_ENGINE_BOOT,
        f"runtime={spec.runtime} rung={command.label} port={command.port} "
        f"argv={shlex.join(argv)}",
        phase=PHASE_STARTED,
        step=rung_index + 1, total_steps=rungs,
    )
    # Capture the engine's merged output instead of inheriting stdio: it is
    # both the boot-PROGRESS signal and the log line an operator needs when a
    # boot goes wrong. The tail thread keeps draining for the whole life of
    # the process — a child whose pipe fills blocks mid-serving.
    try:
        proc = subprocess.Popen(
            argv,
            env=env,
            start_new_session=True,  # own group -> group-wide signals
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except OSError as exc:
        raise EngineBootError(
            f"{spec.runtime}: could not exec {shlex.join(argv)}: {exc}"
        ) from exc
    tail = LineTail(
        proc,
        window_s=float(spec.stall_window_s),
        on_line=lambda line: logger.info("[%s] %s", spec.runtime, line),
        name=f"{spec.runtime}-tail",
    ).start()
    handle = EngineHandle(
        runtime=spec.runtime,
        base_url=base_url,
        process=proc,
        rung=command.label,
    )
    health_url = base_url + spec.health_path
    try:
        _wait_healthy(spec, proc, tail, health_url, argv)
    except BaseException:
        handle.stop()
        raise
    duration_ms = int((time.monotonic() - started) * 1000)
    logger.info("%s healthy at %s after %dms", spec.runtime, base_url,
                duration_ms)
    activity_mod.emit_event(
        activity_mod.KIND_ENGINE_BOOT,
        f"runtime={spec.runtime} rung={command.label} base_url={base_url}",
        phase=PHASE_HEALTHY,
        duration_ms=duration_ms,
        step=rung_index + 1, total_steps=rungs,
    )
    return handle


def _wait_healthy(
    spec: EngineSpec,
    proc: "subprocess.Popen[str]",
    tail: LineTail,
    health_url: str,
    argv: Sequence[str],
) -> None:
    """Boot is OVER when the health URL answers 2xx. It has FAILED only when
    the process died, or stopped emitting output for its stall window."""
    delay = 0.25
    while True:
        code = proc.poll()
        if code is not None:
            raise EngineBootError(
                f"{spec.runtime} exited during boot (code {code}): "
                f"{shlex.join(list(argv))}"
            )
        try:
            with urllib.request.urlopen(health_url, timeout=5) as resp:
                if 200 <= resp.status < 300:
                    return
        except (urllib.error.URLError, OSError, TimeoutError):
            pass
        if tail.stalled():
            raise EngineBootError(
                f"{spec.runtime} produced no output for "
                f"{tail.silent_for():.0f}s (stall window {tail.window_s:.0f}s) "
                f"and never became healthy at {health_url}: "
                f"{shlex.join(list(argv))}"
            )
        time.sleep(delay)
        delay = min(delay * 1.5, 2.0)


def process_vram_bytes(pid: int) -> int:
    """Measured VRAM of one process via nvidia-smi. 0 when unavailable.

    An engine subprocess is invisible to torch's allocator, so residency
    accounting for this tier books THIS number instead of asking torch.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout
    except Exception:
        return 0
    total = 0
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0].isdigit() and int(parts[0]) == pid:
            try:
                total += int(parts[1]) * 1024 * 1024
            except ValueError:
                pass
    return total
