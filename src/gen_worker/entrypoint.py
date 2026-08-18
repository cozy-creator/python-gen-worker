"""
Worker entrypoint module.

This is the main entry point for running a Cozy worker. It loads the manifest,
discovers the image's ``@entrypoint`` declarations, and starts the worker loop.

Usage:
    python -m gen_worker.entrypoint

**THE MODULE NAME IS A WIRE CONTRACT (th#2168).** Every hub-synthesized
Dockerfile writes ``ENTRYPOINT ["python3", "-m", "gen_worker.entrypoint"]``
(`internal/builder/image/generate_dockerfile.go`) and the publish gate REFUSES
an image whose entrypoint does not run it (`matchesWorkerEntrypoint`,
`internal/builder/executor.go`). A `worker_main` rename therefore killed every
image built from a live endpoint pin at CONTAINER START, before any refusal
class could speak. The name is restored here and does not move again without
the hub half moving in the same window.
"""

import os
import faulthandler
import signal

# pgw#1049: the declared process env (PYTORCH_CUDA_ALLOC_CONF, the autograd
# cache disable, PYTHONHASHSEED for children), imposed BEFORE any module
# imports torch — several are read at torch import / first cudaMalloc. The
# entrypoint is the first module loaded in every worker process, so this is
# library-wide coverage; the values and their rationales live in ONE place,
# settings_authority.DECLARED_ENV. Imposed, not setdefault'd: an ambient
# value would be erased by env_seal.scrub_env anyway (never honored).
from .settings_authority import impose_process_env  # noqa: E402

impose_process_env()

# pgw#763: this process becomes the CONTROL PARENT — gRPC stream, identity,
# JWT, child supervision — and must run BEFORE the heavy imports below so it
# never loads torch. It spawns/respawns compute children (this same entrypoint
# with GEN_WORKER_COMPUTE_CHILD=1) and only ever exits deliberately. The split
# is unconditional; only a compute child falls through to the imports below.
if __name__ == "__main__":
    # pgw#1049: interpreter-level declared env (PYTHONHASHSEED) — CPython
    # read it at interpreter start, so a pod whose image env lacks it gets
    # ONE re-exec here, before the procsplit fork; every child inherits it.
    from .settings_authority import ensure_interpreter_env  # noqa: E402

    ensure_interpreter_env()

    from .procsplit import is_compute_child  # noqa: E402

    if not is_compute_child():
        from .procsplit.parent import run_parent  # noqa: E402

        os._exit(run_parent())

    # pgw#975: "the OOM killer picks the fat child, not the reporter" (below)
    # was true only by accident — the margin is the 479 MiB of torch, worth
    # under 4 oom_score_adj points out of 1000 on a real pod and NEGATIVE for
    # the seconds this child spends pre-torch. Declare it here, first thing and
    # before any import of ours, so a child that dies during its own boot is
    # already ranked. Descendants (mint child, AOT entry children) inherit.
    from .procsplit.oom_rank import raise_own_oom_score_adj  # noqa: E402

    raise_own_oom_score_adj()

# gw#640: fork the supervisor BEFORE the heavy imports below. The parent stays
# a bare interpreter (so the OOM killer picks the fat child, not the reporter)
# and outlives the worker to report WTERMSIG / cgroup oom_kill over the wire.
# In the child this returns immediately; the parent never returns from it.
# (In split mode the compute child skips this: the control parent IS the
# survivor, and it sets GEN_WORKER_SUPERVISOR=0 in the child's env.)
if __name__ == "__main__":
    from .supervisor import supervise  # noqa: E402

    supervise()

import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import msgspec

from . import config
from . import worker_credential
from .cuda_probe import CUDA_PROBE_FAILED_MARKER, probe_cuda, should_probe_cuda
from .hardware_report import report_hardware_unsuitable
from .manifest_blocks import (
    DECLARATION_BLOCK,
    declaration_rows,
    declared_row_count,
)
from .models.cache_paths import tensorhub_cas_dir
try:
    from .worker import Worker
except ImportError as e:
    print(f"Error importing Worker: {e}", file=sys.stderr)
    print("Please ensure the gen_worker package is installed.", file=sys.stderr)
    sys.exit(1)

# Default baked container location; overridden by Settings.endpoint_lock_path
# (env ENDPOINT_LOCK_PATH) for non-container runs.
MANIFEST_PATH = Path("/app/.tensorhub/endpoint.lock")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("WorkerEntrypoint")

# FILE-WIDE IMPORT RULE (the one sanctioned exception to top-of-file imports):
# this module runs as TWO process roles. The control parent must stay a bare
# interpreter — no torch, no credentials — for OOM-victim ordering (gw#640,
# pgw#763), and the env `setdefault`s at the top are read once at torch import,
# so anything that could reach torch stays inside a function body below.


def _startup_payload(phase: str, status: str = "ok", **extra: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "phase": str(phase or "").strip(),
        "status": str(status or "ok"),
        "pid": int(os.getpid()),
        "uid": int(os.getuid()) if hasattr(os, "getuid") else None,
        "gid": int(os.getgid()) if hasattr(os, "getgid") else None,
        "cwd": str(os.getcwd()),
    }
    payload.update({k: v for k, v in extra.items() if v is not None})
    return payload


def _log_startup_phase(phase: str, *, status: str = "ok", level: int = logging.INFO, **extra: Any) -> None:
    payload = _startup_payload(phase, status=status, **extra)
    try:
        logger.log(level, "worker.startup.phase %s", json.dumps(payload, separators=(",", ":"), sort_keys=True))
    except Exception:
        logger.log(level, "worker.startup.phase phase=%s status=%s", phase, status)


def _log_worker_fatal(
    phase: str,
    exc: BaseException,
    *,
    exit_code: int,
    settings: Optional[Any] = None,
) -> None:
    """Record this process's cause of death to stdout AND to the hub.

    gw#640/th#1077: stdout alone is unreachable on RunPod (no container-logs
    API), so every cloud-only crash was un-debuggable. The wire report reuses
    the HardwareUnsuitable carrier and lands as a durable pod_events row.
    """
    try:
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    except Exception:
        tb = traceback.format_exc()
    payload = _startup_payload(
        "worker_fatal",
        status="error",
        phase_context=str(phase or ""),
        exception_class=type(exc).__name__,
        exception_message=str(exc),
        traceback=tb,
        exit_code=int(exit_code),
    )
    try:
        logger.error("worker.fatal %s", json.dumps(payload, separators=(",", ":"), sort_keys=True))
    except Exception:
        logger.exception("worker.fatal: %s", exc)
    try:
        from .worker_fatal import report_worker_fatal

        delivered = report_worker_fatal(settings, phase, exc, exit_code=exit_code)
        _log_startup_phase(
            "worker_fatal_report",
            status="ok" if delivered else "error",
            level=logging.INFO if delivered else logging.WARNING,
            delivered=delivered,
            phase_context=str(phase or ""),
        )
    except Exception:
        logger.warning("worker-fatal wire report raised unexpectedly", exc_info=True)


def load_manifest(path: Path = MANIFEST_PATH) -> Optional[Dict[str, Any]]:
    """Load the function manifest if it exists (baked in at build time)."""
    if not path.exists():
        return None
    try:
        raw = path.read_text(encoding="utf-8")
        manifest = msgspec.toml.decode(raw)
        if not isinstance(manifest, dict):
            raise ValueError("endpoint.lock must decode to a TOML table")
        return manifest
    except Exception as e:
        logger.warning("Failed to load manifest from %s: %s", path, e)
        return None


def get_modules_from_manifest(manifest: Dict[str, Any]) -> List[str]:
    """Every user module this image must import, from ``entrypoints[]``.

    One block since pgw#1373; `manifest_blocks` owns the row derivation and
    both this and the CUDA probe feed off it, so neither can go blind on a
    shape the other was written for (pgw#1354, pgw#1395).
    """
    modules = set()
    for row in declaration_rows(manifest):
        module = row.get("module")
        if module:
            modules.add(module)
    return sorted(modules)


def _probe_cache_path_writable(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    probe = path / f".cozy-write-probe-{os.getpid()}"
    with open(probe, "wb") as f:
        f.write(b"ok")
        f.flush()
        os.fsync(f.fileno())
    probe.unlink(missing_ok=True)


def _check_cache_path(label: str, path_str: str) -> tuple[bool, Dict[str, Any]]:
    p = Path(path_str)
    details: Dict[str, Any] = {"label": label, "path": str(p)}
    try:
        _probe_cache_path_writable(p)
        _log_startup_phase("cache_preflight_ok", status="ok", path=str(p), label=label)
        return True, details
    except Exception as e:
        details["exception_class"] = type(e).__name__
        details["exception_message"] = str(e)
        _log_startup_phase(
            "cache_preflight_failed",
            status="error",
            level=logging.ERROR,
            path=str(p),
            label=label,
            exception_class=type(e).__name__,
            exception_message=str(e),
        )
        return False, details


def _preflight_cache_dirs() -> Dict[str, str]:
    """Validate model cache directory writeability before worker startup."""
    primary = str(tensorhub_cas_dir())

    _log_startup_phase(
        "cache_preflight_started",
        status="starting",
        primary_cache_dir=primary,
    )

    ok, details = _check_cache_path("tensorfs CAS", primary)
    if not ok:
        raise RuntimeError(
            "worker cache preflight failed for tensorhub CAS path "
            f"{primary} ({details.get('exception_class')}: {details.get('exception_message')}). "
            "Fix volume permissions/ownership."
        )

    return {
        "model_cache_dir": primary,
        "local_model_cache_dir": "",
    }


def _install_stack_dump_handler() -> None:
    """pgw#639: SIGUSR2 dumps every thread's stack to stderr.

    A wedged worker heartbeats fine (the asyncio loop owns the beat; model
    work runs on threads), so "connected" proves nothing about progress and
    hub-side logs cannot see which thread is stuck. This is the pod-side
    forensic surface: `kill -USR2 <pid>` from any exec channel prints the
    full picture into the pod log. Registration is free and always on —
    the signal is never sent unless a human asks for it. SIGUSR2 is unused
    by CPython and by torch; faulthandler writes without allocating, so it
    works even when the process is wedged on memory.
    """

    try:
        faulthandler.register(signal.SIGUSR2, all_threads=True, chain=True)
    except (AttributeError, ValueError, OSError) as exc:  # non-POSIX / no tty
        logger.debug("stack-dump handler unavailable: %s", exc)
    else:
        logger.info(
            "pgw#639: SIGUSR2 dumps all thread stacks to stderr (pid=%d)",
            os.getpid())
    # pgw#676: fatal signals (SIGSEGV/SIGABRT/SIGBUS/SIGFPE) dump every
    # thread's stack to a file the surviving supervisor attaches to its
    # post-mortem — exit_code=139 carries frames instead of nothing.
    from . import postmortem

    postmortem.enable_fault_dump()


def _establish_env_seal() -> Dict[str, Any]:
    """pgw#694/#696 boot wiring (the ONE executor-side hook): refuse unknown
    ``TORCH*`` env vars, pin the canonical config surface, and record the
    effective seal — BEFORE the CUDA probe or any model/compile work, so
    every graph this process ever mints or serves runs under the sealed
    posture and the ``env_seal`` axis describes reality. A process that
    cannot be sealed must not advertise: the caller exits typed."""
    from . import env_seal

    seal = env_seal.establish()
    _log_startup_phase(
        "env_seal",
        status="ok",
        digest=env_seal.seal_digest(seal),
        config=seal.get("config"),
    )
    _impose_group_host_policy()
    _isolate_group_inductor_cache()
    return seal


def _isolate_group_inductor_cache() -> None:
    """pgw#783: give each compute child its OWN inductor + triton cache dir when
    G children share this pod, so concurrent minting/compilation does not race a
    process-global cache dir.

    Set AFTER the seal — env_seal scrubs the whole ``TORCH*``/``TRITON*``
    namespace at boot, and the sanctioned window to point the SDK's own capture
    redirects is after that scrub (same as compiled graph ``capture_env``). A per-group
    PATH is plumbing, not a behaviour flag, so it does not touch the seal
    digest or minted kernels (inductor keys are content-addressed).

    Gated on ``host_siblings() > 1``: a single child (or no split) keeps torch's
    default dir untouched — byte-identical to today. A respawned child of the
    same group reuses its group dir, so cache hits survive a respawn.
    """
    import tempfile

    from .procsplit import group_ordinal, host_siblings

    if host_siblings() <= 1:
        return
    ordinal = group_ordinal()
    base = (config.current().tensorhub_cache_dir.strip() or tempfile.gettempdir())
    root = os.path.join(base, "gen-worker-inductor", f"g{ordinal}")
    try:
        for sub, var in (("inductor", "TORCHINDUCTOR_CACHE_DIR"),
                         ("triton", "TRITON_CACHE_DIR")):
            d = os.path.join(root, sub)
            os.makedirs(d, exist_ok=True)
            os.environ[var] = d
    except OSError:
        logger.warning("could not isolate the group inductor cache under %s; "
                       "the child keeps the shared default", root, exc_info=True)
        return
    _log_startup_phase(
        "group_inductor_isolation", status="ok", ordinal=ordinal, root=root,
    )


def _impose_group_host_policy() -> None:
    """pgw#782: the host-side posture that depends on HOW MANY execution groups
    share this process — today, the intra-op thread budget.

    Ordered here, right after the seal and before the CUDA probe, for the same
    reason the seal is: the decision belongs to CODE, and every group this
    process ever runs must have been started under it. The executor re-asserts
    it from its authoritative slot count for the cli/serve and harness paths;
    the imposition is idempotent and de-escalation-only.

    Reads the DELIVERED env, deliberately not ``delivered_topology()``, whose
    fabric gate consults the host canary and therefore measures the device —
    this hook runs before anything touches CUDA. The gate only ever demotes
    ``D`` (raising ``G``), so the rare demoted-sharding pod picks its true group
    count up from the executor's re-assertion instead.
    """
    from . import cpu_budget
    from .topology import ExecutionTopology

    try:
        execution_groups = ExecutionTopology.from_env().execution_groups
    except Exception:  # noqa: BLE001  (an illegal topology refuses later, typed)
        return
    threads = cpu_budget.impose_intra_op_threads(execution_groups)
    _log_startup_phase(
        "group_host_policy", status="ok", execution_groups=execution_groups,
        intra_op_threads=threads.get("imposed"),
    )


def _bootstrap_configuration() -> config.Settings:
    """THE bootstrap-owned load for this process entry (§1.18), and the one
    place derived process facts are published from it.

    One function rather than four lines in `_run_main` because these must not
    drift apart: the goal set and the boot credential are DERIVED from these
    exact `Settings`, so a second entry that loaded config without publishing
    them would leave the process holding two answers (§4.22).
    """
    settings = config.install(config.load_settings())
    for name in config.unrecognised_owned_env():
        logger.warning(
            "unknown_owned_env %s: set in a gen-worker-owned namespace but no "
            "reader exists in this build — it is INERT, not applied", name)
    worker_credential.install_bootstrap(settings)
    return settings


def _run_main() -> int:
    _log_startup_phase("boot", status="starting")
    _install_stack_dump_handler()
    try:
        settings = _bootstrap_configuration()
    except Exception as e:
        logger.exception("Failed to load worker settings: %s", e)
        _log_worker_fatal("settings_load", e, exit_code=1)
        return 1
    # pgw#696: seal the execution environment before the CUDA probe touches
    # the device and before any model/compile work. Ordered AFTER settings
    # (which reads no torch config) so a refusal can DIAL THE HUB typed —
    # the 0.70.3 pre-settings ordering made a seal refusal a silent
    # pod_exited the fleet could not attribute.
    try:
        _establish_env_seal()
    except Exception as e:
        _log_worker_fatal("env_seal", e, exit_code=1, settings=settings)
        logger.error(str(e))
        return 1
    manifest_path = Path(settings.endpoint_lock_path or MANIFEST_PATH)
    manifest = load_manifest(manifest_path)
    user_modules: List[str] = []
    if manifest:
        # The decode-set the hub was told about is the one stamped in this
        # lock at IMAGE BUILD. If this process would derive a different one,
        # the code that runs is not the code the hub selected against, and
        # every downstream answer is about the wrong image (pgw#1245).
        try:
            from .discovery.decode_set import assert_matches_baked

            assert_matches_baked(manifest.get("decode_set") or {})
        except Exception as e:
            _log_worker_fatal("decode_set_drift", e, exit_code=1,
                              settings=settings)
            logger.error(str(e))
            return 1
        user_modules = get_modules_from_manifest(manifest)
        _log_startup_phase(
            "manifest_loaded",
            status="ok",
            manifest_path=str(manifest_path),
            entrypoint_count=declared_row_count(manifest),
            module_count=len(user_modules),
        )
    else:
        _log_startup_phase(
            "manifest_loaded",
            status="error",
            level=logging.ERROR,
            manifest_path=str(manifest_path),
            reason="missing_or_invalid_manifest",
        )

    try:
        cache_cfg = _preflight_cache_dirs()
    except Exception as e:
        _log_worker_fatal("cache_preflight", e, exit_code=1, settings=settings)
        logger.error(str(e))
        return 1

    # Boot-time CUDA probe (gw#529): on a GPU-needing manifest, verify the
    # device actually works BEFORE we hello the orchestrator and accept a
    # job — a busy/unavailable GPU (RunPod bad-host fault) must kill this
    # pod now, not terminal-fail a real request at model load.
    if should_probe_cuda(manifest):
        probe = probe_cuda()
        if not probe.ok:
            logger.error("%s: %s", CUDA_PROBE_FAILED_MARKER, probe.reason)
            from .procsplit import is_compute_child

            if is_compute_child():
                # pgw#826: a hardware verdict is terminal for every child this
                # pod could spawn. This process holds no credential — hand the
                # typed report to the parent, which relays it and exits 1.
                from .hardware_report import build_hardware_report
                from .procsplit.child import send_boot_fatal

                report = build_hardware_report(probe, settings)
                relayed = send_boot_fatal(msgspec.to_builtins(report))
                _log_startup_phase(
                    "cuda_probe_boot_fatal",
                    status="ok" if relayed else "error",
                    level=logging.INFO if relayed else logging.WARNING,
                    relayed=relayed,
                    reason_class=report.reason_class,
                )
                _log_worker_fatal("cuda_probe", RuntimeError(probe.reason), exit_code=1)
                return 1
            # gw#619/th#988: dial the hub with a typed hardware-unsuitable
            # report BEFORE exiting — closes the th#986 blindness where this
            # exit was previously silent pre-hello. Best-effort/bounded: the
            # exit below happens regardless of whether the hub is reachable.
            try:
                delivered = report_hardware_unsuitable(settings, probe)
                _log_startup_phase(
                    "cuda_probe_hardware_report",
                    status="ok" if delivered else "error",
                    level=logging.INFO if delivered else logging.WARNING,
                    delivered=delivered,
                )
            except Exception:
                logger.warning("hardware-unsuitable report raised unexpectedly", exc_info=True)
            # settings=None: this path ALREADY dialed the hub with the typed
            # HardwareUnsuitable report just above — a second wire dial would
            # only duplicate it (and double the pre-exit budget).
            _log_worker_fatal("cuda_probe", RuntimeError(probe.reason), exit_code=1)
            return 1
        _log_startup_phase("cuda_probe_ok", status="ok")

    if not settings.orchestrator_public_addr:
        logger.error("Settings.orchestrator_public_addr is empty (set ORCHESTRATOR_PUBLIC_ADDR env). Refusing to start worker.")
        return 1

    # C2PA content-credential signing (th#714): ON iff a cert is configured;
    # logs a loud warning when off, refuses to start when configured-but-broken
    # (a worker that believes it signs but doesn't is a compliance hole).
    try:
        from .content_credentials import configure as _c2pa_configure

        _c2pa_configure(settings)
    except Exception as e:
        _log_worker_fatal("c2pa_configure", e, exit_code=1, settings=settings)
        logger.error(str(e))
        return 1

    logger.info("Starting worker...")
    logger.info("  Orchestrator Public Address: %s", settings.orchestrator_public_addr)
    logger.info("  User Function Modules: %s", user_modules)
    logger.info("  Worker ID: %s", settings.worker_id or "(from JWT)")
    logger.info("  Model Cache Dir: %s", cache_cfg["model_cache_dir"])
    if cache_cfg["local_model_cache_dir"]:
        logger.info("  Local Model Cache Dir: %s", cache_cfg["local_model_cache_dir"])

    if not user_modules:
        # pgw#1354: dial TYPED, like every other fatal in this function. RunPod
        # exposes no container-logs API, so a bare `return 1` reaches the hub as
        # `exit:1` with no reason class and is condemned `[hardware-unsuitable]` —
        # a boot bug wearing a hardware verdict. The message DISCRIMINATES the two
        # gaps, because they have
        # different owners: no manifest at all is a Dockerfile that never ran
        # discovery, while declarations-without-modules is a manifest this wheel
        # cannot read (a block it does not walk, or rows with no `module`).
        declared = declared_row_count(manifest)
        if manifest and declared:
            reason = (
                f"the manifest at {manifest_path} declares {declared} "
                f"entrypoint(s) but no row carries a `module`, so this image "
                f"has nothing to import. The block this build walks is "
                f"{DECLARATION_BLOCK!r}."
            )
        elif manifest:
            reason = (
                f"the manifest at {manifest_path} declares no "
                f"{DECLARATION_BLOCK!r} — this image publishes nothing and "
                f"cannot serve."
            )
        else:
            reason = (
                f"no baked manifest at {manifest_path}. Your Dockerfile should "
                f"run discovery at build time:\n"
                f"  RUN mkdir -p /app/.tensorhub && python -m gen_worker.discovery"
                f" > /app/.tensorhub/endpoint.lock\n"
                f"(non-container runs: set ENDPOINT_LOCK_PATH to the generated file)"
            )
        message = f"no user modules to import: {reason}"
        logger.error("%s", message)
        _log_worker_fatal(
            "no_user_modules", RuntimeError(message), exit_code=1,
            settings=settings,
        )
        return 1

    try:
        worker = Worker(
            settings=settings,
            user_module_names=user_modules,
            manifest=manifest,
        )
        code = worker.run()
        logger.info("Worker process finished gracefully (exit=%d).", code)
        return code
    except ImportError as e:
        logger.exception(
            "Failed to import user module(s) or dependencies: %s. "
            "Make sure modules '%s' and their requirements are installed.",
            e,
            user_modules,
        )
        _log_worker_fatal("import", e, exit_code=1, settings=settings)
        return 1
    except Exception as e:
        logger.exception("Worker failed unexpectedly: %s", e)
        _log_worker_fatal("runtime", e, exit_code=1, settings=settings)
        return 1


if __name__ == "__main__":
    sys.exit(_run_main())
