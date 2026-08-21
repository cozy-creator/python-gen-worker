"""Worker entrypoint — run as `python -m gen_worker.entrypoint`. THE MODULE NAME IS A WIRE CONTRACT: every hub-synthesized Dockerfile writes ENTRYPOINT ["python3","-m","gen_worker.entrypoint"] and the publish gate refuses an image whose entrypoint does not run it, so the name does not move without the hub half moving in the same window."""

import os
import faulthandler
import signal

from .settings_authority import impose_process_env  # noqa: E402

impose_process_env()

if __name__ == "__main__":
    from .settings_authority import ensure_interpreter_env  # noqa: E402

    ensure_interpreter_env()

    from .procsplit import is_compute_child  # noqa: E402

    if not is_compute_child():
        from .procsplit.parent import run_parent  # noqa: E402

        os._exit(run_parent())

    from .procsplit.oom_rank import raise_own_oom_score_adj  # noqa: E402

    raise_own_oom_score_adj()

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

MANIFEST_PATH = Path("/app/.tensorhub/endpoint.lock")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("WorkerEntrypoint")

# File-wide import rule (the sanctioned exception to top-of-file imports): this module runs as TWO process roles, and the control parent must stay a bare interpreter — no torch, no credentials — for OOM-victim ordering, so anything that could reach torch stays inside a function body below.


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
    """Every user module this image must import, from ``entrypoints[]``."""
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

    try:
        faulthandler.register(signal.SIGUSR2, all_threads=True, chain=True)
    except (AttributeError, ValueError, OSError) as exc:
        logger.debug("stack-dump handler unavailable: %s", exc)
    else:
        logger.info(
            "pgw#639: SIGUSR2 dumps all thread stacks to stderr (pid=%d)",
            os.getpid())
    from . import postmortem

    postmortem.enable_fault_dump()


def _establish_env_seal() -> Dict[str, Any]:
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

    if should_probe_cuda(manifest):
        probe = probe_cuda()
        if not probe.ok:
            logger.error("%s: %s", CUDA_PROBE_FAILED_MARKER, probe.reason)
            from .procsplit import is_compute_child

            if is_compute_child():
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
            _log_worker_fatal("cuda_probe", RuntimeError(probe.reason), exit_code=1)
            return 1
        _log_startup_phase("cuda_probe_ok", status="ok")

    if not settings.orchestrator_public_addr:
        logger.error("Settings.orchestrator_public_addr is empty (set ORCHESTRATOR_PUBLIC_ADDR env). Refusing to start worker.")
        return 1

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
