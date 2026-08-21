"""THE single settings authority."""

from __future__ import annotations

import logging
import os
import sys
import threading
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from . import torch_capability

logger = logging.getLogger(__name__)


class SettingsImpositionError(RuntimeError):
    """The declared settings could not be imposed, or an undeclared knob was asked for."""


AUTHORITY_MODULES: Tuple[str, ...] = (
    "settings_authority.py",
    "env_seal.py",
    "host_isa.py",
    "guard_closure.py",
)

DECLARED_ENV: Dict[str, str] = {
    "PYTHONHASHSEED": "0",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "TORCHINDUCTOR_AUTOGRAD_CACHE": "0",
    "NCCL_NVLS_ENABLE": "0",
}

DECLARED_TORCH: Dict[str, str] = {
    "float32_matmul_precision": "high",
    "cuda_matmul_allow_tf32": "True",
    "cudnn_allow_tf32": "True",
    "cudnn_benchmark": "False",
}

DECLARED_DYNAMO: Dict[str, str] = {
    "automatic_dynamic_shapes": "False",
    "assume_static_by_default": "True",
}


def declaration(
    overrides: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """The full declared settings — the ONLY input the env seal digests."""
    from . import guard_closure, host_isa

    if torch_capability.present():
        config = validated_table(overrides)
        dynamo = dict(DECLARED_DYNAMO)
        posture = dict(guard_closure.CANONICAL_POSTURE)
        march = host_isa.mint_march()
        inductor = (
            {"cpp.march": march, "cpp.simdlen": str(host_isa.mint_simdlen(march))}
            if march is not None else {}
        )
    else:
        validated_table(overrides)
        if overrides:
            raise SettingsImpositionError(
                f"config knob(s) {sorted(overrides)!r} declared on a "
                "TORCHLESS worker: every canonical knob is a torch flag, so "
                "there is nothing to impose them on. Either ship torch in "
                "this image or drop the knob (pgw#788)")
        absent = {"torch": torch_capability.ABSENT}
        config, dynamo, posture, inductor = absent, dict(absent), dict(absent), {}
    return {
        "env": dict(DECLARED_ENV),
        "config": config,
        "dynamo": dynamo,
        "inductor": inductor,
        "posture": posture,
    }


def validated_table(
    overrides: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    """:data:`DECLARED_TORCH` with DECLARED knob overrides folded in — the typed-knob surface: keys must exist in the canonical table, so the only route to non-canonical behavior is a declared knob, which..."""
    table = dict(DECLARED_TORCH)
    if overrides:
        unknown = sorted(set(overrides) - set(table))
        if unknown:
            raise SettingsImpositionError(
                f"unknown config knob(s) {unknown!r}: not in the canonical "
                "table (settings_authority.DECLARED_TORCH) — declare the "
                "knob there first (one-way door: knobs in, env vars never)")
        table.update({k: str(v) for k, v in overrides.items()})
    return table


def impose_process_env() -> None:
    """Write :data:`DECLARED_ENV` into ``os.environ`` — unconditionally, so an ambient value never survives."""
    os.environ.update(DECLARED_ENV)


def process_env_diffs() -> List[str]:
    """Where the LIVE ``os.environ`` disagrees with :data:`DECLARED_ENV` — read from the environment itself, never from a "did we call impose" flag, because the environment is what the allocator and every child process actually read."""
    return [
        f"{name}: declared {want!r} but this process has "
        f"{os.environ.get(name)!r}"
        for name, want in sorted(DECLARED_ENV.items())
        if os.environ.get(name) != want
    ]


def verify_process_env() -> None:
    """Fail-closed check that :data:`DECLARED_ENV` is in effect in THIS process."""
    diffs = process_env_diffs()
    if diffs:
        raise SettingsImpositionError(
            "declared process env is not in effect: " + "; ".join(diffs)
            + " — impose it with settings_authority.impose_process_env() "
            "BEFORE torch is imported (PYTORCH_CUDA_ALLOC_CONF is read once, "
            "at CUDA allocator init, so a late imposition is a no-op)")


def process_env_readback() -> Dict[str, str]:
    """The live value of every declared env name (confession surface)."""
    return {name: os.environ.get(name, "") for name in sorted(DECLARED_ENV)}


def _interpreter_env_diffs() -> List[str]:
    diffs: List[str] = []
    want = DECLARED_ENV.get("PYTHONHASHSEED")
    if want == "0" and sys.flags.hash_randomization != 0:
        diffs.append(
            "PYTHONHASHSEED: declared '0' but this interpreter booted with "
            "hash randomization ON")
    return diffs


def ensure_interpreter_env() -> None:
    """Impose the interpreter-level declared env on the CURRENT process, re-exec'ing once if the interpreter booted without it."""
    if not _interpreter_env_diffs():
        impose_process_env()
        return
    if sys.flags.ignore_environment:
        raise SettingsImpositionError(
            "cannot impose the declared interpreter env: python was started "
            "with -E/-I (ignore_environment), so a re-exec cannot deliver "
            "PYTHONHASHSEED. Drop the flag or set the declared env at launch.")
    impose_process_env()
    logger.info("settings authority: re-exec to impose interpreter env "
                "(PYTHONHASHSEED=%s)", DECLARED_ENV.get("PYTHONHASHSEED"))
    os.execv(sys.executable, list(sys.orig_argv))


def verify_interpreter_env() -> None:
    """Fail-closed check that the declared interpreter env is IN EFFECT — ``env_seal.establish`` calls this, so a process that skipped :func:`ensure_interpreter_env` cannot seal an identity claiming the d..."""
    diffs = _interpreter_env_diffs()
    if diffs:
        raise SettingsImpositionError(
            "declared interpreter env is not in effect: "
            + "; ".join(diffs)
            + " — the entrypoint imposes it by re-exec; embedders/tests call "
            "settings_authority.ensure_interpreter_env() before torch")


def torch_readback() -> Dict[str, str]:
    """Live values of the :data:`DECLARED_TORCH` flags (tripwire surface)."""
    torch = torch_capability.torch_or_none()
    if torch is None:
        return {"torch": torch_capability.ABSENT}
    return {
        "float32_matmul_precision": str(torch.get_float32_matmul_precision()),
        "cuda_matmul_allow_tf32": str(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": str(torch.backends.cudnn.allow_tf32),
        "cudnn_benchmark": str(torch.backends.cudnn.benchmark),
    }


def impose_torch(
    overrides: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    """Impose the declared torch table (+ declared knobs), then verify the read-back."""
    table = validated_table(overrides)
    torch = torch_capability.torch_or_none()
    if torch is None:
        if overrides:
            raise SettingsImpositionError(
                f"config knob(s) {sorted(overrides)!r} declared on a "
                "TORCHLESS worker: every canonical knob is a torch flag, so "
                "there is nothing to impose them on (pgw#788)")
        return torch_readback()
    torch.set_float32_matmul_precision(table["float32_matmul_precision"])
    torch.backends.cuda.matmul.allow_tf32 = (
        table["cuda_matmul_allow_tf32"] == "True")
    torch.backends.cudnn.allow_tf32 = (table["cudnn_allow_tf32"] == "True")
    torch.backends.cudnn.benchmark = (table["cudnn_benchmark"] == "True")
    effective = torch_readback()
    diffs = [
        f"{name}: imposed {want!r} != effective {effective.get(name)!r}"
        for name, want in table.items()
        if effective.get(name) != want
    ]
    if diffs:
        raise SettingsImpositionError(
            "torch settings freeze failed: " + "; ".join(diffs))
    return effective


def impose_config_default(
    config_module: object, key: str, value: object,
) -> None:
    """Write the PROCESS-WIDE fallback for a torch ConfigModule key."""
    entry = getattr(config_module, "_config", {}).get(key)
    if entry is None:
        raise SettingsImpositionError(
            f"declared setting cannot reach a process-wide target: torch's "
            f"config has no {key!r} entry to set a default on. torch's "
            f"config internals changed; re-seat the authority before any "
            f"compile can be trusted.")
    entry.default = value


def read_in_fresh_thread(fn: Callable[[], Any]) -> Any:
    """Run ``fn`` on a brand-new thread and return its value."""
    box: Dict[str, Any] = {}

    def _run() -> None:
        box["v"] = fn()

    t = threading.Thread(target=_run, name="settings-readback", daemon=True)
    t.start()
    t.join()
    return box.get("v")


def dynamo_readback() -> Dict[str, str]:
    """Live values of the declared dynamo facts, read on THIS thread — the mint/serve thread is exactly where a stray thread-local override would sit, and a fresh thread would read the (imposed) default a..."""
    if not torch_capability.present():
        return {"torch": torch_capability.ABSENT}
    import torch._dynamo

    return {
        name: str(getattr(torch._dynamo.config, name))
        for name in DECLARED_DYNAMO
    }


def impose_dynamo() -> Dict[str, str]:
    """Impose the declared dynamo shape posture process-wide (default layer + this thread's own override), verify on a FOREIGN thread — the same mechanism and reasoning as ``host_isa.impose``."""
    if not torch_capability.present():
        return {}
    import torch._dynamo

    want = {name: value == "True" for name, value in DECLARED_DYNAMO.items()}
    for name, value in want.items():
        impose_config_default(torch._dynamo.config, name, value)
        setattr(torch._dynamo.config, name, value)
    foreign = read_in_fresh_thread(
        lambda: {n: bool(getattr(torch._dynamo.config, n)) for n in want})
    if foreign != want:
        raise SettingsImpositionError(
            f"dynamo shape posture is thread-local only: imposed {want!r}, "
            f"a fresh thread reads {foreign!r}")
    return dict(DECLARED_DYNAMO)


def disable_autograd_cache() -> None:
    """The AOTAutogradCache key hashes ``fx_kwargs[get_decomp_fn]`` via the function's REPR — a process memory address (ASLR), so AOT keys can NEVER match across processes/pods."""
    os.environ["TORCHINDUCTOR_AUTOGRAD_CACHE"] = (
        DECLARED_ENV["TORCHINDUCTOR_AUTOGRAD_CACHE"])
    if "torch" not in sys.modules:
        return
    try:
        import torch._functorch.config as fconf

        fconf.enable_autograd_cache = False
        fconf._config["enable_autograd_cache"].env_value_force = False  # type: ignore[attr-defined]
    except Exception:
        logger.debug("settings authority: AOT autograd cache disable "
                     "unavailable", exc_info=True)


def set_compiler_cache_tag(tag: str) -> None:
    """Install ``compile_cache``'s semantic tag for an arm's compiles."""
    try:
        import torch.compiler.config as compiler_config

        compiler_config.cache_key_tag = tag
    except Exception:
        logger.debug("settings authority: semantic cache tag unavailable",
                     exc_info=True)


def raise_dynamo_cache_limits(want: int) -> None:
    """Size dynamo's per-code-object recompile ceiling to the declared shape set (a family can declare more graphs than torch's default of 8) — never lower an operator-raised value."""
    if not torch_capability.present():
        return
    try:
        import torch._dynamo

        torch._dynamo.config.cache_size_limit = max(
            int(torch._dynamo.config.cache_size_limit), want)
        torch._dynamo.config.recompile_limit = max(
            int(torch._dynamo.config.recompile_limit), want)
    except Exception:
        logger.warning("settings authority: could not raise the recompile "
                       "limit to %d — declared shapes past torch's default "
                       "ceiling will stop compiling", want, exc_info=True)


__all__ = [
    "AUTHORITY_MODULES",
    "DECLARED_DYNAMO",
    "DECLARED_ENV",
    "DECLARED_TORCH",
    "SettingsImpositionError",
    "declaration",
    "disable_autograd_cache",
    "dynamo_readback",
    "ensure_interpreter_env",
    "impose_config_default",
    "impose_dynamo",
    "impose_process_env",
    "impose_torch",
    "process_env_diffs",
    "process_env_readback",
    "raise_dynamo_cache_limits",
    "read_in_fresh_thread",
    "set_compiler_cache_tag",
    "torch_readback",
    "validated_table",
    "verify_interpreter_env",
    "verify_process_env",
]
