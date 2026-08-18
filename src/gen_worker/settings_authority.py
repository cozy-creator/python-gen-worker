"""THE single settings authority.

``us -> pytorch settings``, never ``[us, ambient world] -> pytorch settings``:
every torch/dynamo/inductor process setting this worker runs under is DECLARED
in this module's tables, imposed from here, and verified against the
declaration. Compiled graph identity (``env_seal``) derives from :func:`declaration` —
the seal digests what WE declared, never a read-back of whatever the process
happens to hold — so ambient mutation is structurally unable to move identity.
It can only trip the drift tripwire (kept in ``env_seal`` as the runtime
detector), which refuses typed before any trace runs under it.

A read-back is a mirror; a declaration is an authority. Two failure modes make
that concrete: torch's own ``aot_compile`` mutates global
``aot_inductor.metadata`` mid-mint, so a read-back seal records the
contamination rather than the boot; and ``cpp.march=None`` hashes identically
on every host while the emitted code differs per host.

Write fence: ``scripts/lint_settings_writers.py`` holds the modules in
:data:`AUTHORITY_MODULES` to be the ONLY writers of torch settings. A write
anywhere else in ``src/gen_worker`` is red unless classified in
``scripts/settings_writers_allowlist.txt``.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from . import torch_capability

logger = logging.getLogger(__name__)


class SettingsImpositionError(RuntimeError):
    """The declared settings could not be imposed, or an undeclared knob was
    asked for. Fail-closed: a process that cannot carry the declaration must
    not mint or serve under its identity."""


#: Modules allowed to WRITE torch settings (the fence's allowed set). Paths
#: relative to ``src/gen_worker``. host_isa and guard_closure hold the
#: ISA-clamp and posture halves of the declaration; env_seal orchestrates
#: boot; everything else calls functions here.
AUTHORITY_MODULES: Tuple[str, ...] = (
    "settings_authority.py",
    "env_seal.py",
    "host_isa.py",
    "guard_closure.py",
)

# ---------------------------------------------------------------------------
# The declaration
# ---------------------------------------------------------------------------

#: Process env WE impose, re-imposed after ``env_seal.scrub_env`` erases the
#: behavior namespaces — an ambient value is deleted, never honored, and the
#: value below is what every child (mint child, AOT entry child, torch's own
#: compile subprocesses) inherits.
#:
#: PYTHONHASHSEED=0: CPython reads it at interpreter start, so imposition for
#: the CURRENT interpreter is :func:`ensure_interpreter_env`'s re-exec;
#: children inherit it from this table.
#:
#: PYTORCH_CUDA_ALLOC_CONF: must be imposed POST-SCRUB. A ``setdefault`` at
#: entrypoint import is dead — ``scrub_env`` erases the ``PYTORCH`` namespace
#: before the first cudaMalloc reads it, silently disabling
#: ``expandable_segments``.
#:
#: TORCHINDUCTOR_AUTOGRAD_CACHE=0: the AOTAutogradCache key embeds a process
#: address (ASLR) and can never hit across pods; compiled graph portability needs the
#: portable FxGraphCache to be the lookup surface.
#:
#: NCCL_NVLS_ENABLE=0: NVLS multicast cannot be bound in our containers
#: (measured: CUDA 401 on the first all-to-all of every group) and Ulysses does
#: not use it. The env survives only as NCCL's own handoff mechanism; it is
#: nobody's choice (``parallel/group.py`` warns at communicator creation if an
#: ambient override was dropped).
DECLARED_ENV: Dict[str, str] = {
    "PYTHONHASHSEED": "0",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "TORCHINDUCTOR_AUTOGRAD_CACHE": "0",
    "NCCL_NVLS_ENABLE": "0",
}

#: Behavior-affecting torch global flags: ONE canonical value each, imposed by
#: :func:`impose_torch` and verified by read-back. The canonical values ARE the
#: ratified SERVING posture, not a preference: the executor serves with TF32 ON
#: (bf16 compute path; TF32 touches residual fp32 matmuls only), and inductor
#: hashes the TF32 state (``cuda_matmul_settings``) into every inner FX key — so
#: a mint sealed with TF32 off could never HIT in a serving process. Note the
#: 2.13 coupling: allow_tf32=True implies float32_matmul_precision "high".
DECLARED_TORCH: Dict[str, str] = {
    "float32_matmul_precision": "high",
    "cuda_matmul_allow_tf32": "True",
    "cudnn_allow_tf32": "True",
    "cudnn_benchmark": "False",
}

#: The dynamo shape posture: nothing becomes dynamic by accident.
#: ``automatic_dynamic_shapes=False`` — never promote a dim on change (a novel
#: signature is a guard miss routed by the consumer guards, never a silent
#: recompile-to-dynamic); ``assume_static_by_default=True`` — unmarked dims
#: are static. Declared dynamism arrives ONLY through explicit
#: ``mark_dynamic`` marks (``compile_cache._with_declared_marks``).
DECLARED_DYNAMO: Dict[str, str] = {
    "automatic_dynamic_shapes": "False",
    "assume_static_by_default": "True",
}


def declaration(
    overrides: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """The full declared settings — the ONLY input the env seal digests.

    Facts here are what WE state, never what the process reads back: the env
    table, the torch flag table (+ declared knob overrides), the dynamo shape
    posture, the host-ISA codegen clamp (a declared rule of the host —
    ``min(host level, BASELINE)`` — like ``sm``, not a config read-back), and
    the canonical process posture. A torchless worker declares the absence
    itself as the fact."""
    from . import guard_closure, host_isa  # lazy: keep this module light

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
        # Knob names are still validated (torch-free contract); a DECLARED
        # knob on a torchless image refuses — honouring it silently would
        # fork compiled graph identity.
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
    """:data:`DECLARED_TORCH` with DECLARED knob overrides folded in — the
    typed-knob surface: keys must exist in the canonical table, so
    the only route to non-canonical behavior is a declared knob, which is
    part of the declaration and therefore keyed. An unknown knob refuses,
    named. One-way door: a scrubbed env var that turns out to be needed
    becomes a knob, never an unscrub."""
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


# ---------------------------------------------------------------------------
# Imposition — process env
# ---------------------------------------------------------------------------


def impose_process_env() -> None:
    """Write :data:`DECLARED_ENV` into ``os.environ`` — unconditionally, so
    an ambient value never survives. Called at entrypoint import (pre-torch,
    covers children) and again by ``env_seal.establish`` after the scrub
    (the scrub erases the whole namespace, including our own entries)."""
    os.environ.update(DECLARED_ENV)


def _interpreter_env_diffs() -> List[str]:
    """Declared entries CPython consumed at interpreter start, checked by
    their EFFECT — the env var alone proves nothing once the interpreter is
    running. PYTHONHASHSEED=0 must have disabled hash randomization."""
    diffs: List[str] = []
    want = DECLARED_ENV.get("PYTHONHASHSEED")
    if want == "0" and sys.flags.hash_randomization != 0:
        diffs.append(
            "PYTHONHASHSEED: declared '0' but this interpreter booted with "
            "hash randomization ON")
    return diffs


def ensure_interpreter_env() -> None:
    """Impose the interpreter-level declared env on the CURRENT process,
    re-exec'ing once if the interpreter booted without it.

    CPython reads ``PYTHONHASHSEED`` at interpreter start, so a running
    process cannot adopt it in place — the sanctioned imposition is exec:
    set the declared env and replace this process with itself
    (``sys.orig_argv``). The re-exec'd interpreter sees the declared env at
    start, so the check passes and the exec runs at most once. Call sites:
    the worker entrypoint's ``__main__`` (before the procsplit fork), test
    ``conftest``, and any embedder that mints — BEFORE torch is imported."""
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
    """Fail-closed check that the declared interpreter env is IN EFFECT —
    ``env_seal.establish`` calls this, so a process that skipped
    :func:`ensure_interpreter_env` cannot seal an identity claiming the
    declared hash seed it does not have."""
    diffs = _interpreter_env_diffs()
    if diffs:
        raise SettingsImpositionError(
            "declared interpreter env is not in effect: "
            + "; ".join(diffs)
            + " — the entrypoint imposes it by re-exec; embedders/tests call "
            "settings_authority.ensure_interpreter_env() before torch")


# ---------------------------------------------------------------------------
# Imposition — torch globals
# ---------------------------------------------------------------------------


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
    """Impose the declared torch table (+ declared knobs), then verify the
    read-back. Torch's backend flags are C-level process globals, so a
    same-thread read-back is a process-wide proof. Returns the read-back.

    On a torchless worker there is nothing to impose; knob names are still
    validated and a declared knob refuses."""
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


# ---------------------------------------------------------------------------
# Imposition — ConfigModule-backed settings (dynamo/inductor/functorch)
# ---------------------------------------------------------------------------


def impose_config_default(
    config_module: object, key: str, value: object,
) -> None:
    """Write the PROCESS-WIDE fallback for a torch ConfigModule key.

    A plain ``module.key = x`` assignment sets a ``user_override``, and torch
    documents that layer as thread-local (a ``ContextVar``) — only the
    assigning thread ever reads it back. Every OTHER thread falls through the
    precedence chain to ``default``, which is the layer this writes. Nothing
    else in the chain is settable at runtime: ``alias`` and the two
    ``env_value_*`` layers are resolved once at config install."""
    entry = getattr(config_module, "_config", {}).get(key)
    if entry is None:
        raise SettingsImpositionError(
            f"declared setting cannot reach a process-wide target: torch's "
            f"config has no {key!r} entry to set a default on. torch's "
            f"config internals changed; re-seat the authority before any "
            f"compile can be trusted.")
    entry.default = value


def read_in_fresh_thread(fn: Callable[[], Any]) -> Any:
    """Run ``fn`` on a brand-new thread and return its value. A fresh thread
    starts with an empty ``ContextVar`` context, so this reads exactly what a
    background compile thread would read."""
    box: Dict[str, Any] = {}

    def _run() -> None:
        box["v"] = fn()

    t = threading.Thread(target=_run, name="settings-readback", daemon=True)
    t.start()
    t.join()
    return box.get("v")


def dynamo_readback() -> Dict[str, str]:
    """Live values of the declared dynamo facts, read on THIS thread — the
    mint/serve thread is exactly where a stray thread-local override would
    sit, and a fresh thread would read the (imposed) default anyway."""
    if not torch_capability.present():
        return {"torch": torch_capability.ABSENT}
    import torch._dynamo

    return {
        name: str(getattr(torch._dynamo.config, name))
        for name in DECLARED_DYNAMO
    }


def impose_dynamo() -> Dict[str, str]:
    """Impose the declared dynamo shape posture process-wide (default layer +
    this thread's own override), verify on a FOREIGN thread — the same
    mechanism and reasoning as ``host_isa.impose``. No-op on torchless."""
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


# ---------------------------------------------------------------------------
# Routed writes — settings mutations other modules NEED, performed here so
# the fence stays airtight. Each names its caller and its invariant.
# ---------------------------------------------------------------------------


def disable_autograd_cache() -> None:
    """The AOTAutogradCache key hashes ``fx_kwargs[get_decomp_fn]`` via the
    function's REPR — a process memory address (ASLR), so AOT keys can NEVER
    match across processes/pods. Compiled graph portability requires the
    (portable) FxGraphCache to be the lookup surface: disable the AOT layer
    symmetrically for producer capture and consumer seeding.

    Process-global disable needs BOTH halves: the pre-torch-import env
    (:data:`DECLARED_ENV`, fresh processes incl. compile-worker subprocesses)
    and, torch already imported, the installed config entry's
    ``env_value_force`` — user overrides are thread-local ContextVars in
    torch>=2.13, and the entry-level env force is consulted by every thread
    with top precedence. Measured: a plain assignment runs on the arming thread
    while the warmup compile runs on another, so it does nothing."""
    os.environ["TORCHINDUCTOR_AUTOGRAD_CACHE"] = (
        DECLARED_ENV["TORCHINDUCTOR_AUTOGRAD_CACHE"])
    if "torch" not in sys.modules:
        return
    try:
        import torch._functorch.config as fconf

        fconf.enable_autograd_cache = False  # this thread (public API)
        fconf._config["enable_autograd_cache"].env_value_force = False  # type: ignore[attr-defined]
    except Exception:
        logger.debug("settings authority: AOT autograd cache disable "
                     "unavailable", exc_info=True)


def set_compiler_cache_tag(tag: str) -> None:
    """Install ``compile_cache``'s semantic tag for an arm's compiles.
    Process-global (torch.compiler.config), retagged at every arm; a
    mid-serve heal recompile under a newer tag can only MISS, never
    cross-consume. Inner-cache identity, not process behavior — deliberately
    NOT in the declaration or the seal."""
    try:
        import torch.compiler.config as compiler_config

        compiler_config.cache_key_tag = tag
    except Exception:
        logger.debug("settings authority: semantic cache tag unavailable",
                     exc_info=True)


def raise_dynamo_cache_limits(want: int) -> None:
    """Size dynamo's per-code-object recompile ceiling to the declared shape
    set (a family can declare more graphs than torch's default of 8) — never
    lower an operator-raised value. Cache
    ADMISSION, not codegen: changes whether dynamo keeps compiling, never
    what it emits, so it is not a seal fact."""
    if not torch_capability.present():
        return
    try:
        import torch._dynamo

        torch._dynamo.config.cache_size_limit = max(
            int(torch._dynamo.config.cache_size_limit), want)
        # `recompile_limit` is unconditional at this repo's torch>=2.13 floor;
        # the hasattr arm it used to carry was for a torch the fleet cannot run.
        torch._dynamo.config.recompile_limit = max(
            int(torch._dynamo.config.recompile_limit), want)
    except Exception:
        # NOT debug: a hub-spawned pod cannot read DEBUG, and failing here
        # silently keeps torch's 8-graph ceiling and stops compiling declared
        # shapes. Announce at WARNING and keep serving (pgw#1307).
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
    "raise_dynamo_cache_limits",
    "read_in_fresh_thread",
    "set_compiler_cache_tag",
    "torch_readback",
    "validated_table",
    "verify_interpreter_env",
]
