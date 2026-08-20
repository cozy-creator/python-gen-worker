"""The compiler-stack facts a mint records, and this worker's own version.

pgw#1573 lifted these three functions out of ``compile_cache`` on the way to
deleting it. They were the ONLY symbols nine live call sites still wanted from
a 2,953-line module — the v1 eager-capable arming brain, orphaned since
pgw#1373 deleted ``executor.py`` — and keeping the module alive for them is
what dragged ``aot_serve``, ``local_compiled_graph_store``, ``hot_swap`` and
``shape_growth`` onto every pod's import path.

Nothing here arms, stamps, or resolves anything. It reads what is installed.
"""

from __future__ import annotations

import functools
import hashlib
import logging
from pathlib import Path
from typing import Dict, Tuple

from . import dist_records, env_seal

logger = logging.getLogger(__name__)


def gen_worker_version() -> str:
    """This install's published version, or ``""`` when it cannot be read.

    A worker reporting a fatal, a hardware report, or a mint verdict states
    which bytes produced it; an unreadable version is answered as absent
    rather than guessed (pgw#1564 — "pin inferred from the build chain" is
    exactly what a stated identity ends).
    """
    try:
        from importlib.metadata import version

        return str(version("gen-worker"))
    except Exception:  # noqa: BLE001 — a missing dist is not a worker failure
        return ""


@functools.lru_cache(maxsize=8192)
def closure_file_digest(path: str, mtime_ns: int, size: int) -> str:
    """Content digest of one source file, keyed on (path, mtime, size) so a
    repeated read of an unchanged file never re-hashes it."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


@functools.lru_cache(None)
def toolchain_digest() -> Tuple[Tuple[str, str], ...]:
    """pgw#710/pgw#1059: CONTENT identity of "the compiler stack AS WE
    CONFIGURE IT", per component — the ``toolchain`` key axis's whole input.

    THE COMPILER, and not the model libraries (pgw#1050): ``diffusers`` /
    ``transformers`` / ``peft`` are excluded because their whole effect on a
    compiled graph arrives through the traced graph, which the ``graph`` axis
    hashes node-for-node — see ``torchcg.identity``'s membership rules for the
    channel-by-channel argument and for the two fences (B1 code-only + the
    pgw#1097 folding fence; ``env_seal.assert_seal_unchanged``) that close the
    routes around it. Folded here, every model-library patch release re-keyed
    every compiled graph in the fleet for a graph that had not moved.
    ``torchcg.identity.toolchain_axis_digest`` is the READER of the same
    membership, and the pair is what keeps one axis one derivation.

    The binary half (pgw#710) is the equivalence precondition that lets
    ``image_digest`` be relaxed (pgw#700) without degrading the compile stack's
    identity to version strings (the ccache ``compiler_check=mtime`` failure
    class; sccache's answer — hash the compiler binary and its runtime libs —
    is the precedent): the dist-info ``RECORD`` of torch/triton and every
    ``nvidia-*`` runtime wheel (RECORD already carries per-file sha256s, so
    hashing it is whole-package content identity with no multi-GB re-walk) plus
    the bundled CUDA tool BINARIES (ptxas/nvdisasm ride triton's wheel; a
    swapped ptxas silently changes emitted cubins).

    The configuration half (pgw#1059 amendment 4, on pgw#1049's seal v4):

    * ``settings_declaration`` — the digest of the settings DECLARATION (env
      table, torch flags + knobs, dynamo posture, host-ISA clamp, process
      posture). Settings are compiler flags: with the single settings authority
      the declaration is one value fleet-wide, so as its own axis it carried
      zero bits — but a deliberate settings change must still re-key, and this
      is the axis that change honestly belongs to.
    * ``loaded_libs`` — the boot-frozen per-file manifest of the native ``.so``
      set the python env ships (pgw#719), which is what covers the
      LD_PRELOAD/LD_LIBRARY_PATH substitution hole: it enumerates the FILES
      rather than the packages, and pgw#1095 derives each digest from the
      RECORD that installed the file while HASHING anything no RECORD covers —
      a preloaded or non-wheel object is therefore still content, not an
      assumption.
    """
    out: Dict[str, str] = {
        "settings_declaration": env_seal.declaration_digest(),
        "loaded_libs": env_seal.loaded_libs_digest(),
    }
    # ONE enumeration of the environment's RECORDs (pgw#1095): the seal's
    # per-FILE digests and this axis's per-PACKAGE digests are two readings of
    # the same manifests, and reading them twice is how two surfaces start
    # disagreeing about what is installed.
    wanted = ("torch", "triton")
    for name, record in dist_records.record_texts().items():
        if name in wanted or name.startswith("nvidia-"):
            out[name] = hashlib.sha256(record.encode()).hexdigest()[:16]
    try:
        import triton

        bin_dir = Path(triton.__file__).parent / "backends" / "nvidia" / "bin"
        if bin_dir.is_dir():
            for tool in sorted(bin_dir.iterdir()):
                if tool.is_file():
                    out[f"bin:{tool.name}"] = hashlib.sha256(
                        tool.read_bytes()).hexdigest()[:16]
    except Exception:  # noqa: BLE001 — a toolchain fact is never fatal
        logger.debug("toolchain_digest: cuda tool hash failed", exc_info=True)
    return tuple(sorted(out.items()))


__all__ = ["closure_file_digest", "gen_worker_version", "toolchain_digest"]
