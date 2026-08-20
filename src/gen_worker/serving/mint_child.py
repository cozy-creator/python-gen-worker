"""pgw#1371: the runtime mint's COMPILE CHILD. Never runs on the serving loop.

One child, one graph. Its whole input is bytes — the serialized
``ExportedProgram`` the release published, fetched by digest (Paul,
2026-08-18) — so no author code runs, no weights are read, and nothing here
touches the parent's live pipeline.

It is a separate PROCESS for three reasons, all of them load-bearing:

* **th#1299**: ``torch.export.load`` and inductor hold the GIL for minutes at
  a time. On the serving process that starves the heartbeat and eager
  serving. This module is the CHILD_ONLY side of that split.
* **Crash isolation**: an inductor OOM or segfault costs one graph, not the
  worker. The parent reads the exit status and the artifact on disk.
* **Condemnation has teeth**: a wedged compile in a thread cannot be stopped
  from Python. A wedged compile in a child can be killed, which is what makes
  the progress guard's verdict actionable rather than advisory.

Invoked as ``python -m gen_worker.serving.mint_child`` with one JSON request
on argv; it writes the artifact and prints nothing the parent parses beyond
its exit status.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Mapping

from . import weightless_program

logger = logging.getLogger(__name__)

#: The modules that DEFINE the parent/child contract: this entrypoint and the
#: mint that spawns it. Their bytes are the request/response shape, so a child
#: whose copies differ from the parent's is not running the parent's contract.
CONTRACT_MODULES = ("mint.py", "mint_child.py")

#: Sentinel for "no source to digest ANYWHERE" — a frozen or zipimported
#: install. The only honest fail-open: we genuinely cannot prove either way.
NO_SOURCE = ""


class ContractModuleMissing(RuntimeError):
    """A NAMED contract module is absent while others are readable.

    Distinguished from :data:`NO_SOURCE` deliberately. pgw#1444: the v1 pool's
    digest returned ``""`` for both cases, so when `cd46c957` deleted the
    module its `_CONTRACT_MODULES` named, the pgw#840 guard's
    ``if not CODE_DIGEST: return`` silently became a no-op — a guard turned off
    by an unrelated deletion, with nothing to say so. An absent named module is
    a BROKEN INSTALL, not an unprovable one, and it refuses.
    """


def contract_digest() -> str:
    """Digest the parent/child contract source, or say why it cannot.

    Taken from the FILES rather than from a version string on purpose: the
    hazard (pgw#840) is a second ``gen_worker`` on the path — a stray checkout,
    an inherited ``PYTHONPATH``, a stale wheel, or this tree edited between the
    parent's import and the child's spawn — and every one of those can carry
    the same version while shipping different bytes.
    """
    here = Path(__file__).resolve().parent
    digests, missing = [], []
    for name in CONTRACT_MODULES:
        try:
            digests.append(hashlib.sha256((here / name).read_bytes()).digest())
        except OSError:
            missing.append(name)
    if not digests:
        return NO_SOURCE  # frozen/zipimport: nothing to compare, and we say so
    if missing:
        raise ContractModuleMissing(
            f"the mint's parent/child contract names {missing} and this "
            f"install does not have them, while {[n for n in CONTRACT_MODULES if n not in missing]} "
            f"IS present — a partial install or a deletion that left its "
            f"callers behind. Refusing rather than skipping the code-identity "
            f"check: an artifact compiled by unknown code must not be "
            f"published under this parent's identity."
        )
    combined = hashlib.sha256()
    for digest in digests:
        combined.update(digest)
    return combined.hexdigest()[:16]



def _place_constants(program: "Any", target_arch: str) -> None:
    """Move the program's lifted constants onto the device being compiled for.

    The derive can only give us real constant VALUES, not real constant
    placement: a cuda trace on a GPU-less publish box fakes them and swaps the
    real cpu tensors back before serializing, so a loaded blob reads
    ``weights: cuda`` / ``constants: cpu``. Measured on a 4070: the values
    survive, the placement does not, and AOTI rejects the mix. One device is
    the compile's own, so the mint owns this move.
    """
    import torch

    if not target_arch.strip().lower().startswith("sm_"):
        return
    constants = getattr(program, "constants", None) or {}
    for name, value in list(constants.items()):
        if isinstance(value, torch.Tensor) and value.device.type != "cuda":
            constants[name] = value.to("cuda")



def _ensure_cuda_home(target_arch: str) -> None:
    """Make the CUDA root exist BEFORE inductor needs it (pgw#1464).

    `cuda_root.py` exists for exactly one failure and nothing on the mint path
    called it. Its own docstring names the shape: AOTI "dies with *CUDA_HOME
    environment variable is not set*" -- and it dies at the LINK step, so the
    codegen and the kernel work are already paid for. Measured on a real
    compile: the whole thing runs, then `InductorError: OSError: CUDA_HOME
    environment variable is not set`. On a pod that is a mint's entire
    wall-clock spent to produce nothing, attributed to nothing.

    A CUDA_HOME already in the environment WINS and is never second-guessed:
    that is the operator stating where their toolkit is (a devel base image, a
    distro package, a developer box that cannot write to `/usr/local`). Only
    when nobody has said does this compose one, and `compose()` is itself
    idempotent -- a pre-existing root is left alone rather than rebuilt.

    CUDA_HOME is set from WHERE THE COMPOSE ACTUALLY WROTE (pgw#1533), not from
    the module constant. Reading the constant made this correct only on a pod:
    on a box whose `/usr/local` is root-owned the compose could not create
    `/usr/local/cuda`, and this pointed torch at it anyway. Fourteen
    specializations died on `FileNotFoundError:
    '/usr/local/cuda/include'` -- while the toolkit was in the venv's own
    `nvidia-*` wheels the whole time, which is what `compose()` builds from.

    Only for `sm_*`: a CPU target needs no CUDA root, and composing one there
    would be work done to answer a question nobody asked.
    """
    if not target_arch.startswith("sm_"):
        return
    if os.environ.get("CUDA_HOME", "").strip():
        return
    from .. import cuda_root

    composed = cuda_root.compose()
    for line in composed.lines():
        logger.info("mint: %s", line)
    if cuda_root.missing_parts(Path(composed.path)):
        # Stated, never fatal: an incomplete root is still worth pointing at
        # (the gaps may be ones this compile does not need), and the refusal
        # belongs to `aot_preconditions`, not here.
        logger.warning(
            "mint: the CUDA root at %s is incomplete (%s) — AOTI resolves it "
            "at the LINK step, so a gap here surfaces after the compile is "
            "already paid for",
            composed.path, ", ".join(cuda_root.missing_parts(Path(composed.path))),
        )
    os.environ["CUDA_HOME"] = composed.path
    logger.info(
        "mint: CUDA_HOME=%s (pgw#1464/pgw#1533) — AOTI resolves it at the link "
        "step, after the compile is already paid",
        composed.path,
    )


def compile_one(request: Mapping[str, Any]) -> Path:
    """Deserialize one graph blob and AOTI-compile it into ``destination``."""
    # pgw#1444/pgw#840, BEFORE any work: this child must BE the parent. It is
    # about to produce the bytes the parent publishes and arms, while every
    # decision around it runs in the parent — an assignment that is only sound
    # while both are the same code. Checked here rather than in the parent's
    # report so a skewed child costs nothing: it refuses before it compiles.
    stated = str(request.get("contract", ""))
    mine = contract_digest()
    if stated and mine and stated != mine:
        raise ContractModuleMissing(
            f"the compile child ran a DIFFERENT gen_worker than the mint that "
            f"spawned it — parent contract {stated}, child contract {mine} "
            f"from {Path(__file__).resolve().parent.parent.parent}. "
            f"`python -m gen_worker.serving.mint_child` resolves whatever the "
            f"interpreter's path yields (a second checkout, an inherited "
            f"PYTHONPATH, a stale wheel, or this tree edited between the "
            f"mint's import and this spawn). Refusing before compiling: this "
            f"artifact would be published and armed under the parent's "
            f"identity."
        )
    import torch

    from .._vendor.torchcg import CallIngress, Engine, GraphSpecialization, RuntimeCompatibility
    from .._vendor.tensorfs import LocalCAS

    _ensure_cuda_home(str(request.get("target_arch", "")))
    # pgw#1468: the derive's blob carries structure and NO weight bytes, and
    # `torch.export.load` cannot read one — every fake parameter dedups onto a
    # single 0-byte payload, so the second and later parameters land out of
    # bounds of a storage sized from the first. This teaches the loader to
    # rebuild the state dict from the archive's own metadata, which is the only
    # description of those tensors that was ever written. Weight-BEARING
    # archives are untouched.
    weightless_program.install()
    program = torch.export.load(str(request["blob"]))
    # pgw#1458: the derive traces on cuda, but the swap-back that keeps the
    # lifted constants' REAL values (they key the graph, so faking them would
    # collide every model that differs only in its constant table) brings them
    # back CPU-placed. Values are authoritative for identity; PLACEMENT is the
    # mint's, and this is the one line that says so. Without it AOTI refuses
    # the mixed placement minutes into the compile, from inside its own
    # assertion, naming no graph specialization.
    _place_constants(program, str(request["target_arch"]))
    # tcg#55: the graph interface is DERIVED, and there is no longer a `graph=`
    # parameter to state it with. This child supplies an ExportedProgram, an
    # identity, and the call INGRESS -- the one fact torch cannot know, because
    # parameter names and argument nesting live in the author's forward
    # signature, not in the traced graph. pgw#1456 was a hand-built two-field
    # dict; its fix was a hand-built five-field dict; both were writing down
    # facts nothing read (`lifted_inputs` and `specialization` had no reader in
    # either repo and are deleted upstream) or facts torchcg already derives
    # (`constant_fqns`, `literal_values`, `placement`). A stub is now
    # unrepresentable rather than merely refused.
    spec = GraphSpecialization(
        name=str(request["graph"]),
        target=str(request["target"]),
        program=program,
        ingress=CallIngress.decode(request["ingress"]),
    )
    runtime = RuntimeCompatibility(
        str(request["target_arch"]), toolchain=dict(request["toolchain"]))
    engine = Engine(LocalCAS(Path(str(request["cas"]))))
    destination = Path(str(request["destination"]))
    # `Engine.compile` resolves the freshly-minted key INTO `destination`, so
    # the unpacked artifact is there; `compiled_graph.artifact` is the CAS ref
    # it came from, not a path a caller can hand anyone.
    engine.compile(spec, runtime, destination)
    return destination


def main(argv: "list[str]") -> int:
    if len(argv) != 2:
        print("usage: python -m gen_worker.serving.mint_child <request.json>",
              file=sys.stderr)
        return 2
    request = json.loads(Path(argv[1]).read_text())
    artifact = compile_one(request)
    Path(str(request["result"])).write_text(str(artifact))
    return 0


if __name__ == "__main__":  # pragma: no cover — process entry
    raise SystemExit(main(sys.argv))
