"""pgw#1371: the runtime mint's COMPILE CHILD. Never runs on the serving loop.

One child, one graph. Its whole input is bytes — the serialized
``ExportedProgram`` the release published, fetched by digest (Paul,
2026-08-18) — so no author code runs, no weights are read, and nothing here
touches the parent's live pipeline.

It is a separate PROCESS for three reasons, all of them load-bearing:

* **th#1299**: ``torch.export.load`` and inductor hold the GIL for minutes at
  a time. On the serving process that starves the heartbeat and eager
  serving. The fence (`scripts/lint_serving_process_compiles.py`) is what
  keeps this honest, and this module is the CHILD_ONLY side of it.
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
import sys
from pathlib import Path
from typing import Any, Mapping

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

    from .._vendor.torchcg import Engine, GraphClassSpec, RuntimeCompatibility
    from .._vendor.tensorfs import LocalCAS

    program = torch.export.load(str(request["blob"]))
    spec = GraphClassSpec(
        graph_class=str(request["graph"]),
        target=str(request["target"]),
        program=program,
        graph={"v": 3, "pytree": {"ingress": request["ingress"]}},
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
