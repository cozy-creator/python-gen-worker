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

import json
import sys
from pathlib import Path
from typing import Any, Mapping


def compile_one(request: Mapping[str, Any]) -> Path:
    """Deserialize one graph blob and AOTI-compile it into ``destination``."""
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
