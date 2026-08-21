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

CONTRACT_MODULES = ("mint.py", "mint_child.py")

NO_SOURCE = ""


class ContractModuleMissing(RuntimeError):
    """A NAMED contract module is absent while others are readable."""


def contract_digest() -> str:
    """Digest the parent/child contract source, or say why it cannot."""
    here = Path(__file__).resolve().parent
    digests, missing = [], []
    for name in CONTRACT_MODULES:
        try:
            digests.append(hashlib.sha256((here / name).read_bytes()).digest())
        except OSError:
            missing.append(name)
    if not digests:
        return NO_SOURCE
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
    import torch

    if not target_arch.strip().lower().startswith("sm_"):
        return
    constants = getattr(program, "constants", None) or {}
    for name, value in list(constants.items()):
        if isinstance(value, torch.Tensor) and value.device.type != "cuda":
            constants[name] = value.to("cuda")



def _ensure_cuda_home(target_arch: str) -> None:
    if not target_arch.startswith("sm_"):
        return
    if os.environ.get("CUDA_HOME", "").strip():
        return
    from .. import cuda_root

    composed = cuda_root.compose()
    for line in composed.lines():
        logger.info("mint: %s", line)
    if cuda_root.missing_parts(Path(composed.path)):
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
    weightless_program.install()
    program = torch.export.load(str(request["blob"]))
    _place_constants(program, str(request["target_arch"]))
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
