"""pgw#997: what the micro-mint rig MINTS, as a selectable vehicle.

The rig (pgw#978) was written against one toy endpoint living in this harness
directory. That endpoint proved the machinery, but it is not an org worker: it
declares one target, one graph class and therefore ONE export entry, and it is
authored inside the SDK's own test tree rather than as a package a build could
consume.

pgw#997 adds a second vehicle — ``examples/micro-diffusion``, a real
org-worker-shaped package with its own ``pyproject.toml``, ``endpoint.toml``,
Dockerfile and deterministic weight generation. Choosing between them is a rig
flag, so the cheap plumbing vehicle stays available and the endpoint-shaped one
is what a mint-path change is proven against.

A vehicle answers six questions and nothing else:

    which modules the mint child imports, and which function it warms
    where its checkpoint comes from (always GENERATED, never fetched)
    which ``CompileCell`` the parent hands across the delegation boundary
    which hub ref names the checkpoint
    which sys.path entries both interpreters need
    how a SECOND process rebuilds enough pipeline to adopt the cell
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

REPO = Path(__file__).resolve().parent.parent.parent
MICRO_SRC = REPO / "examples" / "micro-diffusion" / "src"

#: Listed FIRST in every vehicle's modules: it installs the cardless-box
#: runtime probes before the endpoint module is imported. See rig_runtime.
RUNTIME_HOOK = "harness.rig_runtime"


@dataclass(frozen=True)
class Vehicle:
    name: str
    #: Import names handed to the mint child, in order.
    modules: Tuple[str, ...]
    #: The routable function the child warms.
    function: str
    family: str
    #: The hub ref the parent's ``MintSlot`` claims.
    ref_path: str
    #: Extra sys.path entries the parent and BOTH children need.
    syspath: Tuple[str, ...]
    #: (root) -> the generated checkpoint tree.
    build_checkpoint: Callable[[Path], Path]
    #: (tree) -> total bytes on disk.
    checkpoint_bytes: Callable[[Path], int]
    #: () -> the ``registry.CompileCell`` the delegation boundary carries.
    compile_cell: Callable[[], Any]
    #: (base_url, cache_dir) -> python source for the adopting process.
    adopt_source: Callable[[str, Path], str]
    #: What this vehicle proves that the other does not — printed, so a
    #: reported cycle time is never read against the wrong vehicle.
    covers: str


# ---------------------------------------------------------------------------
# tiny — the pgw#978 plumbing vehicle. One target, one entry, no packaging.
# ---------------------------------------------------------------------------


def _tiny_cell() -> Any:
    from gen_worker.registry import CompileCell

    from harness import tiny_diffusion_endpoint as ep

    return CompileCell(
        shapes=(ep.PIXEL_SHAPE,), targets=("unet",), family=ep.FAMILY,
        regional=False, text_len=ep.TEXT_LEN, dynamic=(), lora_bucket=0,
        guidance_scales=(), text_lens=())


_TINY_ADOPT = """
import json, os, sys
sys.path.insert(0, %(tests)r)
sys.path.insert(0, %(src)r)
from pathlib import Path
from types import SimpleNamespace
from gen_worker import aot_cells
from gen_worker.registry import CompileCell
from harness import tiny_diffusion_endpoint as ep
from harness.tiny_diffusion import TinyUNet

cfg = CompileCell(
    shapes=(ep.PIXEL_SHAPE,), targets=("unet",), family=ep.FAMILY,
    regional=False, text_len=ep.TEXT_LEN, dynamic=(), lora_bucket=0,
    guidance_scales=(), text_lens=())
pipe = SimpleNamespace(unet=TinyUNet().eval())
cell = aot_cells.discover(
    pipe, cfg, base_url=%(base)r,
    worker_jwt=lambda: "local-rig-worker-jwt",
    cache_dir=Path(%(cache)r))
out = {"pid": os.getpid(), "ok": cell is not None}
if cell is not None:
    out.update({
        "cell_key": cell.cell_key, "family": cell.family, "ref": cell.ref,
        "snapshot_digest": cell.snapshot_digest,
        "artifact_bytes": Path(cell.artifact).stat().st_size,
    })
print("RIG_ADOPT " + json.dumps(out))
"""


def _tiny_adopt_source(base: str, cache: Path) -> str:
    return _TINY_ADOPT % {
        "tests": str(REPO / "tests"), "src": str(REPO / "src"),
        "base": base, "cache": str(cache)}


def _tiny_checkpoint(root: Path) -> Path:
    from harness.tiny_diffusion import build_checkpoint

    return build_checkpoint(root)


def _tiny_checkpoint_bytes(tree: Path) -> int:
    from harness.tiny_diffusion import checkpoint_bytes

    return checkpoint_bytes(tree)


TINY = Vehicle(
    name="tiny",
    modules=(RUNTIME_HOOK, "harness.tiny_diffusion_endpoint"),
    function="rig-generate",
    family="microrig",
    ref_path="rig/tiny-diffusion",
    syspath=(),
    build_checkpoint=_tiny_checkpoint,
    checkpoint_bytes=_tiny_checkpoint_bytes,
    compile_cell=_tiny_cell,
    adopt_source=_tiny_adopt_source,
    covers="plumbing: one target, one entry, no container inputs, no packaging",
)


# ---------------------------------------------------------------------------
# micro — the pgw#997 org-worker vehicle. Three entries, container inputs.
# ---------------------------------------------------------------------------


def _micro_cell() -> Any:
    from gen_worker.registry import CompileCell

    from micro_diffusion.aot_declaration import COND_LEN, PIXEL_ROWS

    return CompileCell(
        shapes=PIXEL_ROWS, targets=("denoiser", "decoder"),
        family="micro-diffusion", regional=False, text_len=COND_LEN,
        dynamic=(), lora_bucket=0, guidance_scales=(), text_lens=())


_MICRO_ADOPT = '''
import json, logging, os, sys
for p in %(paths)r:
    sys.path.insert(0, p)
# A MISS is not a crash and its reason lives in the typed `aot-cells` events.
# Without a configured handler they go nowhere and the rig reports "no cell"
# for a filter that rejected twelve of them on one axis (pgw#824).
logging.basicConfig(level=logging.INFO, stream=sys.stderr)
# The SAME runtime hook the mint child imported. `aot_serve.verify` compares
# the cell's stamped sm/torch/cuda against THIS runtime's, so a second process
# that did not get the same supplied probes rejects the cell the first one just
# minted — and reports it as a filter miss, which is the wrong diagnosis.
import harness.rig_runtime  # noqa: F401
import torch
from pathlib import Path
from gen_worker import aot_cells, aot_serve
from gen_worker.models import provision
from gen_worker.registry import CompileCell
from micro_diffusion.aot_declaration import (
    CFG_ARITY, COND_LEN, PIXEL_ROWS, TOKEN_ROWS)
from micro_diffusion.pipeline import MicroPipeline

torch.set_num_threads(2)
cfg = CompileCell(
    shapes=PIXEL_ROWS, targets=("denoiser", "decoder"),
    family="micro-diffusion", regional=False, text_len=COND_LEN,
    dynamic=(), lora_bucket=0, guidance_scales=(), text_lens=())
# The adopting process rebuilds the pipeline from the SAME generated tree,
# which is the whole point of deterministic weights: a second machine with the
# seed has the bytes, and the snapshot digest agrees without a download.
pipe = MicroPipeline.from_pretrained(os.environ["PGW978_CHECKPOINT"])
config = pipe.config


def _feed(arity, tokens):
    """One legal call of each declared arm, built from a FIXED generator so
    the eager pass and the served pass see identical bytes."""
    gen = torch.Generator().manual_seed(997)
    x = [torch.randn(tokens, config.in_channels, generator=gen)
         for _ in range(arity)]
    t = torch.full((arity,), 100.0)
    cond = [torch.randn(COND_LEN, config.cond_dim, generator=gen)
            for _ in range(arity)]
    lat = torch.randn(1, tokens, config.in_channels, generator=gen)
    return x, t, cond, lat


# ARMS: every declared entry. The rows deliberately differ from the seed row
# the mint traced, so the artifact is exercised through its DERIVED range
# rather than at the one coordinate it was built at.
ARMS = [("denoiser/cfg=true", CFG_ARITY, TOKEN_ROWS[0]),
        ("denoiser/cfg=false", 1, TOKEN_ROWS[0]),
        ("decoder", 1, TOKEN_ROWS[-1])]

eager = {}
with torch.no_grad():
    for name, arity, latent in ARMS:
        x, t, cond, lat = _feed(arity, latent)
        eager[name] = (pipe.decoder(lat) if name == "decoder"
                       else pipe.denoiser(x, t, cond)).clone()

cell = aot_cells.discover(
    pipe, cfg, base_url=%(base)r,
    worker_jwt=lambda: "local-rig-worker-jwt",
    cache_dir=Path(%(cache)r))
out = {"pid": os.getpid(), "ok": cell is not None}
if cell is not None:
    meta = aot_serve.unpack_metadata(Path(cell.artifact))
    out.update({
        "cell_key": cell.cell_key, "family": cell.family, "ref": cell.ref,
        "snapshot_digest": cell.snapshot_digest,
        "artifact_bytes": Path(cell.artifact).stat().st_size,
        "entries": sorted((meta.get("entries") or {})),
    })
    # PARITY. Adoption that is never CALLED proves the filter, not the cell —
    # and the serve-side call is exactly where pgw#994 lives: a container
    # input expands to N leaves and every contract position after it shifts.
    #
    # pgw#999: through `provision.arm_aot`, NOT `aot_serve.enable`. They are
    # different gates and the delegated mint uses this one: it adds the mode
    # route, the lifted-LoRA install for a bucket-bearing cfg, and the
    # numerics gate. Arming via `enable` left every one of those uncovered —
    # which is why a cycle could be green while the path that refused sdxl's
    # 36/36 cell had never run locally at all.
    outcome = provision.arm_aot(
        pipe, cfg, Path(%(cache)r), Path(cell.artifact),
        int(getattr(cfg, "lora_bucket", 0) or 0))
    out["armed"] = bool(outcome)
    out["arm_reason"] = str(getattr(outcome, "reason", "") or "")
    out["arm_detail"] = str(getattr(outcome, "detail", "") or
                            getattr(outcome, "identity", ""))[:400]
    if outcome:
        deltas = {}
        with torch.no_grad():
            for name, arity, latent in ARMS:
                x, t, cond, lat = _feed(arity, latent)
                got = (pipe.decoder(lat) if name == "decoder"
                       else pipe.denoiser(x, t, cond))
                deltas[name] = float((got - eager[name]).abs().max())
        out["parity_max_abs"] = deltas
        out["served_entry_calls"] = dict(aot_serve.served_entry_calls(pipe))
        out["execution_count"] = int(aot_serve.execution_count(pipe))
        out["ingress_refusals"] = int(aot_serve.ingress_refusals(pipe))
        # A tolerance, not equality: AOTI fuses and reassociates, so bitwise
        # identity is not the claim. 1e-4 on float32 activations is.
        out["parity_ok"] = all(v <= 1e-4 for v in deltas.values())
        out["ok"] = bool(out["parity_ok"]) and out["execution_count"] > 0
print("RIG_ADOPT " + json.dumps(out))
'''


def _micro_adopt_source(base: str, cache: Path) -> str:
    return _MICRO_ADOPT % {
        "paths": [str(REPO / "tests"), str(REPO / "src"), str(MICRO_SRC)],
        "base": base, "cache": str(cache)}


def _micro_checkpoint(root: Path) -> Path:
    from micro_diffusion.weights import materialize

    return materialize(root)


def _micro_checkpoint_bytes(tree: Path) -> int:
    from micro_diffusion.weights import tree_bytes

    return tree_bytes(tree)


MICRO = Vehicle(
    name="micro",
    modules=(RUNTIME_HOOK, "micro_diffusion.main"),
    function="generate",
    family="micro-diffusion",
    ref_path="cozy/micro-diffusion",
    syspath=(str(MICRO_SRC),),
    build_checkpoint=_micro_checkpoint,
    checkpoint_bytes=_micro_checkpoint_bytes,
    compile_cell=_micro_cell,
    adopt_source=_micro_adopt_source,
    covers=("org-worker packaging: 3 export entries (2 fork arms + a second "
            "target), CONTAINER inputs with a plain input after them "
            "(pgw#993/pgw#994), a derived dynamic range, generated weights"),
)


VEHICLES: Dict[str, Vehicle] = {v.name: v for v in (TINY, MICRO)}
DEFAULT_VEHICLE = TINY.name


def vehicle(name: str) -> Vehicle:
    try:
        return VEHICLES[str(name)]
    except KeyError:
        raise SystemExit(
            f"unknown rig vehicle {name!r} (known: {sorted(VEHICLES)!r})")


__all__ = ["DEFAULT_VEHICLE", "MICRO", "TINY", "VEHICLES", "Vehicle", "vehicle"]
