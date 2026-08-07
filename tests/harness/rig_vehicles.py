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
    #: The outcome this vehicle is EXPECTED to produce. A variant that exists
    #: to demonstrate a refusal is not a failing test — but a variant whose
    #: outcome flips IS news, in either direction, so the gauntlet compares
    #: against this rather than against "green".
    expect: str = "green"
    #: Why, when `expect` is not "green".
    expect_note: str = ""
    #: Requires a real card. `scaled_mm` IS the w8a8 lane, and
    #: `w8a8_gemm_mode()` answers "" without CUDA — which the module class
    #: refuses — so a cardless run of these is NOT-RUN, never a red.
    gpu_only: bool = False


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


def _micro_cell(bucket: int = 0) -> Any:
    from gen_worker.registry import CompileCell

    from micro_diffusion.aot_declaration import COND_LEN, PIXEL_ROWS

    return CompileCell(
        shapes=PIXEL_ROWS, targets=("transformer", "decoder"),
        family="micro-diffusion", regional=False, text_len=COND_LEN,
        dynamic=(), lora_bucket=int(bucket), guidance_scales=(), text_lens=())


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
    shapes=PIXEL_ROWS, targets=("transformer", "decoder"),
    family="micro-diffusion", regional=False, text_len=COND_LEN,
    dynamic=(), lora_bucket=%(bucket)d, guidance_scales=(), text_lens=())
# The adopting process rebuilds the pipeline from the SAME generated tree,
# which is the whole point of deterministic weights: a second machine with the
# seed has the bytes, and the snapshot digest agrees without a download.
# The cell is minted for the device the mint ran on, and a code-only cell
# binds its constants from RESIDENT weights — so the adopting pipeline must
# live on that device or the bind fails inside AOTI itself
# (`update_constant_buffer_func_ ... API call failed`). Measured the first
# time this rig ran on a real card; a CPU-only cycle cannot reach it.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
pipe = MicroPipeline.from_pretrained(os.environ["PGW978_CHECKPOINT"]).to(DEVICE)
config = pipe.config

# pgw#999: a BUCKET-bearing cell is keyed on the branch-bearing lane, and
# `aot_cells.discover` filters candidates by the ADOPTING pipeline's own lane.
# A pod that boots without the LoRA lane computes `lane=plain` and rejects the
# cell it asked for with `lane_mismatch` — BEFORE arming is ever attempted, so
# no `arm_aot` gate ever runs and no adopt reason is produced. Measured here
# the first time this vehicle ran. So the adopting side puts itself on the same
# lane the mint was on, which is what a serving pod with this endpoint's
# declared bucket does at boot.
if int(getattr(cfg, "lora_bucket", 0) or 0):
    from gen_worker import compile_cache as _cc
    _cc.apply_lora_execution_lane(pipe, int(cfg.lora_bucket))


def _feed(arity, tokens):
    """One legal call of each declared arm, built from a FIXED generator so
    the eager pass and the served pass see identical bytes."""
    gen = torch.Generator().manual_seed(997)
    x = [torch.randn(tokens, config.in_channels, generator=gen).to(DEVICE)
         for _ in range(arity)]
    t = torch.full((arity,), 100.0, device=DEVICE)
    cond = [torch.randn(COND_LEN, config.cond_dim, generator=gen).to(DEVICE)
            for _ in range(arity)]
    lat = torch.randn(1, tokens, config.in_channels, generator=gen).to(DEVICE)
    return x, t, cond, lat


# ARMS: every declared entry. The rows deliberately differ from the seed row
# the mint traced, so the artifact is exercised through its DERIVED range
# rather than at the one coordinate it was built at.
ARMS = [("transformer/cfg=true", CFG_ARITY, TOKEN_ROWS[0]),
        ("transformer/cfg=false", 1, TOKEN_ROWS[0]),
        ("decoder", 1, TOKEN_ROWS[-1])]

# pgw#999: the eager reference comes from a SEPARATE, PRISTINE instance, not
# from `pipe` before arming. Arming binds constants from resident weights and
# the numerics gate runs its own forwards through the served module, so a
# reference captured from the object that is about to be armed is a reference
# the arm can contaminate — and a parity number computed against a
# contaminated reference is worse than no parity number.
ref = MicroPipeline.from_pretrained(os.environ["PGW978_CHECKPOINT"]).to(DEVICE)
if int(getattr(cfg, "lora_bucket", 0) or 0):
    from gen_worker import compile_cache as _cc0
    _cc0.apply_lora_execution_lane(ref, int(cfg.lora_bucket))
eager = {}
with torch.no_grad():
    for name, arity, latent in ARMS:
        x, t, cond, lat = _feed(arity, latent)
        eager[name] = (ref.decoder(lat) if name == "decoder"
                       else ref.transformer(x, t, cond)).clone()

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
                       else pipe.transformer(x, t, cond))
                deltas[name] = float((got - eager[name]).abs().max())
        out["parity_max_abs"] = deltas
        out["served_entry_calls"] = dict(aot_serve.served_entry_calls(pipe))
        out["execution_count"] = int(aot_serve.execution_count(pipe))
        out["ingress_refusals"] = int(aot_serve.ingress_refusals(pipe))
        # A tolerance, not equality: AOTI fuses and reassociates, so bitwise
        # identity is not the claim. 1e-4 on float32 activations is.
        out["parity_ok"] = all(v <= 1e-4 for v in deltas.values())
        out["ok"] = bool(out["parity_ok"]) and out["execution_count"] > 0
    else:
        # pgw#999, in the rig's own code: a leg that reports OK while the arm
        # REFUSED is the same defect this issue is about, one layer out. An
        # unarmed cell has served nothing and proved nothing.
        out["ok"] = False
print("RIG_ADOPT " + json.dumps(out))
'''


def _micro_adopt_source_for(bucket: int) -> Any:
    """The adopt child's source for ONE bucket.

    A factory rather than a constant: the adopting process must construct the
    SAME `CompileCell` the mint was keyed on, and a hardcoded `lora_bucket=0`
    made a bucket-bearing vehicle's adopter compute `lane=plain` and reject
    its own cell with `lane_mismatch` before arming (measured, pgw#999).
    """

    def _source(base: str, cache: Path) -> str:
        return _MICRO_ADOPT % {
            "paths": [str(REPO / "tests"), str(REPO / "src"), str(MICRO_SRC)],
            "base": base, "cache": str(cache), "bucket": int(bucket)}

    return _source


def _micro_checkpoint(root: Path) -> Path:
    from micro_diffusion.weights import materialize

    return materialize(root)


def _micro_w8a8_checkpoint(root: Path) -> Path:
    """The fp8 tree, through the SDK's OWN `quantize_tree_w8a8`."""
    from micro_diffusion.weights import materialize_w8a8

    return materialize_w8a8(root)


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
    adopt_source=_micro_adopt_source_for(0),
    covers=("org-worker packaging: 3 export entries (2 fork arms + a second "
            "target), CONTAINER inputs with a plain input after them "
            "(pgw#993/pgw#994), a derived dynamic range, generated weights"),
)


# ---------------------------------------------------------------------------
# micro-lora — pgw#999's diagnosis vehicle. The SAME family on the
# BRANCH-BEARING graph, which is the axis sdxl's refused cell differs on.
# ---------------------------------------------------------------------------

#: sdxl's stamp at the refused mint was `w8a8-lora64`. The w8a8 half cannot be
#: armed on this box (pgw#983: no CUDA, so `w8a8_gemm_mode()` is "" and the
#: module class refuses), but the LORA half is lane-agnostic — gw#558's branch
#: capability covers "plain resident" Linears — so the bucket axis IS
#: reproducible here, on CPU, and it is the axis `arm_aot` grows a whole extra
#: gate for (`install_lifted_lora_forward` -> `assert_lifted_contract`).
MICRO_LORA_BUCKET = 64

MICRO_LORA = Vehicle(
    name="micro-lora",
    modules=(RUNTIME_HOOK, "micro_diffusion.main"),
    function="generate",
    family="micro-diffusion",
    ref_path="cozy/micro-diffusion",
    syspath=(str(MICRO_SRC),),
    build_checkpoint=_micro_checkpoint,
    checkpoint_bytes=_micro_checkpoint_bytes,
    compile_cell=lambda: _micro_cell(MICRO_LORA_BUCKET),
    adopt_source=_micro_adopt_source_for(MICRO_LORA_BUCKET),
    covers=(f"everything `micro` covers, on the BRANCH-BEARING graph "
            f"(lora_bucket={MICRO_LORA_BUCKET}): the mint child applies the "
            f"LoRA execution lane before tracing and the parent's arm runs "
            f"`install_lifted_lora_forward` + `assert_lifted_contract` — the "
            f"gate pair a plain-lane cycle never reaches (pgw#999)"),
)


# ---------------------------------------------------------------------------
# micro-lora-plain-parent — pgw#999's design question, as a standing leg
# ---------------------------------------------------------------------------

#: A `lora64` cell offered to a parent that boots on the PLAIN lane. This is
#: A1 step 8's "boot a SECOND pod cold and adopt", and it is EXPECTED to be
#: refused: `aot_cells.discover` filters candidates by the adopting pipeline's
#: own lane, so a bucket-less parent computes `lane=plain` and rejects the cell
#: with `lane_mismatch` BEFORE any arm runs — no `arm_aot` gate, no adopt
#: reason. Standing here so the day it silently starts passing is visible.
MICRO_LORA_PLAIN_PARENT = Vehicle(
    name="micro-lora-plain-parent",
    modules=(RUNTIME_HOOK, "micro_diffusion.main"),
    function="generate",
    family="micro-diffusion",
    ref_path="cozy/micro-diffusion",
    syspath=(str(MICRO_SRC),),
    build_checkpoint=_micro_checkpoint,
    checkpoint_bytes=_micro_checkpoint_bytes,
    compile_cell=lambda: _micro_cell(MICRO_LORA_BUCKET),
    # The MINT is bucket 64; the ADOPTER is bucket 0. That mismatch is the
    # whole point of the leg.
    adopt_source=_micro_adopt_source_for(0),
    covers=("the pgw#999 design question: a bucket-bearing cell offered to a "
            "parent on the plain lane"),
    expect="red",
    expect_note=("discovery rejects on `lane_mismatch` before arming — the "
                 "adopting pod must carry the cell's own lora bucket "
                 "(required line in attempt 27's choreography)"),
)




# ---------------------------------------------------------------------------
# micro-4d — pgw#998's shape: a NONLINEAR traced extent (z-image's declaration)
# ---------------------------------------------------------------------------


def _micro_4d_cell() -> Any:
    from gen_worker.registry import CompileCell

    from micro_diffusion.aot_declaration_4d import COND_LEN, PIXEL_ROWS

    return CompileCell(
        shapes=PIXEL_ROWS, targets=("transformer",), family="micro-4d",
        regional=False, text_len=COND_LEN, dynamic=(), lora_bucket=0,
        guidance_scales=(), text_lens=())


_MICRO_4D_ADOPT = '''
import json, logging, os, sys
for p in %(paths)r:
    sys.path.insert(0, p)
logging.basicConfig(level=logging.INFO, stream=sys.stderr)
import harness.rig_runtime  # noqa: F401
import torch
from pathlib import Path
from gen_worker import aot_cells, aot_serve
from gen_worker.models import provision
from gen_worker.registry import CompileCell
from micro_diffusion.aot_declaration_4d import ARITY, COND_LEN, LATENT_ROWS, PIXEL_ROWS
from micro_diffusion.pipeline import MicroGridPipeline

torch.set_num_threads(2)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cfg = CompileCell(
    shapes=PIXEL_ROWS, targets=("transformer",), family="micro-4d",
    regional=False, text_len=COND_LEN, dynamic=(), lora_bucket=0,
    guidance_scales=(), text_lens=())
pipe = MicroGridPipeline.from_pretrained(os.environ["PGW978_CHECKPOINT"]).to(DEVICE)
ref = MicroGridPipeline.from_pretrained(os.environ["PGW978_CHECKPOINT"]).to(DEVICE)
config = pipe.config


def _feed(grid):
    gen = torch.Generator().manual_seed(998)
    x = [torch.randn(config.in_channels, grid, grid, generator=gen).to(DEVICE)
         for _ in range(ARITY)]
    t = torch.full((ARITY,), 100.0, device=DEVICE)
    cond = [torch.randn(COND_LEN, config.cond_dim, generator=gen).to(DEVICE)
            for _ in range(ARITY)]
    return x, t, cond


# The row the artifact was NOT seeded on, so the derived range is exercised.
ARMS = [("transformer", LATENT_ROWS[0])]
eager = {}
with torch.no_grad():
    for name, grid in ARMS:
        x, t, cond = _feed(grid)
        eager[name] = pipe.transformer(x, t, cond).clone()

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
    outcome = provision.arm_aot(pipe, cfg, Path(%(cache)r), Path(cell.artifact), 0)
    out["armed"] = bool(outcome)
    out["arm_reason"] = str(getattr(outcome, "reason", "") or "")
    out["arm_detail"] = str(getattr(outcome, "detail", "") or "")[:400]
    if outcome:
        deltas = {}
        with torch.no_grad():
            for name, grid in ARMS:
                x, t, cond = _feed(grid)
                deltas[name] = float(
                    (pipe.transformer(x, t, cond) - eager[name]).abs().max())
        out["parity_max_abs"] = deltas
        out["execution_count"] = int(aot_serve.execution_count(pipe))
        out["parity_ok"] = all(v <= 1e-4 for v in deltas.values())
        out["ok"] = bool(out["parity_ok"]) and out["execution_count"] > 0
    else:
        out["ok"] = False
print("RIG_ADOPT " + json.dumps(out))
'''


def _micro_4d_adopt_source(base: str, cache: Path) -> str:
    return _MICRO_4D_ADOPT % {
        "paths": [str(REPO / "tests"), str(REPO / "src"), str(MICRO_SRC)],
        "base": base, "cache": str(cache)}


MICRO_4D = Vehicle(
    name="micro-4d",
    modules=(RUNTIME_HOOK, "micro_diffusion.main_4d"),
    function="generate-4d",
    family="micro-4d",
    ref_path="cozy/micro-diffusion",
    syspath=(str(MICRO_SRC),),
    build_checkpoint=_micro_checkpoint,
    checkpoint_bytes=_micro_checkpoint_bytes,
    compile_cell=_micro_4d_cell,
    adopt_source=_micro_4d_adopt_source,
    covers=("pgw#998's shape — a 4-D latent with BOTH spatial axes dynamic, so "
            "every matmul's M extent is NONLINEAR in the traced symbols. This "
            "is z-image's declaration, and the seam that was unlowerable "
            "across the mint's export save/load hand-off"),
)




# ---------------------------------------------------------------------------
# micro-w8a8 [+lora] — attempt 26's lane string, on a model that mints in a
# minute. GPU-only: `scaled_mm` is the point.
# ---------------------------------------------------------------------------

_MICRO_W8A8_ADOPT = '''
import json, logging, os, sys
for p in %(paths)r:
    sys.path.insert(0, p)
logging.basicConfig(level=logging.INFO, stream=sys.stderr)
import harness.rig_runtime  # noqa: F401
import torch
from pathlib import Path
from gen_worker import aot_cells, aot_serve, compile_cache as cc
from gen_worker.models import provision
from gen_worker.registry import CompileCell
from micro_diffusion.aot_declaration import CFG_ARITY, COND_LEN, PIXEL_ROWS, TOKEN_ROWS
from micro_diffusion.pipeline import MicroW8a8Pipeline

torch.set_num_threads(2)
DEVICE = "cuda"
BUCKET = %(bucket)d
cfg = CompileCell(
    shapes=PIXEL_ROWS, targets=("transformer", "decoder"),
    family="micro-diffusion", regional=False, text_len=COND_LEN,
    dynamic=(), lora_bucket=BUCKET, guidance_scales=(), text_lens=())
pipe = MicroW8a8Pipeline.from_pretrained(os.environ["PGW978_CHECKPOINT"]).to(DEVICE)
ref = MicroW8a8Pipeline.from_pretrained(os.environ["PGW978_CHECKPOINT"]).to(DEVICE)
if BUCKET:
    cc.apply_lora_execution_lane(pipe, BUCKET)
    cc.apply_lora_execution_lane(ref, BUCKET)
config = pipe.config

# The bf16 tree the producer quantized FROM — `materialize_w8a8` leaves it
# beside the fp8 one. It is the yardstick for how lossy this lane already is.
import json as _json
from safetensors.torch import load_file as _load_file
from micro_diffusion.model import MicroConfig as _MC, MicroDenoiser as _MD
_bf = Path(os.environ["PGW978_CHECKPOINT"])
_bf = _bf.parent / (_bf.name + "-bf16") / "transformer"
_cfgd = _json.loads((_bf / "config.json").read_text())
_ref_mod = _MD(_MC(**{k: v for k, v in _cfgd.items() if k in _MC().as_dict()}))
_ref_mod.load_state_dict(
    _load_file(str(_bf / "diffusion_pytorch_model.safetensors")), strict=True)
_ref_mod = _ref_mod.eval().to(DEVICE)


def _bf16_ref(x, t, cond):
    with torch.no_grad():
        return _ref_mod(x, t, cond)


def _feed(arity, tokens):
    gen = torch.Generator().manual_seed(997)
    x = [torch.randn(tokens, config.in_channels, generator=gen).to(DEVICE)
         for _ in range(arity)]
    t = torch.full((arity,), 100.0, device=DEVICE)
    cond = [torch.randn(COND_LEN, config.cond_dim, generator=gen).to(DEVICE)
            for _ in range(arity)]
    lat = torch.randn(1, tokens, config.in_channels, generator=gen).to(DEVICE)
    return x, t, cond, lat


ARMS = [("transformer/cfg=true", CFG_ARITY, TOKEN_ROWS[0]),
        ("transformer/cfg=false", 1, TOKEN_ROWS[0]),
        ("decoder", 1, TOKEN_ROWS[-1])]
eager = {}
with torch.no_grad():
    for name, arity, tokens in ARMS:
        x, t, cond, lat = _feed(arity, tokens)
        eager[name] = (ref.decoder(lat) if name == "decoder"
                       else ref.transformer(x, t, cond)).clone()

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
        "precision": meta.get("precision"),
    })
    outcome = provision.arm_aot(pipe, cfg, Path(%(cache)r), Path(cell.artifact), BUCKET)
    out["armed"] = bool(outcome)
    out["arm_reason"] = str(getattr(outcome, "reason", "") or "")
    out["arm_detail"] = str(getattr(outcome, "detail", "") or "")[:400]
    if outcome:
        deltas = {}
        with torch.no_grad():
            for name, arity, tokens in ARMS:
                x, t, cond, lat = _feed(arity, tokens)
                got = (pipe.decoder(lat) if name == "decoder"
                       else pipe.transformer(x, t, cond))
                deltas[name] = float((got - eager[name]).abs().max())
        out["parity_max_abs"] = deltas
        out["execution_count"] = int(aot_serve.execution_count(pipe))
        # THE BOUND, MEASURED RATHER THAN PICKED. fp8 storage is a lossy lane:
        # measured on this card, eager-fp8 already differs from the bf16 tree
        # it was quantized from by max|delta| 0.129 (5.3%% of output scale). A
        # fixed max-abs threshold is therefore the wrong instrument here — it
        # would reject the LANE, not the compile.
        #
        # So the invariant is relative to the lane itself: THE COMPILE MUST
        # NOT ADD MORE ERROR THAN THE QUANTIZATION ALREADY DID. The bf16 tree
        # the producer copied from is right next to the fp8 one, so the
        # yardstick is measured every run instead of hardcoded.
        lane = {}
        with torch.no_grad():
            for name, arity, tokens in ARMS:
                if name == "decoder":
                    continue
                x, t, cond, lat = _feed(arity, tokens)
                lane[name] = float(
                    (eager[name] - _bf16_ref(x, t, cond)).abs().max())
        out["lane_noise_max_abs"] = lane
        out["parity_ok"] = all(
            v <= max(lane.get(k, 0.0), 1e-4) for k, v in deltas.items())
        out["ok"] = bool(out["parity_ok"]) and out["execution_count"] > 0
    else:
        out["ok"] = False
print("RIG_ADOPT " + json.dumps(out))
'''


def _micro_w8a8_adopt_source_for(bucket: int) -> Any:
    def _source(base: str, cache: Path) -> str:
        return _MICRO_W8A8_ADOPT % {
            "paths": [str(REPO / "tests"), str(REPO / "src"), str(MICRO_SRC)],
            "base": base, "cache": str(cache), "bucket": int(bucket)}

    return _source


def _w8a8_vehicle(name: str, bucket: int) -> Vehicle:
    lane = f"w8a8-lora{bucket}" if bucket else "w8a8"
    return Vehicle(
        name=name,
        modules=(RUNTIME_HOOK, "micro_diffusion.main_w8a8"),
        function="generate-w8a8",
        family="micro-diffusion",
        ref_path="cozy/micro-diffusion",
        syspath=(str(MICRO_SRC),),
        build_checkpoint=_micro_w8a8_checkpoint,
        checkpoint_bytes=_micro_checkpoint_bytes,
        compile_cell=lambda: _micro_cell(bucket),
        adopt_source=_micro_w8a8_adopt_source_for(bucket),
        gpu_only=True,
        covers=(f"attempt 26's lane string `{lane}` — a REAL fp8 artifact from "
                f"the SDK's own `quantize_tree_w8a8`, loaded by "
                f"`load_w8a8_denoiser` into `_Fp8ScaledLinear` modules with "
                f"`float8_e4m3fn` weights. The gemm mode is chosen by a LIVE "
                f"per-card benchmark (`pertensor` here; an L40S may measure "
                f"`rowwise`) — a per-card fact, never assumed to transfer"),
    )


MICRO_W8A8 = _w8a8_vehicle("micro-w8a8", 0)
MICRO_W8A8_LORA = _w8a8_vehicle("micro-w8a8-lora", MICRO_LORA_BUCKET)


VEHICLES: Dict[str, Vehicle] = {
    v.name: v for v in (TINY, MICRO, MICRO_LORA, MICRO_LORA_PLAIN_PARENT,
                        MICRO_4D, MICRO_W8A8, MICRO_W8A8_LORA)}
DEFAULT_VEHICLE = TINY.name


def vehicle(name: str) -> Vehicle:
    try:
        return VEHICLES[str(name)]
    except KeyError:
        raise SystemExit(
            f"unknown rig vehicle {name!r} (known: {sorted(VEHICLES)!r})")


__all__ = ["DEFAULT_VEHICLE", "MICRO", "MICRO_LORA", "MICRO_LORA_BUCKET",
           "MICRO_4D", "MICRO_LORA_PLAIN_PARENT", "MICRO_W8A8",
           "MICRO_W8A8_LORA", "TINY", "VEHICLES", "Vehicle", "vehicle"]
