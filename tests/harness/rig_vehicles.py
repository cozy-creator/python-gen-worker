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

import importlib
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
    #: (base_url, cache_dir, published checkpoint id) -> python source for
    #: the adopting process (pgw#904: told the exact cell, never listing).
    adopt_source: Callable[[str, Path, str], str]
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
    #: pgw#1042: (tree, device) -> (pipe, cfg) for the PARENT-side handback —
    #: the rig process that opened the mint adopting its own child's cell
    #: through `fleet_cells.adopt_delegated_mint`, exactly as a pod does
    #: BEFORE anything publishes. None skips the leg (the tiny plumbing
    #: vehicle has no loadable pipeline class).
    parent_pipe: Any = None


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
from gen_worker.registry import CompileCell
from harness import tiny_diffusion_endpoint as ep
from harness.tiny_diffusion import TinyUNet

cfg = CompileCell(
    shapes=(ep.PIXEL_SHAPE,), targets=("unet",), family=ep.FAMILY,
    regional=False, text_len=ep.TEXT_LEN, dynamic=(), lora_bucket=0,
    guidance_scales=(), text_lens=())
pipe = SimpleNamespace(unet=TinyUNet().eval())
from harness.rig_fetch import fetch_named_cell

try:
    artifact = fetch_named_cell(
        %(base)r, ep.FAMILY, %(checkpoint)r, Path(%(cache)r))
except Exception:
    artifact = None
out = {"pid": os.getpid(), "ok": artifact is not None}
if artifact is not None:
    from gen_worker import aot_serve, compile_cache as _cc
    meta = aot_serve.unpack_metadata(artifact)
    key = str(meta.get("compiled_graph_key") or "")
    import hashlib
    out.update({
        "compiled_graph_key": key, "family": ep.FAMILY,
        "ref": _cc.system_repo(ep.FAMILY) + "#" + key,
        "snapshot_digest": "sha256:" + hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "artifact_bytes": artifact.stat().st_size,
    })
print("RIG_ADOPT " + json.dumps(out))
"""


def _tiny_adopt_source(base: str, cache: Path, checkpoint: str) -> str:
    return _TINY_ADOPT % {
        "tests": str(REPO / "tests"), "src": str(REPO / "src"),
        "base": base, "cache": str(cache), "checkpoint": checkpoint}


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
from gen_worker import aot_serve
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

# pgw#999: a BUCKET-bearing cell is keyed on the branch-bearing lane, so the
# adopting side puts itself on the same lane the mint was on — what a serving
# pod with this endpoint's declared bucket does at boot — or `arm_aot` refuses
# the very cell it was told to arm.
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

# pgw#904: the adopting process is TOLD the exact cell (checkpoint id from
# the publish leg) — discovery is deleted, and a serving pod is told by
# `Arm.artifact` the same way.
from harness.rig_fetch import fetch_named_cell as _fetch_cell
from types import SimpleNamespace as _CellNS
import hashlib as _hl
from gen_worker import compile_cache as _syscc
try:
    _art = _fetch_cell(%(base)r, cfg.family, %(checkpoint)r, Path(%(cache)r))
except Exception as _exc:
    print("rig-fetch: named cell unavailable: %%s" %% (_exc,), file=sys.stderr)
    cell = None
else:
    _key0 = str(aot_serve.unpack_metadata(_art).get("compiled_graph_key") or "")
    cell = _CellNS(
        artifact=_art, compiled_graph_key=_key0, family=cfg.family,
        ref=_syscc.system_repo(cfg.family) + "#" + _key0,
        snapshot_digest="sha256:" + _hl.sha256(_art.read_bytes()).hexdigest())
out = {"pid": os.getpid(), "ok": cell is not None}
if cell is not None:
    meta = aot_serve.unpack_metadata(Path(cell.artifact))
    out.update({
        "compiled_graph_key": cell.compiled_graph_key, "family": cell.family, "ref": cell.ref,
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

    def _source(base: str, cache: Path, checkpoint: str) -> str:
        return _MICRO_ADOPT % {
            "paths": [str(REPO / "tests"), str(REPO / "src"), str(MICRO_SRC)],
            "base": base, "cache": str(cache), "bucket": int(bucket),
            "checkpoint": checkpoint}

    return _source


def _micro_parent_for(bucket: int, pipeline_cls: str = "MicroPipeline",
                      cell: Any = None) -> Any:
    """(tree, device) -> (pipe, cfg): the mint-opening parent's own pipeline,
    on the mint's lane — what `adopt_delegated_mint` arms on a pod
    (pgw#1042). Mirrors the adopt sources: same class, same device rule,
    same lane application."""

    def _build(tree: Path, device: str) -> Tuple[Any, Any]:
        import micro_diffusion.pipeline as pl
        from gen_worker import compile_cache as cc

        pipe = getattr(pl, pipeline_cls).from_pretrained(str(tree)).to(device)
        if pipeline_cls == "MicroW8a8Pipeline":
            # The production loader stamps the base weight lane at load
            # (`models/w8a8.py`: `pipe._cozy_weight_lane = "w8a8"`), and the
            # parent's arm key reads THAT stamp (`pipeline_weight_lane`).
            # MicroW8a8Pipeline builds its denoiser directly, so the rig
            # parent mirrors the stamp or its lane axis is "" and the
            # pgw#1042 guard (correctly) refuses the handback.
            pipe._cozy_weight_lane = "w8a8"
        if bucket:
            cc.apply_lora_execution_lane(pipe, bucket)
        cfg = cell() if cell is not None else _micro_cell(bucket)
        return pipe, cfg

    return _build


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
    parent_pipe=_micro_parent_for(0),
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
    parent_pipe=_micro_parent_for(MICRO_LORA_BUCKET),
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
#: refused. pgw#904 moved the refusal from discovery's lane filter (deleted)
#: to the arm gate itself: the plain-lane parent fetches the exact named cell
#: and `arm_aot` refuses it on lane drift. Standing here so the day it
#: silently starts passing is visible.
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
    # whole point of the leg. The HANDBACK parent carries the mint's own
    # bucket — the pod that opened the mint always matches its own request.
    adopt_source=_micro_adopt_source_for(0),
    parent_pipe=_micro_parent_for(MICRO_LORA_BUCKET),
    covers=("the pgw#999 design question: a bucket-bearing cell offered to a "
            "parent on the plain lane"),
    expect="red",
    expect_note=("the arm gate refuses the exact named cell on lane drift — "
                 "the adopting pod must carry the cell's own lora bucket "
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
from gen_worker import aot_serve
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

# pgw#904: the adopting process is TOLD the exact cell (checkpoint id from
# the publish leg) — discovery is deleted, and a serving pod is told by
# `Arm.artifact` the same way.
from harness.rig_fetch import fetch_named_cell as _fetch_cell
from types import SimpleNamespace as _CellNS
import hashlib as _hl
from gen_worker import compile_cache as _syscc
try:
    _art = _fetch_cell(%(base)r, cfg.family, %(checkpoint)r, Path(%(cache)r))
except Exception as _exc:
    print("rig-fetch: named cell unavailable: %%s" %% (_exc,), file=sys.stderr)
    cell = None
else:
    _key0 = str(aot_serve.unpack_metadata(_art).get("compiled_graph_key") or "")
    cell = _CellNS(
        artifact=_art, compiled_graph_key=_key0, family=cfg.family,
        ref=_syscc.system_repo(cfg.family) + "#" + _key0,
        snapshot_digest="sha256:" + _hl.sha256(_art.read_bytes()).hexdigest())
out = {"pid": os.getpid(), "ok": cell is not None}
if cell is not None:
    meta = aot_serve.unpack_metadata(Path(cell.artifact))
    out.update({
        "compiled_graph_key": cell.compiled_graph_key, "family": cell.family, "ref": cell.ref,
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


def _micro_4d_adopt_source(base: str, cache: Path, checkpoint: str) -> str:
    return _MICRO_4D_ADOPT % {
        "paths": [str(REPO / "tests"), str(REPO / "src"), str(MICRO_SRC)],
        "base": base, "cache": str(cache), "checkpoint": checkpoint}


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
    parent_pipe=_micro_parent_for(0, "MicroGridPipeline", _micro_4d_cell),
    covers=("pgw#998's shape — a 4-D latent with BOTH spatial axes dynamic, so "
            "every matmul's M extent is NONLINEAR in the traced symbols. This "
            "is z-image's declaration, and the seam that was unlowerable "
            "across the mint's export save/load hand-off"),
)




# ---------------------------------------------------------------------------
# micro-lora16 — pgw#1073: a SECOND rank bucket. The bucket is a KEY axis
# (`<base>-lora<bucket>`), so two buckets are two lanes and two cells; a
# single standing bucket (64) means the axis itself is never varied. This
# member re-proves that the lane label, the branch allocation and the arm
# gate all follow the NUMBER rather than merely following "lora on".
#
# 16, not 8: `RANK_BUCKETS = (16, 32, 64, 128)` — the bucket vocabulary is
# quantized and 8 is NOT a member. The campaign's first cut used 8 and the
# mint child died in `enable_lora_branches` — a correct, typed
# ValidationError that surfaced as an UNCLASSIFIED child crash (the pgw#999
# class one layer down; recorded in pgw#1073's ledger).
# ---------------------------------------------------------------------------

MICRO_LORA16_BUCKET = 16

MICRO_LORA16 = Vehicle(
    name="micro-lora16",
    modules=(RUNTIME_HOOK, "micro_diffusion.main"),
    function="generate",
    family="micro-diffusion",
    ref_path="cozy/micro-diffusion",
    syspath=(str(MICRO_SRC),),
    build_checkpoint=_micro_checkpoint,
    checkpoint_bytes=_micro_checkpoint_bytes,
    compile_cell=lambda: _micro_cell(MICRO_LORA16_BUCKET),
    adopt_source=_micro_adopt_source_for(MICRO_LORA16_BUCKET),
    parent_pipe=_micro_parent_for(MICRO_LORA16_BUCKET),
    covers=(f"pgw#1073: everything `micro-lora` covers at a SECOND rank "
            f"bucket (lora_bucket={MICRO_LORA16_BUCKET}) — the bucket is a "
            f"key axis, so this cell's lane is `lora16`, disjoint from "
            f"`lora64`'s, and the arm gate must follow the number"),
)


# ---------------------------------------------------------------------------
# micro-conv — pgw#1073: the STATIC-ROWS class (sdxl's strategy) at micro
# scale. Conv-bearing, 4 static entries (2 rows x cfg fork), an int64
# timestep (mixed dtype, wan-2.2's shape), deeper module nesting, a named
# persistent buffer (the H3 pattern).
# ---------------------------------------------------------------------------


def _micro_conv_cell() -> Any:
    from gen_worker.registry import CompileCell

    from micro_diffusion.aot_declaration_conv import COND_LEN, PIXEL_ROWS

    return CompileCell(
        shapes=PIXEL_ROWS, targets=("unet",), family="micro-conv",
        regional=False, text_len=COND_LEN, dynamic=(), lora_bucket=0,
        guidance_scales=(), text_lens=())


_MICRO_CONV_ADOPT = '''
import json, logging, os, sys
for p in %(paths)r:
    sys.path.insert(0, p)
logging.basicConfig(level=logging.INFO, stream=sys.stderr)
import harness.rig_runtime  # noqa: F401
import torch
from pathlib import Path
from gen_worker import aot_serve
from gen_worker.models import provision
from gen_worker.registry import CompileCell
from micro_diffusion.aot_declaration_conv import (
    CFG_ARITY, COND_LEN, LATENT_ROWS, PIXEL_ROWS)
from micro_diffusion.model_conv import NUM_TRAIN_TIMESTEPS
from micro_diffusion.pipeline import MicroConvPipeline

torch.set_num_threads(2)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cfg = CompileCell(
    shapes=PIXEL_ROWS, targets=("unet",), family="micro-conv",
    regional=False, text_len=COND_LEN, dynamic=(), lora_bucket=0,
    guidance_scales=(), text_lens=())
pipe = MicroConvPipeline.from_pretrained(os.environ["PGW978_CHECKPOINT"]).to(DEVICE)
ref = MicroConvPipeline.from_pretrained(os.environ["PGW978_CHECKPOINT"]).to(DEVICE)
config = pipe.config


def _feed(batch, grid):
    gen = torch.Generator().manual_seed(1073)
    sample = torch.randn(
        batch, config.in_channels, grid, grid, generator=gen).to(DEVICE)
    timestep = torch.randint(
        0, NUM_TRAIN_TIMESTEPS, (batch,), generator=gen).to(DEVICE)
    cond = torch.randn(
        batch, COND_LEN, config.cond_dim, generator=gen).to(DEVICE)
    return sample, timestep, cond


# THREE of the four static entries, spanning both declared axes (row and
# fork). static-rows means each (row, cfg) coordinate is its own entry — the
# serve side must land each call on ITS entry, which is exactly the
# label-vs-ask identity pgw#1058 broke on.
ARMS = [("unet/cfg=true@24", CFG_ARITY, LATENT_ROWS[0]),
        ("unet/cfg=false@24", 1, LATENT_ROWS[0]),
        ("unet/cfg=true@32", CFG_ARITY, LATENT_ROWS[1])]
eager = {}
with torch.no_grad():
    for name, batch, grid in ARMS:
        sample, timestep, cond = _feed(batch, grid)
        eager[name] = ref.unet(sample, timestep, cond).clone()

from harness.rig_fetch import fetch_named_cell as _fetch_cell
from types import SimpleNamespace as _CellNS
import hashlib as _hl
from gen_worker import compile_cache as _syscc
try:
    _art = _fetch_cell(%(base)r, cfg.family, %(checkpoint)r, Path(%(cache)r))
except Exception as _exc:
    print("rig-fetch: named cell unavailable: %%s" %% (_exc,), file=sys.stderr)
    cell = None
else:
    _key0 = str(aot_serve.unpack_metadata(_art).get("compiled_graph_key") or "")
    cell = _CellNS(
        artifact=_art, compiled_graph_key=_key0, family=cfg.family,
        ref=_syscc.system_repo(cfg.family) + "#" + _key0,
        snapshot_digest="sha256:" + _hl.sha256(_art.read_bytes()).hexdigest())
out = {"pid": os.getpid(), "ok": cell is not None}
if cell is not None:
    meta = aot_serve.unpack_metadata(Path(cell.artifact))
    out.update({
        "compiled_graph_key": cell.compiled_graph_key, "family": cell.family, "ref": cell.ref,
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
        rel = {}
        with torch.no_grad():
            for name, batch, grid in ARMS:
                sample, timestep, cond = _feed(batch, grid)
                got = pipe.unet(sample, timestep, cond)
                deltas[name] = float((got - eager[name]).abs().max())
                rel[name] = float((got - eager[name]).norm()
                                  / eager[name].norm().clamp_min(1e-12))
        out["parity_max_abs"] = deltas
        out["parity_rel_l2"] = rel
        # THE BOUND, PER DEVICE — measured, not picked (pgw#1073 run 1).
        # CPU: compiled conv matches eager to 3.3e-06, so 1e-4 abs holds.
        # GPU: the SAME fp32 graph differs by 1.2-1.6e-3 max|delta| — the
        # fp32-conv kernel-numerics class (TF32/algorithm choice diverging
        # between inductor's triton convs and eager cudnn), which production
        # gates with a COSINE floor, not max-abs. The GPU instrument here is
        # relative L2: measured kernel noise ~1e-3; a real bf16 cast would
        # be ~1e-2 and still fails.
        if DEVICE == "cuda":
            out["parity_ok"] = all(v <= 3e-3 for v in rel.values())
        else:
            out["parity_ok"] = all(v <= 1e-4 for v in deltas.values())
        out["served_entry_calls"] = dict(aot_serve.served_entry_calls(pipe))
        out["execution_count"] = int(aot_serve.execution_count(pipe))
        out["ok"] = bool(out["parity_ok"]) and out["execution_count"] > 0
    else:
        out["ok"] = False
print("RIG_ADOPT " + json.dumps(out))
'''


def _micro_conv_adopt_source(base: str, cache: Path, checkpoint: str) -> str:
    return _MICRO_CONV_ADOPT % {
        "paths": [str(REPO / "tests"), str(REPO / "src"), str(MICRO_SRC)],
        "base": base, "cache": str(cache), "checkpoint": checkpoint}


MICRO_CONV = Vehicle(
    name="micro-conv",
    modules=(RUNTIME_HOOK, "micro_diffusion.main_conv"),
    function="generate-conv",
    family="micro-conv",
    ref_path="cozy/micro-diffusion",
    syspath=(str(MICRO_SRC),),
    build_checkpoint=_micro_checkpoint,
    checkpoint_bytes=_micro_checkpoint_bytes,
    compile_cell=_micro_conv_cell,
    adopt_source=_micro_conv_adopt_source,
    parent_pipe=_micro_parent_for(0, "MicroConvPipeline", _micro_conv_cell),
    covers=("pgw#1073: the STATIC-ROWS class at micro scale — conv-bearing "
            "(so #730 forces the strategy), 4 static entries (2 rows x cfg "
            "fork, the sdxl generator), an INT64 timestep indexing an "
            "embedding (mixed dtype, wan-2.2's shape), three-deep module "
            "nesting, and a named PERSISTENT buffer (the H3 pattern, the "
            "other half of pgw#857's literal seam)"),
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
from gen_worker import aot_serve, compile_cache as cc
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

# pgw#904: the adopting process is TOLD the exact cell (checkpoint id from
# the publish leg) — discovery is deleted, and a serving pod is told by
# `Arm.artifact` the same way.
from harness.rig_fetch import fetch_named_cell as _fetch_cell
from types import SimpleNamespace as _CellNS
import hashlib as _hl
from gen_worker import compile_cache as _syscc
try:
    _art = _fetch_cell(%(base)r, cfg.family, %(checkpoint)r, Path(%(cache)r))
except Exception as _exc:
    print("rig-fetch: named cell unavailable: %%s" %% (_exc,), file=sys.stderr)
    cell = None
else:
    _key0 = str(aot_serve.unpack_metadata(_art).get("compiled_graph_key") or "")
    cell = _CellNS(
        artifact=_art, compiled_graph_key=_key0, family=cfg.family,
        ref=_syscc.system_repo(cfg.family) + "#" + _key0,
        snapshot_digest="sha256:" + _hl.sha256(_art.read_bytes()).hexdigest())
out = {"pid": os.getpid(), "ok": cell is not None}
if cell is not None:
    meta = aot_serve.unpack_metadata(Path(cell.artifact))
    out.update({
        "compiled_graph_key": cell.compiled_graph_key, "family": cell.family, "ref": cell.ref,
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
    def _source(base: str, cache: Path, checkpoint: str) -> str:
        return _MICRO_W8A8_ADOPT % {
            "paths": [str(REPO / "tests"), str(REPO / "src"), str(MICRO_SRC)],
            "base": base, "cache": str(cache), "bucket": int(bucket),
            "checkpoint": checkpoint}

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
        parent_pipe=_micro_parent_for(bucket, "MicroW8a8Pipeline"),
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


# ---------------------------------------------------------------------------
# micro-escape — pgw#1062's standing member: author-defined ops (the pgw#1059
# amendment-7 escape hatch) through the whole mint. GPU-only: inductor lowers
# a Triton impl on every backend that declares one and CPU declares none
# (measured: `0 compatible backends for target (cpu)`).
# ---------------------------------------------------------------------------


def _micro_escape_cell() -> Any:
    from gen_worker.registry import CompileCell

    from micro_diffusion.aot_declaration_escape import COND_LEN, PIXEL_ROWS

    return CompileCell(
        shapes=PIXEL_ROWS, targets=("transformer",), family="micro-escape",
        regional=False, text_len=COND_LEN, dynamic=(), lora_bucket=0,
        guidance_scales=(), text_lens=())


_MICRO_ESCAPE_ADOPT = '''
import json, logging, os, sys
for p in %(paths)r:
    sys.path.insert(0, p)
logging.basicConfig(level=logging.INFO, stream=sys.stderr)
import harness.rig_runtime  # noqa: F401
import torch
from pathlib import Path
from gen_worker import aot_serve
from gen_worker.models import provision
from gen_worker.registry import CompileCell
from micro_diffusion.aot_declaration_escape import (
    ARITY, COND_LEN, PIXEL_ROWS, TOKEN_ROWS)
from micro_diffusion.pipeline import MicroEscapePipeline

torch.set_num_threads(2)
DEVICE = "cuda"  # gpu_only vehicle: the ops in the graph require the card
cfg = CompileCell(
    shapes=PIXEL_ROWS, targets=("transformer",), family="micro-escape",
    regional=False, text_len=COND_LEN, dynamic=(), lora_bucket=0,
    guidance_scales=(), text_lens=())
pipe = MicroEscapePipeline.from_pretrained(os.environ["PGW978_CHECKPOINT"]).to(DEVICE)
ref = MicroEscapePipeline.from_pretrained(os.environ["PGW978_CHECKPOINT"]).to(DEVICE)
config = pipe.config


def _feed(tokens):
    gen = torch.Generator().manual_seed(1062)
    x = [torch.randn(tokens, config.in_channels, generator=gen).to(DEVICE)
         for _ in range(ARITY)]
    t = torch.full((ARITY,), 100.0, device=DEVICE)
    cond = [torch.randn(COND_LEN, config.cond_dim, generator=gen).to(DEVICE)
            for _ in range(ARITY)]
    return x, t, cond


# The row the artifact was NOT seeded on, so the derived range is exercised —
# through the custom-op fallback, the triton_op kernel and the raw HOP.
ARMS = [("transformer", TOKEN_ROWS[0])]
eager = {}
with torch.no_grad():
    for name, tokens in ARMS:
        x, t, cond = _feed(tokens)
        eager[name] = ref.transformer(x, t, cond).clone()

from harness.rig_fetch import fetch_named_cell as _fetch_cell
from types import SimpleNamespace as _CellNS
import hashlib as _hl
from gen_worker import compile_cache as _syscc
try:
    _art = _fetch_cell(%(base)r, cfg.family, %(checkpoint)r, Path(%(cache)r))
except Exception as _exc:
    print("rig-fetch: named cell unavailable: %%s" %% (_exc,), file=sys.stderr)
    cell = None
else:
    _key0 = str(aot_serve.unpack_metadata(_art).get("compiled_graph_key") or "")
    cell = _CellNS(
        artifact=_art, compiled_graph_key=_key0, family=cfg.family,
        ref=_syscc.system_repo(cfg.family) + "#" + _key0,
        snapshot_digest="sha256:" + _hl.sha256(_art.read_bytes()).hexdigest())
out = {"pid": os.getpid(), "ok": cell is not None}
if cell is not None:
    meta = aot_serve.unpack_metadata(Path(cell.artifact))
    out.update({
        "compiled_graph_key": cell.compiled_graph_key, "family": cell.family, "ref": cell.ref,
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
            for name, tokens in ARMS:
                x, t, cond = _feed(tokens)
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


def _micro_escape_adopt_source(base: str, cache: Path, checkpoint: str) -> str:
    return _MICRO_ESCAPE_ADOPT % {
        "paths": [str(REPO / "tests"), str(REPO / "src"), str(MICRO_SRC)],
        "base": base, "cache": str(cache), "checkpoint": checkpoint}


MICRO_ESCAPE = Vehicle(
    name="micro-escape",
    modules=(RUNTIME_HOOK, "micro_diffusion.main_escape"),
    function="generate-escape",
    family="micro-escape",
    ref_path="cozy/micro-diffusion",
    syspath=(str(MICRO_SRC),),
    build_checkpoint=_micro_checkpoint,
    checkpoint_bytes=_micro_checkpoint_bytes,
    compile_cell=_micro_escape_cell,
    adopt_source=_micro_escape_adopt_source,
    parent_pipe=_micro_parent_for(0, "MicroEscapePipeline", _micro_escape_cell),
    gpu_only=True,
    covers=("pgw#1062 (pgw#1059 amendment 7's escape hatch): author-defined "
            "ops through the WHOLE mint — a torch.library custom op with a "
            "fake kernel, a triton_op'd hand-written Triton kernel, and a raw "
            "@triton.jit call, all inside the traced graph. RED proof: drop "
            "the custom op's register_fake and the mint child refuses at "
            "export"),
)


# ---------------------------------------------------------------------------
# micro-pad32 — ie#637's SURVIVING WATCH ITEM, and the gate on the z-image buy.
#
# The token extent is `32*FloorDiv(H*W+31, 32)`: a pad-to-32 over a token count
# that is itself a PRODUCT of two declared symbols. ie#637 proved off-GPU that
# this spelling carries no pinning guard; what no run has ever reached is
# whether AOTInductor CODEGENS it correctly through export -> compile -> load ->
# serve. Two z-image pods died before that phase (the gate, then ie#638's VRAM),
# so it costs a rented A100 to ask there and seconds to ask here.
#
# The RED twin is not decoration. Without it a green here would prove only that
# this graph never reaches the declared-range gate.
# ---------------------------------------------------------------------------


def _declaration_module(module: str, family: str) -> Any:
    """Import a family's declaration module AND GUARANTEE the registration.

    The registration is an import SIDE EFFECT (``register_export_declaration``
    at module scope), so the second import of the same module is a no-op — and
    dozens of test modules call ``reset_export_declarations()``, which empties
    the registry that import populated. The two compose into an order-dependent
    silent failure: ``compile_cell()`` returns a cell whose family nothing has
    declared, and it surfaces much later and elsewhere as
    ``export_declaration(family) -> None``.

    Asking the registry instead of trusting the import is what makes it
    order-free. Re-executing is safe because the declaration modules register
    with ``replace=True``.
    """
    mod = importlib.import_module(module)
    from gen_worker.api.export_contract import export_declaration

    if export_declaration(family) is None:
        mod = importlib.reload(mod)
    return mod


def _micro_pad32_cell(family: str = "micro-pad32") -> Any:
    """The cell, AND the declaration registration the parent needs.

    Importing the family's declaration module is the registration (it calls
    `register_export_declaration` at import), and the parent-side numerics gate
    builds its probe feed from that contract. A variant that imported the OTHER
    family's module minted fine and then refused to arm with
    `no_input_contract` — measured, and the reason this takes a family rather
    than reading one module.
    """
    from gen_worker.registry import CompileCell

    mod = _declaration_module(
        "micro_diffusion.aot_declaration_pad32_branchy"
        if family == "micro-pad32-branchy"
        else "micro_diffusion.aot_declaration_pad32",
        family)

    return CompileCell(
        shapes=mod.PIXEL_ROWS, targets=("transformer",), family=family,
        regional=False, text_len=mod.COND_LEN, dynamic=(), lora_bucket=0,
        guidance_scales=(), text_lens=())


_MICRO_PAD32_ADOPT = '''
import json, logging, os, sys
for p in %(paths)r:
    sys.path.insert(0, p)
logging.basicConfig(level=logging.INFO, stream=sys.stderr)
import harness.rig_runtime  # noqa: F401
import torch
from pathlib import Path
from gen_worker import aot_serve
from gen_worker.models import provision
from gen_worker.registry import CompileCell
from %(decl)s import (
    ARITY, COND_LEN, LATENT_ROWS, PIXEL_ROWS, UNDECLARED_ROW)
from micro_diffusion.model_pad32 import SEQ_MULTIPLE_OF, padded_length
from micro_diffusion.pipeline import %(cls)s

torch.set_num_threads(2)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cfg = CompileCell(
    shapes=PIXEL_ROWS, targets=("transformer",), family=%(family)r,
    regional=False, text_len=COND_LEN, dynamic=(), lora_bucket=0,
    guidance_scales=(), text_lens=())
pipe = %(cls)s.from_pretrained(os.environ["PGW978_CHECKPOINT"]).to(DEVICE)
config = pipe.config


def _feed(grid):
    gen = torch.Generator().manual_seed(637)
    x = [torch.randn(config.in_channels, grid, grid, generator=gen).to(DEVICE)
         for _ in range(ARITY)]
    t = torch.full((ARITY,), 100.0, device=DEVICE)
    cond = [torch.randn(COND_LEN, config.cond_dim, generator=gen).to(DEVICE)
            for _ in range(ARITY)]
    return x, t, cond


# THE POINT OF THIS MEMBER: three L values in one served artifact, whose pads
# are 16, 28 and 0 — three DIFFERENT padded lengths. A graph that decided the
# pad once at trace serves at most one of them, and the middle row is not even
# a declared row, so it can only come from the derived range.
ARMS = [("transformer", LATENT_ROWS[0]),
        ("transformer-undeclared", UNDECLARED_ROW),
        ("transformer-row2", LATENT_ROWS[1])]
pads = {}
for name, grid in ARMS:
    tokens = grid * grid
    pads[name] = {"grid": grid, "tokens": tokens,
                  "padded": padded_length(tokens),
                  "pad": padded_length(tokens) - tokens}
eager = {}
with torch.no_grad():
    for name, grid in ARMS:
        x, t, cond = _feed(grid)
        eager[name] = pipe.transformer(x, t, cond).clone()

from harness.rig_fetch import fetch_named_cell as _fetch_cell
from types import SimpleNamespace as _CellNS
import hashlib as _hl
from gen_worker import compile_cache as _syscc
try:
    _art = _fetch_cell(%(base)r, cfg.family, %(checkpoint)r, Path(%(cache)r))
except Exception as _exc:
    print("rig-fetch: named cell unavailable: %%s" %% (_exc,), file=sys.stderr)
    cell = None
else:
    _key0 = str(aot_serve.unpack_metadata(_art).get("compiled_graph_key") or "")
    cell = _CellNS(
        artifact=_art, compiled_graph_key=_key0, family=cfg.family,
        ref=_syscc.system_repo(cfg.family) + "#" + _key0,
        snapshot_digest="sha256:" + _hl.sha256(_art.read_bytes()).hexdigest())
out = {"pid": os.getpid(), "ok": cell is not None, "pad_classes": pads,
       "distinct_pads": sorted({v["pad"] for v in pads.values()})}
if cell is not None:
    meta = aot_serve.unpack_metadata(Path(cell.artifact))
    out.update({
        "compiled_graph_key": cell.compiled_graph_key, "family": cell.family, "ref": cell.ref,
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
        # A cell that served only ONE of the three rows is not a pass, however
        # small its delta on that row: the whole question is the OTHER rows.
        out["ok"] = (bool(out["parity_ok"])
                     and out["execution_count"] >= len(ARMS)
                     and len(out["distinct_pads"]) >= 2)
    else:
        out["ok"] = False
print("RIG_ADOPT " + json.dumps(out))
'''


def _micro_pad32_adopt_source_for(family: str) -> Any:
    branchy = family == "micro-pad32-branchy"

    def _source(base: str, cache: Path, checkpoint: str) -> str:
        return _MICRO_PAD32_ADOPT % {
            "paths": [str(REPO / "tests"), str(REPO / "src"), str(MICRO_SRC)],
            "base": base, "cache": str(cache), "checkpoint": checkpoint,
            "family": family,
            "decl": ("micro_diffusion.aot_declaration_pad32_branchy" if branchy
                     else "micro_diffusion.aot_declaration_pad32"),
            "cls": ("MicroPad32BranchyPipeline" if branchy
                    else "MicroPad32Pipeline"),
        }

    return _source


MICRO_PAD32 = Vehicle(
    name="micro-pad32",
    modules=(RUNTIME_HOOK, "micro_diffusion.main_pad32"),
    function="generate-pad32",
    family="micro-pad32",
    ref_path="cozy/micro-diffusion",
    syspath=(str(MICRO_SRC),),
    build_checkpoint=_micro_checkpoint,
    checkpoint_bytes=_micro_checkpoint_bytes,
    compile_cell=_micro_pad32_cell,
    adopt_source=_micro_pad32_adopt_source_for("micro-pad32"),
    parent_pipe=_micro_parent_for(0, "MicroPad32Pipeline", _micro_pad32_cell),
    covers=("ie#637's surviving watch item — a dynamic dim carrying the exact "
            "pad-expression class `32*FloorDiv(L+31,32)` over a token count "
            "that is itself a product of two declared symbols, served at three "
            "L values whose pads are 16, 28 and 0. Answers whether AOTI "
            "CODEGENS that expression, which no z-image pod has reached"),
)


MICRO_PAD32_BRANCHY = Vehicle(
    name="micro-pad32-branchy",
    modules=(RUNTIME_HOOK, "micro_diffusion.main_pad32_branchy"),
    function="generate-pad32-branchy",
    family="micro-pad32-branchy",
    ref_path="cozy/micro-diffusion",
    syspath=(str(MICRO_SRC),),
    build_checkpoint=_micro_checkpoint,
    checkpoint_bytes=_micro_checkpoint_bytes,
    compile_cell=lambda: _micro_pad32_cell("micro-pad32-branchy"),
    adopt_source=_micro_pad32_adopt_source_for("micro-pad32-branchy"),
    parent_pipe=_micro_parent_for(
        0, "MicroPad32BranchyPipeline",
        lambda: _micro_pad32_cell("micro-pad32-branchy")),
    expect="green",
    expect_note="",
    covers=("the CONTROL for `micro-pad32` — same declaration, same rows, "
            "upstream's `(-L) %% 32` + `if pad > 0` spelling. Measured "
            "2026-08-10: it is NOT refused, and that is pgw#1077 working. "
            "The `Eq(PythonMod(-L,32), 0)` axioms ie#637 was refused on are "
            "all recorded REFUTED here (test_pad32_codegen_pgw1079), which is "
            "exactly what pgw#1077 taught the gate to evaluate instead of "
            "count. A RED here means the gate started over-refusing again"),
)


# ---------------------------------------------------------------------------
# micro-rope — pgw#1080's RED CONTROL, and the reason ie#628 widened the
# meta-instantiation gate from `__init__` to CALL time.
#
# The denoiser's frequency table is upstream z-image's shape: a plain object
# holding None, built on first use inside `with torch.device("cpu")`. The base
# `micro` vehicle is the GREEN twin — ie#630's registered buffer, no device
# pin — and it mints green in the same gauntlet. The pair is the control: a
# gate that only ever fires, or only ever stays silent, has proved nothing.
# ---------------------------------------------------------------------------


def _micro_rope_cell() -> Any:
    from gen_worker.registry import CompileCell

    from micro_diffusion.aot_declaration_rope import COND_LEN, PIXEL_ROWS

    return CompileCell(
        shapes=PIXEL_ROWS, targets=("transformer",), family="micro-rope",
        regional=False, text_len=COND_LEN, dynamic=(), lora_bucket=0,
        guidance_scales=(), text_lens=())


_MICRO_ROPE_ADOPT = """
import json, os, sys
out = {"pid": os.getpid(), "ok": False,
       "detail": "micro-rope is a REFUSAL vehicle: the mint never publishes a "
                 "cell, so there is nothing to adopt"}
print("RIG_ADOPT " + json.dumps(out))
"""


def _micro_rope_adopt_source(base: str, cache: Path, checkpoint: str) -> str:
    return _MICRO_ROPE_ADOPT


MICRO_ROPE = Vehicle(
    name="micro-rope",
    modules=(RUNTIME_HOOK, "micro_diffusion.main_rope"),
    function="generate-rope",
    family="micro-rope",
    ref_path="cozy/micro-diffusion",
    syspath=(str(MICRO_SRC),),
    build_checkpoint=_micro_checkpoint,
    checkpoint_bytes=_micro_checkpoint_bytes,
    compile_cell=_micro_rope_cell,
    adopt_source=_micro_rope_adopt_source,
    expect="red",
    expect_note=(
        "the denoiser builds its table lazily under `with torch.device('cpu')`"
        ". MEASURED 2026-08-10: inside a fake-mode export that allocation is "
        "itself FAKE, so the MODE-based half of the gate cannot see it — the "
        "residue can. The warm proof runs the handler for real, the lazy "
        "build fires, and the child refuses naming "
        "`transformer.rope.freqs_cis` and the authoring fix. The base "
        "`micro` row is the green half of the same control"),
    covers=("pgw#1080 / ie#628 RED CONTROL: upstream z-image's lazy "
            "CPU-pinned rope table, transcribed. Clean __init__, violation "
            "mid-forward. Green twin: the `micro` row, whose table is a "
            "registered buffer (ie#630)"),
)


VEHICLES: Dict[str, Vehicle] = {
    v.name: v for v in (TINY, MICRO, MICRO_LORA, MICRO_LORA16,
                        MICRO_LORA_PLAIN_PARENT, MICRO_4D, MICRO_CONV,
                        MICRO_ESCAPE, MICRO_W8A8, MICRO_W8A8_LORA,
                        MICRO_PAD32, MICRO_PAD32_BRANCHY, MICRO_ROPE)}
DEFAULT_VEHICLE = TINY.name


def vehicle(name: str) -> Vehicle:
    try:
        return VEHICLES[str(name)]
    except KeyError:
        raise SystemExit(
            f"unknown rig vehicle {name!r} (known: {sorted(VEHICLES)!r})")


__all__ = ["DEFAULT_VEHICLE", "MICRO", "MICRO_CONV", "MICRO_ESCAPE",
           "MICRO_LORA", "MICRO_LORA16", "MICRO_LORA16_BUCKET",
           "MICRO_LORA_BUCKET", "MICRO_4D", "MICRO_LORA_PLAIN_PARENT",
           "MICRO_PAD32", "MICRO_PAD32_BRANCHY", "MICRO_ROPE",
           "MICRO_W8A8",
           "MICRO_W8A8_LORA", "TINY", "VEHICLES", "Vehicle", "vehicle"]
