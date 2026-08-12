"""A REAL packed exported cell: real declaration, real envelope, real tensors.

Promoted out of ``test_numerics_gate_pgw868.py`` by pgw#1152, unchanged. It was
already a harness in everything but location — three other modules imported it
as ``rig868`` — and the adopt-path rig (:mod:`harness.adopt_rig`) needs the same
artifact, so the shared thing now lives where shared things live.

What is REAL here: the ``Compile`` declaration, the packed ``cell.tar.gz`` with
its recorded entry blocks, the class/range digests, the constants manifest, and
a genuine ``nn.Module`` whose eager output the compiled subject is compared
against. The ONE substitution is :class:`ProbePackage`, which stands in for an
entry's ``AOTICompiledModel`` — an AOTI ``.so`` needs a GPU, and it is the only
piece deferred to a pod. It reproduces the eager maths from the constants it was
BOUND with, so the comparison is genuinely compiled-vs-eager on identical
weights, then rotates the result to an exactly declared cosine.

No number produced here may be cited as evidence about a real cell's numerics.
"""

from __future__ import annotations

import math
import platform
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest
import torch

from gen_worker import aot_serve as aot
from gen_worker.api.decorators import Compile
from gen_worker.api.export_contract import (
    Dim, GraphClass, Input, register_export_declaration,
    reset_export_declarations,
)

FAMILY = "pgw868-probe"
RUNTIME = {"sku": "l4", "sm": "sm_89", "torch": "2.13.0+cu130", "cuda": "13.0"}
TARGET = "denoiser"
#: Two declared shape rows -> two packaged entries. The second row is what
#: makes "one shape row" a real axis rather than a word in a docstring.
ROWS = ((8, 8), (16, 16))
#: sdxl's declared band, used verbatim: pgw#868 was a measurement plus wiring,
#: never a re-design of what "good" means.
FLOOR, WARN = 0.995, 0.999


# ---------------------------------------------------------------------------
# the subject — a calibrated blend, so a test can NAME the rung it aims at
# ---------------------------------------------------------------------------


def blend(reference: torch.Tensor, cosine: float) -> torch.Tensor:
    """``reference`` rotated to EXACTLY ``cosine``, at unchanged magnitude.

    Gram-Schmidt against a fixed ramp: the perturbation is deterministic and
    the resulting cosine is analytic, so a threshold test asserts the ladder's
    boundary rather than a tuned fudge factor.
    """
    flat = reference.reshape(-1).to(torch.float64)
    ramp = torch.linspace(-1.0, 1.0, flat.numel(), dtype=torch.float64)
    ramp = ramp - flat * (torch.dot(ramp, flat) / torch.dot(flat, flat))
    ramp = ramp / ramp.norm() * flat.norm()
    sin = math.sqrt(max(0.0, 1.0 - cosine * cosine))
    out = cosine * flat + sin * ramp
    return out.reshape(reference.shape).to(reference.dtype)


class ProbeDenoiser(torch.nn.Module):
    """The eager reference. Deterministic, tiny, and REAL: a genuine
    `nn.Module` whose signature the declaration is positionalized against."""

    def __init__(self, width: int = 16) -> None:
        super().__init__()
        # Built without the global RNG on purpose: the probe must be provable
        # not to disturb the serving generator, and a fixture that seeded it
        # would hide exactly that.
        self.weight = torch.nn.Parameter(
            torch.linspace(0.1, 3.0, width * width).reshape(width, width).sin())

    def forward(self, sample: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        return (sample @ self.weight[: sample.shape[-1], : sample.shape[-1]]) * timestep


class ProbePipeline:
    def __init__(self, module: torch.nn.Module) -> None:
        self.denoiser = module


class ProbePackage:
    """Stands in for one entry's `AOTICompiledModel` — the ONE deferred piece.

    Reproduces the eager maths from the constants it was BOUND with (so the
    comparison is genuinely compiled-vs-eager on identical weights) and then
    rotates the result to a declared cosine.
    """

    def __init__(self, cosine: float = 1.0, *, raises: str = "",
                 drop_output: bool = False, bind_oom: bool = False) -> None:
        self.cosine = float(cosine)
        self.raises = raises
        self.drop_output = drop_output
        #: pgw#1175: this entry's constant bind really runs the card out of
        #: memory. Breaking a REAL input (the bind is the device work an adopt
        #: does), never supplying a fact — the rig's own doctrine.
        self.bind_oom = bool(bind_oom)
        self.loaded: Dict[str, Any] = {}
        self.invocations = 0

    def get_constant_fqns(self) -> List[str]:
        return ["weight"]

    def load_constants(self, values: Dict[str, Any], check_full_update: bool = False,
                       **_kw: Any) -> None:
        if self.bind_oom:
            raise torch.OutOfMemoryError(
                "CUDA out of memory. Tried to allocate 2.00 GiB. GPU 0 has a "
                "total capacity of 47.54 GiB of which 1.88 MiB is free.")
        self.loaded = dict(values)

    def __call__(self, sample: torch.Tensor, timestep: torch.Tensor) -> Any:
        self.invocations += 1
        if self.raises:
            raise RuntimeError(self.raises)
        w = self.loaded["weight"]
        out = (sample @ w[: sample.shape[-1], : sample.shape[-1]]) * timestep
        if self.drop_output:
            return (out, out)
        return out if self.cosine >= 1.0 else blend(out, self.cosine)


# ---------------------------------------------------------------------------
# the declaration + the artifact — both real
# ---------------------------------------------------------------------------


def declaration(floor: float = FLOOR, warn: float = WARN) -> Compile:
    return Compile(
        family=FAMILY,
        targets=(TARGET,),
        dims=(Dim(name="h", carried_by=(("sample", 0),)),
              Dim(name="w", carried_by=(("sample", 1),))),
        classes=tuple(GraphClass(dims={"h": h, "w": w}) for h, w in ROWS),
        inputs=(Input(name="sample", shape=("h", "w"), dtype="float32"),
                Input(name="timestep", shape=(), dtype="float32", value=1.0)),
        shape_strategy="static-rows",
        numerics_floor=floor,
        numerics_warn=warn,
    )


def entry_name(h: int, w: int) -> str:
    return f"{TARGET}/h={h},w={w}"


def _entry(h: int, w: int) -> Dict[str, Any]:
    block = {
        "target": TARGET,
        "fork": [],
        "class_dims": [["h", h], ["w", w]],
        "inputs": [
            {"name": "sample", "position": 0, "dtype": "float32",
             "shape": [h, w]},
            {"name": "timestep", "position": 1, "dtype": "float32",
             "shape": []},
        ],
        "symbols": {},
        "constants": [{"fqn": "weight", "source": aot.SOURCE_STATE_DICT,
                       "dtype": "float32", "shape": [16, 16]}],
        "graph": {},
    }
    block["range_digest"] = aot.range_digest(block)
    block["class_hash"] = aot.class_hash(block, strict=True, lora_bucket=0)
    return block


def metadata(rows: Tuple[Tuple[int, int], ...] = ROWS) -> Dict[str, Any]:
    entries = {entry_name(h, w): _entry(h, w) for h, w in rows}
    meta = {
        "format": aot.ARTIFACT_FORMAT, "kind": aot.ARTIFACT_KIND, **RUNTIME,
        "family": FAMILY, "precision": "w8a8", "cell_key": "cell868",
        "entries": entries, "strict_export": True, "lora_bucket": 0,
        "package_constants_in_so": False, "constant_folding_fenced": True,
        "source_ref": "", "source_digest": "",
        # pgw#950: every mint stamps a host-ISA requirement, and a cell that
        # stamps none is refused rather than sniffed from the .pt2. Satisfiable
        # anywhere: this host's machine, no ISA level.
        "host_isa": {"machine": platform.machine(), "march": "", "simdlen": 0,
                     "level": ""},
    }
    meta["combined_graph_hash"] = aot.combined_graph_hash(
        b["class_hash"] for b in entries.values())
    return meta


def artifact(tmp_path: Path, meta: Dict[str, Any] | None = None) -> Path:
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    (work / aot.PACKAGE_NAME).write_bytes(b"\x00not-a-real-pt2")
    return aot.pack(work, tmp_path / "cell.tar.gz", meta or metadata())


@pytest.fixture
def declared() -> Any:
    reset_export_declarations()
    decl = declaration()
    register_export_declaration(decl, family=FAMILY, replace=True)
    yield decl
    reset_export_declarations()


@pytest.fixture
def events(monkeypatch: pytest.MonkeyPatch) -> List[Tuple[str, str, str]]:
    import gen_worker.activity as activity_mod

    said: List[Tuple[str, str, str]] = []
    monkeypatch.setattr(
        activity_mod, "emit_event",
        lambda kind, detail, **kw: said.append(
            (kind, detail, str(kw.get("phase", "")))))
    return said


def cell_cfg(decl: Any, **enrichments: Any) -> Any:
    """The object the FLEET hands the compile machinery, built its own way.

    pgw#1150 (second pass) / pgw#1152: **raw ``Compile`` never travels past the
    registry** — every production path hands a ``CompileCell``, and it is built
    by exactly one map, ``CompileCell.from_declaration``. A test that passes the
    raw declaration is testing a TYPE no fleet path constructs, which is the
    third variant of this repo's fixture defect class: deleting the two
    ``numerics_floor=`` lines from ``registry.py`` left the old gate suite green
    because every one of its rows passed a shape production never passes.
    """
    from gen_worker.registry import CompileCell

    return CompileCell.from_declaration(decl, **enrichments)


def arm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, decl: Any,
        packages: Dict[str, ProbePackage],
        meta: Dict[str, Any] | None = None,
        verify_numerics: bool = True,
        cfg: Any = None) -> Tuple[Any, Any, Any]:
    """Drive the REAL arm path and return ``(pipeline, module, outcome)``.

    ``cfg`` defaults to :func:`cell_cfg` — the ``CompileCell`` production
    builds. Pass one of :func:`harness.adopt_rig.production_cfgs`' values to
    parametrise a row over every production call site.

    ``verify_numerics=True`` is the MINT arm (pgw#1141 / DESIGN-RULINGS §4.32):
    the gate runs on the pod that minted the bytes, before it publishes them,
    and nowhere else. Adoption passes False and runs no gate — that direction
    belongs to :mod:`harness.adopt_rig`, which enters through the executor.

    pgw#923: the arm returns a typed :class:`AdoptOutcome` rather than a bool,
    so its verdict — armed, or refused with the classified reason — is a value
    the caller can assert on and the executor can put on the wire. It stays
    truthy/falsy, so `assert outcome` reads exactly as `assert armed` did.
    """
    from gen_worker.models import provision

    monkeypatch.setattr(aot, "runtime_key", lambda: dict(RUNTIME))
    monkeypatch.setattr(
        aot, "_entry_admission_drift", lambda *a, **k: None)
    monkeypatch.setattr(
        aot, "_load_package", lambda path, entry="model": packages[entry])
    module = ProbeDenoiser()
    pipeline = ProbePipeline(module)
    outcome = provision.arm_aot(
        pipeline, cell_cfg(decl) if cfg is None else cfg,
        tmp_path / "cache", artifact(tmp_path, meta), 0,
        verify_numerics=verify_numerics)
    return pipeline, module, outcome


def numerics_rows(said: List[Tuple[str, str, str]]) -> List[Tuple[str, str]]:
    import gen_worker.activity as activity_mod

    return [(detail, phase) for kind, detail, phase in said
            if kind == activity_mod.KIND_CELL_NUMERICS]
