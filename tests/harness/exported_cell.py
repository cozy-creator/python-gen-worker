"""A TCG-backed exported graph probe: real declaration and real tensors.

Promoted out of ``test_numerics_gate_pgw868.py`` by pgw#1152, unchanged. It was
already a harness in everything but location — three other modules imported it
as ``rig868`` — and the adopt-path rig (:mod:`harness.adopt_rig`) needs the same
artifact, so the shared thing now lives where shared things live.

What is REAL here: the ``Compile`` declaration, public TCG metadata and ingress,
the constants manifest, and a genuine ``nn.Module`` whose eager output the
compiled subject is compared against. The substitutions are a tiny Engine and
:class:`ProbePackage`, which stands in for an entry's GPU-built AOTI payload.
The worker still exercises its real import/resolve/runner/bind/dispatch path;
the harness does not recreate the deleted worker-owned package format.

No number produced here may be cited as evidence about a real cell's numerics.
"""

from __future__ import annotations

import hashlib
import math
import platform
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast
from types import SimpleNamespace

import pytest
import torch
from gen_worker._vendor.torch_compiled_graphs import (
    CallIngress,
    CallInput,
    CompiledGraphRunner,
    ConstantBindingError,
    StoreOutcome,
)

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
    ingress = CallIngress(
        parameters=("sample", "timestep"),
        flat_arity=2,
        inputs=(
            CallInput(
                "sample", 0, "sample", 0, (), "sample", "float32", (h, w)
            ),
            CallInput(
                "timestep", 1, "timestep", 1, (), "timestep", "float32", ()
            ),
        ),
    )
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
        "constants": [{"fqn": "weight", "source": "state_dict",
                       "dtype": "float32", "shape": [16, 16]}],
        "graph": {
            "pytree": {"ingress": ingress.as_dict()},
            "constant_fqns": ["weight"],
        },
    }
    block["range_digest"] = ingress.digest()
    block["class_hash"] = hashlib.sha256(
        f"{TARGET}\0{h}\0{w}\0{block['range_digest']}".encode("utf-8")
    ).hexdigest()[:16]
    return block


#: The toolchain block every entry of this declaration shares. Recorded because
#: the ``toolchain`` axis is REQUIRED (pgw#1176): an entry that cannot state it
#: has no identity, so a fixture that omitted it would be un-keyable and could
#: never be published, resolved or armed — a shape production cannot produce.
TOOLCHAIN = {"torch": "2.13.0+cu130-record", "triton": "3.6.0-record"}


def metadata(
    row: Tuple[int, int] = ROWS[0],
    rows: Tuple[Tuple[int, int], ...] = ROWS,
) -> Dict[str, Any]:
    """Public TCG metadata for one graph class.

    ``rows`` is accepted for the historical harness call shape but deliberately
    does not enter this class's identity: TCG keys one declared graph class,
    not a worker-owned multi-entry package.
    """
    del rows
    block = _entry(*row)
    name = entry_name(*row)
    key = "cg-key-v1-" + hashlib.sha256(
        f"{FAMILY}\0{name}\0{block['class_hash']}".encode("utf-8")
    ).hexdigest()[:56]
    return {
        "compiled_graph_format": 1,
        "kind": "aot-inductor",
        "compiled_graph_key": key,
        "family": FAMILY,
        "precision": "w8a8",
        "lora_bucket": 0,
        "sm": RUNTIME["sm"],
        "toolchain": dict(TOOLCHAIN),
        "host_isa": {
            "machine": platform.machine(), "march": "", "simdlen": 0,
            "level": "",
        },
        "package_constants_in_so": False,
        "constant_folding_fenced": True,
        "graph_class": {
            "name": name,
            "target": TARGET,
            "class_hash": str(block["class_hash"]),
            "graph": dict(block["graph"]),
            "graph_witness": str(block.get("graph_witness") or "0" * 32),
            "range_digest": str(block["range_digest"]),
            "fork": list(block.get("fork") or ()),
            "class_dims": list(block.get("class_dims") or ()),
            "strict": True,
            "lora_bucket": 0,
            "literal_values": "",
            "literal_payload_values": "",
            "placement": [],
            "constants": [dict(value) for value in block["constants"]],
        },
    }


def declared_names(rows: Tuple[Tuple[int, int], ...] = ROWS) -> Tuple[str, ...]:
    """Every class name this declaration traces to — what a pod hands the
    dispatch so an unarmed class reads as ``pending compile`` rather than as a
    shape gap."""
    return tuple(entry_name(h, w) for h, w in rows)


def artifact(tmp_path: Path, meta: Dict[str, Any] | None = None) -> Path:
    """Create an opaque stand-in for one Engine-owned artifact."""
    meta = meta or metadata()
    key = str(meta["compiled_graph_key"])
    path = tmp_path / f"{key}.tcg"
    path.write_bytes(b"\x00tcg-probe-artifact")
    return path


def _tcg_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Keep the old helper name while its value is already public TCG data."""
    return dict(meta)


class _ProbeTCGRunner:
    """The public TCG runner surface around the probe's fake AOTI package."""

    def __init__(self, package: ProbePackage) -> None:
        self.package = package
        self.bound = False
        self.declared_fqns = tuple(package.get_constant_fqns())
        self.bound_fqns: tuple[str, ...] = ()

    @property
    def calls(self) -> int:
        return int(self.package.invocations)

    def bind(self, state: Dict[str, Any], *, device: str) -> None:
        del device
        values = {name: state[name] for name in self.declared_fqns if name in state}
        missing = sorted(set(self.declared_fqns) - set(values))
        if missing:
            raise ConstantBindingError(
                "constant_unresolved", f"missing constants: {missing!r}"
            )
        try:
            self.package.load_constants(
                values, check_full_update=True, user_managed=True
            )
        except Exception as exc:
            reason = "out_of_memory" if isinstance(exc, torch.OutOfMemoryError) else "injection_failed"
            raise ConstantBindingError(reason, str(exc)) from exc
        self.bound_fqns = tuple(sorted(values))
        self.bound = True

    def __call__(self, *feeds: object) -> Any:
        return self.package(*cast(Tuple[Any, ...], feeds))


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


class ArmOutcomes(tuple):
    """One :class:`AdoptOutcome` PER GRAPH CLASS, in ``rows`` order.

    pgw#1176: the arm is per entry, so there is no single verdict — and a
    plain tuple is what this should be. The two aggregate properties below
    exist because the tests that use them are asking a question the aggregate
    genuinely answers: *"did this whole DECLARATION arm?"*, which is a
    property of a test's fixture, not a claim a pod makes about itself.

    They are DELIBERATELY not what production reports. A pod reports
    per-entry serve state (`aot_serve.entry_states`) precisely because a
    cell-level boolean can advertise more than it serves. Nothing here is
    wired to production; index this to assert per class, which is what the
    per-entry rows do.
    """

    @property
    def armed(self) -> bool:
        """True only when EVERY class of this declaration armed."""
        # len(), never bool() — __bool__ delegates here, so bool(self)
        # would recurse.
        return len(self) > 0 and all(o.armed for o in self)

    @property
    def reason(self) -> str:
        """The first refusal's classified reason, or ""."""
        return next((o.reason for o in self if not o.armed), "")

    @property
    def detail(self) -> str:
        return next((o.detail for o in self if not o.armed), "")

    @property
    def identity(self) -> str:
        return self[0].identity if self else ""

    def __bool__(self) -> bool:
        return self.armed


def arm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, decl: Any,
        packages: Dict[str, ProbePackage],
        meta: Dict[str, Any] | None = None,
        verify_numerics: bool = True,
        cfg: Any = None,
        rows: Tuple[Tuple[int, int], ...] | None = None,
        ) -> Tuple[Any, Any, Any]:
    """Drive the REAL arm path over EVERY class in ``rows`` and return
    ``(pipeline, module, outcomes)``.

    pgw#1176: ``outcomes`` is one :class:`AdoptOutcome` PER GRAPH CLASS, in
    ``rows`` order, because the arm is per entry — one artifact, one class,
    one verdict. A caller that wants "did the whole declaration arm" asks
    ``all(outcomes)``; a caller that wants "did THIS class arm" indexes it.
    There is no combined verdict, deliberately: the object that could report
    one was the wrong atom.

    ``meta`` overrides the metadata of the FIRST row only — it is how a test
    perturbs one artifact (a bad host_isa, a forged stamp) without having to
    rebuild the declaration.

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
    from gen_worker._vendor.torch_compiled_graphs import artifact as tcg_artifact

    metadata_by_artifact: Dict[Path, Dict[str, Any]] = {}
    metadata_by_key: Dict[str, Dict[str, Any]] = {}
    runners_by_key: Dict[str, _ProbeTCGRunner] = {}

    class _Engine:
        def import_artifact(self, key: str, artifact_path: Path) -> Any:
            assert metadata_by_artifact[Path(artifact_path)]["compiled_graph_key"] == key
            return SimpleNamespace(outcome=StoreOutcome.PRESENT)

        def resolve(self, key: str, _destination: Path) -> Any:
            return SimpleNamespace(metadata=metadata_by_key[key])

        def runner(self, key: str, _destination: Path) -> CompiledGraphRunner:
            return runners_by_key[key]  # type: ignore[return-value]

    monkeypatch.setattr(aot, "open_worker_engine", lambda _root=None: _Engine())
    monkeypatch.setattr(
        tcg_artifact,
        "read_metadata",
        lambda path: metadata_by_artifact[Path(path)],
    )
    module = ProbeDenoiser()
    pipeline = ProbePipeline(module)
    use_rows = ROWS if rows is None else rows
    outcomes: List[Any] = []
    for index, row in enumerate(use_rows):
        block = metadata(row, use_rows)
        if meta is not None and index == 0:
            block = meta
        artifact_path = artifact(tmp_path, block)
        tcg_meta = _tcg_metadata(block)
        key = str(tcg_meta["compiled_graph_key"])
        metadata_by_artifact[artifact_path] = tcg_meta
        metadata_by_key[key] = tcg_meta
        runners_by_key[key] = _ProbeTCGRunner(
            packages[str(tcg_meta["graph_class"]["name"])]
        )
        outcomes.append(provision.arm_aot(
            pipeline, cell_cfg(decl) if cfg is None else cfg,
            tmp_path / "cache", artifact_path, 0, meta=tcg_meta,
            verify_numerics=verify_numerics,
            declared=declared_names(use_rows)))
    return pipeline, module, ArmOutcomes(outcomes)


def numerics_rows(said: List[Tuple[str, str, str]]) -> List[Tuple[str, str]]:
    import gen_worker.activity as activity_mod

    return [(detail, phase) for kind, detail, phase in said
            if kind == activity_mod.KIND_COMPILED_GRAPH_NUMERICS]
