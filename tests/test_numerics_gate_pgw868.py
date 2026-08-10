"""pgw#868 — THE NUMERICS GATE. Everything that runs without a card.

The cross-cutting requirement of pgw#868: *everything this program built
refuses a cell for being UNUSABLE; nothing refused one for being WRONG.*

The trap this file exists to keep shut, stated once: `numerics_ladder.gate()`
opens `if comparison is None: return None`. Wiring the call without producing a
measurement passes EVERY cell, always, while looking correct in the diff and in
the call graph. So the tests that matter here are not "is `gate` called" — they
are:

* a cell BELOW its declared floor does not arm, and the pipeline is left eager;
* a cell BETWEEN floor and warn arms *with the warning on the wire*;
* a cell that cannot be MEASURED does not arm either — "nobody could ask" is
  not "it passed";
* and the verdict is bisectable to ONE named axis (one entry x one shape row x
  one lane), because a whole-cell fail nobody can split is the artifact that
  produced three wrong confident diagnoses in this program.

Everything below drives the REAL arm path — `provision.arm_aot` ->
`aot_serve.enable` -> stage/verify/bind/wrap -> the gate — against a real packed
artifact, a real registered `Compile` declaration, real `torch` tensors and the
mint's own input builder. The ONE substitution is `_load_package`: an AOTI
`.so` needs a GPU, and it is the only piece deferred to the pod. The subject it
returns is a CALIBRATED blend of the eager output at an exact cosine, which is
what lets a test name the rung it is aiming at; the real numbers come from a
pod, and no test here may be cited as evidence about a real cell's numerics.
"""

from __future__ import annotations

import math
import platform
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest
import torch

from gen_worker import aot_serve as aot
from gen_worker import numerics_probe
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
#: sdxl's declared band, used verbatim: this issue is a measurement plus
#: wiring, never a re-design of what "good" means.
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
                 drop_output: bool = False) -> None:
        self.cosine = float(cosine)
        self.raises = raises
        self.drop_output = drop_output
        self.loaded: Dict[str, Any] = {}
        self.invocations = 0

    def get_constant_fqns(self) -> List[str]:
        return ["weight"]

    def load_constants(self, values: Dict[str, Any], check_full_update: bool = False,
                       **_kw: Any) -> None:
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


def arm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, decl: Any,
        packages: Dict[str, ProbePackage],
        meta: Dict[str, Any] | None = None) -> Tuple[Any, Any, Any]:
    """Drive the REAL arm path and return ``(pipeline, module, outcome)``.

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
        pipeline, decl, tmp_path / "cache", artifact(tmp_path, meta), 0)
    return pipeline, module, outcome


def numerics_rows(said: List[Tuple[str, str, str]]) -> List[Tuple[str, str]]:
    import gen_worker.activity as activity_mod

    return [(detail, phase) for kind, detail, phase in said
            if kind == activity_mod.KIND_CELL_NUMERICS]


# ---------------------------------------------------------------------------
# THE RED: a cell below its declared floor must not serve
# ---------------------------------------------------------------------------


def test_a_cell_below_its_declared_floor_REFUSES_TO_ARM(
        tmp_path, monkeypatch, declared, events):
    """The headline. Before this gate existed, this exact cell ARMED.

    RED PROOF (recorded rather than asserted, because it is a property of the
    absent call site): with the `gate_cell_numerics(...)` call removed from
    `provision.arm_aot`, this test fails on its first assertion — `arm_aot`
    returns True and the 0.99-cosine cell serves every subsequent request.
    """
    packages = {entry_name(h, w): ProbePackage(cosine=0.99) for h, w in ROWS}
    pipeline, module, outcome = arm(tmp_path, monkeypatch, declared, packages)

    assert outcome.armed is False, "a cell that lost 1% of the output's direction armed"
    # Refused means UNARMED, not merely reported: the module must be eager.
    assert aot.is_armed(pipeline) is False
    assert aot.armed_targets(pipeline) == {}
    assert isinstance(module.forward(torch.zeros(8, 8), torch.tensor(1.0)),
                      torch.Tensor)

    # pgw#923: the adopt ledger cannot need closing, because nothing is
    # announced until the arm is FINAL. `enable` used to say `armed` before
    # this gate ran, so a reader counting armed adoptions over-counted every
    # numerics refusal and a second "retraction" row existed only to correct
    # the first. The arm returns ONE outcome, with the gate's verdict in it.
    assert outcome.reason == "numerics_refused"
    assert "UNARMED by the numerics gate" in outcome.detail

    rows = numerics_rows(events)
    assert rows, "a refused cell said nothing on the wire"
    detail, phase = rows[-1]
    assert phase == "refused"
    # The verdict carries its inputs: the floor, its source, and the AXIS.
    assert "floor=0.995" in detail and "warn=0.999" in detail
    assert "source=declared" in detail
    assert entry_name(*ROWS[0]) in detail or entry_name(*ROWS[1]) in detail
    assert "0.99" in detail


def test_between_floor_and_warn_it_ARMS_and_records_the_warning(
        tmp_path, monkeypatch, declared, events):
    """The gray band is not a refusal and not a silence. It serves, and it
    confesses — a fleet-wide rate is only countable from activity records."""
    packages = {entry_name(h, w): ProbePackage(cosine=0.997) for h, w in ROWS}
    pipeline, _module, outcome = arm(tmp_path, monkeypatch, declared, packages)

    assert outcome.armed is True, "a cell inside the declared gray band failed to arm"
    assert aot.is_armed(pipeline) is True
    rows = numerics_rows(events)
    assert [p for _d, p in rows] == ["degraded"], rows
    detail = rows[0][0]
    assert "cosine=0.997" in detail
    assert "axes=2/2" in detail


def test_a_faithful_cell_arms_AND_THE_PASS_IS_ANNOUNCED(
        tmp_path, monkeypatch, declared, events):
    """A silent pass is indistinguishable from a gate that never ran, which is
    this program's signature failure. So the pass is a hub row too, carrying
    every axis it was taken on."""
    packages = {entry_name(h, w): ProbePackage() for h, w in ROWS}
    pipeline, _module, outcome = arm(tmp_path, monkeypatch, declared, packages)

    assert outcome.armed is True
    assert aot.is_armed(pipeline) is True
    rows = numerics_rows(events)
    assert [p for _d, p in rows] == ["checked"], rows
    detail = rows[0][0]
    assert "axes=2/2" in detail
    assert f"family={FAMILY}" in detail and "key=cell868" in detail
    # Per-axis readings, named — the PoolWidth discipline.
    for h, w in ROWS:
        assert entry_name(h, w) in detail
    assert "cos=1.00000" in detail


# ---------------------------------------------------------------------------
# fail-closed: an unmeasurable cell is NOT a passing cell
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("package,reason", [
    (ProbePackage(raises="dlopen: undefined symbol"), "cell_forward_failed"),
    (ProbePackage(drop_output=True), "output_structure_differs"),
])
def test_a_cell_that_cannot_be_MEASURED_does_not_arm(
        tmp_path, monkeypatch, declared, events, package, reason):
    """"Nobody could ask" must never collapse into "it passed". Staying eager
    is the ordinary miss policy of every other adopt gate, so the cost of a
    probe defect is an un-armed cell — never a silently degraded one."""
    packages = {entry_name(h, w): package for h, w in ROWS}
    pipeline, _module, outcome = arm(tmp_path, monkeypatch, declared, packages)

    assert outcome.armed is False
    assert aot.is_armed(pipeline) is False
    detail, phase = numerics_rows(events)[-1]
    assert phase == "unmeasurable"
    assert reason in detail
    assert "not a pass" in detail


def test_an_undeclared_family_cannot_be_probed_AND_THEREFORE_CANNOT_ARM(
        tmp_path, monkeypatch, events):
    """No export declaration => no feed => no comparison. That is exactly the
    state the whole fleet was in, and it must read as a refusal."""
    reset_export_declarations()
    packages = {entry_name(h, w): ProbePackage() for h, w in ROWS}
    _pipeline, _module, outcome = arm(
        tmp_path, monkeypatch, declaration(), packages)

    assert outcome.armed is False
    detail, phase = numerics_rows(events)[-1]
    assert phase == "unmeasurable"
    assert "no_input_contract" in detail


def test_the_report_cannot_report_a_pass_it_did_not_take():
    """The structural guard against the trap, asserted on the type itself.

    `CellNumerics.measured` is the single predicate the arm consults, and it
    must be False for every shape of partial or absent evidence — an empty
    report, a report short of its own axis count, and a report whose axes
    errored. If this ever returns True for one of these, the gate is passing
    cells nobody measured.
    """
    from gen_worker.numerics_probe import AxisVerdict, CellNumerics, ProbeAxis

    thresholds = numerics_probe.numerics_ladder.DEFAULT_THRESHOLDS
    axis = ProbeAxis(entry="e", target=TARGET)
    empty = CellNumerics(FAMILY, "k", thresholds, "declared", (), 2)
    assert empty.measured is False
    assert empty.comparison() is None

    errored = CellNumerics(
        FAMILY, "k", thresholds, "declared",
        (AxisVerdict(axis=axis, reason="cell_forward_failed"),), 1)
    assert errored.measured is False
    assert errored.comparison() is None


# ---------------------------------------------------------------------------
# bisectability: one entry x one shape row x one lane
# ---------------------------------------------------------------------------


def test_the_verdict_is_bisectable_to_ONE_named_axis(
        tmp_path, monkeypatch, declared, events):
    """A whole-cell "fail" that cannot be split is the artifact that produced
    three wrong confident diagnoses here. So: two shape rows, ONE of them bad,
    and the refusal must NAME it — then that one axis must be re-runnable on
    its own, from the name in the row, with nothing edited.
    """
    from gen_worker.models import provision

    good, bad = entry_name(*ROWS[0]), entry_name(*ROWS[1])
    packages = {good: ProbePackage(), bad: ProbePackage(cosine=0.90)}
    pipeline, _module, outcome = arm(tmp_path, monkeypatch, declared, packages)

    assert outcome.armed is False
    detail, phase = numerics_rows(events)[-1]
    assert phase == "refused"
    # The whole-cell verdict names the class that parted from eager, and the
    # healthy one is still on the record with its own reading.
    assert f"worst axis: entry={bad}" in detail
    assert f"row={'h=16,w=16'}" in detail
    assert f"{good}[h=8,w=8]: cos=1.00000" in detail

    # And now the bisection itself: re-arm and probe ONLY the named axis.
    monkeypatch.setattr(aot, "runtime_key", lambda: dict(RUNTIME))
    monkeypatch.setattr(
        aot, "_entry_admission_drift", lambda *a, **k: None)
    monkeypatch.setattr(
        aot, "_load_package", lambda path, entry="model": packages[entry])
    module = ProbeDenoiser()
    pipeline = ProbePipeline(module)
    assert aot.enable(pipeline, declared, tmp_path / "c2",
                      artifact(tmp_path)).armed is True
    one = numerics_probe.probe_cell(
        pipeline, declared, aot.armed_metadata(pipeline), only=bad)
    assert [v.axis.name for v in one.verdicts] == [bad]
    assert one.measured is True
    assert one.comparison().cosine == pytest.approx(0.90, abs=5e-4)
    assert one.comparison().verdict == "destroyed"
    # The same call against the healthy row is healthy — one variable moved.
    other = numerics_probe.probe_cell(
        pipeline, declared, aot.armed_metadata(pipeline), only=good)
    assert other.comparison().verdict == "healthy"
    del provision  # imported for the arm path above; nothing else needs it


def test_an_axis_names_its_entry_row_execution_lane_and_seed_and_reproduces():
    """Every verdict carries its inputs. The seed is DERIVED from the axis, so
    two rows of one cell are never fed the same latent and any reader can
    rebuild the exact feed."""
    from gen_worker.numerics_probe import ProbeAxis, axes_from_meta

    axes = axes_from_meta(metadata())
    assert [a.name for a in axes] == sorted(entry_name(h, w) for h, w in ROWS)
    a, b = axes
    assert a.execution_lane == "w8a8" and a.target == TARGET
    assert {a.shape_row, b.shape_row} == {"h=8,w=8", "h=16,w=16"}
    assert a.seed != b.seed, "two axes share a seed; a shape-independent bug " \
                             "would read as agreement"
    # Reproducible across processes: the seed is a pure function of the axis.
    assert ProbeAxis(entry=a.entry, target=a.target, execution_lane=a.execution_lane).seed == a.seed
    assert str(a).startswith(f"entry={a.entry} target={TARGET}")


def test_the_probe_does_not_disturb_the_serving_RNG(
        tmp_path, monkeypatch, declared, events):
    """pgw#784: the tenant keeps being served throughout an arm. A probe that
    advanced the global generator would change a paying request's output in
    order to check a cell."""
    torch.manual_seed(4242)
    before = torch.get_rng_state().clone()
    packages = {entry_name(h, w): ProbePackage() for h, w in ROWS}
    _pipeline, _module, outcome = arm(tmp_path, monkeypatch, declared, packages)
    assert outcome.armed is True
    assert torch.equal(torch.get_rng_state(), before)


# ---------------------------------------------------------------------------
# the wiring, tested separately from the helper — this issue's signature failure
# ---------------------------------------------------------------------------


def test_a_REAL_arm_reaches_the_gate(tmp_path, monkeypatch, declared, events):
    """The tests above would all still pass if `arm_aot` never called the gate
    and something else did. This one asserts the CALL SITE: drive the real
    `provision.arm_aot` and require that the cell's own forward was invoked by
    the probe — i.e. a measurement was actually taken on the arm path.

    RED when `gate_cell_numerics(pipe, cfg)` is removed from `arm_aot`:
    `invocations` stays 0 and `armed` is True.
    """
    packages = {entry_name(h, w): ProbePackage() for h, w in ROWS}
    _pipeline, _module, outcome = arm(tmp_path, monkeypatch, declared, packages)
    assert outcome.armed is True
    assert all(p.invocations == 1 for p in packages.values()), (
        "a REAL arm did not run the cell against its eager reference: "
        f"{ {k: p.invocations for k, p in packages.items()} }")
    assert numerics_rows(events), "a REAL arm emitted no cell_numerics row"


def test_the_gate_is_never_reached_without_a_comparison(monkeypatch):
    """The trap, asserted directly: `numerics_ladder.gate(None, ...)` returns
    None and refuses nothing. So `gate_cell_numerics` must never call it with
    a `None` comparison — every path to `gate()` goes through
    `report.measured`, and every other path refuses.
    """
    from gen_worker.models import provision
    from gen_worker import numerics_ladder

    seen: List[Any] = []
    monkeypatch.setattr(
        numerics_ladder, "gate",
        lambda comparison, **kw: seen.append(comparison))
    monkeypatch.setattr(
        numerics_probe, "probe_cell",
        lambda *a, **k: (_ for _ in ()).throw(
            numerics_probe.ProbeUnavailable("not_armed", "nothing armed")))
    assert provision.gate_cell_numerics(object(), declaration()) is False
    assert seen == [], "the ladder was consulted with no measurement to judge"
