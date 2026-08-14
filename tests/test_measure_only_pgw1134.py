"""A blocker whose exit criterion is a MEASUREMENT must be able to gather it,
and gathering it must not be able to publish anything.

The catch-22 this guards against:

* ``mint_supervisor.assert_family_mintable`` refuses a family while ANY blocker is
  open — so the run that would CLOSE a blocker is refused BY it;
* ``boot_trace_child`` is ungated but composes structure-only, and
  ``structure_only._refuse_artifact_lanes`` refuses a w8a8 artifact tree by
  name — so the cheap path is shut too, for a different reason.

Both doors are reproduced here on a REAL vehicle (micro-diffusion, a real
declaration, a real tree carrying a real fp8 weight table), and the third door
— ``gen_worker.measure_child`` — is driven through the same production seams:
``run_setup`` -> the real loader -> ``aot_mint.trace_for_key`` -> the real
``_export_entry``. Nothing about the export is stubbed.

The properties, one section each:

1. **RED both directions.** The measure child runs where a real mint refuses,
   and a real mint STILL refuses afterwards — measuring resolves nothing, which
   is the whole point of a citation-bearing ``resolved=True``.
2. **It cannot publish, structurally.** Its wire struct does not declare the
   output-side fields (so an artifact destination cannot reach the process);
   its report type carries no artifact identity; its source calls no publish
   symbol; and an end-to-end run with every publish seam wired to raise
   completes without touching one.
3. **The compile half is real work with no residue** — the inductor output is
   counted and DELETED, so the one thing a measure run could have left behind
   is gone before the report is written.
4. **Every refusal names itself** in an enumerable vocabulary.

Cardless: CPU fixtures throughout, and the inductor compile is FAKED wherever
it is exercised — this file mints nothing, compiles nothing and touches no GPU.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Tuple

import msgspec
import pytest

from gen_worker import (
    activity,
    aot_compile_child,
    boot_key,
    boot_trace_child,
    measure_child,
    mint_supervisor,
)
from gen_worker.child_contract import CompileSpec, MintSlot

REPO = Path(__file__).resolve().parent.parent
MICRO_SRC = REPO / "examples" / "micro-diffusion" / "src"
FAMILY = "micro-diffusion"
BLOCKER_ID = "OQ-3-whole-graph-OOM-unmeasured-on-the-w8a8-lane"


# ---------------------------------------------------------------------------
# The vehicle: a REAL family, with a REAL open blocker, on a tree that carries
# a REAL fp8 weight table.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def micro_src() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    if str(MICRO_SRC) not in sys.path:
        sys.path.insert(0, str(MICRO_SRC))


@pytest.fixture(scope="module")
def w8a8_tree(
    micro_src: None, tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    """A REAL w8a8 artifact tree, written by the SDK's OWN producer.

    ``materialize_w8a8`` runs ``quantize_tree_w8a8`` over the bf16 tree, so
    the denoiser's eligible linears are fp8 with per-out-channel
    ``weight_scale`` — the gw#534 contract, the thing
    ``detect_w8a8_artifact`` reads headers for, and the exact condition
    ``structure_only._refuse_artifact_lanes`` refuses ltx-video-2.3 for. Not
    a marker, not a monkeypatch: the refusal under test is the production one.
    """
    from micro_diffusion.weights import SEED, materialize_w8a8

    tree = materialize_w8a8(
        tmp_path_factory.mktemp("micro-w8a8") / "w8a8", seed=SEED)

    from gen_worker.models.w8a8 import detect_w8a8_artifact

    assert detect_w8a8_artifact(tree) is not None, (
        "the fixture must reproduce the REFUSED condition, not describe it")
    return tree


@pytest.fixture
def blocked_declaration(micro_src: None) -> Iterator[None]:
    """micro-diffusion's own declaration, carrying ltx's OQ-3 verbatim in
    shape: one OPEN blocker whose exit criterion is one measurement."""
    from gen_worker import MintBlocker
    from gen_worker.api import export_contract as ec

    import micro_diffusion.aot_declaration as decl

    blocked = msgspec.structs.replace(decl.DECLARATION, blockers=(
        MintBlocker(
            id=BLOCKER_ID,
            what="Whole-graph export is assumed and the OOM rationale behind "
                 "the alternative was measured on a lane this endpoint no "
                 "longer serves.",
            evidence="Standing in for ltx-video-2.3's OQ-3 (ie#651): the "
                     "served lane is fp8-w8a8 and the rationale is fp8 "
                     "layerwise-cast provenance.",
            resolves_when="ONE measurement of whole-graph export at the "
                          "largest declared classes, on the served lane.",
        ),))
    ec.register_export_declaration(blocked, replace=True)
    try:
        yield
    finally:
        ec.register_export_declaration(decl.DECLARATION, replace=True)


def _cfg() -> CompileSpec:
    from gen_worker.registry import collect_endpoints

    specs = collect_endpoints(["harness.rig_runtime", "micro_diffusion.main_w8a8"])
    spec = next(s for s in specs if s.name == "generate-w8a8")
    cell = spec.compile_cell()
    return CompileSpec(
        shapes=tuple(tuple(int(v) for v in row) for row in (cell.shapes or ())),
        targets=tuple(str(t) for t in (cell.targets or ())),
        family=str(cell.family or ""),
        lora_bucket=int(cell.lora_bucket or 0),
        guidance_scales=tuple(float(v) for v in (cell.guidance_scales or ())),
        text_lens=tuple(int(v) for v in (cell.text_lens or ())),
    )


def _slots(tree: Path) -> Dict[str, MintSlot]:
    from gen_worker.api.binding import ModelRef

    return {"pipeline": MintSlot(
        ref=ModelRef(source="tensorhub", path="cozy/micro-diffusion",
                     tag="prod"),
        path=str(tree))}


def _measure_job(tree: Path) -> measure_child.MeasureJob:
    cfg = _cfg()
    return measure_child.MeasureJob(
        function="generate-w8a8",
        modules=("harness.rig_runtime", "micro_diffusion.main_w8a8"),
        family=cfg.family, cfg=cfg, slots=_slots(tree))


def _trace_job(tree: Path, report: Path) -> boot_key.TraceJob:
    cfg = _cfg()
    return boot_key.TraceJob(
        function="generate-w8a8",
        modules=("harness.rig_runtime", "micro_diffusion.main_w8a8"),
        family=cfg.family, cfg=cfg, slots=_slots(tree),
        report=str(report), code_digest=boot_key.CODE_DIGEST)


@pytest.fixture
def on_path(monkeypatch: pytest.MonkeyPatch, micro_src: None) -> None:
    monkeypatch.syspath_prepend(str(REPO / "tests"))
    monkeypatch.setenv("PYTHONPATH", ":".join(
        [str(REPO / "src"), str(REPO / "tests"), str(MICRO_SRC)]))


@pytest.fixture
def events(monkeypatch: pytest.MonkeyPatch) -> List[Any]:
    seen: List[Any] = []
    monkeypatch.setattr(activity, "_sink", seen.append, raising=False)
    return seen


def _measure_events(seen: List[Any]) -> List[Any]:
    return [u for u in seen if u.kind == activity.KIND_MEASURE_ONLY]


# ---------------------------------------------------------------------------
# 1. THE CATCH-22, and the third door
# ---------------------------------------------------------------------------


def test_both_front_doors_are_shut_and_the_measure_child_runs_anyway(
    tmp_path: Path, w8a8_tree: Path, blocked_declaration: None, on_path: None,
    events: List[Any],
) -> None:
    """RED on master in the only direction that matters: there is no third
    door there, so this run cannot happen at all.

    Door 1 — the MINT gate refuses the family whose blocker the measurement
    would answer. Door 2 — the ungated boot-trace child refuses the same
    family because its tree is a quantized artifact. Door 3 — the measure
    child runs, on real weights, and says so.
    """
    with pytest.raises(mint_supervisor.DeclaredBlockerRefusal) as refusal:
        mint_supervisor.assert_family_mintable(FAMILY)
    assert BLOCKER_ID in str(refusal.value)

    trace_report = tmp_path / "trace.json"
    rc = boot_trace_child.run(_trace_job(w8a8_tree, trace_report))
    traced = msgspec.json.decode(
        trace_report.read_bytes(), type=boot_key.TraceReport)
    assert rc == boot_key.EXIT_REFUSED and not traced.ok
    assert traced.reason == "structure_unsupported", traced.detail
    assert "w8a8 artifact" in traced.detail

    report_path = tmp_path / "measure.json"
    rc = measure_child.run(
        _measure_job(w8a8_tree), report_path, compile_entries=False)
    report = msgspec.json.decode(
        report_path.read_bytes(), type=measure_child.MeasureReport)

    assert rc == measure_child.EXIT_OK and report.ok, (
        f"{report.reason}: {report.detail[:600]}")
    assert report.weights == "real", (
        "the measurement must state which lane it measured — the whole "
        "argument for the fallback is that the served graph is made of the "
        "artifact's bytes")
    assert report.structure_refusal_token == "structure_unsupported"
    assert "w8a8 artifact" in report.structure_refusal
    assert report.declared_classes == 3 and len(report.entries) == 3
    assert [e.entry for e in report.entries] == [
        "decoder", "transformer/cfg=false", "transformer/cfg=true"], (
        "the rows must arrive in the MINT's own order — the measurement is "
        "worth nothing if it measures a different traversal than the mint")
    assert all(e.ok and e.nodes > 0 for e in report.entries), report.entries
    assert report.compiled is False, "--export-only was asked for"
    assert report.device == "cpu" and report.cuda is False, (
        "a cardless run reports a MEASURED zero under a named device, which "
        "is a different fact from an unmeasured one")

    row = _measure_events(events)[-1]
    assert row.phase == "measured_export"
    assert row.duration_ms > 0
    assert "export_peak_device_bytes=" in row.detail
    assert f"family={FAMILY}" in row.detail and "weights=real" in row.detail


def test_the_mint_gate_is_UNTOUCHED_by_a_measurement(
    tmp_path: Path, w8a8_tree: Path, blocked_declaration: None, on_path: None,
) -> None:
    """The second RED direction. A measure run is evidence, never a
    resolution: the blocker is still open afterwards, the mint still refuses,
    and closing it stays a REVIEWABLE declaration edit that cites the number
    (the SDK refuses an uncited ``resolved=True``, by design)."""
    measure_child.run(
        _measure_job(w8a8_tree), tmp_path / "m.json", compile_entries=False)

    with pytest.raises(mint_supervisor.DeclaredBlockerRefusal) as refusal:
        mint_supervisor.assert_family_mintable(FAMILY)
    assert BLOCKER_ID in str(refusal.value)

    from gen_worker.api.export_contract import export_declaration, open_blockers

    assert [b.id for b in open_blockers(export_declaration(FAMILY))] == [
        BLOCKER_ID]


def test_the_boot_trace_child_did_not_inherit_the_fallback(
    tmp_path: Path, w8a8_tree: Path, on_path: None,
) -> None:
    """pgw#1080's invariant, held where it belongs. §4.27 step 1 forbids
    weights for IDENTITY, so the boot child must keep refusing a stranded
    family rather than downloading its checkpoint to state a key — the
    fallback is scoped to the measurement, which is a different question with
    a different answer."""
    report = tmp_path / "trace.json"
    boot_trace_child.run(_trace_job(w8a8_tree, report))
    traced = msgspec.json.decode(
        report.read_bytes(), type=boot_key.TraceReport)
    assert traced.reason == "structure_unsupported"

    source = (REPO / "src" / "gen_worker" / "boot_trace_child.py").read_text()
    assert "structure_only=tuple(cfg.targets)" in source
    loads = [n for n in ast.walk(ast.parse(source))
             if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == "run_setup"]
    assert len(loads) == 1, (
        "a SECOND run_setup call in the boot-trace child is a real-weight "
        "fallback on the IDENTITY path — the one thing §4.27 step 1 forbids")
    assert any(k.arg == "structure_only" for k in loads[0].keywords)


# ---------------------------------------------------------------------------
# 2. IT CANNOT PUBLISH — four fences, none of them a convention
# ---------------------------------------------------------------------------


def test_the_wire_struct_withholds_every_output_destination() -> None:
    """The strongest of the four: the artifact path never enters the process.

    An operator measures the request they would MINT — the committed
    ``*.mint.json`` — and msgspec drops what the struct does not declare. So
    there is no publish call to audit because there is nowhere to publish to.
    """
    measure_fields = set(measure_child.MeasureJob.__struct_fields__)

    assert set(measure_child.WITHHELD_FIELDS).isdisjoint(measure_fields)
    assert measure_fields == {
        "function", "modules", "cfg", "family", "slots", "device",
        "execution_lane",
    }


def test_an_input_document_decodes_with_its_destinations_dropped(
    tmp_path: Path,
) -> None:
    """The operator document can carry stale output fields, but the closed
    measurement struct admits only diagnostic inputs."""
    cfg = CompileSpec(family=FAMILY, targets=("transformer",))
    document = {
        "function": "generate-w8a8",
        "modules": ["micro_diffusion.main_w8a8"],
        "family": FAMILY,
        "cfg": msgspec.to_builtins(cfg),
        "arm_token": "arm1-deadbeef",
        "target": str(tmp_path / "compiled-graph.tar.gz"),
        "work_root": str(tmp_path / "work"),
        "report": str(tmp_path / "mint.json"),
        "resume": str(tmp_path / "bank"),
    }
    raw = msgspec.json.encode(document)
    assert b"compiled-graph.tar.gz" in raw

    job = msgspec.json.decode(raw, type=measure_child.MeasureJob)

    for field in measure_child.WITHHELD_FIELDS:
        assert not hasattr(job, field), field
    assert job.function == "generate-w8a8" and job.cfg.family == FAMILY


def test_the_report_type_carries_no_artifact_identity() -> None:
    """A measurement that could name an artifact is one field away from being
    mistaken for one. ``MeasureReport`` has no path, no digest, no cell key —
    because a diagnostic report is never an artifact handoff."""

    fields = set(measure_child.MeasureReport.__struct_fields__)
    forbidden = {"artifact", "digest", "cell_key", "cell_ref", "content_digest"}
    assert fields.isdisjoint(forbidden)


PUBLISH_SURFACE = (
    # (module attribute path, why it publishes)
    ("fleet_cells", "publish_self_mint"),
    ("aot_serve", "artifact_metadata"),
    ("aot_mint", "mint_targets"),
    ("aot_delivery", "materialize_named_artifact"),
    ("cell_resolve", "materialize"),
)


def test_the_source_calls_no_publish_symbol() -> None:
    """Read out of the tree, so a publish call added later fails HERE.

    An AST walk over every call in ``measure_child.py``: an attribute call
    whose name is in the publish surface, or a bare call to one, is the
    finding. Name-level rather than resolution-level on purpose — this fence
    is allowed to be over-strict about a name, and a module that must never
    publish has no business spelling one.
    """
    src = (REPO / "src" / "gen_worker" / "measure_child.py").read_text()
    tree = ast.parse(src)
    banned = {name for _mod, name in PUBLISH_SURFACE}
    hits: List[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = (func.attr if isinstance(func, ast.Attribute)
                else func.id if isinstance(func, ast.Name) else "")
        if name in banned:
            hits.append(f"{name} at line {node.lineno}")
    assert not hits, (
        f"the measure-only child reaches the publish surface: {hits}")


def test_an_end_to_end_run_never_touches_a_publish_seam(
    tmp_path: Path, w8a8_tree: Path, blocked_declaration: None, on_path: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runtime half of the same property, on the real path: every publish
    seam is wired to RAISE, and the measurement completes.

    A source fence proves nobody spelled the name; this proves nobody reached
    it transitively either — through ``aot_mint``, which certainly can.
    """
    import importlib

    touched: List[str] = []

    for module_name, attr in PUBLISH_SURFACE:
        module = importlib.import_module(f"gen_worker.{module_name}")
        if not hasattr(module, attr):
            continue

        def _refuse(*_a: Any, _n: str = f"{module_name}.{attr}", **_k: Any) -> Any:
            touched.append(_n)
            raise AssertionError(f"a measure-only run reached {_n}")

        monkeypatch.setattr(module, attr, _refuse)

    report_path = tmp_path / "out" / "measure.json"
    rc = measure_child.run(
        _measure_job(w8a8_tree), report_path, compile_entries=False)

    assert touched == []
    assert rc == measure_child.EXIT_OK
    assert [p.name for p in report_path.parent.iterdir()] == ["measure.json"], (
        "the only thing a measure run may leave behind is its measurement")


# ---------------------------------------------------------------------------
# 3. The inductor half: real work, no residue
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_compiler(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> List[Tuple[str, Path]]:
    """AOTInductor, faked at the ONE seam ``_export_entry`` compiles through.

    Everything before it is real — the export, the gates, the branch-arm
    ordering. This box does not compile (mints and compiles run on remote
    pods), and the property under test is that the measure child DRIVES the
    compile and disposes of its output, which a faked compiler proves exactly
    as well as a real one.
    """
    made: List[Tuple[str, Path]] = []

    def _compile(traced: Any, _spec: Any, _engine: Any, _runtime: Any,
                 *, out_dir: Path, **_kw: Any) -> Any:
        path = out_dir / (traced.name.replace("/", "_") + ".tar.gz")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"tcg diagnostic artifact")
        made.append((traced.name, path))
        traced.release()
        return SimpleNamespace(compile_s=0.01, reuse_s=0.0)

    monkeypatch.setattr(
        aot_compile_child, "compile_traced_class", _compile)
    return made


def test_the_compile_half_runs_and_leaves_nothing_behind(
    tmp_path: Path, w8a8_tree: Path, blocked_declaration: None, on_path: None,
    fake_compiler: List[Tuple[str, Path]], events: List[Any],
) -> None:
    """OQ-3's own words: *"the INDUCTOR half is the half that matters — an
    export-only trace never exercises the whole-graph planner this blocker
    names"*. So the compile runs by default, is measured per entry, and its
    output is counted and deleted before the report is written."""
    report_path = tmp_path / "measure.json"
    rc = measure_child.run(_measure_job(w8a8_tree), report_path)
    report = msgspec.json.decode(
        report_path.read_bytes(), type=measure_child.MeasureReport)

    assert rc == measure_child.EXIT_OK and report.ok, report.detail[:600]
    assert report.compiled is True
    assert len(fake_compiler) == 3, (
        f"every declared class must reach the compiler: {fake_compiler}")
    assert [e.compiled_files for e in report.entries] == [1, 1, 1]
    assert [str(p) for _e, p in fake_compiler if p.exists()] == [], (
        "a measure run that leaves loose .so files behind is one packaging "
        "step away from the artifact it may not produce")
    assert _measure_events(events)[-1].phase == "measured"


def test_an_entry_that_runs_out_of_memory_IS_the_measurement(
    tmp_path: Path, w8a8_tree: Path, blocked_declaration: None, on_path: None,
    monkeypatch: pytest.MonkeyPatch, events: List[Any],
) -> None:
    """The FAIL verdict OQ-3 names (*"MintResourceExhausted ... OUT OF DEVICE
    MEMORY, phase=trace_graph"*) is an ANSWER, not an error: the run reports
    the peaks it reached, names the row that died, and exits refused.

    The compiler is faked into raising — a real OOM is a pod fact, and this is
    the classification path, which is cardless.
    """
    def _oom(traced: Any, *_args: Any, **_kw: Any) -> Any:
        raise MemoryError(f"entry {traced.name!r}: OUT OF DEVICE MEMORY")

    monkeypatch.setattr(
        aot_compile_child, "compile_traced_class", _oom)

    report_path = tmp_path / "measure.json"
    rc = measure_child.run(_measure_job(w8a8_tree), report_path)
    report = msgspec.json.decode(
        report_path.read_bytes(), type=measure_child.MeasureReport)

    assert rc == measure_child.EXIT_REFUSED and not report.ok
    assert report.reason == "export_refused"
    assert "OUT OF DEVICE MEMORY" in report.detail
    assert len(report.entries) == 1 and not report.entries[0].ok
    assert report.entries[0].entry == "decoder", (
        "an exception escaping a generator carries no row identity, and "
        "'something ran out of memory' is not evidence anybody can act on")
    assert _measure_events(events)[-1].phase == "export_refused"


# ---------------------------------------------------------------------------
# 4. Every refusal names itself (pgw#1116's rule, at a new surface)
# ---------------------------------------------------------------------------


def test_every_refusal_token_in_the_measure_child_is_in_its_vocabulary() -> None:
    """Read the sites out of the source — the same scan
    ``test_boot_adopt_observability_pgw1116`` runs over the boot children, so
    adding a refusal without naming it fails here rather than becoming the
    next unenumerable event."""
    src = (REPO / "src" / "gen_worker" / "measure_child.py").read_text()
    found = set(re.findall(r"_fail\(\s*\n?\s*report_path,\s*\"([a-z_]+)\"", src))
    assert found, "the scan found no refusal sites at all — the pattern rotted"
    assert found <= set(measure_child.REASONS), (
        f"unenumerable refusal token(s): {sorted(found - set(measure_child.REASONS))}")
    assert activity.KIND_MEASURE_ONLY not in (
        activity.KIND_AOT_MINT, activity.KIND_JIT_COMPILE), (
        "a measurement counted as a mint would be the first row of a lie")


def test_a_measure_refusal_is_distinguished_by_KIND_not_by_token() -> None:
    """``slots_unresolvable`` means the same thing on the boot path and here,
    and both should keep the word. What must differ is the KIND: a reader
    groups on ``(kind, phase)``, so ``measure_only`` is a countable surface of
    its own and a measurement is never summed into a mint's rows or a
    boot-adopt's. This lane widened ``boot_adopt.REASONS`` with nothing — the
    pgw#1116 fence reads the boot children's sites and is untouched.
    """
    from gen_worker import boot_adopt

    assert activity.KIND_MEASURE_ONLY not in (
        activity.KIND_BOOT_ADOPT, activity.KIND_AOT_MINT,
        activity.KIND_JIT_COMPILE)
    names = [k for k in vars(activity) if k.startswith("KIND_")]
    assert len({getattr(activity, k) for k in names}) == len(names), (
        "two activity kinds sharing a string makes them one column hub-side")
    assert "measure" not in " ".join(boot_adopt.REASONS)
