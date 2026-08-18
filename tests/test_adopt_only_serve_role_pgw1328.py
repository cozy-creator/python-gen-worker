"""pgw#1328: the ADOPT-ONLY serve role, its import guard, and its refusals.

Four claims, and each one has a RED proof beside it — the repo's standard is
that a gate whose green has never been contrasted with a red is a gate nobody
has tested (pgw#1176's measured finding: a fence naming a deleted symbol ran
green for months guarding nothing).

1. The role's declared module set cannot REACH the mint lane (static).
2. In the adopt-only role, importing the mint lane RAISES (runtime), and the
   whole serving host still imports — which is the done-test's premise.
3. A miss produces a typed refusal carrying tcg#37's evidence, and its
   ROUTE/REFUSE disposition is what the wire status is derived from.
4. Nothing in the role can mint: the seam refuses, the deriver is not injected,
   and an eager-first background mint is a loud contradiction.
"""

from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any, List, Tuple

import pytest

from gen_worker import activity, boot_adopt
from gen_worker.serve import boot_miss, guard, mint_seam, refusal, role
from gen_worker.serve import selection as serve_selection
from gen_worker._vendor.torchcg import CallIngress, CallInput
from gen_worker._vendor.torchcg.selection import MissReason, SelectionOutcome

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"


def _fence() -> Any:
    spec = importlib.util.spec_from_file_location(
        "lint_serve_role_closure", REPO / "scripts" / "lint_serve_role_closure.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _restore_role() -> Any:
    """Every test gets the default role back, blocker uninstalled.

    The role is process-global on purpose (it is a fact about the process), so
    a test that declared it and did not put it back would silently decide the
    outcome of every test after it.
    """
    yield
    guard._uninstall_for_test()
    role._reset_for_test()


# ── 1. the static claim ──────────────────────────────────────────────────


def test_the_declared_serve_role_cannot_reach_the_mint_lane() -> None:
    fence = _fence()
    roots = fence._declared_tuple("SERVE_ROLE_MODULES")
    banned = fence._declared_tuple("MINT_MACHINERY")
    assert tuple(roots) == role.SERVE_ROLE_MODULES, (
        "the fence and the role disagree about the role's own module set — "
        "which is the drift channel reading it out of the source exists to "
        "close")
    assert tuple(banned) == role.MINT_MACHINERY
    seen, via, _, _ = fence.closure(roots)
    reached = sorted(name for name in banned if name in seen)
    assert not reached, f"reached {reached} via {[via.get(n) for n in reached]}"
    assert fence.main([]) == 0


def test_red_the_fence_fires_on_a_root_that_reaches_the_lane() -> None:
    """`mint_adapter` is the eager side of the seam and reaches the lane
    exclusively through FUNCTION-LOCAL imports, so this proves both that the
    fence fires and that the walk still follows lazy imports — the shape the
    coupling actually took (`fleet_compiled_graphs`, before it went through the seam)."""
    fence = _fence()
    banned = fence._declared_tuple("MINT_MACHINERY")
    problems = fence.check(("gen_worker.mint_adapter",), banned)
    assert problems
    assert any("mint_supervisor" in line for line in problems)
    assert any("aot_mint" in line for line in problems)
    assert fence.selftest() == 0, "the fence's own selftest stopped proving red"


def test_red_the_fence_refuses_a_role_declaration_it_cannot_read() -> None:
    fence = _fence()
    with pytest.raises(SystemExit):
        fence._declared_tuple(
            "SERVE_ROLE_MODULES", source="SERVE_ROLE_MODULES = tuple(x)")
    with pytest.raises(SystemExit):
        fence._declared_tuple("NOT_A_DECLARATION", source="X = ()")


def test_red_a_fence_root_that_no_longer_exists_is_caught() -> None:
    """A fence naming a deleted module passes vacuously forever. It must not."""
    fence = _fence()
    problems = fence.check(
        ("gen_worker.serve.role", "gen_worker.a_module_that_was_deleted"),
        ("gen_worker.aot_mint",))
    assert any("a_module_that_was_deleted" in line for line in problems)


def test_the_serving_host_names_no_mint_module_at_module_scope() -> None:
    """The structural precondition for the done-test.

    `executor` used to `from . import mint_supervisor` at module scope, so an
    adopt-only interpreter could not import its own serving host. The only
    mint reference left is the key DERIVER, and it is inside a role branch.
    """
    tree = ast.parse((SRC / "gen_worker" / "executor.py").read_text())
    top_level: List[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            top_level.append(base)
            top_level.extend(f"{base}.{a.name}" for a in node.names)
    for banned in role.MINT_MACHINERY:
        tail = banned.rsplit(".", 1)[-1]
        assert not any(
            name == banned or name.endswith(f".{tail}") or name == tail
            for name in top_level), (
            f"executor imports {banned} at module scope again; an adopt-only "
            f"process cannot import its own serving host")


# ── 2. the runtime claim, in a real subprocess ───────────────────────────


_DONE_TEST = """
import sys
sys.path.insert(0, {src!r})
from gen_worker.serve import guard, role
role.declare(role.ServeRole.ADOPT_ONLY)
guard.install()

# The whole serving host, under the blocker.
import gen_worker.executor
import gen_worker.fleet_compiled_graphs
import gen_worker.aot_serve
import gen_worker.boot_adopt
import gen_worker.worker

from gen_worker.serve import mint_seam
from gen_worker.serve.guard import MintMachineryUnavailable
from gen_worker.serve.refusal import AdoptOnlyRefused

for name in role.MINT_MACHINERY:
    try:
        __import__(name)
    except MintMachineryUnavailable:
        continue
    raise SystemExit("IMPORTED " + name)

assert type(mint_seam.supervision()).__name__ == "NoMint"
try:
    mint_seam.supervision().may_delegate()
except AdoptOnlyRefused as exc:
    assert exc.reason == "mint_forbidden", exc.reason
else:
    raise SystemExit("may_delegate did not refuse")
print("ok")
"""


def test_the_serving_host_boots_in_a_process_where_minting_cannot_be_imported(
) -> None:
    """pgw#1328's done-test, minus the card.

    Every module of the serving host imports; every one of the nine declared
    mint modules raises on import; the seam answers `NoMint`; and asking it to
    mint produces the typed refusal rather than an ImportError somebody has to
    attribute. The arm-and-serve half needs a GPU and is recorded separately.
    """
    proof = subprocess.run(
        [sys.executable, "-c", _DONE_TEST.format(src=str(SRC))],
        capture_output=True, text=True, check=False)
    assert proof.returncode == 0, proof.stdout + proof.stderr
    assert proof.stdout.strip().endswith("ok")


def test_the_blocker_refuses_to_promise_something_already_false() -> None:
    """Installing after the lane is imported would be a lie, so it refuses."""
    role.declare(role.ServeRole.ADOPT_ONLY)
    import gen_worker.mint_supervisor  # noqa: F401  (this process is eager)

    with pytest.raises(RuntimeError, match="already imported"):
        guard.install()
    assert guard.present(), "present() cannot see a module that is loaded"


def test_red_the_blocker_refuses_the_wrong_role() -> None:
    with pytest.raises(RuntimeError, match="adopt_only"):
        guard.install()


def test_the_role_cannot_be_changed_once_declared() -> None:
    role.declare(role.ServeRole.ADOPT_ONLY)
    role.declare(role.ServeRole.ADOPT_ONLY)  # idempotent
    with pytest.raises(RuntimeError, match="already declared"):
        role.declare(role.ServeRole.EAGER_CAPABLE)


def test_the_adopt_only_entry_point_declares_before_it_imports() -> None:
    """The ORDER is the whole guarantee, so it is read off the source.

    A `main()` that imported the worker before installing the blocker would
    pass every other test here and give the pod none of the protection.
    """
    entry = ast.parse((SRC / "gen_worker" / "serve" / "__main__.py").read_text())
    fn = next(n for n in ast.walk(entry)
              if isinstance(n, ast.FunctionDef) and n.name == "main")
    steps: List[str] = []
    for statement in fn.body:  # SOURCE order — `ast.walk` does not preserve it
        for node in ast.walk(statement):
            if isinstance(node, ast.Call):
                target = ast.unparse(node.func)
                if target.endswith(("declare", "install")):
                    steps.append(target.rsplit(".", 1)[-1])
            elif isinstance(node, ast.ImportFrom) and "entrypoint" in (
                    node.module or ""):
                steps.append("import_worker")
    assert steps == ["declare", "install", "import_worker"], steps


# ── 3. the refusal, and the evidence it carries ──────────────────────────


def _ingress(name: str, dtype: str, shape: Tuple[int, ...]) -> CallIngress:
    return CallIngress(
        parameters=(name,), flat_arity=1,
        inputs=(CallInput(name, 0, name, 0, (), name, dtype, shape),))


def _dispatch_choice(
    candidates: List[Tuple[str, CallIngress]], value: Any,
) -> serve_selection.EntryChoice[str]:
    return serve_selection.choose(
        [serve_selection.Candidate(name=n, ingress=i, runner=n)
         for n, i in candidates],
        (), {"sample": value})


def test_a_total_miss_carries_the_ranking_and_names_the_closest_class() -> None:
    torch = pytest.importorskip("torch")
    choice = _dispatch_choice(
        [("far", _ingress("sample", "float32", (2, 8))),
         ("near", _ingress("sample", "bfloat16", (4, 8)))],
        torch.ones(4, 8, dtype=torch.float32))
    assert choice.outcome is SelectionOutcome.NO_CLASS_ADMITS
    refused = refusal.from_selection(
        choice.outcome, choice.selection.ranked, choice.selection.ambiguous,
        function="generate", family="flux", compiled_graph_key="cg-key-v1-x")
    assert refused.kind is refusal.MissKind.NO_CLASS_ADMITS
    assert refused.disposition is refusal.Disposition.REFUSE
    # `near` matches every declared DIM and disagrees only on dtype, so tcg#37's
    # dims-last rung table must sort it first. That ordering is the whole point
    # of pgw#1074 and it now comes from the contract rather than from us.
    assert refused.candidates[0].graph_class == "near"
    assert refused.candidates[0].misses[0].reason is MissReason.DTYPE_MISMATCH
    line = refused.wire_detail()
    assert "cg-key-v1-x" in line and "near" in line and "generate" in line
    assert "disposition=refuse" in line


def test_an_unarmed_declared_class_ROUTES_where_a_shape_miss_REFUSES() -> None:
    """The distinction an adopt-only pod has to make and an eager one does not.

    On an eager-capable pod both are "serve eager and wait for the compile". On
    this one, one of them is somebody else's card and the other is nobody's.
    """
    torch = pytest.importorskip("torch")
    choice = _dispatch_choice(
        [("armed", _ingress("sample", "float32", (2, 8)))],
        torch.ones(4, 8, dtype=torch.float32))
    shape_miss = refusal.from_selection(
        choice.outcome, choice.selection.ranked, choice.selection.ambiguous)
    unarmed = refusal.from_selection(
        choice.outcome, choice.selection.ranked, choice.selection.ambiguous,
        unarmed=("not_yet_compiled",))
    assert shape_miss.kind is refusal.MissKind.NO_CLASS_ADMITS
    assert not shape_miss.routable
    assert unarmed.kind is refusal.MissKind.CLASS_UNARMED
    assert unarmed.routable


def test_two_admitting_classes_are_a_defect_and_never_a_coin_flip() -> None:
    torch = pytest.importorskip("torch")
    choice = _dispatch_choice(
        [("a", _ingress("sample", "float32", (4, 8))),
         ("b", _ingress("sample", "float32", (4, 8)))],
        torch.ones(4, 8, dtype=torch.float32))
    assert choice.outcome is SelectionOutcome.CLASS_AMBIGUOUS
    refused = refusal.from_selection(
        choice.outcome, choice.selection.ranked, choice.selection.ambiguous)
    assert refused.kind is refusal.MissKind.CLASS_AMBIGUOUS
    assert not refused.routable
    assert set(refused.unarmed) == {"a", "b"}


def test_every_miss_kind_has_a_disposition_and_no_default() -> None:
    assert set(refusal.DISPOSITIONS) == set(refusal.MissKind)
    source = (SRC / "gen_worker" / "serve" / "refusal.py").read_text()
    assert ".get(" not in source.split("DISPOSITIONS")[-1][:400], (
        "a defaulted disposition lookup is how a new refusal silently "
        "inherits somebody else's routing decision")


def test_every_boot_adopt_reason_has_an_adopt_only_disposition() -> None:
    """Total over the vocabulary, with the tracer tokens excluded BY NAME.

    `boot_adopt.REASONS` is the exhaustive list pgw#1116 fenced. An adopt-only
    pod must have an answer for every one it can reach, and must be unable to
    reach the rest.
    """
    covered = set(boot_miss.BOOT_MISS_KINDS) | set(boot_miss.TRACER_ONLY)
    assert covered == set(boot_adopt.REASONS), (
        f"uncovered: {sorted(set(boot_adopt.REASONS) - covered)}")
    assert not (set(boot_miss.BOOT_MISS_KINDS) & set(boot_miss.TRACER_ONLY))
    assert "trace_failed" in boot_miss.TRACER_ONLY
    assert "keyset_absent" not in boot_miss.TRACER_ONLY


def test_a_boot_that_adopted_produces_no_refusal_and_a_miss_produces_one() -> None:
    hit = boot_adopt.BootAdoptOutcome(
        adoption=None, reason=boot_adopt.HIT)
    # `adopted` is what decides, not the token: a `hit` with nothing attached
    # is a contradiction and must be loud rather than silently "adopted".
    with pytest.raises(boot_miss.TracerReasonInAdoptOnly):
        boot_miss.refusal_for(hit)
    missed = boot_adopt.BootAdoptOutcome(
        reason="miss", function="generate", family="flux",
        derived_key="cg-key-v1-y")
    refused = boot_miss.refusal_for(missed)
    assert refused is not None
    assert refused.kind is refusal.MissKind.ARTIFACT_MISS
    assert refused.routable, "an artifact this pod lacks is a PLACEMENT fact"
    assert refused.compiled_graph_key == "cg-key-v1-y"


def test_red_a_tracer_reason_in_this_role_is_a_broken_premise() -> None:
    traced = boot_adopt.BootAdoptOutcome(reason="trace_failed")
    with pytest.raises(boot_miss.TracerReasonInAdoptOnly, match="key tracer"):
        boot_miss.refusal_for(traced)


def test_the_disposition_decides_the_wire_status() -> None:
    from gen_worker import executor
    from gen_worker.pb import worker_scheduler_pb2 as pb

    routed = refusal.AdoptOnlyRefusal(
        kind=refusal.MissKind.ARTIFACT_MISS, function="generate")
    refused = refusal.AdoptOnlyRefusal(
        kind=refusal.MissKind.NO_CLASS_ADMITS, function="generate")
    status, detail = executor._map_exception(routed.error())
    assert status == pb.JOB_STATUS_RETRYABLE
    assert detail.startswith("artifact_miss:")
    status, detail = executor._map_exception(refused.error())
    assert status == pb.JOB_STATUS_FATAL
    assert detail.startswith("no_class_admits:")


def test_the_refusal_reaches_the_wire_as_one_typed_event() -> None:
    seen: List[Tuple[str, str, str]] = []
    original = activity.emit_event

    def _capture(kind: str, detail: str = "", **kw: Any) -> None:
        seen.append((kind, str(kw.get("phase", "")), detail))

    activity.emit_event = _capture  # type: ignore[assignment]
    try:
        out = refusal.report(refusal.AdoptOnlyRefusal(
            kind=refusal.MissKind.ARM_REFUSED, function="generate"))
    finally:
        activity.emit_event = original  # type: ignore[assignment]
    assert len(seen) == 1
    kind, phase, detail = seen[0]
    assert kind == activity.KIND_ADOPT_REFUSED
    assert phase == "arm_refused", "the phase must be the countable token"
    assert "disposition=route" in detail
    assert out.reported and not refusal.AdoptOnlyRefusal(
        kind=refusal.MissKind.ARM_REFUSED).reported


# ── 4. nothing in this role can mint ─────────────────────────────────────


def test_the_seam_refuses_every_operation_in_the_adopt_only_role() -> None:
    role.declare(role.ServeRole.ADOPT_ONLY)
    mint = mint_seam.supervision()
    assert isinstance(mint, mint_seam.NoMint)
    for call in (
        lambda: mint.may_delegate(),
        lambda: mint.make_task(function="generate"),
        lambda: mint.abandoned("abandoned"),
        lambda: mint.export_spec(None, None),
        lambda: mint.declaration_module_gaps(None, None, None),
    ):
        with pytest.raises(refusal.AdoptOnlyRefused) as caught:
            call()
        assert caught.value.refusal.kind is refusal.MissKind.MINT_FORBIDDEN
        assert not caught.value.refusal.routable


def test_red_no_operation_answers_neutrally_instead_of_refusing() -> None:
    """A `may_delegate` returning "" would read as "yes, mint out of process",
    and a plain reason string would let the caller mint IN process instead —
    the same capability by the other door."""
    role.declare(role.ServeRole.ADOPT_ONLY)
    mint = mint_seam.supervision()
    outcome: Any
    try:
        outcome = mint.may_delegate()
    except refusal.AdoptOnlyRefused:
        outcome = "refused"
    assert outcome == "refused"


def test_the_eager_capable_role_gets_the_registered_implementation() -> None:
    from gen_worker import mint_adapter

    assert isinstance(mint_seam.supervision(), mint_adapter.EagerCapableMint)
    assert mint_seam.supervision().may_delegate() == ""


@pytest.mark.parametrize(
    "module", ["gen_worker.fleet_compiled_graphs", "gen_worker.executor"])
def test_the_seam_is_registered_by_whoever_calls_it(module: str) -> None:
    """A registration that depends on some OTHER module having been imported
    first is an ordering hazard, not a dependency — and this one bit.

    The first shape of this seam registered from the three process entries
    (`entrypoint`, `cli`, `local_serve`). Every test that drives a background
    mint then depended on one of those having been imported into the same
    pytest worker by some unrelated file, so `test_executor_adopt` passed in
    one file ordering and failed in another with the mint never completing.
    Registration now happens in the modules that CALL the seam, and this
    proves it in a process that imports exactly one of them and nothing else.
    """
    proof = subprocess.run(
        [sys.executable, "-c",
         "import sys;"
         f"sys.path.insert(0, {str(SRC)!r});"
         f"import {module};"
         "from gen_worker.serve import mint_seam;"
         "print(type(mint_seam.supervision()).__name__)"],
        capture_output=True, text=True, check=False)
    assert proof.returncode == 0, proof.stderr
    assert proof.stdout.strip() == "EagerCapableMint", proof.stdout + proof.stderr


def test_red_an_unregistered_eager_capable_process_refuses_loudly() -> None:
    """…and when nothing registered, the seam does NOT improvise.

    The alternative failure mode is the one that cost the ordering hazard its
    invisibility: a `supervision()` that quietly answered `NoMint` in the
    eager-capable role would turn "nobody registered" into "this pod stopped
    minting", which is a silent capability loss rather than an error.
    """
    from gen_worker.serve import mint_seam as seam

    saved = seam._registered
    seam._registered = None
    try:
        with pytest.raises(seam.MintSupervisionUnregistered):
            seam.supervision()
    finally:
        seam._registered = saved


def test_the_adopt_only_role_refuses_even_when_the_lane_is_registered() -> None:
    """The role is about what a pod MAY do, not only about what it imported."""
    from gen_worker import mint_adapter  # noqa: F401  (registers)

    role.declare(role.ServeRole.ADOPT_ONLY)
    assert isinstance(mint_seam.supervision(), mint_seam.NoMint)


def test_the_role_declaration_is_not_readable_from_the_environment() -> None:
    """§1.17 / Paul's standing rule: an env may carry a VALUE, never a
    DECISION. pgw#1327 already refused a `GEN_WORKER_ADOPT_ONLY` knob for this
    exact question, and re-introducing one here would be the second answer."""
    for path in sorted((SRC / "gen_worker" / "serve").rglob("*.py")):
        source = path.read_text()
        assert "os.environ" not in source and "getenv" not in source, (
            f"{path.name} reads the environment; the serve role is DECLARED "
            f"by the process entry, never configured")


def test_serve_role_modules_are_all_real_and_the_lane_is_not_among_them() -> None:
    for name in role.SERVE_ROLE_MODULES:
        assert importlib.util.find_spec(name) is not None, name
    assert not set(role.SERVE_ROLE_MODULES) & set(role.MINT_MACHINERY)
    for absent in ("gen_worker.executor", "gen_worker.fleet_compiled_graphs"):
        assert absent not in role.SERVE_ROLE_MODULES, (
            f"{absent} is the eager-capable half and reaches the lane on "
            f"purpose; claiming it here would make the fence unpassable for "
            f"the wrong reason")


def test_the_contract_owns_the_rung_table_rather_than_this_worker() -> None:
    """tcg#37's ranking is READ, not re-derived — two copies of a ranking rule
    is how a second serve host and this one silently disagree about which
    class was closest."""
    from gen_worker import aot_serve
    from gen_worker._vendor.torchcg import selection as contract

    assert aot_serve.MISS_RUNGS == {
        reason.value: rung for reason, rung in contract.MISS_RUNGS.items()}
    assert aot_serve.AOTI_ALIGNMENT == contract.AOTI_ALIGNMENT
    assert aot_serve.RECAST_TARGETS == contract.RECAST_TARGETS
    source = (SRC / "gen_worker" / "aot_serve.py").read_text()
    assert '"dtype_mismatch": 1' not in source, (
        "the rung table was re-typed into aot_serve beside the contract's")


def test_the_selection_verdict_is_a_value_before_it_is_an_exception() -> None:
    """The property that lets ONE walk serve both roles: an eager-capable host
    renders the miss as `IngressContractError` and falls back, and the
    adopt-only one renders the same walk as a refusal."""
    torch = pytest.importorskip("torch")
    choice = _dispatch_choice(
        [("only", _ingress("sample", "float32", (2, 8)))],
        torch.ones(4, 8, dtype=torch.float32))
    assert isinstance(choice, serve_selection.EntryChoice)
    assert not choice.admitted and choice.runner is None
    assert choice.selection.ranked, "the refusal path computed no ranking"


def test_an_undescribable_candidate_set_is_a_declaration_defect() -> None:
    """tcg#37's `input_name_collision`: two classes spell one input from
    different coordinates — here the same parameter NAME at two different
    positions in the two signatures — so no single presented call stands for
    both. A declaration defect, in the same class as `class_ambiguous`, and it
    must refuse rather than resolve one class against the other's coordinate.
    """
    torch = pytest.importorskip("torch")
    with pytest.raises(serve_selection.CallUndescribable) as caught:
        serve_selection.choose(
            [serve_selection.Candidate(
                name="a", runner="a",
                ingress=CallIngress(
                    parameters=("sample",), flat_arity=1,
                    inputs=(CallInput("sample", 0, "sample", 0, (), "sample",
                                      "float32", (4, 8)),))),
             serve_selection.Candidate(
                name="b", runner="b",
                ingress=CallIngress(
                    parameters=("hidden", "sample"), flat_arity=1,
                    inputs=(CallInput("sample", 0, "sample", 1, (), "sample",
                                      "float32", (4, 8)),)))],
            (), {"sample": torch.ones(4, 8)})
    assert caught.value.reason == "input_name_collision"
