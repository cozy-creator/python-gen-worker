"""A mint REFUSAL is DATA on the declaration, and it fails CLOSED.

The defect this prevents: a family that expresses "may not mint yet" by
registering a THUNK raising ``MintRefused`` loses that refusal the moment the
declaration folds onto ``@endpoint(compile=)``, which accepts a ``Compile`` and
never a callable — the family then mints against unanswered design questions,
silently.

So the refusal is vocabulary: ``Compile(blockers=(MintBlocker(...),))``.
Four properties, one section each:

1. it is DATA — SDK vocabulary plus literals, nothing callable, nothing to
   evaluate, and NOT a contract axis (resolving a blocker must not re-key);
2. the endpoint repo's TORCH-FREE declaration lint can read it out of an
   AST-extracted ``compile=`` expression, which is the constraint that decides
   the shape (a design the lint cannot read is the wrong design);
3. minting FAILS CLOSED while any blocker is unresolved — in the serving
   parent's recipe gate AND in the mint child, each naming the ids — and a
   family with none mints normally;
4. serving is UNAFFECTED, and a fold cannot lose a refusal.

Cardless: no GPU, no pod, no mint, no weights.
"""

from __future__ import annotations

import ast
import asyncio
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Tuple

import msgspec
import pytest

from gen_worker import child_preflight
from gen_worker import child_contract
from gen_worker import Compile, MintBlocker, fleet_cells
from gen_worker import mint_child, mint_process
from gen_worker import mint_process as mp
from gen_worker import config as gw_config
from gen_worker.api.derive import (
    DeclarationMismatch, assert_blockers, assert_faithful, blocker_delta,
    contract_delta,
)
from gen_worker.api.export_contract import (
    DeclarationError, open_blockers, reset_export_declarations,
)
from gen_worker.cell_adopt import AdoptOutcome
from gen_worker.registry import CompileCell

from harness import blocked_endpoint_pgw1115 as blocked

HARNESS_MODULE = "harness.blocked_endpoint_pgw1115"
FAMILY = blocked.FAMILY
GIB = 1 << 30


def _clean(**over: Any) -> Compile:
    """The same declaration with NO open blockers — the mintable control."""
    over.setdefault("blockers", ())
    return msgspec.structs.replace(blocked.BLOCKED_COMPILE, **over)


# ---------------------------------------------------------------------------
# 1. The refusal is DATA
# ---------------------------------------------------------------------------


def test_a_declaration_carries_its_open_blockers_as_values() -> None:
    decl = blocked.BLOCKED_COMPILE
    assert [b.id for b in decl.open_blockers] == list(blocked.OPEN_IDS)
    # The resolved row is still carried — it is a record, not a hole.
    assert len(decl.blockers) == len(blocked.OPEN_IDS) + 1
    assert open_blockers(_clean()) == ()


@pytest.mark.parametrize("hole", ["what", "evidence", "resolves_when"])
def test_a_blocker_missing_its_claim_evidence_or_exit_is_refused(hole: str) -> None:
    """A blocker that cannot be reviewed is not a blocker: without evidence it
    is an opinion, and without an exit criterion it is a permanent stall."""
    kwargs = {"id": "OQ-x", "what": "w", "evidence": "e", "resolves_when": "r"}
    kwargs[hole] = "   "
    with pytest.raises(DeclarationError) as exc:
        MintBlocker(**kwargs)  # type: ignore[arg-type]
    assert hole in str(exc.value)


def test_resolving_a_blocker_requires_a_CITATION_not_a_bool_flip() -> None:
    """The unblock has to be reviewable. A bare ``resolved=True`` is exactly
    the silent unblock this vocabulary exists to prevent."""
    with pytest.raises(DeclarationError) as exc:
        MintBlocker(id="OQ-x", what="w", evidence="e", resolves_when="r",
                    resolved=True)
    assert "resolution" in str(exc.value)
    ok = MintBlocker(id="OQ-x", what="w", evidence="e", resolves_when="r",
                     resolved=True, resolution="measured 2026-08-11 on the "
                                               "w8a8 lane")
    assert open_blockers(SimpleNamespace(blockers=(ok,))) == ()


def test_an_id_is_a_single_token_because_the_refusal_prints_a_list() -> None:
    with pytest.raises(DeclarationError):
        MintBlocker(id="OQ 2 audio rank", what="w", evidence="e",
                    resolves_when="r")


def test_a_repeated_blocker_id_is_refused_at_declaration_time() -> None:
    row = blocked.BLOCKERS[0]
    with pytest.raises(DeclarationError) as exc:
        _clean(blockers=(row, row))
    assert "repeats a blocker id" in str(exc.value)


def test_a_callable_is_not_a_blocker() -> None:
    """No thunks, at any level: the whole point is that nothing evaluates."""
    with pytest.raises(TypeError):
        _clean(blockers=(lambda: None,))  # type: ignore[arg-type]


def test_blockers_are_NOT_a_contract_axis_so_resolving_one_never_re_keys() -> None:
    """A blocked family has no published cell to re-key, but a family that
    RESOLVES its last blocker must not find its first mint keyed differently
    from the declaration it was reviewed against."""
    assert contract_delta(blocked.BLOCKED_COMPILE, _clean()) == {}
    assert "blockers" not in blocked.BLOCKED_COMPILE.contract_axes()


# ---------------------------------------------------------------------------
# 2. The TORCH-FREE lint can read them
#
#    `inference-endpoints/scripts/lint_declarations.py` AST-extracts a family's
#    `compile=` expression and evaluates it in an SDK-only namespace with no
#    torch installed. This section is that reader, run against the harness
#    endpoint with torch BLOCKED at the import system — so a design that needed
#    torch, a callable or a runtime value to state its blockers fails here.
# ---------------------------------------------------------------------------


def _extract(module_path: Path) -> Tuple[List[str], str]:
    """``(prelude, compile_expression)``, read statically — the fold lane's
    reader, kept in step with it deliberately."""
    tree = ast.parse(module_path.read_text(encoding="utf-8"), str(module_path))
    prelude: List[str] = []
    expr = ""
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] == "gen_worker" and not node.level:
                prelude.append(ast.unparse(node))
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.ClassDef,
                               ast.FunctionDef)):
            prelude.append(ast.unparse(node))
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for deco in node.decorator_list:
            if not isinstance(deco, ast.Call):
                continue
            name = (deco.func.id if isinstance(deco.func, ast.Name)
                    else getattr(deco.func, "attr", ""))
            if name == "endpoint":
                for kw in deco.keywords:
                    if kw.arg == "compile":
                        expr = ast.unparse(kw.value)
    return prelude, expr


_TORCH_FREE_PROBE = '''
import json, sys

class _NoTorch:
    """Anything the fleet's pinned wheel does NOT install alongside the SDK."""
    BANNED = ("torch", "diffusers", "transformers", "triton", "torchvision")

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in self.BANNED:
            raise ModuleNotFoundError(f"{name} is not installed (lint env)")
        return None

sys.meta_path.insert(0, _NoTorch())

from gen_worker.api.export_contract import blocker_rows

ns = {"__name__": "declaration", "__builtins__": __builtins__}
for stmt in PRELUDE:
    try:
        exec(compile(stmt, "<prelude>", "exec"), ns)
    except Exception:
        pass
decl = eval(compile(EXPR, "<declaration>", "eval"), ns)
print(json.dumps({
    "family": decl.family,
    "blockers": blocker_rows(decl),
    "torch_imported": "torch" in sys.modules,
}))
'''


def test_the_torch_free_lint_reads_and_reports_the_open_blockers(
    tmp_path: Path,
) -> None:
    """THE CONSTRAINT. A declaration the lint cannot read is the wrong design:
    the lint is the only automated feedback on a declaration that does not cost
    a rented pod, and after the fold it is also the only place a
    family's refusal is visible without one."""
    prelude, expr = _extract(Path(blocked.__file__))
    assert expr, "the reader found no compile= on the @endpoint decorator"

    probe = tmp_path / "probe.py"
    probe.write_text(
        f"PRELUDE = {prelude!r}\nEXPR = {expr!r}\n" + _TORCH_FREE_PROBE)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(Path(__file__).resolve().parent), env.get("PYTHONPATH", "")])
    proc = subprocess.run([sys.executable, str(probe)], env=env,
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-3000:]
    out = json.loads(proc.stdout.strip().splitlines()[-1])

    assert out["torch_imported"] is False, "reading a declaration imported torch"
    assert out["family"] == FAMILY
    open_ids = [r["id"] for r in out["blockers"] if not r["resolved"]]
    assert open_ids == list(blocked.OPEN_IDS)
    # A reported blocker carries its exit criterion, or the report tells a
    # reader a family is stuck without telling them how it gets unstuck.
    for row in out["blockers"]:
        assert row["resolves_when"] and row["evidence"]
    resolved = [r for r in out["blockers"] if r["resolved"]]
    assert resolved and all(r["resolution"] for r in resolved)


# ---------------------------------------------------------------------------
# 3. Minting FAILS CLOSED — in the parent's recipe gate and in the child
# ---------------------------------------------------------------------------


class _Pipe:
    pass


@dataclass
class _Cfg:
    family: str = FAMILY
    lora_bucket: int = 0
    shapes: Tuple[Tuple[int, int], ...] = ((64, 64),)
    targets: Tuple[str, ...] = ("transformer",)
    text_lens: Tuple[int, ...] = (128,)
    guidance_scales: Tuple[float, ...] = ()
    regional: bool = False


class _Publisher:
    base_url = "http://hub.invalid"

    def enabled(self) -> bool:
        return True

    def worker_jwt(self) -> str:
        return "jwt"


@pytest.fixture()
def _miss(monkeypatch: pytest.MonkeyPatch) -> Any:
    """A real AOT discovery miss on an otherwise mint-capable pod (the
    pgw#853 fixture, which is the production shape of this decision)."""
    gw_config.reload_for_test()
    monkeypatch.setattr(
        fleet_cells.provision, "enable_compiled",
        lambda pipe, cfg, cache_dir, artifact: AdoptOutcome.miss("no_cell"))
    monkeypatch.setattr(fleet_cells.cc, "has_compile_target", lambda p, c, **_kw: True)
    monkeypatch.setattr(fleet_cells.cc, "toolchain_present", lambda: True)
    monkeypatch.setattr(fleet_cells.cc, "apply_lora_execution_lane", lambda p, b, **_kw: None)
    monkeypatch.setattr(fleet_cells.cc, "drop_lora_execution_lane", lambda p: None)
    monkeypatch.setattr(fleet_cells, "_cuda_ready", lambda: True)
    monkeypatch.setattr(fleet_cells, "_PENDING", {})
    monkeypatch.setattr(
        fleet_cells, "arm_identity",
        lambda *a, **k: type("_A", (), {
            "token": "arm1-" + "a" * 56,
            "facts_dict": lambda self: {}})())
    monkeypatch.setattr(fleet_cells.cc, "mandatory_serving", lambda p: False)
    monkeypatch.setattr(fleet_cells.cc, "arm_jit_intake", lambda p, c, **_kw: None)
    monkeypatch.setattr(
        fleet_cells.loading, "pipeline_weight_lane", lambda pipe: "w8a8")
    reset_export_declarations()
    yield
    reset_export_declarations()
    gw_config.reload_for_test()


@pytest.fixture()
def _events(monkeypatch: pytest.MonkeyPatch) -> List[Tuple[str, str, str]]:
    seen: List[Tuple[str, str, str]] = []
    monkeypatch.setattr(
        fleet_cells.activity_mod, "emit_event",
        lambda kind, detail, phase="", duration_ms=0, **_kw: seen.append(
            (kind, phase, detail)))
    return seen


def _enable(decl: Compile, events: List[Tuple[str, str, str]]) -> Any:
    from gen_worker.api.export_contract import register_export_declaration

    register_export_declaration(decl, replace=True)
    return fleet_cells.enable_compiled(
        _Pipe(), _Cfg(), publisher=_Publisher(), delegate=True)  # type: ignore[arg-type]


def test_the_recipe_gate_declines_a_blocked_family_and_names_the_ids(
    _miss: Any, _events: List[Tuple[str, str, str]],
) -> None:
    """The refusal must still be SAID, typed and groupable, with its evidence
    — not swallowed, and not a quieter string than the thunk's was."""
    outcome = _enable(blocked.BLOCKED_COMPILE, _events)

    assert outcome.self_mint is None, "a blocked family started a mint"
    skipped = [(p, d) for k, p, d in _events if k == "self_mint_skipped"]
    phases = [p for p, _ in skipped]
    assert "declaration_blocked" in phases, phases
    detail = next(d for p, d in skipped if p == "declaration_blocked")
    for ident in blocked.OPEN_IDS:
        assert ident in detail, detail
    assert "RESOLVES WHEN" in detail
    assert "OQ-9-already-settled" not in detail, "a RESOLVED blocker still gates"


def test_the_blocked_pod_keeps_serving(
    _miss: Any, _events: List[Tuple[str, str, str]],
) -> None:
    """Blockers gate MINTING only. The arm is not failed, and nothing about
    the request path changes — this is the property that makes recording a
    refusal cheap enough to be honest about."""
    outcome = _enable(blocked.BLOCKED_COMPILE, _events)

    assert outcome.armed is True
    assert not [k for k, _p, _d in _events if k == "self_mint_started"]


def test_a_family_with_no_open_blockers_is_not_blocked(
    _miss: Any, _events: List[Tuple[str, str, str]],
) -> None:
    """The control, and it is load-bearing: a gate that blocks everything is
    indistinguishable from one that works. The clean declaration declines
    later, for a reason that is about the composed pipeline, never about
    blockers."""
    _enable(_clean(), _events)

    phases = [p for k, p, _d in _events if k == "self_mint_skipped"]
    assert "declaration_blocked" not in phases, phases


def _blocked_request(tmp_path: Path) -> mp.MintRequest:
    """A mint request for the blocked family, built through the REAL parent
    chain and round-tripped through msgspec — the boundary IS a file."""
    tree = tmp_path / "weights"
    tree.mkdir(parents=True, exist_ok=True)
    pending = SimpleNamespace(
        family=FAMILY, arm_token="ck1-blocked", recipe="aot",
        cfg=CompileCell(shapes=((64, 64),), targets=("transformer",),
                        family=FAMILY, regional=False, text_len=128,
                        dynamic=(), lora_bucket=0, guidance_scales=(),
                        text_lens=()),
        target=tmp_path / "cell.tar.gz", mint_root=tmp_path)
    task = mint_process.MintTask(
        pending=pending, pipe=object(), function="blocked-echo",
        modules=(HARNESS_MODULE,),
        slots={"pipeline": child_contract.MintSlot(
            ref=blocked.DECLARED_PIPELINE, path=str(tree))},
        execution_lane="w8a8", device=-1)
    request = mint_process.build_request(
        task, workdir=tmp_path / "w")
    return msgspec.json.decode(msgspec.json.encode(request), type=mp.MintRequest)


def test_the_mint_child_refuses_a_blocked_family_before_it_reads_a_weight(
    tmp_path: Path,
) -> None:
    """The FAIL-CLOSED half, on the real entrypoint: ``python -m
    gen_worker.mint_child request.json``, spawned by the parent's real
    supervisor. The parent declines a blocked family in ``mint_recipe``, so a
    request that reaches a child came from somewhere else — an operator CLI, a
    delegated request built against a stale declaration — and a refusal only
    one of the two paths honours is not a refusal (pgw#1080's lesson: split by
    type and fail closed; never degrade).
    """
    request = _blocked_request(tmp_path)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(Path(__file__).resolve().parent), env.get("PYTHONPATH", "")])
    outcome = asyncio.run(mp.run_mint(
        request, workdir=tmp_path / "w", python=sys.executable, env=env))

    assert outcome.status == mp.REFUSED, (
        f"{outcome.status}: {outcome.detail or outcome.stderr_tail}")
    assert outcome.exit_code == mp.EXIT_REFUSED, "a refusal must be TERMINAL"
    report = outcome.report
    assert report is not None and report.status == "refused"
    for ident in blocked.OPEN_IDS:
        assert ident in report.detail, report.detail
    # Before any side effect the child could have: no artifact, no weights.
    assert not Path(request.target).exists()
    assert "warmup_forward" not in report.phases, report.phases


def test_the_mint_child_mints_a_family_whose_blockers_are_all_resolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The control for the child gate — in-process, because the point is the
    gate and not the export: a declaration with no open blocker passes it
    silently, and one with an open blocker does not."""
    reset_export_declarations()
    try:
        from gen_worker.api.export_contract import register_export_declaration

        register_export_declaration(_clean(), replace=True)
        mint_child._assert_family_mintable(FAMILY)  # no raise

        register_export_declaration(blocked.BLOCKED_COMPILE, replace=True)
        with pytest.raises(child_preflight.PreflightRefused) as exc:
            mint_child._assert_family_mintable(FAMILY)
        assert all(i in str(exc.value) for i in blocked.OPEN_IDS)
    finally:
        reset_export_declarations()


def test_the_image_BUILD_says_a_family_is_blocked_without_renting_a_pod() -> None:
    """pgw#996's property, preserved across the fold: the static gate already
    recorded a thunk's ``MintRefused`` as a ``blocked`` verdict, and it must
    recognise the declared form too — otherwise folding a family turns a
    build-time sentence into silence."""
    from gen_worker import aot_preconditions as pre
    from gen_worker.api.export_contract import register_export_declaration

    reset_export_declarations()
    try:
        register_export_declaration(blocked.BLOCKED_COMPILE, replace=True)
        rows = pre.static_mint_preconditions({FAMILY: 0}, torch_available=True,
                                             torch_version="2.13.0+cu130")
        verdicts = {r.check: r for r in rows}
        row = verdicts[pre.CHECK_DECLARATION_EVALUATES]
        assert row.verdict == pre.BLOCKED, [r.manifest_row() for r in rows]
        assert all(i in row.detail for i in blocked.OPEN_IDS)
        # A family that will never mint owes this image no toolchain row.
        assert pre.CHECK_CXX_TOOLCHAIN not in verdicts

        register_export_declaration(_clean(), replace=True)
        rows = pre.static_mint_preconditions({FAMILY: 0}, torch_available=True,
                                             torch_version="2.13.0+cu130")
        assert {r.check: r.verdict for r in rows}[
            pre.CHECK_DECLARATION_EVALUATES] == pre.OK
    finally:
        reset_export_declarations()


# ---------------------------------------------------------------------------
# 4. A fold cannot LOSE a refusal
# ---------------------------------------------------------------------------


def test_the_migration_gate_stops_a_fold_that_drops_a_blocker() -> None:
    """pgw#1107's per-family gate is ``assert_faithful(standing, migrated)``.
    A dropped blocker re-keys nothing and loosens nothing, so neither
    ``contract_delta`` nor ``override_delta`` sees it — it just starts minting
    against an open design question."""
    standing = blocked.BLOCKED_COMPILE
    migrated = _clean()
    assert contract_delta(standing, migrated) == {}
    assert blocker_delta(standing, migrated) == blocked.OPEN_IDS

    with pytest.raises(DeclarationMismatch) as exc:
        assert_faithful(standing, migrated, family=FAMILY)
    assert "REFUSAL dropped" in str(exc.value)
    for ident in blocked.OPEN_IDS:
        assert ident in str(exc.value)


def test_a_fold_that_ADDS_a_blocker_is_faithful() -> None:
    """Directional on purpose. ltx-video-2.3's blockers live OUTSIDE its
    Compile today (a module table read by a refusing thunk), so its fold
    ADDS them to the declaration — and that is the fold working, not a
    mismatch."""
    assert blocker_delta(_clean(), blocked.BLOCKED_COMPILE) == ()
    assert_faithful(_clean(), blocked.BLOCKED_COMPILE, family=FAMILY)


def test_resolving_a_blocker_in_the_MIGRATED_half_counts_as_dropping_it() -> None:
    """Resolving is a reviewable edit to the standing declaration, never a
    side effect of a move."""
    rows = tuple(
        msgspec.structs.replace(b, resolved=True, resolution="in the fold")
        for b in blocked.BLOCKED_COMPILE.blockers)
    assert blocker_delta(
        blocked.BLOCKED_COMPILE, _clean(blockers=rows)) == blocked.OPEN_IDS


def test_assert_blockers_is_the_per_family_guard() -> None:
    """The gate above can only compare two Compiles, and the family that most
    needs the guard keeps its blockers outside its Compile today — so the
    expectation is stated in the family's OWN test, where it survives the file
    the blockers used to live in."""
    assert_blockers(blocked.BLOCKED_COMPILE, ids=blocked.OPEN_IDS, family=FAMILY)
    assert_blockers(_clean(), ids=(), family=FAMILY)

    with pytest.raises(DeclarationMismatch) as exc:
        assert_blockers(_clean(), ids=blocked.OPEN_IDS, family=FAMILY)
    assert "MISSING" in str(exc.value)

    with pytest.raises(DeclarationMismatch) as exc:
        assert_blockers(blocked.BLOCKED_COMPILE, ids=(), family=FAMILY)
    assert "UNEXPECTED" in str(exc.value)
