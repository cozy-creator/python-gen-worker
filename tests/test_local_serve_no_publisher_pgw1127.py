"""pgw#1127 S1 — cozy-local can REACH the sink pgw#1096 built, and can never publish.

Two things are proven here, and they are different in kind.

**Reachability** (the live defect). pgw#1096 built ``local_cell_store``, keyed it
by ``ck1``, and wired it into ``fleet_cells._arming_policy`` local-first. It left
``cli/run.py`` pointing at ``local_cells.enable_compiled`` — the JIT path — so
``_arming_policy`` was never entered from ``cozy serve`` and the machine §4.28 was
written about got none of it. RED before this issue on every test in section 1:
the arming import in ``gen_worker.cli`` was ``..local_cells``, and
``gen_worker.local_serve`` did not exist.

**Never-publish, STRUCTURALLY** (§4.28's *"never uploaded"*). Before S1 that
property held because the local CLI could not reach the publishing module at
all. After S1 it runs ``fleet_cells``, which imports ``CellPublisher`` — so
"cozy-local never publishes" would rest on one keyword argument at one call
site. Section 2 is the fence that replaces the convention: the local serve entry
names no publisher, constructs none, and reaches no publish call, read out of
the source tree rather than asserted about a run. Each of those is RED-provable
by deleting ``publisher=None`` or adding a ``CellPublisher(...)`` — which is the
only reason to prefer it to a mock.
"""

from __future__ import annotations

import ast
import io
import json
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

from gen_worker import fleet_cells, local_serve
from gen_worker.cell_adopt import AdoptOutcome
from gen_worker.cli import run as cli_run
from gen_worker.child_contract import MintSlot

KEY_A = "cg-key-v1-" + "a" * 56
ARM_A = fleet_cells.ARM_SCHEME + "-" + "1" * fleet_cells.ARM_DIGEST_HEX

SRC = Path(fleet_cells.__file__).parent

#: The local serve ENTRY: every file a `cozy serve` / `cozy run` arm passes
#: through on its way to the arming brain. The fence in section 2 reads all of
#: them, because a publisher constructed one frame up the stack is a publisher.
LOCAL_SERVE_ENTRY = (
    SRC / "local_serve.py",
    SRC / "cli" / "run.py",
    SRC / "cli" / "serve.py",
)


class _Pipe:
    pass


@dataclass
class _Cfg:
    family: str = "micro-diffusion"
    lora_bucket: int = 0
    shapes: Tuple[Tuple[int, int], ...] = ((64, 64),)
    targets: Tuple[str, ...] = ("transformer",)
    text_lens: Tuple[int, ...] = (16,)
    guidance_scales: Tuple[float, ...] = (1.0,)
    regional: bool = False


class _Arm:
    def __init__(self, token: str = ARM_A) -> None:
        self.token = token

    def facts_dict(self) -> Dict[str, str]:
        return {}


def _armable_artifact(tmp_path: Path, *, key: str = KEY_A) -> Path:
    """A cell with a READABLE envelope — `_arm_exported_cell` refuses an
    unreadable one before every other gate (pgw#1098), local store included."""
    p = tmp_path / "mint" / "cell.tar.gz"
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        {"kind": "aot-inductor", "cell_key": key, "family": "micro-diffusion"}
    ).encode()
    with tarfile.open(p, mode="w:gz") as tar:
        info = tarfile.TarInfo("metadata.json")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    return p


@pytest.fixture()
def machine(monkeypatch: pytest.MonkeyPatch) -> None:
    """The FACTS about this box that a CPU runner cannot supply: a card, a C
    toolchain, a resolvable compile target, a family that declares an export.

    Deliberately only facts. Nothing in the decision under test —  the store
    consult, its ordering against the pending, the sink choice, the arm gate —
    is stubbed, because a test that stubs the thing it is measuring proves the
    stub.
    """
    monkeypatch.setattr(fleet_cells, "_cuda_ready", lambda: True)
    monkeypatch.setattr(fleet_cells.cc, "toolchain_present", lambda: True)
    monkeypatch.setattr(
        fleet_cells.cc, "has_compile_target", lambda pipe, cfg: True)
    monkeypatch.setattr(
        fleet_cells, "mint_recipe",
        lambda pipe, cfg, **kw: fleet_cells.RECIPE_AOT)
    monkeypatch.setattr(
        fleet_cells, "arm_identity", lambda *a, **k: _Arm())
    monkeypatch.setattr(
        fleet_cells.provision, "enable_compiled",
        lambda pipe, cfg, cache_dir=None, artifact=None: AdoptOutcome.miss(
            "no_cell", "no artifact was delivered to this machine"))


@pytest.fixture()
def armable(monkeypatch: pytest.MonkeyPatch) -> List[Path]:
    """`provision.arm_aot` succeeds; record WHICH artifact it was handed."""
    seen: List[Path] = []

    def _arm(pipe: Any, cfg: Any, cache_dir: Any, artifact: Path,
             bucket: int, expected: Any = None, *,
             verify_numerics: bool = False, **_kw: Any) -> AdoptOutcome:
        # pgw#1141 / §4.32: the local store's route is an ADOPTION — these
        # bytes were proven at their own mint — so it must not ask for the
        # mint-time gate. Asserted rather than absorbed: a bare `**kwargs`
        # shim would keep this file green if the per-adopter tax came back.
        assert verify_numerics is False, (
            "the local store's ADOPT path asked for the mint-time gate")
        seen.append(Path(artifact))
        return AdoptOutcome.hit(KEY_A)

    monkeypatch.setattr(fleet_cells.provision, "arm_aot", _arm)
    monkeypatch.setattr(
        fleet_cells.artifact_meta, "read_metadata",
        lambda p: {"cell_key": KEY_A, "family": "micro-diffusion"})
    monkeypatch.setattr(
        fleet_cells.artifact_meta, "try_read_metadata",
        lambda p: {"cell_key": KEY_A, "family": "micro-diffusion"})
    monkeypatch.setattr(fleet_cells, "arm_axis_divergence", lambda key, meta: "")
    return seen


def _ctx() -> local_serve.LocalMintContext:
    return local_serve.mint_context(
        function="generate", module="micro_diffusion.endpoint",
        slots={"pipeline": MintSlot(ref="cozy/micro#1", path="/tmp/micro")},
    )


# ---------------------------------------------------------------------------
# 1. REACHABILITY — the local CLI arms through the AOT sink, not the JIT one
# ---------------------------------------------------------------------------


def _arming_import(module_file: Path) -> str:
    """The module `_load_injected_model` imports its arming entry from."""
    tree = ast.parse(module_file.read_text())
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_load_injected_model")
    names: List[str] = []
    for node in ast.walk(fn):
        if isinstance(node, ast.ImportFrom):
            # `from .. import local_serve` carries the name on the ALIAS, not
            # on `.module` (which is None at level 2) — so both are read, or
            # the fence answers "" for exactly the form this issue introduced.
            names.append(node.module or "")
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
    return ";".join(names)


def test_the_local_cli_arms_through_the_AOT_sink_and_never_the_JIT_one() -> None:
    """§4.28's second clause, at the ONE call site that decides whether cozy-local
    has it at all.

    RED before pgw#1127: this read ``..local_cells`` — the JIT path, which
    imports ``compile_cache`` and nothing that mints AOT. ``_arming_policy``,
    and therefore the entire local store, was unreachable from ``cozy serve``.
    """
    imported = _arming_import(SRC / "cli" / "run.py")
    assert "local_serve" in imported, (
        "the local serve entry must arm through `local_serve` — the fleet "
        "arming brain with no sink — so a `cozy serve` miss consults THIS "
        "MACHINE's ck1 store before it mints anything")
    assert "local_cells" not in imported, (
        "the JIT local-serve path is what pgw#1086 wave 1 deletes; arming "
        "through it is how cozy-local loses compiled serving outright")


def test_nothing_under_the_cli_imports_the_JIT_local_cell_module() -> None:
    """Reachability has to be a property of the PACKAGE, not of one function.

    A sibling module re-importing `local_cells` would restore the coupling this
    issue exists to cut, and would put `cozy` back on pgw#1086 wave 1's
    critical path.
    """
    offenders: List[str] = []
    for path in sorted((SRC / "cli").rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and "local_cells" in (
                    node.module or ""):
                offenders.append(f"{path.name}:{node.lineno}")
            elif isinstance(node, ast.ImportFrom) and any(
                    a.name == "local_cells" for a in node.names):
                offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, f"cli still reaches the JIT local store: {offenders}"


def test_a_mint_context_missing_its_module_is_INCOMPLETE_not_silently_empty(
) -> None:
    """A module list a child cannot import is a machine that compiles nothing,
    forever, and says nothing about it."""
    assert local_serve.mint_context(
        function="f", module="pkg.mod", slots={}).incomplete == ""
    assert local_serve.mint_context(
        function="f", module="", slots={}).incomplete
    assert local_serve.mint_context(
        function="", module="pkg.mod", slots={}).incomplete


def test_a_slot_with_bytes_and_no_identity_is_ABSENT_never_half_present(
) -> None:
    """``MintSlot``'s rule (pgw#974), enforced where the local CLI builds them:
    ``{"pipeline": "/tmp/x"}`` with no binding decoded, type-checked and looked
    complete, and the child died 0.0 s into ``warmup_forward``."""
    slots = local_serve.slot_map(
        {"pipeline": "/tmp/a", "refiner": "/tmp/b", "empty": ""},
        {"pipeline": "cozy/x#1", "empty": "cozy/z#1"},
    )
    assert set(slots) == {"pipeline"}
    assert slots["pipeline"].path == "/tmp/a"


def test_the_local_run_resolves_the_WHOLE_setup_not_one_slot_at_a_time(
) -> None:
    """The child re-runs ``setup()``, so it needs every slot the endpoint
    declares — a context built inside the per-slot loader would carry whichever
    slot happened to reach the arm first."""
    tree = ast.parse((SRC / "cli" / "run.py").read_text())
    loader = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_load_injected_model")
    built_in_loader = [
        c for c in ast.walk(loader)
        if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
        and c.func.id == "_local_mint_context"
    ]
    assert not built_in_loader, (
        "the mint context must be built once from the whole resolution, above "
        "the per-slot loader")
    assert any(
        isinstance(n, ast.FunctionDef) and n.name == "run_setup"
        and any(
            isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
            and c.func.id == "_local_mint_context" for c in ast.walk(n))
        for n in ast.walk(tree)), "run_setup must build it"


def test_compile_children_never_open_a_mint_of_their_own() -> None:
    """``aot_compile_child`` and ``boot_trace_child`` call ``run_setup``. Neither
    may arm — a child that opened its own mint is the recursion this must not
    have — and neither names a function, so the context is None by
    construction as well as by ``arm_compile=False``."""
    for module in ("aot_compile_child.py", "boot_trace_child.py"):
        tree = ast.parse((SRC / module).read_text())
        for call in ast.walk(tree):
            if not (isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Name)
                    and call.func.id == "run_setup"):
                continue
            kwargs = {kw.arg: kw.value for kw in call.keywords}
            assert "arm_compile" in kwargs, f"{module}: run_setup must be explicit"
            value = kwargs["arm_compile"]
            assert isinstance(value, ast.Constant) and value.value is False, (
                f"{module}: a mint/trace child must never arm a cell under itself")
            assert "selected" not in kwargs, (
                f"{module}: a child must not be able to open a mint")


def test_the_cli_serve_path_names_the_function_so_it_can_mint() -> None:
    """The reachability half's other end: without ``selected=`` the local entry
    can only ADOPT, so a machine with an empty store would stay eager forever
    and §4.28's *"compile ONCE"* would never happen at all."""
    tree = ast.parse((SRC / "cli" / "serve.py").read_text())
    calls = [
        c for c in ast.walk(tree)
        if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
        and c.func.attr == "run_setup"
    ]
    assert calls, "cli/serve.py must still be the serve path"
    for call in calls:
        assert any(kw.arg == "selected" for kw in call.keywords), (
            "`cozy serve` must name the function it is arming, or it can "
            "never mint the first cell")


def test_run_setup_signature_carries_the_selected_function() -> None:
    """Cheap, and it is the thing every other test in section 1 rests on."""
    import inspect

    params = inspect.signature(cli_run.run_setup).parameters
    assert "selected" in params and params["selected"].default is None
