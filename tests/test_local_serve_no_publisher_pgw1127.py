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
import atexit
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

import tcg_artifacts
from gen_worker import fleet_cells, local_cell_store, local_serve, mint_supervisor
from gen_worker.cell_adopt import AdoptOutcome
from gen_worker.cli import run as cli_run
from gen_worker.child_contract import MintSlot

# pgw#1283: a REAL TCG envelope, and a key DERIVED from it. `local_cell_store`
# hands its bytes to `Engine.import_artifact` now, which refuses an artifact
# whose metadata does not restate the key it is filed under.
_FIXTURE_DIR = Path(tempfile.mkdtemp(prefix="pgw1127-local-serve-"))
atexit.register(shutil.rmtree, _FIXTURE_DIR, True)
ARTIFACT_A = tcg_artifacts.build(_FIXTURE_DIR / "a.tar.gz", witness="a" * 16)
KEY_A = tcg_artifacts.key_of(ARTIFACT_A)
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


@pytest.fixture()
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "cozy-cells"
    monkeypatch.setenv(local_cell_store.ENV_STORE_DIR, str(root))
    return root


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


def _armable_artifact(tmp_path: Path) -> Path:
    """A cell with a READABLE envelope — `_arm_exported_cell` refuses an
    unreadable one before every other gate (pgw#1098), local store included.
    Since pgw#1283 the store refuses one too, so this is TCG's own envelope."""
    p = tmp_path / "mint" / "cell.tar.gz"
    p.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ARTIFACT_A, p)
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
            "no_compiled_graph", "no artifact was delivered to this machine"))


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
        lambda p: {"compiled_graph_key": KEY_A, "family": "micro-diffusion"})
    monkeypatch.setattr(
        fleet_cells.artifact_meta, "try_read_metadata",
        lambda p: {"compiled_graph_key": KEY_A, "family": "micro-diffusion"})
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


def test_a_second_run_on_this_machine_arms_from_its_own_store_and_never_mints(
    store: Path, tmp_path: Path, machine: None, armable: List[Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compile-once-run-forever, through the entry `cozy serve` actually calls.

    The whole ordering runs for real: delivered-artifact miss -> in-process
    ledgers -> **this machine's store** -> (never reached) the pending. A mint
    opened here would mean the machine recompiled what it already had, so the
    delegated mint is wired to FAIL the test rather than to be counted.

    RED before pgw#1127: `gen_worker.local_serve` did not exist, and the entry
    that did exist could not address a ck1-keyed cell at all.
    """
    monkeypatch.setattr(
        mint_supervisor, "supervise",
        lambda *a, **k: pytest.fail("a machine holding its own cell re-minted"))
    local_cell_store.store(
        _armable_artifact(tmp_path), key=KEY_A, family="micro-diffusion",
        arm_token=ARM_A)

    armed = local_serve.enable_compiled(_Pipe(), _Cfg(), None, mint=_ctx())

    assert armed is True
    assert armable and armable[0] == store / "aot-cells" / KEY_A / "cell.tar.gz"


def test_the_local_entry_hands_the_arming_brain_no_sink_at_all(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """§4.28 at the seam: ``publisher=None`` is what makes ``local_keep_reason``
    answer ``no_publish_sink``, which is what makes the mint's own cell land in
    this machine's store instead of being rmtree'd behind a failed publish."""
    seen: Dict[str, Any] = {}

    def _enable(pipe: Any, cfg: Any, cache_dir: Any = None, artifact: Any = None,
                **kw: Any) -> fleet_cells.ArmOutcome:
        seen.update(kw)
        return fleet_cells.ArmOutcome(armed=True)

    monkeypatch.setattr(fleet_cells, "enable_compiled", _enable)
    assert local_serve.enable_compiled(_Pipe(), _Cfg(), None, mint=_ctx())
    assert "publisher" in seen and seen["publisher"] is None
    assert fleet_cells.no_publish_sink_reason(seen["publisher"]) == (
        fleet_cells.KEEP_NO_PUBLISHER)


# ---------------------------------------------------------------------------
# 2. THE FENCE — never-publish is STRUCTURAL, not a keyword argument
# ---------------------------------------------------------------------------


def test_the_local_serve_entry_constructs_no_publisher(
) -> None:
    """The fence pgw#1127 §4 says is owed.

    RED-provable in one edit: add ``CellPublisher(...)`` anywhere on the local
    serve entry and this fails. That is the difference between a property and a
    habit — ``publisher=None`` at one call site is a habit, and after S1 the
    module the local CLI now runs is one that CAN publish.
    """
    banned = {"CellPublisher", "publish_self_mint", "_publish_async",
              "publish_intent", "presign", "put_object", "upload"}
    offenders: List[str] = []
    for path in LOCAL_SERVE_ENTRY:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            name = (
                node.id if isinstance(node, ast.Name)
                else node.attr if isinstance(node, ast.Attribute)
                else "")
            if name in banned:
                offenders.append(f"{path.name}:{node.lineno} {name}")
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    if alias.name in banned:
                        offenders.append(
                            f"{path.name}:{node.lineno} import {alias.name}")
    assert not offenders, (
        "§4.28: the local serve entry must never name a publish route — "
        f"{offenders}")


def test_the_sinkless_call_is_a_LITERAL_none_and_not_a_variable() -> None:
    """A ``publisher=publisher`` that is None today is a sink tomorrow.

    The keyword is pinned to the literal so that wiring one in is an EDIT to
    this line, seen in review, rather than a value that changes upstream.
    """
    tree = ast.parse((SRC / "local_serve.py").read_text())
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "enable_compiled")
    passed = [
        kw for call in ast.walk(fn)
        if isinstance(call, ast.Call)
        for kw in call.keywords if kw.arg == "publisher"
    ]
    assert passed, "the local entry must state its sink, not inherit a default"
    for kw in passed:
        assert isinstance(kw.value, ast.Constant) and kw.value.value is None, (
            "publisher= must be the literal None on the local serve entry")


def test_the_obligation_ends_at_a_terminus_that_cannot_publish(
    store: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A driven local mint ends at ``keep_self_mint_local`` — pgw#815's rule
    (every obligation ends somewhere nameable) met by a function that takes no
    publisher and therefore cannot grow into one.

    ``publish_self_mint`` would also have "worked": its sinkless branch is a
    WIRING ALARM for a fleet pod that lost its `file_base_url`, and firing it on
    the one machine §4.28 was written about would make correct and broken
    indistinguishable forever.
    """
    monkeypatch.setattr(
        fleet_cells, "publish_self_mint",
        lambda pending: pytest.fail("cozy-local reached the publish gate"))
    pending = fleet_cells.PendingSelfMint(
        family="micro-diffusion", arm_token=ARM_A, ref="repo#x", cfg=_Cfg(),
        target=tmp_path / "cell.tar.gz", mint_root=tmp_path / "mint",
        publisher=None, cache_dir=None, arm_key=_Arm(),
    )
    (tmp_path / "mint").mkdir(exist_ok=True)
    pending._state["minted"] = object()

    fleet_cells.keep_self_mint_local(pending)

    assert fleet_cells.terminus_of(pending) == fleet_cells.TERMINUS_WITHHELD
    assert not (tmp_path / "mint").exists(), "the capture dir must be cleaned"


def test_a_pending_this_process_cannot_drive_is_ended_not_dropped(
    store: Path, tmp_path: Path, machine: None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pgw#815, on the local path: a mint context that names no importable
    module cannot be handed to a child, and the obligation it would have
    discharged must not simply be forgotten."""
    monkeypatch.setattr(
        mint_supervisor, "supervise",
        lambda *a, **k: pytest.fail("an undrivable pending spawned a child"))
    monkeypatch.setattr(
        fleet_cells.provision, "arm_aot",
        lambda *a, **k: pytest.fail("nothing is armed on a miss (pgw#784)"))

    armed = local_serve.enable_compiled(
        _Pipe(), _Cfg(), None,
        mint=local_serve.mint_context(function="f", module="", slots={}))

    assert armed is False
    pending = fleet_cells._PENDING.get(ARM_A)
    assert pending is None or fleet_cells.terminus_of(pending)


def test_storing_a_cell_imports_no_transport_at_all(tmp_path: Path) -> None:
    """§4.28's *"never uploaded"*, proven by ABSENCE in a real interpreter.

    The AST fence in ``test_aot_local_mint_pgw1096`` proves the module names no
    transport. This proves the RUN does not: a cell enters the store in a fresh
    process and no HTTP client, no CAS client and no upload module is resident
    afterwards. A store that reached transport lazily — the one shape an AST
    scan of the module cannot see — fails here.
    """
    root = tmp_path / "store"
    cas = tmp_path / "cas"
    artifact = tmp_path / "cell.tar.gz"
    shutil.copyfile(ARTIFACT_A, artifact)
    program = f"""
import os, sys
os.environ["GEN_WORKER_LOCAL_CELLS_DIR"] = {str(root)!r}
from pathlib import Path
from gen_worker import local_cell_store
cell = local_cell_store.store(
    Path({str(artifact)!r}), key={KEY_A!r}, family="micro-diffusion",
    arm_token={ARM_A!r}, cas_root=Path({str(cas)!r}))
assert cell is not None, "the store refused a well-formed cell"
assert local_cell_store.lookup({KEY_A!r}, cas_root=Path({str(cas)!r})) is not None
banned = [
    m for m in sys.modules
    if m in ("httpx", "requests", "urllib3", "boto3", "aiohttp")
    or m.startswith(("gen_worker.convert", "gen_worker.presigned_upload",
                     "gen_worker.hubio.transport", "gen_worker.transport"))
]
print(",".join(sorted(banned)))
"""
    out = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True,
        timeout=300)
    assert out.returncode == 0, out.stderr
    resident = out.stdout.strip()
    assert not resident, (
        "storing a cell pulled transport into the process: " + resident)


def _store_artifact(tmp_path: Path) -> Path:
    p = tmp_path / "cell.tar.gz"
    shutil.copyfile(ARTIFACT_A, p)
    return p


# ---------------------------------------------------------------------------
# 3. The mint context — what a child needs, resolved once, or declared absent
# ---------------------------------------------------------------------------


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


def test_the_mint_child_path_never_opens_a_mint_of_its_own() -> None:
    """``mint_child`` and ``boot_trace_child`` both call ``run_setup``. Neither
    may arm — a child that opened its own mint is the recursion this must not
    have — and neither names a function, so the context is None by
    construction as well as by ``arm_compile=False``."""
    for module in ("mint_child.py", "boot_trace_child.py"):
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
