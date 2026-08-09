"""pgw#1053 — 25.7 GiB of the L40S was held by two residents that did NOTHING
through the 97-minute compile phase.

The two residents, measured (attempts 24/26/30):

* the serving PARENT's eager pipeline — 9.54 GiB on a pod whose goal set
  admits no tenant dispatch (it serves nobody; the heartbeat needs no VRAM);
* the MINT CHILD's own pipeline plus the retained programs' weight aliases —
  16.2 GiB of dead weight from the moment the last row exports.

What is asserted here:

* the mint-child release is a code-only PROJECTION: weights go to meta,
  literals keep their bytes, and every identity fact the parent-side gates
  read (FQN sets, graph hash, literal digest) is unchanged — so no gate is
  dropped and the cell key cannot move (pgw#846);
* a full mint with ``release_residents=True`` still packs, still keys
  identically, and provably released the pipeline;
* the pool REGRANTS K when the residents come back — through the same
  pgw#992-bounded arithmetic, never past the card's simultaneous budget, and
  never touching the co-tenant term (the census predates every child and the
  release re-baselines only the OWN floor);
* the PARENT-side park is decided by the GOALS machinery (pgw#930): a pod
  holding a serve goal keeps eager resident and hot (Paul ruling 2 — this is
  the serving-pod half of the asymmetry), a pod with no serve goal parks to
  host RAM for the mint and restores before ADOPT.
"""

from __future__ import annotations

import types
from pathlib import Path
from typing import Any, List

import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn  # noqa: E402

from gen_worker import aot_compile_pool as pool_mod  # noqa: E402
from gen_worker import (  # noqa: E402
    aot_flatten, aot_mint, aot_package, aot_serve, compile_cache, graph_hash,
    mint_delegate, worker_goals)
from gen_worker.api.decorators import Compile  # noqa: E402
from gen_worker.api.export_contract import (  # noqa: E402
    Dim,
    GraphClass,
    Input,
    register_export_declaration,
    reset_export_declarations,
)

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")

_GIB = 1024 ** 3


# ---------------------------------------------------------------------------
# The projection: code-only, gates intact, identity frozen
# ---------------------------------------------------------------------------


class _WithLiteral(nn.Module):
    """A weight (state_dict), a buffer (state_dict) and a LITERAL — a plain
    attribute tensor with no state_dict counterpart, the qwen/z-image rope
    shape pgw#857 keys by value."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(8, 8)
        self.register_buffer("scale", torch.ones(8))
        self.table = torch.linspace(0.0, 1.0, 8)   # literal: plain attribute

    def forward(self, x: Any) -> Any:
        return self.lin(x) * self.scale + self.table


def _minted_entry(module: nn.Module) -> Any:
    program = torch.export.export(module.eval(), (torch.randn(2, 8),))
    return aot_mint._MintedEntry(
        name="unet/B=2", spec=aot_mint.ExportSpec(family="f", target="unet"),
        module=None, owner=module, program=program,
        input_names=("x",),
        flat_leaves=(aot_flatten.Leaf(param="x", param_position=0, path=()),),
        files=[], timings={})


def test_release_projects_programs_code_only() -> None:
    module = _WithLiteral()
    row = _minted_entry(module)
    program = row.program

    weights_before = aot_package.program_state_dict_fqns(program)
    literal_digest_before = aot_package.literal_values_digest(program)
    graph_before = graph_hash.graph_hash(program)
    assert literal_digest_before, "the fixture must actually lift a literal"

    pipe = types.SimpleNamespace(unet=module)
    facts = aot_mint._release_mint_residents(pipe, [row])

    # The weights are GONE from residency...
    for fqn in weights_before:
        value = program.state_dict.get(fqn)
        assert value is not None and value.device.type == "meta", (
            f"{fqn}: a state_dict constant survived the projection resident")
    for p in module.parameters():
        assert p.device.type == "meta", "the module's storage stayed resident"
    # ...and every identity fact the gates read is untouched.
    assert aot_package.program_state_dict_fqns(program) == weights_before
    assert aot_package.literal_values_digest(program) == literal_digest_before, (
        "the literal's BYTES are part of the artifact (pgw#857) and must "
        "survive the release")
    assert graph_hash.graph_hash(program) == graph_before
    assert facts.get("residents_release_modules", 0) >= 1


def test_release_is_not_wired_into_library_mints_by_default(
    tmp_path: Path, monkeypatch,
) -> None:
    """`release_residents` is the CALLER's lifecycle statement. A library
    caller that keeps its pipeline must find it untouched after `mint()`."""
    _fake_sm(monkeypatch)
    _declare()
    pipe = types.SimpleNamespace(unet=TinyUNet())
    spec = aot_mint.ExportSpec(family=FAMILY, target="")
    aot_mint.mint(pipe, spec, tmp_path / "out")
    assert all(p.device.type != "meta" for p in pipe.unet.parameters()), (
        "a default mint released a pipeline it does not own")


# ---------------------------------------------------------------------------
# A full mint with the release: packs, keys identically, provably released
# ---------------------------------------------------------------------------

FAMILY = "tiny1053"


class TinyUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, sample: Any) -> Any:
        return torch.tanh(self.lin(sample)) + 1.0


def _declare() -> Any:
    return register_export_declaration(Compile(
        family=FAMILY,
        targets=("unet",),
        dims=(Dim("B", carried_by=(("sample", 0),)),),
        classes=(GraphClass(dims={"B": 2}), GraphClass(dims={"B": 1})),
        inputs=(Input("sample", shape=("B", 4)),),
        shape_strategy="static-rows",
        warm_changes_key=False,
    ))


@pytest.fixture(autouse=True)
def _fresh_registry():
    reset_export_declarations()
    yield
    reset_export_declarations()


def _fake_sm(monkeypatch) -> None:
    full = {"sku": "", "sm": "sm_89", "torch": str(torch.__version__),
            "cuda": ""}
    monkeypatch.setattr(compile_cache, "runtime_key", lambda: dict(full))
    monkeypatch.setattr(aot_serve, "runtime_key", lambda: {
        "sku": full["sku"], "sm": full["sm"], "torch": full["torch"],
        "cuda": full["cuda"]})


def test_full_mint_with_release_packs_and_keys_identically(
    tmp_path: Path, monkeypatch,
) -> None:
    """The whole path: overlapped mint, release at producer exhaustion, every
    package gate runs against the projection, the artifact keys identically —
    and the pipeline is PROVABLY released (weights on meta afterwards)."""
    _fake_sm(monkeypatch)
    _declare()
    spec = aot_mint.ExportSpec(family=FAMILY, target="")

    keeper = types.SimpleNamespace(unet=TinyUNet())
    kept = aot_mint.mint(keeper, spec, tmp_path / "kept")

    surrendered = types.SimpleNamespace(unet=TinyUNet())
    released = aot_mint.mint(
        surrendered, spec, tmp_path / "released", release_residents=True)

    assert kept.cell_key == released.cell_key, (
        "the release re-keyed the cell — the projection leaked into identity "
        "(pgw#846)")
    assert all(
        p.device.type == "meta" for p in surrendered.unet.parameters()), (
        "release_residents=True did not release the pipeline")
    assert "residents_release_s" in released.timings
    assert released.timings.get("entry_workers", 0) > 1, (
        "the release path only runs on the pooled path; this mint went "
        "serial and proved nothing")


# ---------------------------------------------------------------------------
# The pool regrants K when the residents come back (the pgw#992 pod, freed)
# ---------------------------------------------------------------------------

# The incident pod's numbers, verbatim from test_pool_simultaneity_pgw992.
CARD_TOTAL = 47661043712          # 44.39 GiB
FREE_AT_OPEN = 31664532480        # 29.49 GiB
SERVING_PARENT = 10243173417      # 9.54 GiB co-tenant
OWN_AT_OPEN = CARD_TOTAL - FREE_AT_OPEN - SERVING_PARENT
OWN_PEAK = 17394617548            # 16.20 GiB — the mint child's pipeline
MEASURED_ENTRY_PEAK = 6461325312  # 6.02 GiB


def _census() -> pool_mod.CardCensus:
    return pool_mod.CardCensus(CARD_TOTAL, FREE_AT_OPEN, OWN_AT_OPEN, "sampled")


def _width() -> pool_mod.PoolWidth:
    return pool_mod.entry_workers(
        36, vcpus=256, available_bytes=116 * _GIB, peak_rss_bytes=3 * _GIB,
        free_vram_bytes=FREE_AT_OPEN, device_bytes=int(9.9 * _GIB),
        device_basis="estimated", device_lock=True,
        goals=worker_goals.MINT_ONLY)


def _pool(tmp_path: Path, monkeypatch, *, own_peak: int) -> pool_mod.EntryCompilePool:
    monkeypatch.setattr(pool_mod, "card_census", lambda device=-1: _census())
    monkeypatch.setattr(
        pool_mod, "own_device_high_water", lambda device=-1: own_peak)
    return pool_mod.EntryCompilePool(tmp_path / "pool", width=_width())


def test_release_regrants_K_within_the_card_budget(
    tmp_path: Path, monkeypatch,
) -> None:
    """The incident pod, with the mint child's 16.2 GiB handed back: the SAME
    bounded arithmetic that held K=2 now grants more — and still never more
    than the card's simultaneous budget with the 9.54 GiB co-tenant priced."""
    box = _pool(tmp_path, monkeypatch, own_peak=OWN_PEAK)
    # Two real entry reports first (the measured 6.02 GiB ask), as on the pod.
    for i in range(2):
        box.observe_entry_device(pool_mod.EntryReport(
            entry=f"unet/dim={i}", status=pool_mod.COMPILED,
            peak_device_reserved_bytes=MEASURED_ENTRY_PEAK))
        box._rewiden()
    held = box.width.workers
    assert held == 2, box.width.reason   # the pgw#992 bound, still holding

    # The release: the pipeline is gone, the high-water restarts small.
    residue = 1 * _GIB
    monkeypatch.setattr(
        pool_mod, "own_reserved_now", lambda device=-1: residue)
    monkeypatch.setattr(
        pool_mod, "own_device_high_water", lambda device=-1: residue)
    box.note_residents_released()

    assert box.width.workers > held, (
        f"the release freed {OWN_PEAK / _GIB:.1f} GiB and K did not move "
        f"({box.width.reason})")
    # Bounded: K children's simultaneous peak + co-tenant + residue <= card.
    entry_ask = MEASURED_ENTRY_PEAK + 1 * _GIB  # +context floor
    assert (box.width.workers * entry_ask + SERVING_PARENT + residue
            <= CARD_TOTAL), box.width.reason
    assert box.simultaneity.get("residents_released_bytes", 0) > 0


def test_release_regrant_never_touches_the_cotenant_term(
    tmp_path: Path, monkeypatch,
) -> None:
    """The census predates every child (pgw#992/pgw#1000 ordering) and the
    release re-baselines ONLY the own floor — the 9.54 GiB serving parent
    stays priced. A regrant that erased the co-tenant would re-create the
    incident with a new receipt."""
    box = _pool(tmp_path, monkeypatch, own_peak=OWN_PEAK)
    before = box.census.resident_other_bytes
    monkeypatch.setattr(pool_mod, "own_reserved_now", lambda device=-1: 0)
    monkeypatch.setattr(pool_mod, "own_device_high_water", lambda device=-1: 0)
    box.note_residents_released()
    assert box.census.resident_other_bytes == before
    budget, terms = box.entry_budget_bytes(MEASURED_ENTRY_PEAK)
    assert budget is not None
    assert budget <= CARD_TOTAL - SERVING_PARENT, (
        "the freed budget swallowed the co-tenant's residency")


def test_release_on_an_unreadable_card_regrants_nothing(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setattr(
        pool_mod, "card_census",
        lambda device=-1: pool_mod.CardCensus(0, 0, 0, "unreadable"))
    box = pool_mod.EntryCompilePool(tmp_path / "pool", width=_width())
    held = box.width.workers
    box.note_residents_released()
    assert box.width.workers == held


def test_live_spawn_admission_holds_a_spawn_the_budget_cannot(
    tmp_path: Path, monkeypatch,
) -> None:
    """pgw#992 under overlap: the residents keep growing through the export
    phase, and `_spawn_admitted` re-asks the budget with the LIVE own
    high-water before every spawn. At the incident pod's grown residency the
    third child is held; after the release it is admitted."""
    from dataclasses import replace as dc_replace

    box = _pool(tmp_path, monkeypatch, own_peak=OWN_PEAK)
    # Give the width room so admission, not construction, is the bound.
    box.width = dc_replace(
        box.width, workers=6,
        per_entry_device_bytes=MEASURED_ENTRY_PEAK + 1 * _GIB)
    # budget = 44.39 - 9.54 (co-tenant) - 16.20 (own) = 18.65 GiB -> k_cap 2
    assert box._spawn_admitted(1) is True
    assert box._spawn_admitted(2) is False, (
        "a third child was admitted against a budget that holds two — the "
        "pgw#992 incident, live")
    assert box._spawn_admitted(0) is True, "the serial floor must always spawn"
    # After the release the same question admits more.
    monkeypatch.setattr(pool_mod, "own_reserved_now", lambda device=-1: 1 * _GIB)
    monkeypatch.setattr(
        pool_mod, "own_device_high_water", lambda device=-1: 1 * _GIB)
    box.note_residents_released()
    assert box._spawn_admitted(2) is True


# ---------------------------------------------------------------------------
# The PARENT half: goals decide, serving pods keep eager hot
# ---------------------------------------------------------------------------


class _Poisoned:
    """An object no park may touch when the goals say serve."""

    def __getattr__(self, name: str) -> Any:  # pragma: no cover — the trap
        raise AssertionError(f"a serving pod's pipeline was touched ({name})")


def test_serving_goal_keeps_eager_resident() -> None:
    parked = mint_delegate.maybe_park_eager(
        _Poisoned(), goals=worker_goals.SERVE_ONLY)
    assert parked is None


def test_dual_goal_pod_keeps_eager_resident() -> None:
    """Paul's serve+mint case: the tenant reserve is real and eager stays."""
    goals = worker_goals.WorkerGoals(serve=True, mint=True)
    assert mint_delegate.maybe_park_eager(_Poisoned(), goals=goals) is None


def test_mint_only_goal_parks_cuda_modules_and_restores(monkeypatch) -> None:
    """The decision seam, cardless: a mint-only pod PARKS what is on the
    device and restores it on demand. The move itself is recorded through
    the module's own `.to`, so this passes on the real GPU leg unchanged."""

    class _FakeParam:
        is_cuda = True
        device = "cuda:0"

    class _FakeModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.moves: List[Any] = []

        def parameters(self, recurse: bool = True):  # type: ignore[override]
            yield _FakeParam()

        def to(self, target: Any, *a: Any, **k: Any):  # type: ignore[override]
            self.moves.append(target)
            return self

    module = _FakeModule()
    pipe = types.SimpleNamespace(unet=module)
    parked = mint_delegate.maybe_park_eager(
        pipe, goals=worker_goals.MINT_ONLY)
    assert parked is not None
    assert module.moves == ["cpu"]
    assert mint_delegate.restore_eager_pipeline(parked) is True
    assert module.moves == ["cpu", "cuda:0"]


def test_mint_only_goal_with_nothing_on_device_is_a_noop() -> None:
    pipe = types.SimpleNamespace(unet=nn.Linear(4, 4))  # CPU weights
    assert mint_delegate.maybe_park_eager(
        pipe, goals=worker_goals.MINT_ONLY) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a real card")
def test_park_and_restore_on_the_real_card() -> None:
    """The GPU leg (rig box / GPU CI lane): park frees reserved bytes, the
    weights land in host RAM, restore puts them back."""
    module = nn.Linear(256, 256).cuda()
    pipe = types.SimpleNamespace(unet=module)
    torch.cuda.synchronize()
    parked = mint_delegate.maybe_park_eager(
        pipe, goals=worker_goals.MINT_ONLY)
    assert parked is not None
    assert all(p.device.type == "cpu" for p in module.parameters())
    assert mint_delegate.restore_eager_pipeline(parked) is True
    assert all(p.is_cuda for p in module.parameters())
