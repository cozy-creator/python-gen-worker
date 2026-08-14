"""pgw#1053 — 25.7 GiB of the L40S was held by two residents that did NOTHING
through the 97-minute compile phase.

The two residents:

* the serving PARENT's eager pipeline — 9.54 GiB, now always a live tenant's
  and therefore never parked (see the last section);
* the MINT CHILD's own pipeline plus the retained programs' weight aliases —
  16.2 GiB of dead weight from the moment the last row exports.

What is asserted here:

* the mint-child release is a code-only PROJECTION: weights go to meta,
  literals keep their bytes, and every identity fact the parent-side gates
  read (FQN sets, graph hash, literal digest) is unchanged — so no gate is
  dropped and the cell key cannot move;
* a full mint with ``release_residents=True`` still packs, still keys
  identically, and provably released the pipeline;
* the pool REGRANTS K when the residents come back — through the same
  pgw#992-bounded arithmetic, never past the card's simultaneous budget, and
  never touching the co-tenant term (the census predates every child and the
  release re-baselines only the OWN floor);
* the PARENT-side park is DELETED (§4.28 / pgw#1092): it was reachable only
  on a pod holding no serve goal, and that pod class no longer exists.
"""

from __future__ import annotations

import types
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn  # noqa: E402

from gen_worker import aot_compile_pool as pool_mod  # noqa: E402
from gen_worker import (  # noqa: E402
    aot_flatten, aot_mint, aot_package, aot_serve, compile_cache, graph_hash,
    mint_delegate)
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
        inputs=(Input("sample", shape=("B", 4), dtype="model"),),
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


def _wide_pool(monkeypatch) -> None:
    """The width is STATED, not derived (the pgw#809 discipline): a 4-vCPU CI
    runner honestly derives K=1 and the release below only runs on the pooled
    path. The REAL policy runs on pinned resource inputs."""
    real = pool_mod.entry_workers

    def _wide(entries: int, **kw: Any) -> Any:
        kw.update(vcpus=16, available_bytes=64 * _GIB, device_lock=True)
        return real(entries, **kw)

    monkeypatch.setattr(pool_mod, "entry_workers", _wide)


def test_full_mint_with_release_packs_and_keys_identically(
    tmp_path: Path, monkeypatch,
) -> None:
    """The whole path: mint, release once every row has exported, every
    package gate runs against the projection, the artifact keys identically —
    and the pipeline is PROVABLY released (weights on meta afterwards).

    pgw#1215: the release used to be reachable only on the OVERLAPPED pooled
    path (it fired when that path's producer was exhausted), which meant a
    serial mint held its residents through packaging for no reason. The
    overlapped path is gone — a compile child traces its own share, so the
    parent no longer exports at all when it is K-wide — and the release now
    runs at the same point on the one path there is: after the last row
    exports, before anything packs."""
    _fake_sm(monkeypatch)
    _wide_pool(monkeypatch)
    _declare()
    spec = aot_mint.ExportSpec(family=FAMILY, target="")

    keeper = types.SimpleNamespace(unet=TinyUNet())
    kept = aot_mint.mint(keeper, spec, tmp_path / "kept")

    surrendered = types.SimpleNamespace(unet=TinyUNet())
    released = aot_mint.mint(
        surrendered, spec, tmp_path / "released", release_residents=True)

    # a mint produces N independently keyed artifacts, not "a cell",
    # so the pgw#846 claim is now per ENTRY. The declaration above traces two
    # graph classes; the length assert is what stops an empty `entries` from
    # making the comparison below pass vacuously.
    kept_keys = {r.entry: r.key for r in kept.entries}
    released_keys = {r.entry: r.key for r in released.entries}
    assert len(kept_keys) == 2, kept_keys
    assert kept_keys == released_keys, (
        "the release re-keyed an entry — the projection leaked into identity "
        "(pgw#846)")
    assert all(
        p.device.type == "meta" for p in surrendered.unet.parameters()), (
        "release_residents=True did not release the pipeline")
    assert "residents_release_s" in released.timings
    # ...and the mint that did NOT ask for it kept its weights, so the row
    # above is a property of `release_residents=True` and not of minting.
    assert all(
        p.device.type != "meta" for p in keeper.unet.parameters()), (
        "the mint released a pipeline whose owner never surrendered it")


# ---------------------------------------------------------------------------
# The pool REGRANT — DELETED WITH THE BUDGET IT REGRANTED
# ---------------------------------------------------------------------------
#
# Four rows stood here. They drove the incident pod's real numbers (44.39 GiB
# card, 9.54 GiB serving parent, 16.20 GiB mint-child pipeline, 6.02 GiB
# measured entry peak) through `card_census` -> `entry_budget_bytes` ->
# `_apply_simultaneity_bound` / `_spawn_admitted` / `_rewiden`, and asserted
# that handing 16.2 GiB back moved K. Every one of those functions is deleted:
# §4.33 forbids predicting VRAM, and K is f(cores, one measured child RSS).
#
# WHAT THE ROWS GUARDED, AND WHERE IT GOES. Their subject was "K children's
# simultaneous device peak must fit beside the residents" — a prediction, and
# a wrong one wherever the child's own weights were its leading term. It is
# replaced by the attempt: a compile child that exceeds the card dies in its
# own process and is classified `MintResourceExhausted`
# (`test_mint_oom_classification_pgw848`), which is also what teaches the next
# attempt (the host-RSS bank).
#
# THE RELEASE ITSELF SURVIVES and is covered above: `release_residents=True`
# still projects the mint child's pipeline to meta and hands the memory back,
# and `residents_release_s` / `residents_released_bytes` still ride the
# timings. What is gone is the arithmetic that used to spend it.


# ---------------------------------------------------------------------------
# The PARENT half is GONE (§4.28 / pgw#1092)
# ---------------------------------------------------------------------------


def test_the_parent_never_parks_its_eager_pipeline_any_more() -> None:
    """pgw#1053's park was reachable ONLY on a pod holding no serve goal — a
    forge pod. §4.28 deleted that pod class, so the park is unreachable code
    and is deleted with it: every mint now runs beside a live tenant whose
    eager pipeline stays resident and hot (Paul ruling 2).

    RED before this change: `mint_delegate.maybe_park_eager` existed and
    `build_cell` called it before the first budget probe.
    """
    for gone in ("ParkedEager", "maybe_park_eager", "restore_eager_pipeline",
                 "_eager_modules"):
        assert not hasattr(mint_delegate, gone), gone
    assert "park" not in mint_delegate.__all__
