"""pgw#1149 — the family's DECLARED compile-vs-eager bar reaches the hub.

th#1811 built the publish-time validation session and the promotability gate:
at initial publish the hub boots one pod, drives real inferences compiled and
eager, and judges the steady-state medians *against the family's own declared
bar*. Its hub-side ingest for `compile.speed_metric` / `min_speedup` /
`blockers` landed with it — and **discovery emitted none of them**, so every
compile-declaring release resolved `bar_undeclared` and enforcement had to ship
defaulted to observe.

The bar therefore lives on the DECLARATION (`Compile`), beside `numerics_floor`
whose precedent it copies field for field — not in the endpoint repo's
`author-ci.toml`, which the wheel cannot see at discovery and which would be a
second file feeding the manifest (the drift pgw#1107 spent a campaign deleting).

What this file pins:

1. the bar is validated where it is DECLARED, with the hub's own rules — a
   stage name never the round trip, `>= 1.0`, and metric+bar as a PAIR
   (the hub's `Bar.Declared` is `metric != "" && min_speedup >= 1.0`);
2. it is NOT a contract axis — declaring or raising a bar must never re-key a
   cell — but it IS an OVERRIDE_FACT, so a migration cannot silently drop it;
3. discovery emits all three onto `fn["compile"]`, blockers as the OPEN ids
   only: the hub reads `len(blockers) > 0` as "the author refuses to mint", so
   a resolved id would park the family in `blocked-by-declaration` forever;
4. one readable shape (`speed_bar`) beside `blocker_rows`, so the endpoint
   repo's torch-free lint reads the bar from the SDK and never re-implements it.

Cardless: no GPU, no pod, no mint, no weights.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any, Dict

import pytest

from gen_worker import Compile
from gen_worker.api.derive import (
    OVERRIDE_FACTS, contract_delta, override_delta,
)
from gen_worker.api.export_contract import DeclarationError, speed_bar

BAR = {"speed_metric": "stage_ms.denoise", "min_speedup": 1.10}


@pytest.fixture()
def tmp_pkg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.syspath_prepend(str(tmp_path))
    return tmp_path


def _write(pkg: Path, main_src: str) -> None:
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "main.py").write_text(textwrap.dedent(main_src))


def _compile_block(tmp_pkg: Path, name: str, decl_src: str) -> Dict[str, Any]:
    """The `compile` block discovery emits for a one-function endpoint whose
    `@endpoint(compile=)` is ``decl_src``. Real discovery, no fixture manifest."""
    from gen_worker.discovery.discover import discover_functions

    _write(tmp_pkg / name, f"""
        import msgspec
        from gen_worker import (
            Compile, Dim, GraphClass, Input, MintBlocker, RequestContext,
            endpoint,
        )

        class In_(msgspec.Struct):
            prompt: str = ""

        class Out_(msgspec.Struct):
            y: str

        @endpoint(compile={decl_src})
        class Gen:
            def generate(self, ctx: RequestContext, data: In_) -> Out_:
                return Out_(y="x")
    """)
    fns = {f["name"]: f
           for f in discover_functions(tmp_pkg, main_module=f"{name}.main")}
    return dict(fns["generate"]["compile"])


# ---------------------------------------------------------------------------
# 1. the bar is declared, and validated where it is declared
# ---------------------------------------------------------------------------

def test_a_declaration_carries_the_family_s_own_bar() -> None:
    decl = Compile(family="probe", shapes=((1024, 1024),), **BAR)
    assert decl.speed_metric == "stage_ms.denoise"
    assert decl.min_speedup == pytest.approx(1.10)


def test_a_bar_below_1_0_is_refused_at_DECLARATION_time() -> None:
    # The hub refuses it at publish as "asking the platform to certify a
    # slowdown"; refusing it here means no such manifest is ever built.
    with pytest.raises(ValueError, match="certify a slowdown"):
        Compile(family="probe", shapes=((1024, 1024),),
                speed_metric="stage_ms.denoise", min_speedup=0.9)


@pytest.mark.parametrize("metric", [
    "total_round_trip_ms", "stage_ms.total_round_trip_ms"])
def test_the_round_trip_is_refused_BY_NAME(metric: str) -> None:
    # a bar declared against the round trip measures the network and
    # the queue and calls it the model — the "10.9x" that corrected to 1.3x.
    with pytest.raises(ValueError, match="round trip"):
        Compile(family="probe", shapes=((1024, 1024),),
                speed_metric=metric, min_speedup=1.1)


def test_a_metric_that_names_no_stage_is_refused() -> None:
    with pytest.raises(ValueError, match="stage_ms"):
        Compile(family="probe", shapes=((1024, 1024),),
                speed_metric="denoise", min_speedup=1.1)


@pytest.mark.parametrize("half", [
    {"speed_metric": "stage_ms.denoise"}, {"min_speedup": 1.2}])
def test_the_bar_is_a_PAIR_and_half_of_one_is_refused(half: Dict[str, Any]) -> None:
    # The hub's `Bar.Declared` is `metric != "" && min_speedup >= 1.0`, so half
    # a bar is `bar_undeclared` with extra steps: refuse it at the source.
    with pytest.raises(ValueError, match="a bar is a PAIR"):
        Compile(family="probe", shapes=((1024, 1024),), **half)


# ---------------------------------------------------------------------------
# 2. not a contract axis; IS a must-survive override fact
# ---------------------------------------------------------------------------

def test_declaring_a_bar_NEVER_re_keys_a_cell() -> None:
    bare = Compile(family="probe", shapes=((1024, 1024),))
    barred = Compile(family="probe", shapes=((1024, 1024),), **BAR)
    assert contract_delta(bare, barred) == {}
    assert "min_speedup" not in barred.contract_axes()
    assert "speed_metric" not in barred.contract_axes()


def test_a_migration_that_DROPS_the_declared_bar_is_caught() -> None:
    # numerics_floor's precedent exactly: outside contract_axes(), so
    # contract_delta alone would wave the loss through.
    standing = Compile(family="probe", shapes=((1024, 1024),), **BAR)
    migrated = Compile(family="probe", shapes=((1024, 1024),))
    assert contract_delta(standing, migrated) == {}
    assert override_delta(standing, migrated) == {
        "speed_metric": ("stage_ms.denoise", ""),
        "min_speedup": (1.10, None),
    }
    assert {"speed_metric", "min_speedup"} <= set(OVERRIDE_FACTS)


def test_a_migration_that_LOWERS_the_declared_bar_is_caught() -> None:
    standing = Compile(family="probe", shapes=((1024, 1024),), **BAR)
    lowered = Compile(family="probe", shapes=((1024, 1024),),
                      speed_metric="stage_ms.denoise", min_speedup=1.0)
    assert override_delta(standing, lowered) == {"min_speedup": (1.10, 1.0)}


# ---------------------------------------------------------------------------
# 3. discovery EMITS it — the whole point of this issue
# ---------------------------------------------------------------------------

def test_the_manifest_carries_the_declared_bar(tmp_pkg: Path) -> None:
    block = _compile_block(
        tmp_pkg, "ep_bar",
        'Compile(family="probe", shapes=((1024, 1024),), text_len=77, '
        'speed_metric="stage_ms.denoise", min_speedup=1.10)')
    assert block["speed_metric"] == "stage_ms.denoise"
    assert block["min_speedup"] == pytest.approx(1.10)


def test_an_undeclared_bar_is_ABSENT_never_defaulted(tmp_pkg: Path) -> None:
    # The hub reports `bar_undeclared` by name. A default emitted here would
    # make the SDK the author of the bar the platform verifies.
    block = _compile_block(
        tmp_pkg, "ep_nobar",
        'Compile(family="probe", shapes=((1024, 1024),), text_len=77)')
    assert "speed_metric" not in block
    assert "min_speedup" not in block
    assert "blockers" not in block


#: `blockers=` is export-contract vocabulary (EXPORT_CONTRACT_FIELDS), so a
#: declaration carrying one must name its graph classes like any other — the
#: minimum shape that registers.
_BLOCKED_DECL = (
    'Compile(family="{family}", targets=("transformer",), text_len=128, '
    'shapes=((64, 64),), shape_strategy="static-rows", warm_changes_key=False, '
    'dims=(Dim("B", carried_by=(("hidden_states", 0),)),), '
    'classes=(GraphClass(dims={{"B": 1}}),), '
    'inputs=(Input("hidden_states", shape=("B", 4, 8, 8), dtype="model"),), '
    'blockers=({blockers}))')

_OPEN = ('MintBlocker(id="open-q", what="w", evidence="e", '
         'resolves_when="r"),')
_RESOLVED = ('MintBlocker(id="settled", what="w", evidence="e", '
             'resolves_when="r", resolved=True, resolution="pgw#1149"),')


def test_the_manifest_carries_the_OPEN_blockers_only(tmp_pkg: Path) -> None:
    # The hub reads `len(blockers) > 0` as "the author refuses to mint" and
    # marks the mint check blocked-by-declaration. A RESOLVED id emitted here
    # would park the family in that state forever, so open-vs-resolved is the
    # whole content of this field — the prose stays in the declaration.
    block = _compile_block(tmp_pkg, "ep_blocked", _BLOCKED_DECL.format(
        family="probe-blocked", blockers=_OPEN + _RESOLVED))
    assert block["blockers"] == ["open-q"]


def test_a_family_whose_blockers_are_ALL_resolved_emits_none(tmp_pkg: Path) -> None:
    block = _compile_block(tmp_pkg, "ep_unblocked", _BLOCKED_DECL.format(
        family="probe-unblocked", blockers=_RESOLVED))
    assert "blockers" not in block


# ---------------------------------------------------------------------------
# 4. one readable shape, for the endpoint repo's torch-free lint
# ---------------------------------------------------------------------------

def test_speed_bar_is_the_one_readable_shape() -> None:
    decl = Compile(family="probe", shapes=((1024, 1024),), **BAR)
    assert speed_bar(decl) == {"family": "probe",
                               "metric": "stage_ms.denoise",
                               "min_speedup": 1.10}
    assert speed_bar(Compile(family="probe", shapes=((1024, 1024),))) is None
    # Reads any object carrying the attributes, like `blocker_rows` — the lint
    # evaluates an AST-extracted declaration, not an imported endpoint.
    assert speed_bar(None) is None


def test_a_bar_that_is_not_a_bar_is_refused() -> None:
    with pytest.raises((ValueError, TypeError, DeclarationError)):
        Compile(family="probe", shapes=((1024, 1024),),
                speed_metric="stage_ms.denoise", min_speedup=float("inf"))
