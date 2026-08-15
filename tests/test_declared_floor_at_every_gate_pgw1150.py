"""pgw#1150 (second pass) — the DECLARED numerics band reaches every gate that
judges parity, and ``threshold_source`` never lies about which band decided.

## The two properties

`registry.CompileCell` carries `numerics_floor`/`numerics_warn`, so the object
the FLEET mint parent hands the gate carries the family's declaration.

PROVENANCE is answered ONCE. If `numerics_probe.probe_cell` re-derives
`threshold_source` from `cfg.numerics_floor` alone while
`numerics_ladder.declared_thresholds` decides the band from `floor` **or**
`warn`, a family declaring only `numerics_warn` is judged at its DECLARED band
while every wire row, every `author-ci.toml` `[proof]` record and
`Parity.floor_source` report `sdk-default` — the measurement is right and the
record of it is wrong, and the record is what anyone acts on.
`declared_thresholds` is the ONE authority and stamps `Thresholds.source`.

## Why this file exists as well as `test_numerics_gate_pgw868.py`

pgw#868's rows drive the real arm — and every one of them passes a raw
`Compile`, which is a type NO fleet path ever hands the gate. That single
substitution is why `Compile.numerics_floor` reached nobody for 300+ commits
with a green suite the whole time: deleting the two `numerics_floor=` lines
from `registry.py` leaves every pre-existing test green.

So the rows below are parametrized over the cfg objects the gate ACTUALLY
receives in production — the raw `Compile` (author CI), the registry's
`CompileCell` (fleet mint parent), and the local CLI's `CompileCell` (§4.28
desktop) — and each drives the real path rather than `declared_thresholds` in
isolation. The pgw#868 rig is reused verbatim: real `arm_aot`, real
`aot_serve.enable`, real ladder, real packed artifact, real torch tensors.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Tuple

import pytest

from gen_worker import numerics_ladder, numerics_probe
from gen_worker.api import derive
from gen_worker.registry import CompileCell, extract_specs

import test_numerics_gate_pgw868 as rig  # noqa: E402
from harness import author_ci_endpoints_pgw1150 as endpoints  # noqa: E402

declared = rig.declared
events = rig.events

#: Between the SDK default floor (0.98) and the declared one (0.995). The whole
#: file turns on this number: a cell here is DEGRADED against the default and
#: DESTROYED against the declaration, so the two bands produce different
#: verdicts, different wire phases and different outcomes on one measurement.
BETWEEN = 0.99


# ---------------------------------------------------------------------------
# the three cfg objects a gate can actually be handed, built the production way
# ---------------------------------------------------------------------------

def _registry_cell() -> CompileCell:
    """The FLEET mint parent's object: `pending.cfg` is the `spec.compile_cell()`
    the executor opened its `ArmingScope` with."""
    cell = extract_specs(endpoints.Fast)[0].compile_cell()
    assert cell is not None
    return cell


def _local_cell() -> CompileCell:
    """The §4.28 desktop object `cli.run` builds for `local_serve`.

    Spelled out field by field rather than through
    :meth:`CompileCell.from_declaration`, deliberately: this row has to stay
    RED-able about the BAND on any revision, and a fixture that called the
    helper this change introduces would fail on the older tree for the wrong
    reason (no such attribute) and prove nothing about which floor decided.
    The helper has its own structural row below.
    """
    decl = rig.declaration()
    return CompileCell(
        shapes=tuple(decl.shapes), targets=tuple(decl.targets),
        family=str(decl.family or ""), regional=bool(decl.regional),
        text_len=decl.text_len, dynamic=tuple(decl.dynamic),
        lora_bucket=0, guidance_scales=(), text_lens=(),
        numerics_floor=decl.numerics_floor, numerics_warn=decl.numerics_warn)


def _raw_compile() -> Any:
    """Author CI's: `_Subject.declaration` is the endpoint's raw `Compile`."""
    return rig.declaration()


CFGS = {
    "author-ci/raw-Compile": _raw_compile,
    "fleet-mint-parent/registry.CompileCell": _registry_cell,
    "local-serve/cli.CompileCell": _local_cell,
}


def _undeclare(cfg: Any) -> Any:
    """The same cfg with the band removed — the negative control."""
    if isinstance(cfg, CompileCell):
        return dataclasses.replace(cfg, numerics_floor=None, numerics_warn=None)
    return rig.declaration(floor=None, warn=None)


def _arm(tmp_path, monkeypatch, cfg: Any, cosine: float,
         *, verify_numerics: bool) -> Any:
    packages = {rig.entry_name(h, w): rig.ProbePackage(cosine=cosine)
                for h, w in rig.ROWS}
    _pipe, _module, outcome = rig.arm(
        tmp_path, monkeypatch, cfg, packages,
        verify_numerics=verify_numerics)
    return outcome


def _phases(said: List[Tuple[str, str, str]]) -> List[str]:
    return [phase for _detail, phase in rig.numerics_rows(said)]


# ---------------------------------------------------------------------------
# 1. THE MINT-PARENT PUBLISH GATE — every cfg type, the declaration decides
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(CFGS))
def test_the_mint_parent_gate_judges_at_the_familys_declared_floor(
        name, tmp_path, monkeypatch, declared, events):
    """`adopt_delegated_mint` -> `arm_aot(verify_numerics=True)`, §4.32.

    cos=0.99 is DESTROYED against the declared 0.995 and merely DEGRADED
    against the SDK default 0.98, so the band that decided is legible from the
    wire phase alone — not inferred from a number this test also chose.
    """
    outcome = _arm(tmp_path, monkeypatch, CFGS[name](), BETWEEN,
                   verify_numerics=True)

    assert outcome.armed is False
    assert outcome.reason == "numerics_refused"
    assert numerics_ladder.PHASE_REFUSED in _phases(events), (
        "the cell was DEGRADED, not DESTROYED — the gate scored it against "
        "the SDK default 0.98 while this family declares 0.995")

    report = numerics_probe.last_report()
    assert report is not None
    assert report.thresholds.floor == endpoints.FLOOR
    assert report.thresholds.warn == endpoints.WARN
    assert report.threshold_source == "declared"


@pytest.mark.parametrize("name", sorted(CFGS))
def test_an_undeclared_family_still_gets_the_sdk_default_and_says_so(
        name, tmp_path, monkeypatch, declared, events):
    """The negative. A family that declares nothing must NOT silently inherit
    some other family's band, and must report the default as the default —
    `threshold_source` is what tells an operator which of the two happened."""
    outcome = _arm(tmp_path, monkeypatch, _undeclare(CFGS[name]()), BETWEEN,
                   verify_numerics=True)

    # Still refused — §4.32 is strict, and the gray band does not publish —
    # but for a DIFFERENT reason, and the wire says which.
    assert outcome.armed is False
    assert numerics_ladder.PHASE_DEGRADED in _phases(events)
    assert numerics_ladder.PHASE_REFUSED not in _phases(events)

    report = numerics_probe.last_report()
    assert report is not None
    assert report.thresholds == numerics_ladder.DEFAULT_THRESHOLDS
    assert report.threshold_source == "sdk-default"


# ---------------------------------------------------------------------------
# 2. `provision.gate_cell_numerics` — the function the author-CI adopt leg calls
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(CFGS))
def test_gate_cell_numerics_refuses_on_the_declared_floor_alone(
        name, tmp_path, monkeypatch, declared, events):
    """Called directly, non-strict, so the FLOOR is the only thing separating
    the two answers: 0.99 is below the declared 0.995 (refuse) and inside the
    default's gray band (serve, confess). One measurement, two verdicts,
    decided by the declaration and nothing else.

    `author_ci.read_parity` reaches this exact call for an ADOPTED cell.
    """
    from gen_worker.models import provision

    cfg = CFGS[name]()
    packages = {rig.entry_name(h, w): rig.ProbePackage(cosine=BETWEEN)
                for h, w in rig.ROWS}
    pipe, _module, outcome = rig.arm(
        tmp_path, monkeypatch, cfg, packages, verify_numerics=False)
    assert outcome.armed, "the rig did not arm without the gate"

    assert provision.gate_cell_numerics(pipe, cfg, strict=False) is False
    assert numerics_probe.last_report().threshold_source == \
        "declared"

    assert provision.gate_cell_numerics(
        pipe, _undeclare(cfg), strict=False) is True
    assert numerics_probe.last_report().threshold_source == \
        "sdk-default"


# ---------------------------------------------------------------------------
# 3. `threshold_source` IS TRUE — RED on 88af2c9b
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(CFGS))
def test_a_warn_only_declaration_is_reported_as_DECLARED(
        name, tmp_path, monkeypatch, declared, events):
    """RED on `88af2c9b`: `sdk-default` while a declared band decided.

    `Compile(numerics_warn=…)` with no floor is a legal declaration — the floor
    refuses, the warn only confesses, and a family may reasonably tighten just
    the confession band. `declared_thresholds` honours it (the returned band is
    NOT `DEFAULT_THRESHOLDS`), but `probe_cell` used to answer the provenance
    question a SECOND time from `cfg.numerics_floor` alone and got `sdk-default`
    — the one field whose whole job is to say which band decided, saying the
    wrong one. Nobody re-derives it now.
    """
    cfg = CFGS[name]()
    if isinstance(cfg, CompileCell):
        cfg = dataclasses.replace(cfg, numerics_floor=None, numerics_warn=0.9999)
    else:
        cfg = rig.declaration(floor=None, warn=0.9999)

    from gen_worker.models import provision

    band = numerics_ladder.declared_thresholds(cfg)
    assert band != numerics_ladder.DEFAULT_THRESHOLDS
    assert band.warn == 0.9999

    # Armed without the gate, then gated explicitly, so the report read below
    # is THIS row's and never a stale one from the process.
    packages = {rig.entry_name(h, w): rig.ProbePackage(cosine=0.9995)
                for h, w in rig.ROWS}
    pipe, _module, outcome = rig.arm(
        tmp_path, monkeypatch, cfg, packages, verify_numerics=False)
    assert outcome.armed
    # 0.9995 sits under the declared 0.9999 warn and over the default 0.999:
    # the confession itself is the declaration's, not the SDK's.
    assert provision.gate_cell_numerics(pipe, cfg, strict=True) is False
    report = numerics_probe.last_report()
    assert report is not None
    assert report.thresholds.warn == 0.9999, \
        "the gate did not use the declared warn band"
    assert report.threshold_source == "declared", \
        "the gate scored against a DECLARED band and reported `sdk-default`"


def test_the_provenance_is_decided_in_exactly_one_place():
    """The structural half of the row above: `Thresholds` carries its own
    source, so a reader cannot disagree with the band it is reporting on."""
    assert numerics_ladder.DEFAULT_THRESHOLDS.source == \
        "sdk-default"
    for name, build in CFGS.items():
        cfg = build()
        assert numerics_ladder.declared_thresholds(cfg).source == \
            "declared", name
        assert numerics_ladder.declared_thresholds(_undeclare(cfg)).source == \
            "sdk-default", name


# ---------------------------------------------------------------------------
# 4. the fences: ONE constructor, no re-key, and the band survives a migration
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(CFGS))
def test_declaring_a_band_never_moves_a_compiled_graph_key(name):
    """A numerics band is a GATE, not a graph axis. If declaring one re-keyed,
    every family that states its floor would re-mint the whole fleet — which is
    the cost that makes "just leave it at the default" tempting."""
    cfg = CFGS[name]()
    bare = _undeclare(cfg)

    if isinstance(cfg, CompileCell):
        assert "numerics_floor" not in cfg.contract_facts()
        assert "numerics_warn" not in cfg.contract_facts()
        assert cfg.contract_digest() == bare.contract_digest()
    else:
        assert "numerics_floor" not in cfg.contract_axes()
        assert "numerics_warn" not in cfg.contract_axes()
        assert cfg.contract_axes() == bare.contract_axes()
        # ...and BECAUSE it does not re-key, `contract_delta` alone would wave a
        # dropped band through. That is what OVERRIDE_FACTS is for.
        assert derive.contract_delta(cfg, bare) == {}
        assert set(derive.override_delta(cfg, bare)) == {
            "numerics_floor", "numerics_warn"}


def test_one_constructor_maps_a_declaration_onto_a_cell():
    """Both production sites go through `CompileCell.from_declaration`, so a
    must-survive field cannot reach one path and not the other — which is
    precisely how the band reached no gate at all."""
    decl = rig.declaration()
    for name, build in CFGS.items():
        cell = build()
        if not isinstance(cell, CompileCell):
            continue
        assert cell.numerics_floor == decl.numerics_floor, name
        assert cell.numerics_warn == decl.numerics_warn, name

    every: Dict[str, Any] = dataclasses.asdict(
        CompileCell.from_declaration(decl, lora_bucket=0))
    assert every["numerics_floor"] == rig.FLOOR
    assert every["numerics_warn"] == rig.WARN
