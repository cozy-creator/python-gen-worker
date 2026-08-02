"""pgw#847: one export + per-row re-specialization, and the gate that must
PROVE it or fall back.

Real exports, real `aot_compile`, real generated C++ compiled by the real g++,
CPU target — no mocks, because the whole claim is about what the toolchain
emits. The gate's own contract is tested the pgw#832 way: `torch.export.export`
is monkeypatched to RAISE across the reuse arm, so a byte equality cannot come
from an accidental re-export.
"""
from __future__ import annotations

import copy
import hashlib
import subprocess

import pytest
import torch
from torch import nn

from gen_worker import aot_export_reuse, host_isa

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


# --------------------------------------------------------------------------
# probe modules
# --------------------------------------------------------------------------


class _Stack(nn.Module):
    """A shape-STABLE module: the same ops at every row."""

    def __init__(self, depth: int = 6, channels: int = 8) -> None:
        super().__init__()
        self.inp = nn.Conv2d(4, channels, 3, padding=1)
        self.body = nn.ModuleList(
            [nn.Conv2d(channels, channels, 3, padding=1) for _ in range(depth)])
        self.out = nn.Conv2d(channels, 4, 3, padding=1)

    def forward(self, x):
        h = self.inp(x)
        for conv in self.body:
            h = torch.nn.functional.silu(conv(h)) + h
        return self.out(h)


class _BranchesOnSize(nn.Module):
    """The family the gate exists for: its GRAPH moves with the shape row."""

    def __init__(self, channels: int = 8) -> None:
        super().__init__()
        self.a = nn.Conv2d(4, channels, 3, padding=1)
        self.b = nn.Conv2d(channels, 4, 3, padding=1)

    def forward(self, x):
        h = self.a(x)
        if x.shape[-1] >= 64:              # traced away per row -> two graphs
            h = torch.nn.functional.silu(h)
        else:
            h = torch.tanh(h) * 2.0
        return self.b(h)


def _export(module, b: int, h: int, w: int):
    example = (torch.randn(b, 4, h, w),)
    with torch.no_grad():
        return torch.export.export(module, example, strict=False), example


@pytest.fixture(autouse=True)
def _clamped():
    host_isa.impose()


@pytest.fixture
def _refuse_export(monkeypatch):
    """Make the real computation REFUSE, so equality cannot be accidental."""

    def _factory():
        def refuse(*_a, **_kw):
            raise AssertionError(
                "torch.export.export was called where the test forbids it")

        monkeypatch.setattr(torch.export, "export", refuse)

    return _factory


# --------------------------------------------------------------------------
# the primitive
# --------------------------------------------------------------------------


def test_respecialized_program_is_a_real_exported_program(tmp_path):
    """It must survive `torch.export.save`/`load` — that round trip is how
    pgw#809's pool hands an entry to its child."""
    module = _Stack()
    base, _ = _export(module, 1, 32, 32)
    _, example = _export(module, 1, 24, 40)

    out = aot_export_reuse.respecialize(base, example, {})
    assert isinstance(out, type(base))

    path = tmp_path / "program.pt2"
    torch.export.save(out, path)
    reloaded = torch.export.load(path)
    assert reloaded.graph_module.code == out.graph_module.code
    assert reloaded.example_inputs[0][0].shape == example[0].shape


def test_respecialize_carries_the_new_row_not_the_base_row():
    """The naive form of this optimisation compiled the BASE row's kernels
    under another row's name. Pin that the metadata actually moves."""
    module = _Stack()
    base, _ = _export(module, 1, 32, 32)
    _, example = _export(module, 1, 24, 40)

    out = aot_export_reuse.respecialize(base, example, {})
    placeholders = [n for n in out.graph_module.graph.nodes
                    if n.op == "placeholder" and "val" in n.meta]
    shapes = {tuple(n.meta["val"].shape) for n in placeholders
              if hasattr(n.meta["val"], "shape")}
    assert (1, 4, 24, 40) in shapes, shapes
    assert (1, 4, 32, 32) not in shapes, shapes


def test_respecialize_declines_a_row_it_cannot_place():
    module = _Stack()
    base, _ = _export(module, 1, 32, 32)
    with pytest.raises(aot_export_reuse.ReuseUnproven):
        aot_export_reuse.respecialize(base, (), {})
    with pytest.raises(aot_export_reuse.ReuseUnproven):
        aot_export_reuse.respecialize(
            base, (torch.randn(1, 4, 8, 8), torch.randn(1, 4, 8, 8)), {})


# --------------------------------------------------------------------------
# the gate
# --------------------------------------------------------------------------


def test_source_and_command_determine_the_object(tmp_path):
    """The premise the cheap gate rests on, re-checked on THIS toolchain.

    Measured separately on the real 6.3 MB sdxl wrapper TU
    (`scripts/pgw847/source_implies_so.py`): identical source + identical
    command + the same build path => byte-identical object, and a different
    build path moves 156 bytes of 15 MB. Pinned here on a small generated TU
    so the claim cannot rot silently.

    NOTE the trap this test exists to avoid: `-g1` embeds the OBJECT path as
    well as the source path, so two compiles that differ in `-o` are not the
    same command. A first pass read a false NEGATIVE off exactly that.
    """
    src = tmp_path / "u.cpp"
    src.write_text(
        "#include <vector>\n#include <string>\n"
        "std::vector<std::string> f() { return {\"a\", \"b\"}; }\n")
    obj = tmp_path / "u.o"
    cmd = ["g++", "-O1", "-g1", "-fPIC", "-std=c++20", "-c", str(src),
           "-o", str(obj)]
    digests = []
    for _ in range(2):
        assert subprocess.run(cmd, capture_output=True).returncode == 0
        digests.append(hashlib.sha256(obj.read_bytes()).hexdigest())
        obj.unlink()
    assert digests[0] == digests[1], "g++ is not deterministic here"


def test_gate_admits_a_shape_stable_family_and_proves_it_on_the_artifact(
        tmp_path, _refuse_export):
    """The whole claim, end to end: same graph text AND byte-identical
    generated C++ under a byte-identical host command, from two arms that
    each stop before `g++` and run concurrently."""
    module = _Stack()
    base, _ = _export(module, 1, 32, 32)
    witness, example = _export(module, 1, 24, 40)

    # The real computation REFUSES for the whole gate: `respecialize` runs in
    # THIS process, so a secret re-export would raise here. (The two arms are
    # subprocesses that only `torch.export.load` a `.pt2`; neither calls
    # `export` at all, which is the point.)
    _refuse_export()
    verdict = aot_export_reuse.prove(
        base, witness, example, {}, workdir=tmp_path, entry="probe")

    assert verdict.admitted, verdict.reason
    assert verdict.code_equal is True
    assert verdict.artifacts_equal is True
    assert verdict.own_digests == verdict.reuse_digests
    # the host command is part of the evidence, not just the sources
    assert "__cmd__" in verdict.own_digests, verdict.own_digests
    assert any(k.endswith(".cpp") for k in verdict.own_digests), \
        verdict.own_digests


def test_the_gate_never_reaches_the_wrapper_host_compile(tmp_path, monkeypatch):
    """The whole cost argument: the 180 s `g++` must not run in either arm."""
    module = _Stack()
    base, _ = _export(module, 1, 32, 32)
    witness, example = _export(module, 1, 24, 40)

    seen: list[str] = []
    real = aot_export_reuse._capture_codegen

    def watched(program, entry, cache_dir, configs):
        out = real(program, entry, cache_dir, configs)
        seen.extend(sorted(cache_dir.rglob("*.wrapper.o")) and ["WRAPPER.O"])
        return out

    monkeypatch.setattr(aot_export_reuse, "_capture_codegen", watched)
    aot_export_reuse.prove(
        base, witness, example, {}, workdir=tmp_path, entry="probe")
    assert seen == [], "an arm compiled the wrapper TU"


def test_gate_declines_a_family_whose_graph_moves_with_the_row(tmp_path):
    """RED arm: the gate's reason for existing. Cheap — it never reaches a
    compile, because the graph text already differs."""
    module = _BranchesOnSize()
    base, _ = _export(module, 1, 4, 64)      # >= 64 -> the silu branch
    witness, example = _export(module, 1, 4, 32)   # < 64  -> the tanh branch

    verdict = aot_export_reuse.prove(
        base, witness, example, {}, workdir=tmp_path, entry="probe")

    assert not verdict.admitted
    assert verdict.code_equal is False
    assert "structure moves with the shape row" in verdict.reason
    assert verdict.artifacts_equal is None      # never got that far


def test_gate_declines_a_WRONG_respecialization_on_real_codegen(
        tmp_path, monkeypatch):
    """The artifact check's RED arm, with no mocking of the thing under test.

    `respecialize` is replaced by the NAIVE form this lane measured and
    rejected — hand back the base program untouched, so the row's metadata
    never moves. That is exactly the failure that compiles row 0's kernels
    under row R's name, and the gate must catch it from the generated C++
    alone."""
    module = _Stack()
    base, _ = _export(module, 1, 32, 32)
    witness, example = _export(module, 1, 24, 40)

    monkeypatch.setattr(
        aot_export_reuse, "respecialize", lambda b, a, k: b)
    verdict = aot_export_reuse.prove(
        base, witness, example, {}, workdir=tmp_path, entry="probe")

    assert not verdict.admitted, verdict.reason
    assert verdict.code_equal is True          # the graph TEXT is identical
    assert verdict.artifacts_equal is False    # the generated C++ is not
    assert verdict.own_digests != verdict.reuse_digests


def test_gate_declines_when_it_cannot_build_its_evidence(tmp_path, monkeypatch):
    """Absence of evidence is a FALLBACK, never a pass."""
    module = _Stack()
    base, _ = _export(module, 1, 32, 32)
    witness, example = _export(module, 1, 24, 40)

    def boom(*_a, **_kw):
        raise RuntimeError("no compiler today")

    monkeypatch.setattr(aot_export_reuse, "_run_arms", boom)
    verdict = aot_export_reuse.prove(
        base, witness, example, {}, workdir=tmp_path, entry="probe")

    assert not verdict.admitted
    assert verdict.code_equal is True
    assert "could not build its evidence" in verdict.reason


def test_gate_declines_when_an_arm_emits_nothing(tmp_path, monkeypatch):
    module = _Stack()
    base, _ = _export(module, 1, 32, 32)
    witness, example = _export(module, 1, 24, 40)

    monkeypatch.setattr(
        aot_export_reuse, "_run_arms",
        lambda *a, **k: {"gate-own": {}, "gate-reuse": {}})
    verdict = aot_export_reuse.prove(
        base, witness, example, {}, workdir=tmp_path, entry="probe")

    assert not verdict.admitted
    assert "emitted no files at all" in verdict.reason


def test_gate_declines_on_any_artifact_difference(tmp_path, monkeypatch):
    module = _Stack()
    base, _ = _export(module, 1, 32, 32)
    witness, example = _export(module, 1, 24, 40)

    def two_arms(*_a, **_kw):
        return {"gate-own": {"__cmd__": "c", ".wrapper.cpp": "aaa"},
                "gate-reuse": {"__cmd__": "c", ".wrapper.cpp": "bbb"}}

    monkeypatch.setattr(aot_export_reuse, "_run_arms", two_arms)
    verdict = aot_export_reuse.prove(
        base, witness, example, {}, workdir=tmp_path, entry="probe")

    assert not verdict.admitted
    assert verdict.artifacts_equal is False
    assert ".wrapper.cpp" in verdict.reason


# --------------------------------------------------------------------------
# the state machine
# --------------------------------------------------------------------------


def _state(tmp_path, *, active: bool = True) -> aot_export_reuse.ReuseState:
    return aot_export_reuse.ReuseState(tmp_path, active=active)


def test_flag_is_off_by_default(monkeypatch, tmp_path):
    monkeypatch.delenv(aot_export_reuse.ENV_FLAG, raising=False)
    assert aot_export_reuse.enabled() is False
    assert aot_export_reuse.ReuseState(tmp_path).active is False
    for value in ("0", "", "no", "off", "maybe"):
        monkeypatch.setenv(aot_export_reuse.ENV_FLAG, value)
        assert aot_export_reuse.enabled() is False, value
    for value in ("1", "true", "TRUE", "yes", "on"):
        monkeypatch.setenv(aot_export_reuse.ENV_FLAG, value)
        assert aot_export_reuse.enabled() is True, value


def test_inactive_state_exports_every_row(tmp_path):
    state = _state(tmp_path, active=False)
    calls = {"n": 0}

    def full():
        calls["n"] += 1
        return f"program{calls['n']}"

    for _ in range(6):
        program, how = state.program(
            ("unet", True), entry="e", rows=6, args=(), kwargs={},
            full_export=full)
        assert how == "full"
        assert program.startswith("program")
    assert calls["n"] == 6
    assert state.reused == 0


def test_a_declined_gate_exports_every_remaining_row(tmp_path, monkeypatch):
    """The fallback is per row and complete — not 'mostly'."""
    monkeypatch.setattr(
        aot_export_reuse, "prove",
        lambda *a, **k: aot_export_reuse.GateVerdict(False, "nope"))
    state = _state(tmp_path)
    calls = {"n": 0}

    def full():
        calls["n"] += 1
        return f"program{calls['n']}"

    hows = [state.program(("unet", True), entry="e", rows=8, args=(),
                          kwargs={}, full_export=full)[1] for _ in range(8)]
    assert hows == ["full"] * 8
    assert calls["n"] == 8
    assert state.reused == 0


def test_an_admitted_gate_reuses_from_the_third_row(tmp_path, monkeypatch):
    monkeypatch.setattr(
        aot_export_reuse, "prove",
        lambda *a, **k: aot_export_reuse.GateVerdict(True, "ok"))
    monkeypatch.setattr(
        aot_export_reuse, "respecialize",
        lambda base, args, kwargs: f"reused-from-{base}")
    state = _state(tmp_path)
    calls = {"n": 0}

    def full():
        calls["n"] += 1
        return f"program{calls['n']}"

    results = [state.program(("unet", True), entry="e", rows=8, args=(),
                             kwargs={}, full_export=full) for _ in range(8)]
    assert [how for _, how in results] == ["full", "full"] + ["reused"] * 6
    assert calls["n"] == 2
    assert all(p == "reused-from-program1" for p, how in results
               if how == "reused")
    assert state.reused == 6 and state.exported == 2


def test_a_row_that_cannot_be_respecialized_falls_back_alone(
        tmp_path, monkeypatch):
    """An admitted family does NOT license a row the code cannot place."""
    monkeypatch.setattr(
        aot_export_reuse, "prove",
        lambda *a, **k: aot_export_reuse.GateVerdict(True, "ok"))
    seen = {"n": 0}

    def flaky(base, args, kwargs):
        seen["n"] += 1
        if seen["n"] == 2:
            raise aot_export_reuse.ReuseUnproven("not this one")
        return "reused"

    monkeypatch.setattr(aot_export_reuse, "respecialize", flaky)
    state = _state(tmp_path)
    hows = [state.program(("unet", True), entry="e", rows=6, args=(),
                          kwargs={}, full_export=lambda: "program")[1]
            for _ in range(6)]
    assert hows == ["full", "full", "reused", "full", "reused", "reused"]


def test_a_short_family_never_reuses(tmp_path, monkeypatch):
    """Row 0 is the base and row 1 is the evidence, so under MIN_ROWS there is
    nothing left to pay for the gate."""
    monkeypatch.setattr(
        aot_export_reuse, "prove",
        lambda *a, **k: pytest.fail("the gate must not run below MIN_ROWS"))
    state = _state(tmp_path)
    for rows in (1, 2):
        hows = [state.program(("unet", True), entry="e", rows=rows, args=(),
                              kwargs={}, full_export=lambda: "p")[1]
                for _ in range(rows)]
        assert hows == ["full"] * rows


def test_the_verdict_is_per_key_and_never_memoised_across_families(
        tmp_path, monkeypatch):
    """A verdict reached for one (target, arm) must not license another, and
    nothing may survive the ReuseState the mint created."""
    gates = {"n": 0}

    def counting_prove(*_a, **_kw):
        gates["n"] += 1
        return aot_export_reuse.GateVerdict(True, "test")

    monkeypatch.setattr(aot_export_reuse, "prove", counting_prove)
    monkeypatch.setattr(
        aot_export_reuse, "respecialize", lambda b, a, k: "reused")

    state = _state(tmp_path)
    for _ in range(4):
        state.program(("unet", True), entry="e", rows=4, args=(), kwargs={},
                      full_export=lambda: "p")
    # a SECOND key starts from scratch: two full exports before any reuse
    hows = [state.program(("vae", True), entry="e", rows=4, args=(),
                          kwargs={}, full_export=lambda: "p")[1]
            for _ in range(4)]
    assert hows == ["full", "full", "reused", "reused"]
    # one gate PER KEY — the first key's verdict licensed nothing here
    assert gates["n"] == 2

    # and a fresh state shares nothing with the old one
    fresh = _state(tmp_path)
    assert fresh.verdict(("unet", True)) is None
    assert fresh.reused == 0


def test_no_module_level_verdict_cache_exists():
    """Guards the invariant by inspection: a memoised verdict is a verdict
    about a module nobody checked."""
    import functools

    for name in dir(aot_export_reuse):
        obj = getattr(aot_export_reuse, name)
        assert not isinstance(obj, functools._lru_cache_wrapper), name
        assert not hasattr(obj, "cache_clear"), name


def test_telemetry_is_flat_and_reports_both_routes(tmp_path, monkeypatch):
    monkeypatch.setattr(
        aot_export_reuse, "prove",
        lambda *a, **k: aot_export_reuse.GateVerdict(
            True, "ok", code_equal=True, artifacts_equal=True, gate_s=1.5))
    monkeypatch.setattr(
        aot_export_reuse, "respecialize", lambda b, a, k: "reused")
    state = _state(tmp_path)
    for _ in range(5):
        state.program(("unet", True), entry="e", rows=5, args=(), kwargs={},
                      full_export=lambda: "p")
    telemetry = state.telemetry()
    assert telemetry["rows_exported"] == 2
    assert telemetry["rows_reused"] == 3
    assert telemetry["gates"][0]["admitted"] is True
    assert telemetry["gates"][0]["gate_s"] == 1.5


def test_the_gate_never_mutates_the_base_program(tmp_path):
    """`FakeTensorProp` mutates in place, so the re-specialization MUST work
    on a copy — a base that drifted would poison every later row."""
    module = _Stack()
    base, _ = _export(module, 1, 32, 32)
    before = copy.deepcopy(base.graph_module.code)
    placeholders_before = [
        tuple(n.meta["val"].shape) for n in base.graph_module.graph.nodes
        if n.op == "placeholder" and hasattr(n.meta.get("val"), "shape")]

    _, example = _export(module, 1, 24, 40)
    aot_export_reuse.respecialize(base, example, {})
    aot_export_reuse.respecialize(base, example, {})

    assert base.graph_module.code == before
    placeholders_after = [
        tuple(n.meta["val"].shape) for n in base.graph_module.graph.nodes
        if n.op == "placeholder" and hasattr(n.meta.get("val"), "shape")]
    assert placeholders_after == placeholders_before
