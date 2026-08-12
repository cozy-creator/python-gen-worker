"""pgw#758 — multi-graph cells: one .pt2 per (family x lane x contract),
every declared graph class a NAMED ENTRY.

Paul's ruling: "generate and generate_turbo are separate functions, they
have separate graphs, but they are COMBINED TOGETHER INTO ONE FILE."

The headline red test mints a two-class declared family (a cfg fork — the
generate / generate-turbo shape) into ONE artifact and serves BOTH
functions through the single loaded cell in one process. Real torch.export
+ real AOTI compiles on CPU — no mocks on the packaging path, because the
packaging path IS what this issue changes.
"""

from __future__ import annotations

import json
import tarfile
import types
from pathlib import Path
from typing import Any, Dict, Tuple

import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn  # noqa: E402

from gen_worker import aot_declaration, aot_mint, aot_package, aot_serve  # noqa: E402
from gen_worker import cell_key  # noqa: E402
from gen_worker import compile_cache  # noqa: E402
from gen_worker.api.decorators import Compile  # noqa: E402
from gen_worker.api.export_contract import (  # noqa: E402
    Dim,
    Fork,
    GraphClass,
    Input,
    register_export_declaration,
    reset_export_declarations,
)

FAMILY = "tiny758"


@pytest.fixture(autouse=True)
def _fresh_registry():
    reset_export_declarations()
    yield
    reset_export_declarations()


@pytest.fixture
def fake_sm(monkeypatch):
    """A CPU box has no sm; identity requires one (same convention as the
    pgw#723 tests)."""
    full = {"sku": "", "sm": "sm_89", "torch": str(torch.__version__), "cuda": ""}
    monkeypatch.setattr(compile_cache, "runtime_key", lambda: dict(full))
    # The consumer probe must agree with the mint probe: aot_serve.verify
    # rules on sm (pgw#765), so a fake sm is only coherent in both.
    monkeypatch.setattr(aot_serve, "runtime_key", lambda: {
        "sku": full["sku"], "sm": full["sm"], "torch": full["torch"],
        "cuda": full["cuda"]})
    return full


class TinyUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, sample: Any) -> Any:
        return torch.tanh(self.lin(sample)) + 1.0


def _declare(family: str = FAMILY) -> Any:
    return register_export_declaration(Compile(
        family=family,
        targets=("unet",),
        dims=(Dim("B", carried_by=(("sample", 0),)),),
        forks=(Fork("cfg", served=(True, False)),),
        classes=(GraphClass(dims={"B": 2}, fork={"cfg": True}),
                 GraphClass(dims={"B": 1}, fork={"cfg": False})),
        inputs=(Input("sample", shape=("B", 4), dtype="model"),),
        shape_strategy="static-rows",
        warm_changes_key=False,
    ))


def _mint(tmp_path: Path, pipe: Any = None) -> Tuple[Any, aot_mint.MintResult]:
    pipe = pipe or types.SimpleNamespace(unet=TinyUNet())
    spec = aot_mint.ExportSpec(family=FAMILY, target="")
    result = aot_mint.mint(
        pipe, spec, tmp_path / "out")
    return pipe, result


@pytest.fixture(scope="module")
def minted_cell(tmp_path_factory, request) -> Dict[str, Any]:
    """ONE real mint shared by the read-only assertions (an AOTI compile
    costs ~10s/entry on CPU; the mutating tests mint their own)."""
    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    request.addfinalizer(mp.undo)
    full = {"sku": "", "sm": "sm_89", "torch": str(torch.__version__), "cuda": ""}
    mp.setattr(compile_cache, "runtime_key", lambda: dict(full))
    mp.setattr(aot_serve, "runtime_key", lambda: {
        "sku": full["sku"], "sm": full["sm"], "torch": full["torch"],
        "cuda": full["cuda"]})
    reset_export_declarations()
    _declare()
    tmp = tmp_path_factory.mktemp("cell758")
    pipe, result = _mint(tmp)
    reset_export_declarations()
    return {"pipe": pipe, "result": result, "tmp": tmp}


# ---------------------------------------------------------------------------
# Entry naming — deterministic, declaration-derived
# ---------------------------------------------------------------------------


def test_entry_names_derive_from_declaration_coordinates() -> None:
    decl = _declare()
    plans = aot_declaration.cell_plans(decl)
    names = sorted(aot_declaration.plan_entry_name(p) for p in plans)
    assert names == ["unet/cfg=false/B=1", "unet/cfg=true/B=2"]


def test_dynamic_collapse_entry_omits_the_dims_segment() -> None:
    name = aot_declaration.entry_name(
        "transformer", (("expand_timesteps", False),), ())
    assert name == "transformer/expand_timesteps=false"


# ---------------------------------------------------------------------------
# The combined key — pgw#716's formula, verbatim
# ---------------------------------------------------------------------------


def test_the_manifest_digest_keeps_the_verbatim_ck6_formula() -> None:
    """pgw#1176 moved this arithmetic, not changed it: `combined_graph_hash`
    became `cell_key.manifest_digest` — same bytes, demoted from IDENTITY to a
    coverage label. The formula is a cross-repo wire fact (the hub folds
    compile-health rows under it), so the verbatim pin survives the move."""
    import hashlib

    hashes = ["bbb", "aaa", "ccc"]
    want = hashlib.sha256("\n".join(sorted(hashes)).encode()).hexdigest()[:16]
    assert cell_key.manifest_digest(hashes) == want
    # order-independent, content-sensitive
    a = cell_key.manifest_digest(["x", "y"])
    assert cell_key.manifest_digest(["y", "x"]) == a
    assert cell_key.manifest_digest(["x", "z"]) != a


def test_per_class_hashes_ride_metadata_and_name_the_class(minted_cell) -> None:
    row = sorted(minted_cell["result"].entries, key=lambda r: r.entry)[0]
    meta = json.loads(json.dumps(row.metadata))
    block = meta[cell_key.ENTRY_BLOCK_KEY]
    assert block["class_hash"]
    block["class_hash"] = "f" * 16
    reason = aot_serve.verify(meta, family=FAMILY)
    assert row.entry in reason and "class_hash" in reason


def test_range_digest_is_folded_per_class() -> None:
    entry = {
        "target": "unet",
        "fork": [["cfg", True]],
        "class_dims": [["B", 2]],
        "range_digest": "abc",
        "graph": {},
    }
    one = aot_serve.class_hash(entry, strict=True, lora_bucket=0)
    entry["range_digest"] = "def"
    assert aot_serve.class_hash(entry, strict=True, lora_bucket=0) != one


# ---------------------------------------------------------------------------
# The mint — one invocation, one cell, every declared class
# ---------------------------------------------------------------------------


def test_one_mint_packages_EVERY_declared_class_AS_ITS_OWN_ARTIFACT(
        minted_cell) -> None:
    """pgw#1176: the mint still covers the whole declared class set — what
    changed is that each class is its OWN keyed, publishable, armable
    artifact. `every declared class` survives as the assertion; `one cell`
    does not, because that unit is what forbade incremental adoption.
    """
    result = minted_cell["result"]
    assert sorted(r.entry for r in result.entries) == [
        "unet/cfg=false/B=1", "unet/cfg=true/B=2"]
    # Independently keyed — two classes, two identities, no shared unit.
    assert len({r.key for r in result.entries}) == 2
    assert all(cell_key.is_key(r.key) for r in result.entries)
    # ...and ONE manifest label over the set, which is telemetry not identity.
    assert result.manifest
    assert all(r.metadata["manifest_digest"] == result.manifest
               for r in result.entries)
    for i, row in enumerate(result.entries):
        assert row.metadata["format"] == 3
        assert "entries" not in row.metadata
        with tarfile.open(row.artifact) as tar:
            tar.extractall(minted_cell["tmp"] / f"x{i}", filter="data")
        names = aot_package.package_entry_names(
            minted_cell["tmp"] / f"x{i}" / aot_serve.PACKAGE_NAME)
        assert list(names) == [row.entry], (
            "an entry artifact carries ONE named model")


def test_package_gates_run_per_entry(minted_cell) -> None:
    for i, row in enumerate(minted_cell["result"].entries):
        with tarfile.open(row.artifact) as tar:
            tar.extractall(minted_cell["tmp"] / f"g{i}", filter="data")
        package = minted_cell["tmp"] / f"g{i}" / aot_serve.PACKAGE_NAME
        assert aot_package.code_only_violations(package, row.entry) == []
        assert aot_package.declared_constants(package, row.entry)


def test_entry_scoped_reads_refuse_an_unknown_entry(minted_cell) -> None:
    with tarfile.open(minted_cell["result"].entries[0].artifact) as tar:
        tar.extractall(minted_cell["tmp"] / "u", filter="data")
    package = minted_cell["tmp"] / "u" / aot_serve.PACKAGE_NAME
    with pytest.raises(aot_package.PackageIntrospectionError) as err:
        aot_package.declared_constants(package, "unet/cfg=true/B=99")
    assert "unet/cfg=true/B=99" in str(err.value)


def test_mint_refuses_a_family_with_no_declaration(tmp_path, fake_sm) -> None:
    pipe = types.SimpleNamespace(unet=TinyUNet())
    spec = aot_mint.ExportSpec(family="undeclared", target="")
    with pytest.raises(aot_mint.MintRefused, match="declaration"):
        aot_mint.mint(pipe, spec, tmp_path)


def test_mint_request_refuses_coordinate_subsets(tmp_path) -> None:
    """A request naming a target, a fork/class coordinate, or hand dynamic
    rows (the pod-9-era per-coordinate shapes) would mint a SUBSET of the
    contract the key advertises — each is refused by name."""
    bodies = [
        {"family": FAMILY, "target": "unet"},
        {"family": FAMILY, "fork": {"cfg": True}, "class_dims": {"B": 2}},
        {"family": FAMILY,
         "dynamic": [{"input": "sample", "axis": 0, "min": 1, "max": 2}]},
    ]
    for index, body in enumerate(bodies):
        p = tmp_path / f"req{index}.json"
        p.write_text(json.dumps(body))
        with pytest.raises(aot_mint.MintRefused,
                           match="WHOLE declared class set"):
            aot_mint._load_spec(p)


def test_export_failure_names_the_entry(tmp_path, fake_sm) -> None:
    _declare()

    class Broken(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)

        def forward(self, sample: Any) -> Any:
            if int(sample.shape[0]) == 1:  # data-dependent python branch
                raise RuntimeError("boom on the turbo class")
            return self.lin(sample)

    pipe = types.SimpleNamespace(unet=Broken())
    spec = aot_mint.ExportSpec(family=FAMILY, target="")
    with pytest.raises(aot_mint.MintRefused) as err:
        aot_mint.mint(pipe, spec, tmp_path)
    assert "unet/cfg=false/B=1" in str(err.value)


# ---------------------------------------------------------------------------
# The WARM CANON — declared, and now EXECUTED
# ---------------------------------------------------------------------------


class WarmSensitive(nn.Module):
    """A z-image-shaped module: the first forward caches a table that the
    exported graph then bakes.

    The cache is a REGISTERED BUFFER, per the pgw#857 tensor-binding contract,
    and pgw#1097 is why that stopped being optional. As a plain attribute,
    ``torch.export`` lifts the table under its ATTRIBUTE PATH (``_table``,
    measured) while AOTInductor's own constant table names it
    ``_tensor_constant0`` — and the mint reconciles the two by name. So a
    declared plain-attribute literal fails ``program_package_drift`` ("declares
    a constant the program never lifted") or, past it, ``_write_literals``
    ("declared literal constant(s) have no value in their exported program").
    **That is already true on master for any such literal big enough to be
    declared**; this one survived only because ``arange(4)`` is 4 elements and
    therefore met ``GraphLowering.can_inline_constant``, so inductor rendered
    its values into the kernel and no row ever appeared. With the folding fence
    on, nothing is inlined, and the contract violation surfaces as the typed
    refusal it always was. A buffer's program name and package ``original_fqn``
    agree, which is what makes a literal packable and bindable at all — and it
    is why `micro-4d`, whose literal IS a buffer, has always been green.

    Nothing about what this fixture TESTS changes: the first forward still
    populates the table, still bumps ``warm_calls``, and still changes the
    graph that gets exported.
    """

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.warm_calls = 0
        self.register_buffer("_table", None, persistent=False)

    def forward(self, sample: Any) -> Any:
        if self._table is None:
            self.warm_calls += 1
            self._table = torch.arange(4, dtype=sample.dtype) * 0.5
        return self.lin(sample) + self._table


def _declare_warm(warm: bool, family: str) -> None:
    register_export_declaration(Compile(
        family=family,
        targets=("unet",),
        dims=(Dim("B", carried_by=(("sample", 0),)),),
        classes=(GraphClass(dims={"B": 2}),),
        inputs=(Input("sample", shape=("B", 4), dtype="model"),),
        shape_strategy="static-rows",
        warm_changes_key=warm,
    ))


def test_declared_warm_family_mints_the_warmed_graph(tmp_path, fake_sm) -> None:
    _declare_warm(True, "warm758")
    module = WarmSensitive()
    pipe = types.SimpleNamespace(unet=module)
    spec = aot_mint.ExportSpec(family="warm758", target="")
    result = aot_mint.mint(pipe, spec, tmp_path)
    # The pre-warm RAN before export (the cache branch is warm at trace time),
    # and the mint recorded its cost.
    assert module.warm_calls == 1
    entry = next(iter(
        result.entries[0].metadata["mint_phases"]["entries"].values()))
    assert "warm_s" in entry


def test_undeclared_warm_family_is_not_warmed(tmp_path, fake_sm) -> None:
    _declare_warm(False, "cold758")
    module = WarmSensitive()
    pipe = types.SimpleNamespace(unet=module)
    spec = aot_mint.ExportSpec(family="cold758", target="")
    result = aot_mint.mint(pipe, spec, tmp_path)
    # Export itself traces the cold branch; no separate warm forward ran and
    # none was recorded.
    entry = next(iter(
        result.entries[0].metadata["mint_phases"]["entries"].values()))
    assert "warm_s" not in entry


def test_failed_declared_warm_is_a_named_refusal(tmp_path, fake_sm) -> None:
    _declare_warm(True, "warmfail758")

    class Fails(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)
            self.armed = False

        def forward(self, sample: Any) -> Any:
            if not self.armed:
                self.armed = True
                raise RuntimeError("warm blew up")
            return self.lin(sample)

    pipe = types.SimpleNamespace(unet=Fails())
    spec = aot_mint.ExportSpec(family="warmfail758", target="")
    with pytest.raises(aot_mint.MintRefused, match="mint-warm"):
        aot_mint.mint(pipe, spec, tmp_path)


# ---------------------------------------------------------------------------
# Mint-phase telemetry (#757's instrument-first deliverable)
# ---------------------------------------------------------------------------


def test_mint_records_the_phase_table(minted_cell) -> None:
    result = minted_cell["result"]
    # The phase table is a property of the MINT, not of one artifact, so every
    # entry's metadata carries the same one — it is the run's record.
    table = result.entries[0].metadata["mint_phases"]
    assert table["n_entries"] == 2
    assert set(table["entries"]) == {r.entry for r in result.entries}
    for row in table["entries"].values():
        assert row["export_s"] >= 0 and row["compile_s"] > 0
    # The host C++ compile+link phase — the stage the JIT path skips — is a
    # labeled number, not folklore.
    assert table["phases"].get("host_compile_s", 0) > 0
    assert "max_autotune" in table["autotune"]
    assert table["totals"]["compile_s"] > 0
    # The ONE resolved inductor config every entry compiled under is
    # recorded verbatim (#757's open per-call seal-bypass concern), and the
    # #757-measured mint default rides it: compile_threads 32 -> 4 is FREE
    # (-2% wall) and identity-inert.
    resolved = table["inductor_configs"]
    assert resolved["compile_threads"] == aot_mint.MINT_COMPILE_THREADS == 4
    assert resolved["aot_inductor.package_constants_in_so"] is False


# ---------------------------------------------------------------------------
# Serve — ONE loaded cell, every declared function
# ---------------------------------------------------------------------------


def _armed(minted_cell) -> Any:
    """Arm EVERY minted entry onto one pipeline, one artifact at a time.

    pgw#1176: this is what accretion looks like from the outside — the second
    call joins the first's registry, the first's target pool and the first's
    live wrap. The old shape (one `load_and_wrap` over a multi-entry cell) is
    gone with the cell.
    """
    pipe = minted_cell["pipe"]
    if not aot_serve.is_armed(pipe):
        cfg = types.SimpleNamespace(family=FAMILY, lora_bucket=0)
        for row in minted_cell["result"].entries:
            aot_serve.arm_entry(
                pipe, cfg, row.artifact,
                cache_dir=minted_cell["tmp"] / "cache",
                declared=[r.entry for r in minted_cell["result"].entries])
    return pipe


def test_one_resident_cell_serves_two_functions(minted_cell) -> None:
    """THE pgw#758 red test: generate and generate-turbo — two different
    graph classes — served by a single loaded artifact in one process."""
    pipe = _armed(minted_cell)
    w, b = pipe.unet.lin.weight, pipe.unet.lin.bias
    before = aot_serve.execution_count(pipe)
    x2, x1 = torch.randn(2, 4), torch.randn(1, 4)
    y2 = pipe.unet.forward(x2)   # cfg graph
    y1 = pipe.unet.forward(x1)   # turbo graph
    assert torch.allclose(y2, torch.tanh(x2 @ w.T + b) + 1.0, atol=1e-5)
    assert torch.allclose(y1, torch.tanh(x1 @ w.T + b) + 1.0, atol=1e-5)
    assert aot_serve.execution_count(pipe) == before + 2
    assert aot_serve.is_armed(pipe)


def test_off_contract_call_is_a_named_refusal_and_stays_armed(minted_cell) -> None:
    pipe = _armed(minted_cell)
    refusals = aot_serve.ingress_refusals(pipe)
    out = pipe.unet.forward(torch.randn(3, 4))  # no declared class admits B=3
    assert out is not None  # served eagerly
    assert aot_serve.ingress_refusals(pipe) == refusals + 1
    assert aot_serve.is_armed(pipe)


def test_dispatch_refuses_ambiguity_by_name() -> None:
    contract = aot_serve.contract_from_meta({
        "inputs": [{"name": "sample", "position": 0, "dtype": "float32",
                    "shape": [2, 4]}],
        "symbols": {},
    })

    class _Pkg:
        def get_constant_fqns(self):
            return []

        def load_constants(self, values, check_full_update=True,
                           user_managed=False):
            pass

        def __call__(self, *feeds):
            return feeds[0]

    def runner(entry: str) -> aot_serve.ArtifactRunner:
        r = aot_serve.ArtifactRunner(
            package=_Pkg(), contract=contract, constants=(), entry=entry)
        r.bind({}, {})
        return r

    dispatch = aot_serve.EntryDispatch(
        (("unet/a", runner("unet/a")), ("unet/b", runner("unet/b"))))
    with pytest.raises(aot_serve.IngressContractError) as err:
        dispatch(torch.randn(2, 4))
    assert err.value.reason == "entry_ambiguous"
    assert "unet/a" in str(err.value) and "unet/b" in str(err.value)


def test_unwrap_restores_every_target(minted_cell) -> None:
    pipe = _armed(minted_cell)
    eager = pipe.unet.__class__.forward
    assert aot_serve.unwrap(pipe)
    assert not aot_serve.is_armed(pipe)
    # The restored callable is the module's own eager forward again.
    x = torch.randn(2, 4)
    w, b = pipe.unet.lin.weight, pipe.unet.lin.bias
    assert torch.allclose(
        pipe.unet.forward(x), torch.tanh(x @ w.T + b) + 1.0, atol=1e-5)
    del eager


# ---------------------------------------------------------------------------
# Envelope v2 — refusals name the entry
# ---------------------------------------------------------------------------


def test_verify_names_a_malformed_entry(minted_cell) -> None:
    row = sorted(minted_cell["result"].entries, key=lambda r: r.entry)[0]
    meta = json.loads(json.dumps(row.metadata))
    meta[cell_key.ENTRY_BLOCK_KEY]["inputs"] = []
    reason = aot_serve.verify(meta, family=FAMILY)
    assert row.entry in reason


def test_literals_are_namespaced_per_entry() -> None:
    split = aot_serve.split_literals({
        "unet/cfg=true/B=2::pos.table": 1,
        "unet/cfg=false/B=1::pos.table": 2,
    })
    assert split["unet/cfg=true/B=2"] == {"pos.table": 1}
    assert split["unet/cfg=false/B=1"] == {"pos.table": 2}
    with pytest.raises(ValueError, match="not namespaced"):
        aot_serve.split_literals({"pos.table": 3})


def test_entry_names_reserving_the_literal_separator_are_refused() -> None:
    with pytest.raises(ValueError, match="reserves"):
        aot_serve.entry_from_meta({cell_key.ENTRY_BLOCK_KEY: {
            "name": "unet::x", "target": "unet", "inputs": [
                {"name": "s", "position": 0, "dtype": "float32", "shape": [1]}],
            "symbols": {}, "constants": []}})
