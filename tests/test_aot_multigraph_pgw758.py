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
        inputs=(Input("sample", shape=("B", 4)),),
        shape_strategy="static-rows",
        warm_changes_key=False,
    ))


def _mint(tmp_path: Path, pipe: Any = None) -> Tuple[Any, aot_mint.MintResult]:
    pipe = pipe or types.SimpleNamespace(unet=TinyUNet())
    spec = aot_mint.ExportSpec(family=FAMILY, target="")
    result = aot_mint.mint(
        pipe, spec, tmp_path / "out", allow_regressed_lanes=True)
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


def test_entry_names_are_stable_across_enumerations() -> None:
    decl = _declare()
    first = [aot_declaration.plan_entry_name(p)
             for p in aot_declaration.cell_plans(decl)]
    second = [aot_declaration.plan_entry_name(p)
              for p in aot_declaration.cell_plans(decl)]
    assert first == second


def test_dynamic_collapse_entry_omits_the_dims_segment() -> None:
    name = aot_declaration.entry_name(
        "transformer", (("expand_timesteps", False),), ())
    assert name == "transformer/expand_timesteps=false"


def test_colliding_entry_names_are_refused_by_name() -> None:
    # Two targets that render the same label cannot happen via the
    # vocabulary (targets differ), so collide via duplicated plans instead.
    decl = _declare()
    plans = aot_declaration.cell_plans(decl)
    assert len({aot_declaration.plan_entry_name(p) for p in plans}) == len(plans)


# ---------------------------------------------------------------------------
# The combined key — pgw#716's formula, verbatim
# ---------------------------------------------------------------------------


def test_combined_graph_hash_is_the_verbatim_ck6_formula() -> None:
    import hashlib

    hashes = ["bbb", "aaa", "ccc"]
    want = hashlib.sha256("\n".join(sorted(hashes)).encode()).hexdigest()[:16]
    assert aot_serve.combined_graph_hash(hashes) == want


def test_combined_hash_is_order_independent_and_content_sensitive() -> None:
    a = aot_serve.combined_graph_hash(["x", "y"])
    assert aot_serve.combined_graph_hash(["y", "x"]) == a
    assert aot_serve.combined_graph_hash(["x", "z"]) != a


def test_cell_key_folds_the_combined_hash(minted_cell) -> None:
    meta = dict(minted_cell["result"].metadata)
    spec = aot_mint.ExportSpec(family=FAMILY, target="")
    assert aot_mint.cell_identity(meta, spec).digest == meta["cell_key"]
    # Perturb one entry's class hash: the combined hash — and the key — move.
    meta2 = json.loads(json.dumps(meta))
    name = next(iter(meta2["entries"]))
    meta2["entries"][name]["class_hash"] = "0" * 16
    meta2["combined_graph_hash"] = aot_serve.combined_graph_hash(
        row["class_hash"] for row in meta2["entries"].values())
    assert aot_mint.cell_identity(meta2, spec).digest != meta["cell_key"]


def test_per_class_hashes_ride_metadata_and_name_the_class(minted_cell) -> None:
    meta = json.loads(json.dumps(minted_cell["result"].metadata))
    name = sorted(meta["entries"])[0]
    assert meta["entries"][name]["class_hash"]
    meta["entries"][name]["class_hash"] = "f" * 16
    reason = aot_serve.verify(meta, family=FAMILY)
    assert name in reason and "class_hash" in reason


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


def test_one_mint_packages_every_declared_class(minted_cell) -> None:
    meta = minted_cell["result"].metadata
    assert sorted(meta["entries"]) == [
        "unet/cfg=false/B=1", "unet/cfg=true/B=2"]
    assert meta["format"] == 2
    assert meta["combined_graph_hash"]
    # The .pt2 itself carries both named models.
    with tarfile.open(minted_cell["result"].artifact) as tar:
        tar.extractall(minted_cell["tmp"] / "x", filter="data")
    names = aot_package.package_entry_names(
        minted_cell["tmp"] / "x" / aot_serve.PACKAGE_NAME)
    assert sorted(names) == sorted(meta["entries"])


def test_package_gates_run_per_entry(minted_cell) -> None:
    with tarfile.open(minted_cell["result"].artifact) as tar:
        tar.extractall(minted_cell["tmp"] / "g", filter="data")
    package = minted_cell["tmp"] / "g" / aot_serve.PACKAGE_NAME
    for entry in aot_package.package_entry_names(package):
        assert aot_package.code_only_violations(package, entry) == []
        assert aot_package.declared_constants(package, entry)


def test_entry_scoped_reads_refuse_an_unknown_entry(minted_cell) -> None:
    with tarfile.open(minted_cell["result"].artifact) as tar:
        tar.extractall(minted_cell["tmp"] / "u", filter="data")
    package = minted_cell["tmp"] / "u" / aot_serve.PACKAGE_NAME
    with pytest.raises(aot_package.PackageIntrospectionError) as err:
        aot_package.declared_constants(package, "unet/cfg=true/B=99")
    assert "unet/cfg=true/B=99" in str(err.value)


def test_mint_refuses_a_family_with_no_declaration(tmp_path, fake_sm) -> None:
    pipe = types.SimpleNamespace(unet=TinyUNet())
    spec = aot_mint.ExportSpec(family="undeclared", target="")
    with pytest.raises(aot_mint.MintRefused, match="declaration"):
        aot_mint.mint(pipe, spec, tmp_path, allow_regressed_lanes=True)


def test_mint_request_refuses_coordinate_subsets(tmp_path) -> None:
    body = {"family": FAMILY, "target": "unet"}
    p = tmp_path / "req.json"
    p.write_text(json.dumps(body))
    with pytest.raises(aot_mint.MintRefused, match="WHOLE declared class set"):
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
        aot_mint.mint(pipe, spec, tmp_path, allow_regressed_lanes=True)
    assert "unet/cfg=false/B=1" in str(err.value)


# ---------------------------------------------------------------------------
# The WARM CANON — declared, and now EXECUTED
# ---------------------------------------------------------------------------


class WarmSensitive(nn.Module):
    """A z-image-shaped module: the first forward caches a table that the
    exported graph then bakes."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.warm_calls = 0
        self._table: Any = None

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
        inputs=(Input("sample", shape=("B", 4)),),
        shape_strategy="static-rows",
        warm_changes_key=warm,
    ))


def test_declared_warm_family_mints_the_warmed_graph(tmp_path, fake_sm) -> None:
    _declare_warm(True, "warm758")
    module = WarmSensitive()
    pipe = types.SimpleNamespace(unet=module)
    spec = aot_mint.ExportSpec(family="warm758", target="")
    result = aot_mint.mint(pipe, spec, tmp_path, allow_regressed_lanes=True)
    # The pre-warm RAN before export (the cache branch is warm at trace time),
    # and the mint recorded its cost.
    assert module.warm_calls == 1
    entry = next(iter(result.metadata["mint_phases"]["entries"].values()))
    assert "warm_s" in entry


def test_undeclared_warm_family_is_not_warmed(tmp_path, fake_sm) -> None:
    _declare_warm(False, "cold758")
    module = WarmSensitive()
    pipe = types.SimpleNamespace(unet=module)
    spec = aot_mint.ExportSpec(family="cold758", target="")
    result = aot_mint.mint(pipe, spec, tmp_path, allow_regressed_lanes=True)
    # Export itself traces the cold branch; no separate warm forward ran and
    # none was recorded.
    entry = next(iter(result.metadata["mint_phases"]["entries"].values()))
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
        aot_mint.mint(pipe, spec, tmp_path, allow_regressed_lanes=True)


# ---------------------------------------------------------------------------
# Mint-phase telemetry (#757's instrument-first deliverable)
# ---------------------------------------------------------------------------


def test_mint_records_the_phase_table(minted_cell) -> None:
    table = minted_cell["result"].metadata["mint_phases"]
    assert table["n_entries"] == 2
    assert set(table["entries"]) == set(minted_cell["result"].metadata["entries"])
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


def test_caller_compile_threads_override_wins() -> None:
    configs = aot_mint._entry_configs({"compile_threads": 16})
    assert configs["compile_threads"] == 16
    assert aot_mint._entry_configs(None)["compile_threads"] == 4


# ---------------------------------------------------------------------------
# Serve — ONE loaded cell, every declared function
# ---------------------------------------------------------------------------


def _armed(minted_cell) -> Any:
    pipe = minted_cell["pipe"]
    if not aot_serve.is_armed(pipe):
        cfg = types.SimpleNamespace(family=FAMILY, lora_bucket=0)
        aot_serve.load_and_wrap(
            pipe, cfg, minted_cell["result"].artifact,
            cache_dir=minted_cell["tmp"] / "cache")
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

        def load_constants(self, values, check_full_update=True):
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


def test_unbound_entry_is_refused_before_the_segfault() -> None:
    contract = aot_serve.contract_from_meta({
        "inputs": [{"name": "sample", "position": 0, "dtype": "float32",
                    "shape": [2, 4]}],
        "symbols": {},
    })
    runner = aot_serve.ArtifactRunner(
        package=object(), contract=contract, constants=(),
        entry="unet/cfg=true/B=2")
    with pytest.raises(aot_serve.ConstantsUnboundError) as err:
        runner(torch.randn(2, 4))
    assert "unet/cfg=true/B=2" in str(err.value)


def test_all_entries_bind_before_any_wrap(minted_cell, monkeypatch, tmp_path) -> None:
    """A cell that cannot bind ONE class arms NONE: a partial arm would be a
    silent subset of the contract the key advertises."""
    pipe = types.SimpleNamespace(unet=TinyUNet())
    cfg = types.SimpleNamespace(family=FAMILY, lora_bucket=0)
    real_bind = aot_serve.ArtifactRunner.bind
    calls: list = []

    def bind_second_fails(self, state_dict, literals):
        calls.append(self.entry)
        if len(calls) == 2:
            raise aot_serve.ConstantsUnboundError("constant_unresolved", "boom")
        return real_bind(self, state_dict, literals)

    monkeypatch.setattr(aot_serve.ArtifactRunner, "bind", bind_second_fails)
    with pytest.raises(compile_cache.AdoptError):
        aot_serve.load_and_wrap(
            pipe, cfg, minted_cell["result"].artifact, cache_dir=tmp_path)
    assert not aot_serve.is_armed(pipe)
    assert not hasattr(pipe.unet, "_cozy_aot")


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
    meta = json.loads(json.dumps(minted_cell["result"].metadata))
    name = sorted(meta["entries"])[0]
    meta["entries"][name]["inputs"] = []
    reason = aot_serve.verify(meta, family=FAMILY)
    assert name in reason


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
        aot_serve.entries_from_meta({"entries": {
            "unet::x": {"target": "unet", "inputs": [
                {"name": "s", "position": 0, "dtype": "float32", "shape": [1]}],
                "symbols": {}, "constants": []},
        }})
