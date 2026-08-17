"""Arming a compiled graph from the STORE, with no eager module anywhere.

pgw#1329. ``arm_compiled_graph`` binds ``resident_constants(module)``, so
serving one compiled graph required the whole diffusers pipeline resident as
the constant SOURCE — the single reason adopt-only serving (pgw#1328) could
not shed eager loading. The artifact already names every constant by FQN,
dtype and exact shape, so the same by-reference table is buildable from
safetensors bytes.

What is REAL here: real safetensors shards on disk, the real manifest parser,
the real two-phase plan/realize, the real ``EntryDispatch`` registration, and
a runner double that enforces TCG's actual ``bind`` contract (once-only,
exact table, ``user_managed``). What is NOT here is the AOTI compile — it
cannot run on the dev box (weights/compile locality) and proving it in
software would prove nothing about a GPU. The bitwise-equality bar lives on a
real pod: ``gen_worker.benchmarks.store_arm_parity``.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import pytest

torch = pytest.importorskip("torch")

from safetensors.torch import save_file

from gen_worker import aot_constants, aot_serve
from gen_worker.compile_cache import AdoptError
from gen_worker._vendor.torchcg import CallIngress, CallInput

FAMILY = "store-arm-1329"
KEY = "cg-key-v1-" + "b" * 56
GRAPH_CLASS = "denoiser"
TARGET = "transformer"
WEIGHT_SET = "cozy://weights/tiny-1329@sha256:" + "c" * 8


# ---------------------------------------------------------------------------
# Fixtures: a real shard, a real graph_class block, a contract-enforcing runner
# ---------------------------------------------------------------------------


def _shard(directory: Path, name: str, tensors: Mapping[str, Any]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    save_file(dict(tensors), str(path))
    return path


def _weights() -> Dict[str, Any]:
    generator = torch.Generator().manual_seed(1329)
    return {
        "lin.weight": torch.randn(8, 8, generator=generator, dtype=torch.float32),
        "lin.bias": torch.randn(8, generator=generator, dtype=torch.float32),
    }


def _constant_rows(weights: Mapping[str, Any]) -> List[Dict[str, Any]]:
    return [
        {
            "fqn": name,
            "source": "state_dict",
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "shape": list(tensor.shape),
        }
        for name, tensor in sorted(weights.items())
    ]


def _contract() -> CallIngress:
    return CallIngress(
        parameters=("sample",),
        flat_arity=1,
        inputs=(
            CallInput("sample", 0, "sample", 0, (), "sample", "float32", (1, 8)),
        ),
    )


def _graph_block(weights: Mapping[str, Any]) -> Dict[str, Any]:
    """A graph_class block shaped exactly as TCG admits one."""

    return {
        "name": GRAPH_CLASS,
        "target": TARGET,
        "class_hash": "d" * 16,
        "constants": _constant_rows(weights),
        "graph": {"pytree": {"ingress": _contract().as_dict()}},
    }


class _RunnerDouble:
    """TCG's ``bind`` contract, enforced — not a permissive stand-in.

    Once-only, exact-table, and it REFUSES a value that is not the declared
    dtype/shape. A double that accepted anything would let this file pass
    while the real runner rejected the same table on a pod.
    """

    def __init__(self, rows: Sequence[Mapping[str, Any]]) -> None:
        self._rows = [dict(row) for row in rows]
        self.bound = False
        self.calls = 0
        self.bound_values: Dict[str, Any] = {}

    @property
    def declared_fqns(self) -> Tuple[str, ...]:
        return tuple(str(row["fqn"]) for row in self._rows)

    def bind(self, state: Mapping[str, Any], *, device: str) -> None:
        if self.bound:
            raise AssertionError("bind is once-only and was called twice")
        wanted = {
            str(row["fqn"]) for row in self._rows if row["source"] == "state_dict"
        }
        if set(state) != wanted:
            raise AssertionError(
                f"bind table != declared state_dict set: "
                f"extra={sorted(set(state) - wanted)!r} "
                f"missing={sorted(wanted - set(state))!r}"
            )
        for row in self._rows:
            if row["source"] != "state_dict":
                continue
            value = state[str(row["fqn"])]
            assert str(value.dtype).removeprefix("torch.") == row["dtype"]
            assert list(value.shape) == list(row["shape"])
            assert str(value.device).startswith(str(device))
        self.bound_values = dict(state)
        self.bound = True

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        weight = self.bound_values["lin.weight"]
        bias = self.bound_values["lin.bias"]
        return torch.nn.functional.linear(args[0], weight, bias)


def _resolved(weights: Mapping[str, Any]) -> aot_serve.ResolvedGraphClass:
    block = _graph_block(weights)
    return aot_serve.ResolvedGraphClass(
        key=KEY,
        runner=_RunnerDouble(block["constants"]),  # type: ignore[arg-type]
        metadata={
            aot_serve.COMPILED_GRAPH_FORMAT_KEY: aot_serve.COMPILED_GRAPH_FORMAT,
            "compiled_graph_key": KEY,
        },
        graph_class=block,
        graph=block["graph"],
        name=GRAPH_CLASS,
        target=TARGET,
    )


@pytest.fixture()
def armed_store(tmp_path: Path) -> Tuple[Dict[str, Any], aot_constants.ConstantStore]:
    weights = _weights()
    _shard(tmp_path / "transformer", "model.safetensors", weights)
    store = aot_constants.SafetensorsConstantStore.for_component(
        tmp_path / "transformer", weight_set=WEIGHT_SET, why="pgw#1329 arm"
    )
    return weights, store


def _patch_resolve(
    monkeypatch: pytest.MonkeyPatch, weights: Mapping[str, Any]
) -> aot_serve.ResolvedGraphClass:
    resolved = _resolved(weights)
    monkeypatch.setattr(
        aot_serve, "_resolve_graph_class", lambda _key, _cache: resolved
    )
    return resolved


# ---------------------------------------------------------------------------
# The manifest is a versioned, typed schema
# ---------------------------------------------------------------------------


def test_manifest_refuses_an_envelope_version_it_does_not_understand() -> None:
    with pytest.raises(aot_constants.ConstantManifestError) as excinfo:
        aot_constants.parse_constant_manifest(
            _graph_block(_weights()), compiled_graph_format=2
        )
    assert excinfo.value.reason == "manifest_version_unsupported"


def test_manifest_refuses_a_row_whose_field_set_is_not_v1() -> None:
    block = _graph_block(_weights())
    block["constants"][0]["provenance"] = "somewhere"
    with pytest.raises(aot_constants.ConstantManifestError) as excinfo:
        aot_constants.parse_constant_manifest(block, compiled_graph_format=1)
    assert excinfo.value.reason == "constant_row_malformed"


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (lambda row: row.update(source="whatever"), "constant_source_unknown"),
        (lambda row: row.update(dtype="torch.float32"), "dtype_malformed"),
        (lambda row: row.update(dtype=""), "dtype_malformed"),
        (lambda row: row.update(fqn=" lin.weight"), "fqn_malformed"),
        (lambda row: row.update(fqn="lin..weight"), "fqn_malformed"),
        (lambda row: row.update(shape=[-1, 8]), "shape_malformed"),
        (lambda row: row.update(shape=[True]), "shape_malformed"),
    ],
)
def test_manifest_refuses_each_malformed_row(
    mutate: Any, reason: str
) -> None:
    block = _graph_block(_weights())
    mutate(block["constants"][0])
    with pytest.raises(aot_constants.ConstantManifestError) as excinfo:
        aot_constants.parse_constant_manifest(block, compiled_graph_format=1)
    assert excinfo.value.reason == reason


def test_manifest_refuses_a_duplicated_fqn() -> None:
    block = _graph_block(_weights())
    block["constants"].append(dict(block["constants"][0]))
    with pytest.raises(aot_constants.ConstantManifestError) as excinfo:
        aot_constants.parse_constant_manifest(block, compiled_graph_format=1)
    assert excinfo.value.reason == "constant_duplicate"


def test_only_state_dict_rows_are_the_store_s_problem() -> None:
    block = _graph_block(_weights())
    block["constants"].append(
        {"fqn": "folded.0", "source": "computed", "dtype": "float32", "shape": [4]}
    )
    block["constants"].append(
        {"fqn": "baked.0", "source": "literal", "dtype": "float32", "shape": [4]}
    )
    manifest = aot_constants.parse_constant_manifest(block, compiled_graph_format=1)
    assert len(manifest.constants) == 4
    assert sorted(str(spec.fqn) for spec in manifest.store_sourced) == [
        "lin.bias",
        "lin.weight",
    ]


# ---------------------------------------------------------------------------
# The plan refuses on HEADERS — before any device allocation
# ---------------------------------------------------------------------------


class _ExplodingReadStore:
    """A store whose ``read`` is a test failure. Planning must never call it."""

    def __init__(self, index: Mapping[str, aot_constants.TensorFacts]) -> None:
        self._index = {
            aot_constants.ConstantFQN(name): facts for name, facts in index.items()
        }

    @property
    def weight_set(self) -> aot_constants.WeightSetRef:
        return aot_constants.WeightSetRef(WEIGHT_SET)

    def describe(self) -> Mapping[aot_constants.ConstantFQN, aot_constants.TensorFacts]:
        return dict(self._index)

    def read(self, fqn: aot_constants.ConstantFQN, *, device: str) -> Any:
        raise AssertionError(f"planning read {fqn!r}: no byte may be read to plan")


def _facts(dtype: str, shape: Tuple[int, ...]) -> aot_constants.TensorFacts:
    return aot_constants.TensorFacts(
        dtype=aot_constants.TorchDtype(dtype), shape=shape
    )


@pytest.mark.parametrize(
    ("index", "reason", "named"),
    [
        ({"lin.weight": _facts("float32", (8, 8))}, "constant_absent", "lin.bias"),
        (
            {
                "lin.weight": _facts("bfloat16", (8, 8)),
                "lin.bias": _facts("float32", (8,)),
            },
            "constant_dtype_mismatch",
            "lin.weight",
        ),
        (
            {
                "lin.weight": _facts("float32", (8, 16)),
                "lin.bias": _facts("float32", (8,)),
            },
            "constant_shape_mismatch",
            "lin.weight",
        ),
    ],
)
def test_plan_refuses_by_name_without_reading_a_byte(
    index: Mapping[str, aot_constants.TensorFacts], reason: str, named: str
) -> None:
    manifest = aot_constants.parse_constant_manifest(
        _graph_block(_weights()), compiled_graph_format=1
    )
    with pytest.raises(aot_constants.ConstantResolutionError) as excinfo:
        aot_constants.plan_store_constants(manifest, _ExplodingReadStore(index))
    assert excinfo.value.reason == reason
    assert any(named in fault for fault in excinfo.value.fqns)


def test_plan_carries_the_weight_set_that_answers_which_checkpoint() -> None:
    manifest = aot_constants.parse_constant_manifest(
        _graph_block(_weights()), compiled_graph_format=1
    )
    plan = aot_constants.plan_store_constants(
        manifest,
        _ExplodingReadStore(
            {
                "lin.weight": _facts("float32", (8, 8)),
                "lin.bias": _facts("float32", (8,)),
            }
        ),
    )
    # Class identity is checkpoint-free (§4.27); the plan is what carries the
    # instance-level fact, and it is a parsed type rather than a bare str.
    assert plan.weight_set == WEIGHT_SET
    assert plan.graph_class == GRAPH_CLASS
    assert plan.elements == 8 * 8 + 8
    assert len(plan) == 2


def test_a_store_that_cannot_name_its_weight_set_is_refused() -> None:
    with pytest.raises(aot_constants.ConstantManifestError) as excinfo:
        aot_constants.parse_weight_set_ref("")
    assert excinfo.value.reason == "weight_set_ref_malformed"


# ---------------------------------------------------------------------------
# The safetensors store is fail-CLOSED
# ---------------------------------------------------------------------------


def test_store_indexes_real_shards_by_fqn(tmp_path: Path) -> None:
    weights = _weights()
    _shard(tmp_path / "c", "a.safetensors", {"lin.weight": weights["lin.weight"]})
    _shard(tmp_path / "c", "b.safetensors", {"lin.bias": weights["lin.bias"]})
    store = aot_constants.SafetensorsConstantStore.for_component(
        tmp_path / "c", weight_set=WEIGHT_SET, why="index"
    )
    index = store.describe()
    assert sorted(str(name) for name in index) == ["lin.bias", "lin.weight"]
    assert index[aot_constants.ConstantFQN("lin.weight")] == _facts(
        "float32", (8, 8)
    )


def test_store_refuses_a_shard_whose_header_will_not_parse(tmp_path: Path) -> None:
    """The fail-OPEN spelling of this loop is the pgw#1330 defect.

    ``key_topology.tensor_keys`` swallows exactly this error and yields no
    keys, so an unreadable shard is indistinguishable from a shard that does
    not hold the tensor — and the caller concludes "absent" about a file it
    could not read.
    """

    directory = tmp_path / "c"
    directory.mkdir()
    (directory / "broken.safetensors").write_bytes(
        struct.pack("<Q", 16) + b"not json at all!"
    )
    with pytest.raises(aot_constants.ConstantStoreError) as excinfo:
        aot_constants.SafetensorsConstantStore.for_component(
            directory, weight_set=WEIGHT_SET, why="index"
        )
    assert excinfo.value.reason == "shard_unreadable"


def test_store_refuses_a_dtype_whose_element_width_it_does_not_know(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "c"
    directory.mkdir()
    header = json.dumps({"lin.weight": {"dtype": "F4_E2M1", "shape": [2], "data_offsets": [0, 1]}})
    raw = header.encode()
    (directory / "odd.safetensors").write_bytes(
        struct.pack("<Q", len(raw)) + raw + b"\x00"
    )
    with pytest.raises(aot_constants.ConstantStoreError) as excinfo:
        aot_constants.SafetensorsConstantStore.for_component(
            directory, weight_set=WEIGHT_SET, why="index"
        )
    assert excinfo.value.reason == "dtype_unknown"


def test_store_refuses_one_tensor_named_by_two_shards(tmp_path: Path) -> None:
    weights = _weights()
    _shard(tmp_path / "c", "a.safetensors", {"lin.weight": weights["lin.weight"]})
    _shard(tmp_path / "c", "b.safetensors", {"lin.weight": weights["lin.weight"]})
    with pytest.raises(aot_constants.ConstantStoreError) as excinfo:
        aot_constants.SafetensorsConstantStore.for_component(
            tmp_path / "c", weight_set=WEIGHT_SET, why="index"
        )
    assert excinfo.value.reason == "tensor_duplicated"


def test_store_reads_the_exact_bytes_the_shard_holds(
    armed_store: Tuple[Dict[str, Any], aot_constants.ConstantStore]
) -> None:
    weights, store = armed_store
    for name, expected in weights.items():
        got = store.read(aot_constants.ConstantFQN(name), device="cpu")
        assert torch.equal(got, expected)
        assert got.dtype == expected.dtype


# ---------------------------------------------------------------------------
# The arm itself
# ---------------------------------------------------------------------------


def test_store_sourced_arm_binds_and_serves_with_no_module(
    armed_store: Tuple[Dict[str, Any], aot_constants.ConstantStore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    weights, store = armed_store
    resolved = _patch_resolve(monkeypatch, weights)

    armed = aot_serve.arm_compiled_graph_from_store(
        SimpleNamespace(family=FAMILY), KEY, store, device="cpu"
    )

    assert armed.key == KEY
    assert armed.graph_class == GRAPH_CLASS
    assert armed.target == TARGET
    assert armed.weight_set == WEIGHT_SET
    assert armed.meta["constant_source"] == "store"
    assert armed.meta["weight_set"] == WEIGHT_SET
    assert resolved.runner.bound  # type: ignore[attr-defined]
    # The table is the store's tensors, by reference — not a copy of a module.
    for name, expected in weights.items():
        assert torch.equal(armed.constants[name], expected)
    assert aot_serve.entry_states(SimpleNamespace()) == {}
    assert [name for name, _ in armed.dispatch.runners] == [GRAPH_CLASS]

    sample = torch.ones(1, 8)
    got = armed(sample)
    assert torch.equal(
        got, torch.nn.functional.linear(sample, weights["lin.weight"], weights["lin.bias"])
    )
    assert armed.runner.calls == 1


def test_an_absent_fqn_fails_closed_before_any_dispatch_state_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The checklist row, stated as an ordering property.

    ``CompiledGraphRunner.bind`` is once-only and marks itself ``_failed`` on
    a partial table, so a miss discovered mid-bind costs the runner as well as
    the memory. The refusal must therefore land before the bind AND before any
    entry joins a dispatch.
    """

    weights = _weights()
    _shard(tmp_path / "c", "model.safetensors", {"lin.weight": weights["lin.weight"]})
    store = aot_constants.SafetensorsConstantStore.for_component(
        tmp_path / "c", weight_set=WEIGHT_SET, why="short store"
    )
    resolved = _patch_resolve(monkeypatch, weights)
    joined: List[str] = []
    monkeypatch.setattr(
        aot_serve.EntryDispatch,
        "add",
        lambda self, name, runner: joined.append(str(name)),
    )

    with pytest.raises(aot_constants.ConstantResolutionError) as excinfo:
        aot_serve.arm_compiled_graph_from_store(
            SimpleNamespace(family=FAMILY), KEY, store, device="cpu"
        )

    assert excinfo.value.reason == "constant_absent"
    assert excinfo.value.fqns == ("lin.bias",)
    assert joined == []
    assert not resolved.runner.bound  # type: ignore[attr-defined]


def test_a_dtype_drift_refuses_before_the_bind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    weights = _weights()
    drifted = {name: tensor.to(torch.bfloat16) for name, tensor in weights.items()}
    _shard(tmp_path / "c", "model.safetensors", drifted)
    store = aot_constants.SafetensorsConstantStore.for_component(
        tmp_path / "c", weight_set=WEIGHT_SET, why="drifted store"
    )
    resolved = _patch_resolve(monkeypatch, weights)

    with pytest.raises(aot_constants.ConstantResolutionError) as excinfo:
        aot_serve.arm_compiled_graph_from_store(
            SimpleNamespace(family=FAMILY), KEY, store, device="cpu"
        )

    assert excinfo.value.reason == "constant_dtype_mismatch"
    assert not resolved.runner.bound  # type: ignore[attr-defined]


def test_a_store_arm_must_be_told_its_device(
    armed_store: Tuple[Dict[str, Any], aot_constants.ConstantStore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No module means no ``module.device`` to fall back on, and no default.

    ``arm_compiled_graph`` reads the device off the module and defaults to
    ``"cuda"``. Carrying that default here would put the constant table on a
    device the caller never named.
    """

    weights, store = armed_store
    _patch_resolve(monkeypatch, weights)
    with pytest.raises(AdoptError) as excinfo:
        aot_serve.arm_compiled_graph_from_store(
            SimpleNamespace(family=FAMILY), KEY, store, device="  "
        )
    assert excinfo.value.reason == "device_missing"


def test_the_store_arm_keeps_the_same_ingress_contract_as_the_module_arm(
    armed_store: Tuple[Dict[str, Any], aot_constants.ConstantStore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A call outside the declared envelope is refused, exactly as it is on a
    wrapped module: nothing about ingress is different because the constants
    came from the store."""

    weights, store = armed_store
    _patch_resolve(monkeypatch, weights)
    armed = aot_serve.arm_compiled_graph_from_store(
        SimpleNamespace(family=FAMILY), KEY, store, device="cpu"
    )
    assert isinstance(armed.runner.contract, CallIngress)
    with pytest.raises(aot_serve.IngressContractError):
        armed(torch.ones(4, 8))


def test_the_store_sourced_path_never_touches_diffusers_or_nn_module(
    armed_store: Tuple[Dict[str, Any], aot_constants.ConstantStore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The checklist's "no diffusers/module import on the store-sourced path".

    Asserted the only way that binds: ``diffusers`` is made UNIMPORTABLE for
    the duration, and ``nn.Module.__init__`` is made a failure, so an arm that
    reached for either would raise rather than quietly succeed because the
    import happened to be cached from an earlier test.
    """

    import builtins

    weights, store = armed_store
    _patch_resolve(monkeypatch, weights)
    real_import = builtins.__import__

    def _fenced(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "diffusers" or name.startswith("diffusers."):
            raise AssertionError(f"the store-sourced arm imported {name!r}")
        return real_import(name, *args, **kwargs)

    def _no_modules(self: Any, *args: Any, **kwargs: Any) -> None:
        raise AssertionError("the store-sourced arm constructed an nn.Module")

    monkeypatch.setattr(builtins, "__import__", _fenced)
    monkeypatch.setattr(torch.nn.Module, "__init__", _no_modules)

    armed = aot_serve.arm_compiled_graph_from_store(
        SimpleNamespace(family=FAMILY), KEY, store, device="cpu"
    )
    assert armed.runner.bound


def test_two_weight_sets_over_one_key_are_two_instances_of_one_class(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The family/instance split the arming surface must not foreclose.

    Same compiled-graph key, same graph class, same ``.so`` — two
    checkpoints. The class-level facts are identical (§4.27: identity is
    checkpoint-free) and the instance-level facts differ, which is exactly
    what a module-sourced arm could never state.
    """

    first = _weights()
    second = {name: tensor + 1.0 for name, tensor in first.items()}
    _shard(tmp_path / "a", "model.safetensors", first)
    _shard(tmp_path / "b", "model.safetensors", second)

    armed: List[aot_serve.StoreArmedGraph] = []
    for component, ref in (("a", WEIGHT_SET + "-a"), ("b", WEIGHT_SET + "-b")):
        _patch_resolve(monkeypatch, first)
        store = aot_constants.SafetensorsConstantStore.for_component(
            tmp_path / component, weight_set=ref, why="instances"
        )
        armed.append(
            aot_serve.arm_compiled_graph_from_store(
                SimpleNamespace(family=FAMILY), KEY, store, device="cpu"
            )
        )

    assert armed[0].key == armed[1].key
    assert armed[0].graph_class == armed[1].graph_class
    assert armed[0].weight_set != armed[1].weight_set
    assert not torch.equal(
        armed[0].constants["lin.weight"], armed[1].constants["lin.weight"]
    )
