"""pgw#721 AOT serve path: the two pgw#704 blockers, plus envelope,
adoption dispatch, and rotation — everything that runs without a GPU or a
real ``.pt2``.

The two tests that carry the issue:

* **B1** — invoking a code-only artifact before ``load_constants``
  SEGFAULTS the worker process. A segfault cannot be asserted on (it kills
  the test runner too), so the red test asserts the CALL NEVER HAPPENS: a
  fake package records every invocation, and the gate must refuse by name
  with the record still empty.
* **B2** — an exported graph carries zero symbolic-range assertions, so
  out-of-declared-range input is silently accepted. The red test asserts
  the refusal is named AND that the artifact was not reached.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gen_worker import aot_serve as aot

FAMILY = "sdxl-base"
RUNTIME = {"sku": "l4", "sm": "sm_89", "torch": "2.13.0+cu130",
           "cuda": "13.0"}

# 1024x1024 exported with symbolic latent dims: the pgw#704 headline is that
# the SAME artifact serves 1152x896 (144x112 latent units), so the declared
# range must admit it and the assertion must not reject it.
SYMBOLS = {"h": [64, 160], "w": [64, 160]}
INPUTS = [
    {"name": "sample", "position": 0, "dtype": "bfloat16",
     "shape": [2, 4, "h", "w"]},
    {"name": "timestep", "position": 1, "dtype": "int64", "shape": []},
    {"name": "encoder_hidden_states", "position": 2, "dtype": "bfloat16",
     "shape": [2, 77, 2048]},
]
CONSTANTS = [
    {"fqn": "conv_in.weight", "source": aot.SOURCE_STATE_DICT,
     "dtype": "bfloat16", "shape": [320, 4, 3, 3]},
    {"fqn": "conv_in.bias", "source": aot.SOURCE_STATE_DICT,
     "dtype": "bfloat16", "shape": [320]},
]


# ---------------------------------------------------------------------------
# fakes — no torch, no CUDA, no real artifact
# ---------------------------------------------------------------------------


class FakeTensor:
    """Just enough tensor for the ingress assertion (shape + dtype)."""

    def __init__(self, shape, dtype="torch.bfloat16"):
        self.shape = tuple(shape)
        self.dtype = dtype


class FakePackage:
    """Stands in for ``AOTICompiledModel``.

    ``invocations`` is the whole point: in production a call made here with
    ``loaded is False`` is a segfault, so the tests assert this list stays
    empty rather than trying to catch anything.
    """

    def __init__(self, fqns=("conv_in.weight", "conv_in.bias"), *, table_raises=False):
        self._fqns = list(fqns)
        self._table_raises = table_raises
        self.invocations: list[tuple] = []
        self.loaded: dict | None = None
        self.full_update_asked: bool | None = None

    def get_constant_fqns(self):
        if self._table_raises:
            raise RuntimeError("no constant table in this artifact")
        return list(self._fqns)

    def load_constants(self, values, check_full_update=False):
        self.full_update_asked = check_full_update
        if check_full_update and set(values) != set(self._fqns):
            raise RuntimeError("partial constant update refused by torch")
        self.loaded = dict(values)

    def __call__(self, *args, **kwargs):
        self.invocations.append((args, kwargs))
        return "ARTIFACT_OUTPUT"


class FakeModule:
    def __init__(self, weights=("conv_in.weight", "conv_in.bias")):
        self.device = "cpu"
        self._weights = {name: FakeTensor([1]) for name in weights}
        self.eager_calls = 0

    def state_dict(self):
        return dict(self._weights)

    def forward(self, *args, **kwargs):
        self.eager_calls += 1
        return "EAGER_OUTPUT"


class FakePipeline:
    def __init__(self, module=None):
        self.unet = module or FakeModule()


class Cfg:
    family = FAMILY
    lora_bucket = 0
    targets = ("unet",)
    regional = False


#: The single entry name every fixture cell carries (format 2: one entry is
#: simply the N=1 case of a multi-graph cell).
ENTRY = "unet/g"


def _entry(**over):
    """One format-2 entry block, range_digest/class_hash stamped the way
    the mint stamps them (best effort for deliberately-malformed blocks)."""
    e = {
        "target": "unet", "fork": [], "class_dims": [],
        "inputs": [dict(r) for r in INPUTS],
        "symbols": {k: list(v) for k, v in SYMBOLS.items()},
        "constants": [dict(r) for r in CONSTANTS],
        "graph": {},
    }
    e.update(over)
    try:
        e["range_digest"] = aot.range_digest(e)
    except (ValueError, TypeError):
        e.setdefault("range_digest", "")
    e["class_hash"] = aot.class_hash(e, strict=True, lora_bucket=0)
    return e


def _meta(**over):
    entry_over = {
        k: over.pop(k) for k in ("inputs", "symbols", "constants")
        if k in over}
    entries = over.pop("entries", None) or {ENTRY: _entry(**entry_over)}
    m = {
        "format": aot.ARTIFACT_FORMAT, "kind": aot.ARTIFACT_KIND, **RUNTIME,
        "family": FAMILY, "precision": "w8a8",
        "cell_key": "deadbeef", "entries": entries,
        "strict_export": True, "lora_bucket": 0,
        "package_constants_in_so": False,
        "source_ref": "", "source_digest": "",
    }
    m["combined_graph_hash"] = aot.combined_graph_hash(
        str((b or {}).get("class_hash") or "")
        for b in entries.values() if isinstance(b, dict))
    m.update(over)
    return m


def _contract(meta=None):
    return aot.contract_from_meta((meta or _meta())["entries"][ENTRY])


def _specs(meta=None):
    return aot.constants_from_meta((meta or _meta())["entries"][ENTRY])


def _runner(package=None, meta=None):
    return aot.ArtifactRunner(
        package=package or FakePackage(), contract=_contract(meta),
        constants=_specs(meta), module_name="unet", entry=ENTRY)


def _in_range_call(h=128, w=128):
    return (
        (FakeTensor([2, 4, h, w]), FakeTensor([], "torch.int64"),
         FakeTensor([2, 77, 2048])),
        {},
    )


def _tar(tmp_path: Path, meta=None, literals=False, name="cell.tar.gz") -> Path:
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    (work / aot.PACKAGE_NAME).write_bytes(b"\x00not-a-real-pt2")
    if literals:
        (work / aot.LITERALS_NAME).write_bytes(b"\x00fake-safetensors")
    return aot.pack(work, tmp_path / name, meta or _meta())


@pytest.fixture
def stub_runtime(monkeypatch):
    """Pin the runtime key so verify() passes without a GPU."""
    monkeypatch.setattr(aot, "runtime_key", lambda: dict(RUNTIME))


# ---------------------------------------------------------------------------
# B1 — the constants-bound gate. THE non-negotiable test.
# ---------------------------------------------------------------------------


def test_b1_pre_binding_invocation_refused_by_name_and_never_reaches_artifact():
    """Calling a code-only artifact before load_constants segfaults the
    worker. The gate must refuse BY NAME with zero invocations recorded."""
    package = FakePackage()
    runner = _runner(package)
    args, kwargs = _in_range_call()

    assert runner.bound is False
    with pytest.raises(aot.ConstantsUnboundError) as excinfo:
        runner(*args, **kwargs)

    # The refusal is named...
    assert excinfo.value.reason == "constants_unbound"
    assert "segfault" in str(excinfo.value).lower()
    # ...and — the actual assertion of record — the artifact was NEVER called.
    assert package.invocations == []


def test_b1_bind_then_call_reaches_the_artifact():
    """The gate opens once, and only once, binding has been proven."""
    package = FakePackage()
    runner = _runner(package)
    module = FakeModule()

    runner.bind(module.state_dict(), {})
    assert runner.bound is True
    assert runner.bound_fqns == ("conv_in.bias", "conv_in.weight")
    # torch's own full-coverage assertion is demanded, not merely hoped for.
    assert package.full_update_asked is True
    assert set(package.loaded) == {"conv_in.weight", "conv_in.bias"}

    args, kwargs = _in_range_call()
    assert runner(*args, **kwargs) == "ARTIFACT_OUTPUT"
    assert len(package.invocations) == 1
    assert runner.calls == 1


def test_b1_declared_constant_missing_from_weights_refuses_unbound():
    """A partial bind is the segfault precondition — refuse, stay unbound."""
    package = FakePackage()
    runner = _runner(package)
    module = FakeModule(weights=("conv_in.weight",))  # bias absent

    with pytest.raises(aot.ConstantsUnboundError) as excinfo:
        runner.bind(module.state_dict(), {})
    assert excinfo.value.reason == "constant_unresolved"
    assert "conv_in.bias" in str(excinfo.value)
    assert runner.bound is False
    assert package.loaded is None
    # Still closed afterwards.
    with pytest.raises(aot.ConstantsUnboundError):
        runner(*_in_range_call()[0])
    assert package.invocations == []


def test_b1_artifact_wants_an_undeclared_constant():
    """An FQN the artifact wants but the manifest omits would be left
    UNBOUND — and, for a LoRA branch, is also the folded-adapter bug."""
    package = FakePackage(
        fqns=("conv_in.weight", "conv_in.bias", "down.lora_A"))
    runner = _runner(package)
    with pytest.raises(aot.ConstantsUnboundError) as excinfo:
        runner.bind(FakeModule().state_dict(), {})
    assert excinfo.value.reason == "constant_set_mismatch"
    assert "down.lora_A" in str(excinfo.value)
    assert runner.bound is False


def test_b1_unreadable_constant_table_refuses():
    package = FakePackage(table_raises=True)
    runner = _runner(package)
    with pytest.raises(aot.ConstantsUnboundError) as excinfo:
        runner.bind(FakeModule().state_dict(), {})
    assert excinfo.value.reason == "constant_table_unreadable"
    assert runner.bound is False


def test_b1_literal_constants_bind_from_the_artifact_payload():
    """Non-weight lifted constants come from the artifact, not the CAS."""
    meta = _meta(constants=[
        dict(CONSTANTS[0]),
        {"fqn": "_tensor_constant0", "source": aot.SOURCE_LITERAL,
         "dtype": "int64", "shape": [4]},
    ])
    package = FakePackage(fqns=("conv_in.weight", "_tensor_constant0"))
    runner = _runner(package, meta)
    runner.bind(
        FakeModule().state_dict(), {"_tensor_constant0": FakeTensor([4])})
    assert runner.bound is True
    assert set(package.loaded) == {"conv_in.weight", "_tensor_constant0"}


# ---------------------------------------------------------------------------
# B2 — the ingress range assertion. THE doctrine-blocking test.
# ---------------------------------------------------------------------------


def test_b2_out_of_declared_range_refused_by_name():
    """The measured silent acceptance: 2048x2048 (256 latent units) through
    an artifact declaring max=160. Must be refused, naming symbol + bound."""
    contract = _contract()
    with pytest.raises(aot.IngressContractError) as excinfo:
        aot.assert_ingress(
            contract, (FakeTensor([2, 4, 256, 256]),
                       FakeTensor([], "torch.int64"),
                       FakeTensor([2, 77, 2048])), {})
    exc = excinfo.value
    assert exc.reason == "range_violation"
    text = str(exc)
    assert "sample" in text and "'h'" in text and "256" in text
    assert "[64, 160]" in text

    # the min bound is enforced the same way
    with pytest.raises(aot.IngressContractError) as low:
        aot.assert_ingress(
            contract, (FakeTensor([2, 4, 32, 128]),
                       FakeTensor([], "torch.int64"),
                       FakeTensor([2, 77, 2048])), {})
    assert low.value.reason == "range_violation"
    assert "32" in str(low.value)


def test_b2_in_range_second_aspect_ratio_is_admitted():
    """pgw#704's headline: 1024x1024 export serves 1152x896 (144x112)."""
    contract = _contract()
    assert aot.assert_ingress(
        contract, (FakeTensor([2, 4, 144, 112]),
                   FakeTensor([], "torch.int64"),
                   FakeTensor([2, 77, 2048])), {}) == {"h": 144, "w": 112}


def test_b2_range_bounds_are_inclusive():
    for h in (64, 160):
        assert aot.assert_ingress(
            _contract(), (FakeTensor([2, 4, h, h]),
                          FakeTensor([], "torch.int64"),
                          FakeTensor([2, 77, 2048])), {})["h"] == h


def test_b2_static_dim_is_not_a_range():
    """A statically specialized dim admits exactly one value."""
    with pytest.raises(aot.IngressContractError) as excinfo:
        aot.assert_ingress(
            _contract(), (FakeTensor([4, 4, 128, 128]),
                          FakeTensor([], "torch.int64"),
                          FakeTensor([2, 77, 2048])), {})
    assert excinfo.value.reason == "static_dim_mismatch"
    assert "dim 0 = 4" in str(excinfo.value)


def test_b2_dtype_is_asserted():
    with pytest.raises(aot.IngressContractError) as excinfo:
        aot.assert_ingress(
            _contract(), (FakeTensor([2, 4, 128, 128], "torch.float16"),
                          FakeTensor([], "torch.int64"),
                          FakeTensor([2, 77, 2048])), {})
    assert excinfo.value.reason == "dtype_mismatch"
    assert "float16" in str(excinfo.value) and "bfloat16" in str(excinfo.value)


def test_b2_rank_is_asserted():
    with pytest.raises(aot.IngressContractError) as excinfo:
        aot.assert_ingress(
            _contract(), (FakeTensor([2, 4, 128]),
                          FakeTensor([], "torch.int64"),
                          FakeTensor([2, 77, 2048])), {})
    assert excinfo.value.reason == "rank_mismatch"


def test_b2_symbol_consistency_across_inputs():
    """One symbol in two shapes must take ONE value. range_constraints
    cannot express this; the graph requires it."""
    meta = _meta(inputs=[
        {"name": "sample", "position": 0, "dtype": "bfloat16",
         "shape": [2, 4, "h", "w"]},
        {"name": "mask", "position": 1, "dtype": "bfloat16",
         "shape": [2, 1, "h", "w"]},
    ], symbols={"h": [64, 160], "w": [64, 160]})
    contract = _contract(meta)
    # both individually in range, but inconsistent
    with pytest.raises(aot.IngressContractError) as excinfo:
        aot.assert_ingress(
            contract, (FakeTensor([2, 4, 128, 128]),
                       FakeTensor([2, 1, 96, 128])), {})
    assert excinfo.value.reason == "symbol_inconsistent"
    assert "'h'" in str(excinfo.value)
    assert "mask" in str(excinfo.value) and "sample" in str(excinfo.value)
    # consistent passes
    assert aot.assert_ingress(
        contract, (FakeTensor([2, 4, 128, 128]),
                   FakeTensor([2, 1, 128, 128])), {}) == {"h": 128, "w": 128}


def test_b2_missing_declared_input_refused():
    with pytest.raises(aot.IngressContractError) as excinfo:
        aot.assert_ingress(_contract(), (FakeTensor([2, 4, 128, 128]),), {})
    assert excinfo.value.reason == "input_missing"
    assert "timestep" in str(excinfo.value)


def test_b2_optional_input_may_be_absent():
    meta = _meta(inputs=INPUTS + [
        {"name": "added_cond", "position": 3, "dtype": "bfloat16",
         "shape": [2, 2816], "optional": True}])
    assert aot.assert_ingress(_contract(meta), *_in_range_call())
    with pytest.raises(aot.IngressContractError) as excinfo:
        aot.assert_ingress(
            _contract(meta), (*_in_range_call()[0], FakeTensor([2, 99])), {})
    assert excinfo.value.reason == "static_dim_mismatch"


def test_b2_non_tensor_argument_refused():
    with pytest.raises(aot.IngressContractError) as excinfo:
        aot.assert_ingress(
            _contract(), (FakeTensor([2, 4, 128, 128]), 7,
                          FakeTensor([2, 77, 2048])), {})
    assert excinfo.value.reason == "input_not_tensor"


# ---------------------------------------------------------------------------
# the fail-soft module swap
# ---------------------------------------------------------------------------


def test_swap_serves_out_of_contract_requests_eagerly_and_stays_armed():
    """B2's designed outcome: named refusal, eager service, artifact intact
    for in-contract traffic — one bad request says nothing about health."""
    module, package = FakeModule(), FakePackage()
    pipeline = FakePipeline(module)
    runner = _runner(package)
    runner.bind(module.state_dict(), {})
    meta = _meta()
    aot.wrap_module(module, runner, meta)
    setattr(pipeline, "_cozy_aot", {"meta": meta, "module": module,
                                    "state": getattr(module, "_cozy_aot")["state"]})

    # in-contract: the artifact serves
    assert module.forward(*_in_range_call()[0]) == "ARTIFACT_OUTPUT"
    assert aot.execution_count(pipeline) == 1
    assert module.eager_calls == 0

    # out-of-contract: eager serves, refusal counted, artifact untouched
    assert module.forward(
        FakeTensor([2, 4, 256, 256]), FakeTensor([], "torch.int64"),
        FakeTensor([2, 77, 2048])) == "EAGER_OUTPUT"
    assert module.eager_calls == 1
    assert aot.ingress_refusals(pipeline) == 1
    assert len(package.invocations) == 1
    assert aot.is_armed(pipeline) is True

    # still armed for the next in-contract request
    assert module.forward(*_in_range_call(144, 112)[0]) == "ARTIFACT_OUTPUT"
    assert aot.execution_count(pipeline) == 2


def test_swap_falls_permanently_eager_and_revokes_proof_on_artifact_error():
    class Exploding(FakePackage):
        def __call__(self, *args, **kwargs):
            self.invocations.append((args, kwargs))
            raise RuntimeError("cuda kernel launch failed")

    module, package = FakeModule(), Exploding()
    pipeline = FakePipeline(module)
    runner = _runner(package)
    runner.bind(module.state_dict(), {})
    meta = _meta()
    aot.wrap_module(module, runner, meta)
    setattr(pipeline, "_cozy_aot", {"meta": meta, "module": module,
                                    "state": getattr(module, "_cozy_aot")["state"]})
    seen: list[str] = []
    assert aot.set_guard_failure_callback(pipeline, seen.append) is True

    assert module.forward(*_in_range_call()[0]) == "EAGER_OUTPUT"
    assert module.forward(*_in_range_call()[0]) == "EAGER_OUTPUT"
    # permanently eager: the second call never re-entered the artifact
    assert len(package.invocations) == 1
    assert module.eager_calls == 2
    # ...and the compiled proof is revoked, through the failure callback.
    assert len(seen) == 1 and "cuda kernel launch failed" in seen[0]
    assert aot.is_armed(pipeline) is False


def test_swap_never_invokes_an_unbound_runner():
    """Belt-and-braces on B1: a wrap installed over an UNBOUND runner (an
    arm-order violation) must still not reach the artifact."""
    module, package = FakeModule(), FakePackage()
    runner = _runner(package)           # deliberately not bound
    aot.wrap_module(module, runner, _meta())
    assert module.forward(*_in_range_call()[0]) == "EAGER_OUTPUT"
    assert package.invocations == []


# ---------------------------------------------------------------------------
# envelope: key, verify, pack/unpack
# ---------------------------------------------------------------------------


def test_labels_and_refs():
    assert aot.torch_maj_min("2.13.0+cu130") == "2.13"
    assert aot.torch_maj_min("") == ""
    assert aot.flavor_label("l4", "2.13.0+cu130", "w8a8") == "aot-l4-torch2.13-w8a8"
    assert aot.flavor_label("", "2.13.0", "w8a8") == ""
    assert aot.is_aot_ref(f"root/family-{FAMILY}#aot-l4-torch2.13-w8a8") is True
    assert aot.is_aot_ref(
        f"root/family-{FAMILY}#aot-l4-torch2.13-w8a8", family="other") is False
    assert aot.is_aot_ref(f"root/family-{FAMILY}#trt-l4-trt10.16-fp16") is False
    # A dynamo cache ref must not be mistaken for an exported cell (pgw#735:
    # it would then be scored by FX hits).
    assert aot.is_aot_ref(
        f"root/family-{FAMILY}#ck5-0123456789abcdef") is False


def test_verify_refuses_baked_weights(stub_runtime):
    """A weights-baked .pt2 would duplicate GiB per cell and break the CAS
    distribution model — refused outright, not warned about."""
    assert aot.verify(_meta()) == ""
    for bad in (True, None):
        meta = _meta()
        if bad is None:
            meta.pop("package_constants_in_so")
        else:
            meta["package_constants_in_so"] = bad
        assert "package_constants_in_so" in aot.verify(meta)


def test_verify_is_abi_exact(stub_runtime):
    assert "torch" in aot.verify(_meta(torch="2.12.0+cu130"))
    assert "sm" in aot.verify(_meta(sm="sm_80"))
    assert "cuda" in aot.verify(_meta(cuda="12.8"))
    # pgw#765: the GPU MODEL is not an axis — an h100-minted cell would be
    # refused on sm_90-vs-sm_89, never on the marketing name.
    assert aot.verify(_meta(sku="h100-80gb-hbm3")) == ""
    assert "kind" in aot.verify(_meta(kind="trt-engine"))
    assert "format" in aot.verify(_meta(format=99))
    assert "family" in aot.verify(_meta(family="flux2"), family=FAMILY)


def test_verify_rejects_malformed_contract(stub_runtime):
    assert "malformed" in aot.verify(_meta(inputs=[]))
    assert "malformed" in aot.verify(_meta(constants="nope"))
    assert "malformed" in aot.verify(_meta(inputs=[
        {"name": "sample", "dtype": "bfloat16", "shape": [2, 4, "zz", "w"]}]))


def test_verify_rejects_tampered_range_digest(stub_runtime):
    meta = aot.artifact_metadata(
        family=FAMILY, precision="w8a8", cell_key="k",
        entries=_entries_arg())
    assert aot.verify(meta) == ""
    meta["entries"][ENTRY]["symbols"]["h"] = [64, 4096]
    reason = aot.verify(meta)
    assert "range_digest" in reason and ENTRY in reason


def _entries_arg(**entry_over):
    block = {
        "target": "unet", "fork": [], "class_dims": [],
        "inputs": [dict(r) for r in INPUTS],
        "symbols": {k: list(v) for k, v in SYMBOLS.items()},
        "constants": [dict(r) for r in CONSTANTS],
        "graph": {}}
    block.update(entry_over)
    return {ENTRY: block}


def test_artifact_metadata_validates_at_mint():
    """A malformed contract must fail on the pod, not on a paid request."""
    with pytest.raises(ValueError, match="no declared range"):
        aot.artifact_metadata(
            family=FAMILY, precision="w8a8", cell_key="k",
            entries=_entries_arg(symbols={"h": [64, 160]}))
    with pytest.raises(ValueError, match="unknown source"):
        aot.artifact_metadata(
            family=FAMILY, precision="w8a8", cell_key="k",
            entries=_entries_arg(
                constants=[{"fqn": "a", "source": "guess", "shape": []}]))
    meta = aot.artifact_metadata(
        family=FAMILY, precision="w8a8", cell_key="k", entries=_entries_arg())
    assert meta["package_constants_in_so"] is False
    entry = meta["entries"][ENTRY]
    assert entry["range_digest"] == aot.range_digest(entry)
    assert entry["class_hash"] == aot.class_hash(
        entry, strict=True, lora_bucket=0)
    assert meta["combined_graph_hash"] == aot.combined_graph_hash(
        [entry["class_hash"]])


def test_constant_manifest_rejects_duplicates():
    with pytest.raises(ValueError, match="repeats"):
        _specs(_meta(constants=[CONSTANTS[0], CONSTANTS[0]]))


def test_pack_unpack_roundtrip_is_deterministic(tmp_path):
    a = _tar(tmp_path, name="a.tar.gz")
    b = _tar(tmp_path, name="b.tar.gz")
    assert a.read_bytes() == b.read_bytes()
    meta = aot.unpack(a, tmp_path / "out")
    assert meta["kind"] == aot.ARTIFACT_KIND
    assert (tmp_path / "out" / aot.PACKAGE_NAME).read_bytes() == b"\x00not-a-real-pt2"
    assert aot.unpack_metadata(a)["cell_key"] == "deadbeef"


def test_unpack_refuses_incomplete_or_foreign_members(tmp_path):
    work = tmp_path / "bare"
    work.mkdir()
    with pytest.raises(ValueError, match="model.pt2 missing"):
        aot.pack(work, tmp_path / "x.tar.gz", _meta())

    import gzip
    import io
    import tarfile
    from gen_worker.compile_cache import _clean_tarinfo
    path = tmp_path / "sneaky.tar.gz"
    with open(path, "wb") as raw, gzip.GzipFile(
            filename="", fileobj=raw, mode="wb", mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w") as tar:
            for name, data in ((aot.METADATA_NAME,
                                json.dumps(_meta()).encode()),
                               (aot.PACKAGE_NAME, b"x"),
                               ("../evil.so", b"x")):
                ti = _clean_tarinfo(tarfile.TarInfo(name))
                ti.size = len(data)
                tar.addfile(ti, io.BytesIO(data))
    with pytest.raises(ValueError, match="unexpected member"):
        aot.unpack(path, tmp_path / "out2")


def test_unpack_refuses_declared_literals_with_no_payload(tmp_path):
    meta = _meta(constants=[
        dict(CONSTANTS[0]),
        {"fqn": "_tensor_constant0", "source": aot.SOURCE_LITERAL,
         "dtype": "int64", "shape": [4]}])
    path = _tar(tmp_path, meta, literals=False)
    with pytest.raises(ValueError, match="carries no constants.safetensors"):
        aot.unpack(path, tmp_path / "out")
    ok = _tar(tmp_path, meta, literals=True, name="ok.tar.gz")
    assert aot.unpack(ok, tmp_path / "out2")["kind"] == aot.ARTIFACT_KIND


def test_stage_artifact_leaves_nothing_behind_on_rejection(tmp_path, monkeypatch):
    monkeypatch.setattr(aot, "runtime_key", lambda: dict(RUNTIME, torch="9.9.9"))
    cache = tmp_path / "cache"
    path = _tar(tmp_path)
    from gen_worker.compile_cache import AdoptError
    with pytest.raises(AdoptError) as excinfo:
        aot.stage_artifact(path, FAMILY, cache_dir=cache)
    assert excinfo.value.reason == "key_mismatch"
    assert list(cache.iterdir()) == []


def test_is_aot_artifact_sniffs_kind(tmp_path):
    assert aot.is_aot_artifact(_tar(tmp_path)) is True
    assert aot.is_aot_artifact(
        _tar(tmp_path, _meta(kind="trt-engine"), name="t.tar.gz")) is False
    junk = tmp_path / "junk.tar.gz"
    junk.write_bytes(b"not a tarball")
    assert aot.is_aot_artifact(junk) is False


def test_find_artifact(tmp_path):
    path = _tar(tmp_path)
    assert aot.find_artifact(path) == path
    snap = tmp_path / "snap"
    (snap / "nested").mkdir(parents=True)
    target = snap / "nested" / "cell.tar.gz"
    target.write_bytes(b"x")
    assert aot.find_artifact(snap) == target
    assert aot.find_artifact(tmp_path / "empty") is None


# ---------------------------------------------------------------------------
# adoption: enable() + the provision dispatch (same HIT path as dynamo)
# ---------------------------------------------------------------------------


def test_enable_arms_and_binds_from_resident_weights(tmp_path, monkeypatch, stub_runtime):
    package = FakePackage()
    monkeypatch.setattr(aot, "_load_package", lambda p, e="model": package)
    module = FakeModule()
    pipeline = FakePipeline(module)
    original = module.forward

    assert aot.enable(pipeline, Cfg(), tmp_path / "cache", _tar(tmp_path)) is True
    assert aot.is_armed(pipeline) is True
    assert module.forward is not original
    # constants came from the RESIDENT weights, not from the artifact
    assert set(package.loaded) == {"conv_in.weight", "conv_in.bias"}
    assert module.forward(*_in_range_call()[0]) == "ARTIFACT_OUTPUT"
    assert aot.execution_count(pipeline) == 1


def test_enable_stays_eager_on_any_miss(tmp_path, monkeypatch, stub_runtime):
    module = FakeModule()
    pipeline = FakePipeline(module)
    original = module.forward

    # no artifact
    assert aot.enable(pipeline, Cfg(), tmp_path / "c") is False
    # wrong runtime key
    monkeypatch.setattr(aot, "_load_package", lambda p, e="model": FakePackage())
    assert aot.enable(
        pipeline, Cfg(), tmp_path / "c",
        _tar(tmp_path, _meta(torch="2.12.0"), name="old.tar.gz")) is False
    # package will not load
    def _boom(path, entry="model"):
        raise RuntimeError("dlopen failed: undefined symbol")
    monkeypatch.setattr(aot, "_load_package", _boom)
    assert aot.enable(pipeline, Cfg(), tmp_path / "c", _tar(tmp_path)) is False
    # constants cannot bind
    monkeypatch.setattr(
        aot, "_load_package",
        lambda p, e="model": FakePackage(fqns=("nope.weight",)))
    assert aot.enable(pipeline, Cfg(), tmp_path / "c", _tar(tmp_path)) is False

    # pipeline without the target module
    class Bare:
        pass

    assert aot.enable(Bare(), Cfg(), tmp_path / "c", _tar(tmp_path)) is False

    # every miss leaves the pipeline exactly eager
    assert module.forward == original
    assert aot.is_armed(pipeline) is False


def test_provision_dispatches_aot_kind_and_reports_a_hit(tmp_path, monkeypatch, stub_runtime):
    """A .pt2 cell must flow through the SAME hit path as a dynamo cell:
    provision.enable_compiled returning True IS the HIT that fleet_cells
    treats as a genuine match (no self-mint, no executor change)."""
    from gen_worker import compile_cache as cc
    from gen_worker.models import provision

    package = FakePackage()
    monkeypatch.setattr(aot, "_load_package", lambda p, e="model": package)
    # the inductor lane must never be consulted for an aot-kind artifact
    def _unexpected(*a, **k):
        raise AssertionError("compile_cache.enable must not be reached")
    monkeypatch.setattr(cc, "enable", _unexpected)

    module = FakeModule()
    pipeline = FakePipeline(module)
    assert provision.enable_compiled(
        pipeline, Cfg(), tmp_path / "cache", _tar(tmp_path)) is True
    assert aot.is_armed(pipeline) is True


def test_provision_falls_through_to_inductor_when_the_pt2_is_unusable(
        tmp_path, monkeypatch, stub_runtime):
    from gen_worker import compile_cache as cc
    from gen_worker.models import provision

    def _boom(path, entry="model"):
        raise RuntimeError("dlopen failed")
    monkeypatch.setattr(aot, "_load_package", _boom)
    seen: list = []

    def _inductor(pipe, cfg, cache_dir=None, artifact=None):
        seen.append(artifact)
        return False
    monkeypatch.setattr(cc, "enable", _inductor)

    pipeline = FakePipeline()
    assert provision.enable_compiled(
        pipeline, Cfg(), tmp_path / "cache", _tar(tmp_path)) is False
    # the unusable .pt2 was DROPPED, not handed to the inductor lane
    assert seen == [None]


def test_provision_still_routes_trt_and_inductor_kinds(tmp_path, monkeypatch, stub_runtime):
    from gen_worker import compile_cache as cc
    from gen_worker import trt_engine as te
    from gen_worker.models import provision

    monkeypatch.setattr(aot, "_load_package", lambda p, e="model": FakePackage())
    monkeypatch.setattr(te, "enable", lambda *a, **k: True)
    pipeline = FakePipeline()
    trt = _tar(tmp_path, _meta(kind="trt-engine"), name="trt.tar.gz")
    assert provision.enable_compiled(
        pipeline, Cfg(), tmp_path / "cache", trt) is True
    assert aot.is_armed(pipeline) is False

    calls: list = []

    def _inductor(pipe, cfg, cache_dir=None, artifact=None):
        calls.append(artifact)
        return True
    monkeypatch.setattr(cc, "enable", _inductor)
    inductor = _tar(tmp_path, _meta(kind="compile-cache"), name="cc.tar.gz")
    assert provision.enable_compiled(
        FakePipeline(), Cfg(), tmp_path / "cache", inductor) is True
    assert calls == [inductor]


def test_provision_drops_the_artifact_when_the_receipts_gate_refuses(
        tmp_path, monkeypatch, stub_runtime):
    """pgw#709 ordering: a .pt2 is an ELF opened with dlopen, so the receipt
    gate must run BEFORE the kind sniff ever reaches aot_serve."""
    from gen_worker import compile_cache as cc
    from gen_worker import receipts
    from gen_worker.models import provision

    monkeypatch.setattr(receipts, "gate_delivered_artifact", lambda *a, **k: False)
    def _unexpected(p, e="model"):
        raise AssertionError("a refused artifact must never be dlopen-ed")
    monkeypatch.setattr(aot, "_load_package", _unexpected)
    seen: list = []

    def _inductor(pipe, cfg, cache_dir=None, artifact=None):
        seen.append(artifact)
        return False
    monkeypatch.setattr(cc, "enable", _inductor)

    assert provision.enable_compiled(
        FakePipeline(), Cfg(), tmp_path / "cache", _tar(tmp_path)) is False
    assert seen == [None]


# ---------------------------------------------------------------------------
# pgw#725 LoRA seam — lifted adapters ride as MANDATORY call kwargs
# ---------------------------------------------------------------------------


def _binding(a_numel=64 * 2048, b_numel=2048 * 64):
    """A real LiftedLoraBinding carrying just the flat pair.

    Built with ``__new__`` to skip the torch allocation: ``call_kwargs()``
    reads only the two slots, and using the REAL class means the production
    ``lifted_binding()`` isinstance check is what the tests exercise.
    """
    from gen_worker.models.lora_lifted import LiftedLoraBinding

    binding = object.__new__(LiftedLoraBinding)
    binding._a = FakeTensor([a_numel])
    binding._b = FakeTensor([b_numel])
    return binding


def _lift(module, binding=None):
    """Install a lifted binding the way the LoRA lane's installer does."""
    binding = binding or _binding()
    setattr(module, "_cozy_lora_lifted", binding)
    return binding


LIFTED_INPUTS = INPUTS + [
    # The lifted pair is FLAT rank-1 — one window per layer inside one tensor.
    {"name": "lora_a", "position": 3, "dtype": "bfloat16", "shape": [64 * 2048]},
    {"name": "lora_b", "position": 4, "dtype": "bfloat16", "shape": [2048 * 64]},
]


def test_wrapper_splats_the_lifted_pair_into_the_artifact_call():
    meta = _meta(inputs=LIFTED_INPUTS)
    package = FakePackage()
    module = FakeModule()
    binding = _lift(module)
    runner = _runner(package, meta)
    runner.bind(module.state_dict(), {})
    aot.wrap_module(module, runner, meta)

    # the pipeline calls the denoiser WITHOUT the adapter kwargs
    assert module.forward(*_in_range_call()[0]) == "ARTIFACT_OUTPUT"
    args, kwargs = package.invocations[0]
    # POSITIONAL-only, measured on pod: the package takes a fixed flat arity
    # and refuses kwargs, so the adapter must land in its declared SLOT.
    assert kwargs == {}
    assert args[3] is binding.tensors[0]
    assert args[4] is binding.tensors[1]


def test_adapter_swap_needs_no_artifact_interaction():
    """The adapter machinery mutates the owned pair in place through views, so
    a swap touches neither the runner nor the constant table — the pointers
    the graph captured stay valid."""
    meta = _meta(inputs=LIFTED_INPUTS)
    package = FakePackage()
    module = FakeModule()
    binding = _lift(module)
    runner = _runner(package, meta)
    runner.bind(module.state_dict(), {})
    bound_before = dict(package.loaded or {})
    aot.wrap_module(module, runner, meta)

    module.forward(*_in_range_call()[0])
    first = package.invocations[0][0][3]

    # a swap writes THROUGH the same tensor object (in place); nothing rebinds
    module.forward(*_in_range_call()[0])
    second = package.invocations[1][0][3]
    assert first is second is binding.tensors[0]
    assert package.loaded == bound_before   # constant table untouched
    assert package.full_update_asked is True


def test_lifted_pair_is_resolved_per_call_not_captured():
    """The LoRA lane may re-install or drop lifting independently of arming,
    so a stale capture would feed the graph a dead pointer."""
    meta = _meta(inputs=LIFTED_INPUTS)
    package = FakePackage()
    module = FakeModule()
    first = _lift(module)
    runner = _runner(package, meta)
    runner.bind(module.state_dict(), {})
    aot.wrap_module(module, runner, meta)

    module.forward(*_in_range_call()[0])
    assert package.invocations[0][0][3] is first.tensors[0]

    # a re-install swaps the binding OBJECT: the next call must follow it
    second = _lift(module, _binding())
    assert second is not first
    module.forward(*_in_range_call()[0])
    assert package.invocations[1][0][3] is second.tensors[0]

    # dropping lifting entirely leaves a MANDATORY declared input absent —
    # B2 refuses by name and serves eager rather than reusing a dead pointer
    delattr(module, "_cozy_lora_lifted")
    assert module.forward(*_in_range_call()[0]) == "EAGER_OUTPUT"
    assert len(package.invocations) == 2
    assert aot.ingress_refusals(module) == 1
    assert "lora_a" in getattr(module, "_cozy_aot")["state"]["last_refusal"]


def test_lifted_pair_is_asserted_at_ingress_like_any_input():
    """A wrong-layout flat pair is refused by name, not silently broadcast."""
    contract = _contract(_meta(inputs=LIFTED_INPUTS))
    good = (*_in_range_call()[0],
            FakeTensor([64 * 2048]), FakeTensor([2048 * 64]))
    assert aot.assert_ingress(contract, good, {}) == {"h": 128, "w": 128}

    bad = (*_in_range_call()[0],
           FakeTensor([32 * 2048]), FakeTensor([2048 * 64]))
    with pytest.raises(aot.IngressContractError) as excinfo:
        aot.assert_ingress(contract, bad, {})
    assert excinfo.value.reason == "static_dim_mismatch"
    assert "lora_a" in str(excinfo.value)


def test_assert_lifted_contract_agrees_both_ways():
    lifted_contract = _contract(_meta(inputs=LIFTED_INPUTS))
    plain_contract = _contract()

    plain_module = FakeModule()
    lifted_module = FakeModule()
    _lift(lifted_module)

    # matching pairs are fine
    aot.assert_lifted_contract(plain_module, plain_contract)
    aot.assert_lifted_contract(lifted_module, lifted_contract)

    # lifted module, artifact that never declared the adapter => the branch
    # was traced away and every request would serve the base model
    from gen_worker.compile_cache import AdoptError
    with pytest.raises(AdoptError) as excinfo:
        aot.assert_lifted_contract(lifted_module, plain_contract)
    assert excinfo.value.reason == "lifted_inputs_undeclared"

    # artifact wants the adapter, module cannot supply it
    with pytest.raises(AdoptError) as excinfo:
        aot.assert_lifted_contract(plain_module, lifted_contract)
    assert excinfo.value.reason == "lifted_inputs_unbindable"


def test_enable_refuses_a_lifted_module_against_a_branchless_cell(
        tmp_path, monkeypatch, stub_runtime):
    """The mismatch is caught at ARM time, by name, not at first call."""
    monkeypatch.setattr(aot, "_load_package", lambda p, e="model": FakePackage())
    module = FakeModule()
    _lift(module)
    pipeline = FakePipeline(module)
    original = module.forward

    assert aot.enable(pipeline, Cfg(), tmp_path / "c", _tar(tmp_path)) is False
    assert aot.is_armed(pipeline) is False
    assert module.forward == original


def test_enable_arms_a_lifted_module_against_a_lifted_cell(
        tmp_path, monkeypatch, stub_runtime):
    package = FakePackage()
    monkeypatch.setattr(aot, "_load_package", lambda p, e="model": package)
    module = FakeModule()
    binding = _lift(module)
    pipeline = FakePipeline(module)
    art = _tar(tmp_path, _meta(inputs=LIFTED_INPUTS), name="lifted.tar.gz")

    assert aot.enable(pipeline, Cfg(), tmp_path / "c", art) is True
    assert module.forward(*_in_range_call()[0]) == "ARTIFACT_OUTPUT"
    assert package.invocations[0][0][3] is binding.tensors[0]
    assert aot.execution_count(pipeline) == 1


def test_artifact_is_called_positionally_never_with_kwargs():
    """MEASURED on the first-light pod: an AOTI package takes POSITIONAL
    inputs only — splatting kwargs at it dies with "Ran into a kwarg keyword
    mismatch: Got [...] but expected []" before any compiled code runs.
    Diffusers calls a denoiser with a mix of positional and keyword args, so
    the serve path must flatten to the order export recorded."""
    package = FakePackage()
    runner = _runner(package)
    runner.bind(FakeModule().state_dict(), {})
    a, t, e = _in_range_call()[0]

    # caller uses kwargs for everything except the first arg
    runner(a, timestep=t, encoder_hidden_states=e)
    args, kwargs = package.invocations[0]
    assert kwargs == {}, "the package refuses kwargs"
    assert args == (a, t, e), "flattened into declared position order"

    # and a caller that reverses the kwarg order still lands correctly
    runner(a, encoder_hidden_states=e, timestep=t)
    assert package.invocations[1][0] == (a, t, e)


def test_marshal_positional_refuses_a_missing_input_rather_than_shifting():
    """A package has a fixed flat arity: skipping an absent input would slide
    every later argument into the wrong graph slot."""
    contract = _contract()
    a, t, e = _in_range_call()[0]
    assert aot.marshal_positional(contract, (a, t, e), {}) == [a, t, e]

    # a required input absent -> refused by name (bind_call_inputs)
    with pytest.raises(aot.IngressContractError) as excinfo:
        aot.marshal_positional(contract, (a, t), {})
    assert excinfo.value.reason == "input_missing"
    assert "encoder_hidden_states" in str(excinfo.value)

    # an OPTIONAL declared input absent is what ingress alone would permit —
    # marshalling still refuses it, because skipping it would shift every
    # later argument into the wrong graph slot.
    meta = _meta(inputs=INPUTS + [
        {"name": "added_cond", "position": 3, "dtype": "bfloat16",
         "shape": [2, 2816], "optional": True}])
    opt = _contract(meta)
    assert aot.assert_ingress(opt, (a, t, e), {})          # ingress: fine
    with pytest.raises(aot.IngressContractError) as excinfo:
        aot.marshal_positional(opt, (a, t, e), {})
    assert excinfo.value.reason == "input_missing"
    assert "wrong graph slot" in str(excinfo.value)
