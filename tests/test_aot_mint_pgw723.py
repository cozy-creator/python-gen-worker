"""pgw#723 — the AOT mint path: export, code-only AOTI package, gates, pack.

Integration tests over REAL ``torch.export`` + REAL ``aoti_compile_and_package``
runs on tiny CPU modules. Nothing that carries meaning is mocked: the packages
under test are genuine ``.pt2`` archives, and the B1 gate is proven by compiling
a package that ACTUALLY bakes its constants and asserting the gate refuses it by
name. A fabricated "baked" fixture would only prove the gate reads our fixture.

The single unavoidable seam is the compute-capability probe: a cell key requires
an ``sm``, this box has no usable GPU, and no test design changes that.
``_gpu_runtime`` patches the PROBES and nothing else — every gate, digest, and
pack path exercised below is the production one.

Envelope ownership: ``aot_serve`` owns ``artifact_metadata``/``pack``/``verify``
(#721 S1) and this lane imports it. ``aot_package`` owns the mint-side ``.pt2``
introspection and the B1 gate. ``lora_lifted`` owns the no-baked-adapter gate,
which must run on the ExportedProgram — see the test that proves the
package-side alternative is a false pass.

CPU AOTI compiles cost real seconds, so packages are session-scoped.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import pytest

torch = pytest.importorskip("torch")

from gen_worker import aot_mint, aot_package, aot_serve, cell_key  # noqa: E402
from gen_worker.api.errors import ValidationError  # noqa: E402
from gen_worker.models import lora_lifted  # noqa: E402


LORA_RANK, WIDTH = 8, 16


# ---------------------------------------------------------------------------
# Tiny modules + real packages
# ---------------------------------------------------------------------------


class LiftedModule(torch.nn.Module):
    """The #725 option-2 shape: the adapter arrives as CALL ARGUMENTS.

    ``lora_a``/``lora_b`` are forward parameters, so they can never reach the
    package's constant table — the structural guarantee option 2 was chosen for.
    """

    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(WIDTH, WIDTH)
        self.register_buffer("scale_table", torch.randn(WIDTH, WIDTH))

    def forward(
        self, x: torch.Tensor, lora_a: torch.Tensor, lora_b: torch.Tensor,
    ) -> torch.Tensor:
        base = self.proj(x)
        delta = (x @ lora_a.t()) @ lora_b.t()
        return base + delta + self.scale_table.sum()


class BakedBufferModule(torch.nn.Module):
    """The bug the adapter gate exists to make unpublishable: the adapter is a
    registered BUFFER, so export lifts it into the constant table and a serving
    pod could never swap it."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(WIDTH, WIDTH)
        self.register_buffer("lora_a", torch.randn(LORA_RANK, WIDTH))
        self.register_buffer("lora_b", torch.randn(WIDTH, LORA_RANK))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x) + (x @ self.lora_a.t()) @ self.lora_b.t()


class PlainAttrModule(torch.nn.Module):
    """The measured false-PASS case (pgw#725): the adapter lives in plain
    ``__dict__``, so packing renames it to ``_tensor_constant0`` and a
    package-side FQN scan finds no ``lora_*`` name at all."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(WIDTH, WIDTH)
        object.__setattr__(self, "lora_a", torch.randn(LORA_RANK, WIDTH))
        object.__setattr__(self, "lora_b", torch.randn(WIDTH, LORA_RANK))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x) + (x @ self.lora_a.t()) @ self.lora_b.t()


def _example_inputs() -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    # A NON-ZERO adapter, deliberately: exporting with zeros lets constant
    # folding erase the branch and leaves a gate passing on absence, which
    # pgw#704 S9 calls out as the same bug in a different hat.
    return (
        torch.randn(4, WIDTH),
        torch.randn(LORA_RANK, WIDTH),
        torch.randn(WIDTH, LORA_RANK),
    ), {}


def _export_lifted(*, lo: int = 2, hi: int = 32) -> Any:
    # The seeding batch must sit INSIDE the declared range; which value it is
    # decides nothing about what the artifact admits (that is the range's job).
    batch = min(max(4, lo), hi)
    args = (
        torch.randn(batch, WIDTH),
        torch.randn(LORA_RANK, WIDTH),
        torch.randn(WIDTH, LORA_RANK),
    )
    kwargs: Dict[str, Any] = {}
    return torch.export.export(
        LiftedModule().eval(), args, kwargs,
        dynamic_shapes={
            "x": {0: torch.export.Dim("x_0", min=lo, max=hi)},
            "lora_a": None, "lora_b": None,
        },
        strict=True,
    )


def _export_module(module: torch.nn.Module) -> Any:
    return torch.export.export(
        module, (torch.randn(4, WIDTH),),
        dynamic_shapes={"x": {0: torch.export.Dim("x_0", min=2, max=32)}},
        strict=True,
    )


def _compile(program: Any, out: Path, *, code_only: bool) -> Path:
    return Path(torch._inductor.aoti_compile_and_package(
        program, package_path=str(out),
        inductor_configs={
            "aot_inductor.package_constants_in_so": not code_only},
    ))


@pytest.fixture(scope="session")
def packages(tmp_path_factory: pytest.TempPathFactory) -> Dict[str, Path]:
    """Real ``.pt2`` packages: code-only, weights-baked, and the two adapter
    shapes the gate has to distinguish."""
    root = tmp_path_factory.mktemp("aot-packages")
    lifted = _export_lifted()
    return {
        "code_only": _compile(lifted, root / "code_only.pt2", code_only=True),
        "baked": _compile(lifted, root / "baked.pt2", code_only=False),
        "buffer_adapter": _compile(
            _export_module(BakedBufferModule().eval()),
            root / "buffer_adapter.pt2", code_only=True),
        "plain_attr_adapter": _compile(
            _export_module(PlainAttrModule().eval()),
            root / "plain_attr.pt2", code_only=True),
    }


@pytest.fixture
def _gpu_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the runtime key-complete on a GPU-less box.

    Patches only the PROBES. Every value is a real fact of some runtime; nothing
    about how they are combined into a key is faked.
    """
    from gen_worker import compile_cache

    full = {
        "sku": "l4", "sm": "sm_89", "torch": str(torch.__version__),
        "triton": "3.6.0", "cuda": "13.0", "cuda_driver": "13000",
        "image_digest": "sha256:" + "ab" * 32,
    }
    monkeypatch.setattr(compile_cache, "runtime_key", lambda: dict(full))
    monkeypatch.setattr(aot_serve, "runtime_key", lambda: {
        "sku": full["sku"], "torch": full["torch"], "cuda": full["cuda"]})


# ---------------------------------------------------------------------------
# Reading the package's own declaration
# ---------------------------------------------------------------------------


def test_constants_in_so_reads_the_rendered_flag(packages: Dict[str, Path]) -> None:
    """The flag we passed, round-tripped through the artifact's own codegen."""
    assert aot_package.constants_in_so(packages["baked"]) is True
    assert aot_package.constants_in_so(packages["code_only"]) is False


def test_declared_constants_parses_the_full_table(
    packages: Dict[str, Path],
) -> None:
    constants = aot_package.declared_constants(packages["code_only"])
    by_fqn = {c.fqn: c for c in constants}
    assert "proj.weight" in by_fqn, sorted(by_fqn)
    assert "scale_table" in by_fqn, sorted(by_fqn)

    weight = by_fqn["proj.weight"]
    # The C++ identifier is mangled; the FQN is what a state_dict lookup needs.
    assert weight.name == "proj_weight"
    assert weight.shape == (WIDTH, WIDTH)
    assert weight.dtype == "float32"
    assert weight.kind == "Parameter"
    assert weight.data_size == WIDTH * WIDTH * 4
    assert weight.source == aot_serve.SOURCE_STATE_DICT
    assert weight.from_folded is False


def test_manifest_rows_carry_the_source_class(packages: Dict[str, Path]) -> None:
    """``aot_serve.constants_from_meta`` refuses a row whose ``source`` is not
    one of its two classes, so the producer must classify every entry."""
    rows = aot_package.constants_manifest(packages["code_only"])
    assert rows
    for row in rows:
        assert set(row) == {"fqn", "source", "dtype", "shape"}
        assert row["source"] in (
            aot_serve.SOURCE_STATE_DICT, aot_serve.SOURCE_LITERAL)
    # And the envelope accepts them as-is — the contract is verified, not assumed.
    specs = aot_serve.constants_from_meta({"constants": rows})
    assert {s.fqn for s in specs} == {row["fqn"] for row in rows}


def test_declaration_is_identical_whether_or_not_constants_are_baked(
    packages: Dict[str, Path],
) -> None:
    """Baking changes where the BYTES live, never the declaration — which is why
    the constant table is a usable seam between the two lanes."""
    def rows(p: Path) -> list:
        return [
            (c.fqn, c.data_size, c.kind, c.dtype, c.shape)
            for c in aot_package.declared_constants(p)
        ]

    assert rows(packages["code_only"]) == rows(packages["baked"])


def test_lifted_adapter_is_absent_from_the_constant_table(
    packages: Dict[str, Path],
) -> None:
    """Option 2's guarantee, measured on a real package."""
    fqns = set(aot_package.constant_names(packages["code_only"]))
    assert "lora_a" not in fqns
    assert "lora_b" not in fqns


# ---------------------------------------------------------------------------
# B1 — the code-only gate
# ---------------------------------------------------------------------------


def test_code_only_gate_accepts_a_code_only_package(
    packages: Dict[str, Path],
) -> None:
    assert aot_package.code_only_violations(packages["code_only"]) == []


def test_code_only_gate_refuses_a_baked_package_naming_the_tensors(
    packages: Dict[str, Path],
) -> None:
    """The gate standing between us and multi-GiB weights in every cell.

    Red without the gate: a baked package is a perfectly valid ``.pt2`` that
    loads and runs, so nothing else in the pipeline would object to it.
    """
    reasons = aot_package.code_only_violations(packages["baked"])
    assert reasons, "a weights-baked package must not pass the code-only gate"
    joined = " ".join(reasons)
    assert "load_constants_from_blob=true" in joined
    assert "package_constants_in_so=False" in joined
    assert "B1" in joined
    # Actionability: the offending tensors are NAMED, not merely counted.
    assert "proj.weight" in joined or "scale_table" in joined


def test_lrodata_proof_is_independent_of_the_rendered_flag(
    packages: Dict[str, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Proof 2 must catch a baked package on its own.

    Proof 1 reads codegen we do not own; if a torch upgrade moved the literal,
    the fleet would silently start shipping weights. So the ELF proof is checked
    with proof 1 forced to agree with the regression.
    """
    monkeypatch.setattr(aot_package, "constants_in_so", lambda _p: False)
    reasons = aot_package.code_only_violations(packages["baked"])
    assert reasons, ".lrodata proof must refuse a baked package unaided"
    assert ".lrodata" in " ".join(reasons)
    assert aot_package.code_only_violations(packages["code_only"]) == []


def test_lrodata_predicate_has_no_tunable_threshold(
    packages: Dict[str, Path],
) -> None:
    """The predicate is ".lrodata >= declared bytes", so the two packages must
    fall on opposite sides of the SAME declared-bytes figure."""
    declared = sum(
        c.data_size for c in aot_package.declared_constants(packages["code_only"]))

    def lrodata(p: Path) -> int:
        _name, blob = aot_package.packaged_so(p)
        return aot_package.elf_section_sizes(blob).get(".lrodata", 0)

    assert lrodata(packages["baked"]) >= declared > lrodata(packages["code_only"])


def test_elf_parser_refuses_non_elf_bytes() -> None:
    with pytest.raises(aot_package.PackageIntrospectionError, match="not an ELF"):
        aot_package.elf_section_sizes(b"not an elf at all" * 8)


def test_unbindable_constants_are_named(packages: Dict[str, Path]) -> None:
    """A cell that could only ever fail to arm must not be published — the fleet
    learns on the mint pod, not by every serving pod refusing it in turn."""
    reasons = aot_package.unbindable_constants(
        packages["code_only"], ("proj.weight", "proj.bias"))
    assert reasons
    assert "scale_table" in reasons[0]


def test_unbindable_constants_passes_a_complete_state_dict(
    packages: Dict[str, Path],
) -> None:
    keys = LiftedModule().state_dict().keys()
    assert aot_package.unbindable_constants(packages["code_only"], keys) == []


# ---------------------------------------------------------------------------
# pgw#728 — strict is doctrine, and the manifest must prove it describes
# the package it ships with
# ---------------------------------------------------------------------------


def test_program_and_package_constant_sets_agree_for_a_real_mint(
    packages: Dict[str, Path],
) -> None:
    """The matched pair: two independent derivations of the same fact."""
    assert aot_package.program_package_drift(
        _export_lifted(), packages["code_only"]) == []


def test_package_only_drift_is_refused_and_names_the_fqns(
    packages: Dict[str, Path],
) -> None:
    """The FATAL direction: the artifact wants a constant nothing declared.

    Nothing would bind it, and invoking a code-only package with an unbound
    constant segfaults inside AOTICompiledModel.__call__. A strict/non-strict
    trace mix (pgw#728) surfaces here too.
    """
    class _Empty:
        graph_signature = None
        constants: Dict[str, Any] = {}

    reasons = aot_package.program_package_drift(_Empty(), packages["code_only"])
    assert reasons
    joined = " ".join(reasons)
    assert "never lifted" in joined and "segfault" in joined
    assert "proj.weight" in joined or "scale_table" in joined


def test_program_only_drift_is_BENIGN(packages: Dict[str, Path]) -> None:
    """The routine direction: the compiler fused a lifted constant away.

    Measured on the first real SDXL w8a8 mint — program 2423 vs package 2422,
    the difference being ``unet.conv_out.bias`` folded into the conv epilogue.
    An equality-demanding gate refused EVERY real mint, so this test exists to
    stop anyone restoring the symmetry.
    """
    real = set(aot_package.program_constant_fqns(_export_lifted()))
    assert real

    class _Extra:
        graph_signature = None
        constants = {name: None for name in real | {"fused.away.bias"}}

    assert aot_package.program_package_drift(_Extra(), packages["code_only"]) == []
    assert "fused.away.bias" in aot_package.eliminated_constants(
        _Extra(), packages["code_only"])


def test_mint_refuses_a_package_wanting_undeclared_constants(
    tmp_path: Path, _gpu_runtime: None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wired into the mint, not merely available."""
    monkeypatch.setattr(aot_package, "program_constant_fqns", lambda _p: ())
    with pytest.raises(aot_mint.MintRefused, match="constant-set drift"):
        aot_mint.mint(
            LiftedModule().eval(), _spec(), tmp_path,
            example_inputs=_example_inputs)
    assert not list(tmp_path.glob("*.tar.gz"))


def test_mint_records_fused_constants_without_refusing(
    tmp_path: Path, _gpu_runtime: None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compiler fusion must be observable but never fatal."""
    real = aot_package.program_constant_fqns

    monkeypatch.setattr(
        aot_package, "program_constant_fqns",
        lambda p: tuple(real(p)) + ("definitely.fused.away",))
    result = aot_mint.mint(
        LiftedModule().eval(), _spec(), tmp_path,
        example_inputs=_example_inputs)
    meta = aot_serve.unpack_metadata(result.artifact)
    assert "definitely.fused.away" in meta["graph"]["fused_constants"]


def test_strict_mode_drift_is_refused_by_name() -> None:
    """Drift the env seal cannot see: both trace modes run identically sealed,
    with the identical toolchain, so nothing else in the key would notice."""
    assert aot_package.strict_mode_drift({"strict_export": True}, True) == []
    reasons = aot_package.strict_mode_drift({"strict_export": False}, True)
    assert reasons
    assert "strict=False" in reasons[0] and "strict=True" in reasons[0]
    assert "pgw#728" in reasons[0]


def test_an_artifact_silent_on_its_trace_mode_is_refused() -> None:
    """Silence is not "probably strict" — the constant set it declares cannot be
    trusted without knowing which trace produced it."""
    reasons = aot_package.strict_mode_drift({}, True)
    assert reasons
    assert "unprovable" in reasons[0]


def test_minted_metadata_pins_the_trace_mode(
    tmp_path: Path, _gpu_runtime: None,
) -> None:
    """Recorded on the artifact so a consumer can refuse a mode mismatch rather
    than discover it as a constant-set surprise."""
    result = aot_mint.mint(
        LiftedModule().eval(), _spec(), tmp_path,
        example_inputs=_example_inputs)
    meta = aot_serve.unpack_metadata(result.artifact)
    assert meta["strict_export"] is True
    assert aot_package.strict_mode_drift(meta, True) == []
    assert aot_package.strict_mode_drift(meta, False)


# ---------------------------------------------------------------------------
# The no-baked-adapter gate belongs on the ExportedProgram
# ---------------------------------------------------------------------------


def test_package_side_adapter_scan_is_a_false_pass(
    packages: Dict[str, Path],
) -> None:
    """pgw#725's measured reason the gate cannot live on the package.

    A plain-``__dict__`` adapter IS baked, but packing renames it to
    ``_tensor_constant0`` — so a package-side name scan reports "no lora
    constant" and would wave the artifact through. This test is the evidence
    for that design constraint, and it fails if anyone re-adds a package-side
    gate.
    """
    names = aot_package.constant_names(packages["plain_attr_adapter"])
    assert not [n for n in names if "lora" in n], names
    # ...while the ep-level gate catches the very same module.
    with pytest.raises(ValidationError):
        lora_lifted.assert_no_baked_adapter(_export_module(PlainAttrModule().eval()))


def test_ep_gate_refuses_a_registered_buffer_adapter() -> None:
    with pytest.raises(ValidationError, match="BAKED"):
        lora_lifted.assert_no_baked_adapter(
            _export_module(BakedBufferModule().eval()))


def test_ep_gate_accepts_a_lifted_adapter() -> None:
    lora_lifted.assert_no_baked_adapter(_export_lifted())


def test_mint_refuses_a_baked_adapter_before_compiling(
    tmp_path: Path, _gpu_runtime: None,
) -> None:
    """The pack path must call the ep-level gate, and must do it BEFORE the AOTI
    compile — the point is not to spend the minutes, not merely to discard the
    result."""
    spec = aot_mint.ExportSpec(
        family="tiny", target="forward", weight_lane="w8a8",
        shapes=((1024, 1024),), lora_bucket=LORA_RANK,
        dynamic=(aot_mint.DynamicDim("x", 0, 2, 32),),
    )
    with pytest.raises(aot_mint.MintRefused, match="#725"):
        aot_mint.mint(
            BakedBufferModule().eval(), spec, tmp_path,
            example_inputs=lambda: ((torch.randn(4, WIDTH),), {}))
    assert not list(tmp_path.glob("*.tar.gz"))


def test_mint_refuses_a_plain_attribute_adapter(
    tmp_path: Path, _gpu_runtime: None,
) -> None:
    """The case a package-side gate would wave through."""
    spec = aot_mint.ExportSpec(
        family="tiny", target="forward", weight_lane="w8a8",
        shapes=((1024, 1024),), lora_bucket=LORA_RANK,
        dynamic=(aot_mint.DynamicDim("x", 0, 2, 32),),
    )
    with pytest.raises(aot_mint.MintRefused, match="#725"):
        aot_mint.mint(
            PlainAttrModule().eval(), spec, tmp_path,
            example_inputs=lambda: ((torch.randn(4, WIDTH),), {}))


def test_lifted_input_gate_names_an_unlifted_declaration() -> None:
    """Caught on the PROGRAM, and it covers declared lifted inputs generally —
    not only the two the LoRA gate knows by name."""
    spec = aot_mint.ExportSpec(
        family="tiny", target="forward", weight_lane="w8a8",
        lifted_inputs=("some_other_tensor",))
    gaps = aot_mint.lifted_input_gaps(_export_lifted(), spec)
    assert gaps
    assert "some_other_tensor" in " ".join(gaps)


def test_lifted_input_gate_passes_when_declared_inputs_are_lifted() -> None:
    spec = aot_mint.ExportSpec(
        family="tiny", target="forward", weight_lane="w8a8",
        lifted_inputs=("lora_a", "lora_b"))
    assert aot_mint.lifted_input_gaps(_export_lifted(), spec) == []


# ---------------------------------------------------------------------------
# B2 — the declared range, recorded so it CAN be asserted
# ---------------------------------------------------------------------------


def test_input_contract_binds_each_symbol_to_an_input_dim() -> None:
    """Without this the consumer has nothing to assert, which is exactly the B2
    hole pgw#704 measured (2048x2048 accepted by a max=160 artifact)."""
    program = _export_lifted(lo=4, hi=24)
    inputs, symbols = aot_package.input_contract(
        program, ("x", "lora_a", "lora_b"))
    row = inputs[0]
    assert row["name"] == "x"
    assert row["dtype"] == "float32"
    symbol = row["shape"][0]
    assert isinstance(symbol, str), row
    assert symbols[symbol] == [4, 24]
    # A static dim stays an int, so the consumer checks it exactly.
    assert row["shape"][1] == WIDTH


def test_input_contract_is_accepted_by_the_envelope_parser() -> None:
    """The producer and the consumer must agree on the contract's shape, so the
    rows are round-tripped through the parser that will actually read them."""
    inputs, symbols = aot_package.input_contract(
        _export_lifted(lo=4, hi=24), ("x", "lora_a", "lora_b"))
    contract = aot_serve.contract_from_meta(
        {"inputs": inputs, "symbols": symbols})
    assert [s.name for s in contract.inputs] == ["x", "lora_a", "lora_b"]
    assert contract.symbols


def test_input_contract_refuses_an_unbounded_symbol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An open bound admits any traffic — B2 with extra steps. Recording a fake
    bound would be worse than refusing."""
    program = _export_lifted()

    class _Open:
        lower = 2
        upper = float("inf")

    monkeypatch.setattr(
        type(program), "range_constraints",
        property(lambda _self: {"s0": _Open()}), raising=False)
    with pytest.raises(
        aot_package.PackageIntrospectionError, match="unbounded range",
    ):
        aot_package.input_contract(program, ("x", "lora_a", "lora_b"))


def test_declared_range_gate_catches_a_specialized_dim() -> None:
    """An export that specialized a dim we advertise as dynamic must fail the
    mint: the artifact would serve one shape while its metadata claims a range,
    and B2 means nothing at runtime would notice."""
    args, kwargs = _example_inputs()
    static = torch.export.export(LiftedModule().eval(), args, kwargs, strict=True)
    gaps = aot_mint.declared_range_gaps(
        static, (aot_mint.DynamicDim("x", 0, 2, 32),))
    assert gaps
    assert "specialized a dim" in " ".join(gaps)


def test_declared_range_gate_passes_a_matching_export() -> None:
    assert aot_mint.declared_range_gaps(
        _export_lifted(lo=8, hi=16),
        (aot_mint.DynamicDim("x", 0, 8, 16),)) == []


def test_declared_range_gate_compares_bounds_not_symbol_names() -> None:
    """Symbol names are trace-order artifacts; the bound pairs are the contract."""
    a = _export_lifted(lo=8, hi=16)
    b = _export_lifted(lo=8, hi=16)
    names_a = set(getattr(a, "range_constraints", {}))
    names_b = set(getattr(b, "range_constraints", {}))
    dims = (aot_mint.DynamicDim("x", 0, 8, 16),)
    assert aot_mint.declared_range_gaps(a, dims) == []
    assert aot_mint.declared_range_gaps(b, dims) == []
    assert names_a and names_b


# ---------------------------------------------------------------------------
# The declared dynamic-shape spec
# ---------------------------------------------------------------------------


def test_dynamic_shapes_spec_refuses_bounds_off_the_multiple() -> None:
    """A range not expressible as the required multiple must be refused, not
    silently 0/1-specialized — otherwise "one artifact serves every aspect
    ratio" quietly stops holding."""
    with pytest.raises(aot_mint.MintRefused, match="not multiples of 8"):
        aot_mint.dynamic_shapes_spec(
            (aot_mint.DynamicDim("sample", 2, 65, 160, multiple_of=8),),
            ("sample",))


def test_dynamic_shapes_spec_refuses_an_unknown_input() -> None:
    with pytest.raises(aot_mint.MintRefused, match="does not take"):
        aot_mint.dynamic_shapes_spec(
            (aot_mint.DynamicDim("nope", 0, 2, 4),), ("sample",))


def test_dynamic_shapes_spec_refuses_an_empty_range() -> None:
    with pytest.raises(aot_mint.MintRefused, match="empty range"):
        aot_mint.dynamic_shapes_spec(
            (aot_mint.DynamicDim("sample", 0, 32, 8),), ("sample",))


def test_dynamic_shapes_spec_leaves_undeclared_inputs_static() -> None:
    spec = aot_mint.dynamic_shapes_spec(
        (aot_mint.DynamicDim("x", 0, 2, 32),), ("x", "lora_a"))
    assert spec["lora_a"] is None
    assert 0 in spec["x"]


def test_multiple_of_dim_actually_exports_with_the_declared_range() -> None:
    """The multiple-of factor must produce a REAL exportable guard, not just a
    tidy declaration — so it is exercised through a genuine export."""
    class Down(torch.nn.Module):
        def forward(self, sample: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.avg_pool2d(sample, 8)

    spec = aot_mint.dynamic_shapes_spec(
        (aot_mint.DynamicDim("sample", 2, 64, 160, multiple_of=8),
         aot_mint.DynamicDim("sample", 3, 64, 160, multiple_of=8)),
        ("sample",))
    program = torch.export.export(
        Down().eval(), (torch.randn(2, 4, 128, 128),),
        dynamic_shapes=spec, strict=True)
    assert aot_mint.declared_range_gaps(
        program, (aot_mint.DynamicDim("sample", 2, 64, 160, multiple_of=8),)) == []


# ---------------------------------------------------------------------------
# Lane order — w8a8 first
# ---------------------------------------------------------------------------


def test_w8a8_lane_is_admitted() -> None:
    spec = aot_mint.ExportSpec(family="sdxl", target="unet", weight_lane="w8a8")
    assert aot_mint.lane_admitted(spec, allow_regressed_lanes=False) == ""


def test_plain_lane_is_held_by_default() -> None:
    """#730 holds the lanes measured 6.9-7.0% slower under AOTI."""
    spec = aot_mint.ExportSpec(family="sdxl", target="unet", weight_lane="")
    reason = aot_mint.lane_admitted(spec, allow_regressed_lanes=False)
    assert reason
    assert "#730" in reason
    assert aot_mint.lane_admitted(spec, allow_regressed_lanes=True) == ""


def test_mint_holds_a_regressed_lane_before_exporting(
    tmp_path: Path, _gpu_runtime: None,
) -> None:
    spec = aot_mint.ExportSpec(family="tiny", target="forward", weight_lane="")
    with pytest.raises(aot_mint.MintRefused, match="#730"):
        aot_mint.mint(
            LiftedModule().eval(), spec, tmp_path,
            example_inputs=_example_inputs)
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# The mint, end to end
# ---------------------------------------------------------------------------


def _spec(**over: Any) -> aot_mint.ExportSpec:
    kwargs: Dict[str, Any] = dict(
        family="tiny", target="forward", weight_lane="w8a8",
        shapes=((1024, 1024),),
        dynamic=(aot_mint.DynamicDim("x", 0, 2, 32),),
        specialization={"gemm_mode": "pertensor"},
        lifted_inputs=("lora_a", "lora_b"),
    )
    kwargs.update(over)
    return aot_mint.ExportSpec(**kwargs)


def test_mint_produces_a_packed_keyed_gated_cell(
    tmp_path: Path, _gpu_runtime: None,
) -> None:
    """The whole produce half on a real module: export -> ep gates -> code-only
    AOTI -> B1 gate -> declared contract -> keyed pack."""
    result = aot_mint.mint(
        LiftedModule().eval(), _spec(), tmp_path,
        example_inputs=_example_inputs)

    assert result.artifact.exists()
    assert result.artifact.name == f"{result.cell_key}.tar.gz"
    assert cell_key.is_key(result.cell_key)
    assert result.timings["aoti_compile_s"] > 0

    meta = aot_serve.unpack_metadata(result.artifact)
    assert meta["kind"] == aot_serve.ARTIFACT_KIND
    assert meta["package_constants_in_so"] is False
    assert meta["strict_export"] is True
    assert meta["weight_lane"] == "w8a8"
    assert meta["cell_key"] == result.cell_key

    # The envelope's own parsers accept what the producer wrote — the two lanes
    # agree about the bytes, verified rather than assumed.
    assert aot_serve.verify(meta, family="tiny") == ""
    contract = aot_serve.contract_from_meta(meta)
    assert [s.name for s in contract.inputs] == ["x", "lora_a", "lora_b"]
    assert list(contract.symbols.values()) == [(2, 32)]
    specs = aot_serve.constants_from_meta(meta)
    fqns = {s.fqn for s in specs}
    assert "proj.weight" in fqns
    assert "lora_a" not in fqns, "the adapter must be a call input, not a constant"

    # The identity blocks the key is recomputed FROM ride the metadata.
    assert meta["graph"]["lifted_inputs"] == ["lora_a", "lora_b"]
    assert meta["graph"]["specialization"]["gemm_mode"] == "pertensor"
    assert meta["toolchain"] and meta["code_closure"] and meta["sm"] == "sm_89"
    assert aot_mint.cell_identity(meta, _spec()).digest == result.cell_key


def test_minted_artifact_stages_on_the_consumer_side(
    tmp_path: Path, _gpu_runtime: None,
) -> None:
    """The produce half's real acceptance test: the consumer's own staging path
    accepts the artifact this lane packed."""
    result = aot_mint.mint(
        LiftedModule().eval(), _spec(), tmp_path,
        example_inputs=_example_inputs)
    staged = aot_serve.stage_artifact(
        result.artifact, "tiny", cache_dir=tmp_path / "cache")
    try:
        assert staged.metadata["cell_key"] == result.cell_key
        assert (staged.root / aot_serve.PACKAGE_NAME).exists()
        assert aot_package.constants_in_so(
            staged.root / aot_serve.PACKAGE_NAME) is False
    finally:
        staged.close()


def test_mint_cannot_be_talked_out_of_code_only(
    tmp_path: Path, _gpu_runtime: None,
) -> None:
    """B1 is a fleet correctness requirement, not a default a caller may
    override — a caller passing the flag True is ignored."""
    result = aot_mint.mint(
        LiftedModule().eval(), _spec(), tmp_path,
        example_inputs=_example_inputs,
        inductor_configs={"aot_inductor.package_constants_in_so": True},
    )
    dest = tmp_path / "dest"
    aot_serve.unpack(result.artifact, dest)
    assert aot_package.constants_in_so(dest / aot_serve.PACKAGE_NAME) is False


def test_mint_is_byte_deterministic(tmp_path: Path, _gpu_runtime: None) -> None:
    """#699 double-mint on the ``.pt2`` payload: the same recipe must produce the
    same key, or a byte-comparison between two mint pods can never confirm a
    cell. The artifact carries no timestamps and no durations for this reason."""
    a = aot_mint.mint(
        LiftedModule().eval(), _spec(), tmp_path / "a",
        example_inputs=_example_inputs)
    b = aot_mint.mint(
        LiftedModule().eval(), _spec(), tmp_path / "b",
        example_inputs=_example_inputs)
    assert a.cell_key == b.cell_key

    meta_a = aot_serve.unpack_metadata(a.artifact)
    meta_b = aot_serve.unpack_metadata(b.artifact)
    # Weights differ between the two module instances, so the PACKAGE bytes may
    # differ; the recorded contract and identity must not.
    for field_name in (
        "inputs", "symbols", "range_digest", "graph", "toolchain",
        "code_closure", "cell_key", "package_constants_in_so",
    ):
        assert meta_a[field_name] == meta_b[field_name], field_name


def test_mint_refuses_a_dim_the_graph_cannot_make_dynamic(
    tmp_path: Path, _gpu_runtime: None,
) -> None:
    """Declaring a dim the graph specializes must fail the mint, NAMING the dim.

    Here export itself raises the constraint violation, which is the good case —
    ``declared_range_gaps`` covers the other one, where export specializes
    silently and only the recorded range would disagree with reality.
    """
    with pytest.raises(aot_mint.MintRefused, match="lora_a"):
        aot_mint.mint(
            LiftedModule().eval(),
            _spec(dynamic=(aot_mint.DynamicDim("lora_a", 0, 2, 32),)),
            tmp_path, example_inputs=_example_inputs)
    assert not list(tmp_path.glob("*.tar.gz"))


def test_mint_refuses_an_unbindable_constant(
    tmp_path: Path, _gpu_runtime: None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cell whose declared constants cannot bind from the resident module is
    refused on the mint pod, not by every serving pod in turn."""
    monkeypatch.setattr(
        aot_mint, "_state_dict_keys", lambda _m: ("proj.weight", "proj.bias"))
    with pytest.raises(aot_mint.MintRefused, match="bindability gate"):
        aot_mint.mint(
            LiftedModule().eval(), _spec(), tmp_path,
            example_inputs=_example_inputs)


# ---------------------------------------------------------------------------
# Identity — the ck6 owed item
# ---------------------------------------------------------------------------


def _meta_for(program: Any, package: Path, spec: aot_mint.ExportSpec) -> Dict[str, Any]:
    inputs, symbols = aot_package.input_contract(
        program, ("x", "lora_a", "lora_b"))
    meta = aot_serve.artifact_metadata(
        family="tiny", module="forward", precision="bf16", cell_key="",
        inputs=inputs, symbols=symbols,
        constants=aot_package.constants_manifest(package),
    )
    meta.update(aot_mint.identity_blocks(program, package, spec))
    meta["cell_key"] = aot_mint.cell_identity(meta, spec).digest
    return meta


def test_cell_key_differs_when_ONLY_the_declared_range_differs(
    _gpu_runtime: None, packages: Dict[str, Path],
) -> None:
    """pgw#704's measured soundness bug, owed to #716/#717.

    Three exports differing only in declared range produced the IDENTICAL
    node-only digest, so a node-only graph hash collides artifacts admitting
    different traffic — and B2 means the runtime would not refuse the
    difference. Folding the range into the contract axis is the fix; this is the
    test that goes red if it is ever dropped.
    """
    spec = _spec()
    narrow = _meta_for(_export_lifted(lo=8, hi=16), packages["code_only"], spec)
    wide = _meta_for(_export_lifted(lo=8, hi=32), packages["code_only"], spec)
    assert narrow["range_digest"] != wide["range_digest"]
    assert narrow["cell_key"] != wide["cell_key"], (
        "an artifact admitting 8..16 must not share a key with one admitting "
        "8..32")


def test_cell_key_is_stable_for_an_identical_declaration(
    _gpu_runtime: None, packages: Dict[str, Path],
) -> None:
    spec = _spec()
    a = _meta_for(_export_lifted(lo=8, hi=16), packages["code_only"], spec)
    b = _meta_for(_export_lifted(lo=8, hi=16), packages["code_only"], spec)
    assert a["cell_key"] == b["cell_key"]


def test_cell_key_differs_by_lane(
    _gpu_runtime: None, packages: Dict[str, Path],
) -> None:
    program = _export_lifted()
    plain = _meta_for(program, packages["code_only"], _spec(weight_lane=""))
    w8a8 = _meta_for(program, packages["code_only"], _spec(weight_lane="w8a8"))
    assert plain["cell_key"] != w8a8["cell_key"]


def test_cell_key_differs_by_frozen_specialization(
    _gpu_runtime: None, packages: Dict[str, Path],
) -> None:
    """Export bakes a python branch into the graph, so two artifacts differing
    only in a frozen branch are different artifacts."""
    program = _export_lifted()
    a = _meta_for(program, packages["code_only"],
                  _spec(specialization={"gemm_mode": "pertensor"}))
    b = _meta_for(program, packages["code_only"],
                  _spec(specialization={"gemm_mode": "rowwise"}))
    assert a["cell_key"] != b["cell_key"]


def test_cell_key_differs_between_strict_and_non_strict(
    _gpu_runtime: None, packages: Dict[str, Path],
) -> None:
    """A non-strict trace is a different guarantee and must not share identity
    with a strict one."""
    program = _export_lifted()
    a = _meta_for(program, packages["code_only"], _spec(strict=True))
    b = _meta_for(program, packages["code_only"], _spec(strict=False))
    assert a["cell_key"] != b["cell_key"]


def test_cell_identity_refuses_an_artifact_with_no_range_digest(
    _gpu_runtime: None, packages: Dict[str, Path],
) -> None:
    """An exported cell must not be keyable without its admissible range."""
    spec = _spec()
    meta = _meta_for(_export_lifted(), packages["code_only"], spec)
    meta["range_digest"] = ""
    with pytest.raises(aot_mint.MintRefused, match="range"):
        aot_mint.cell_identity(meta, spec)


def test_cell_identity_refuses_an_artifact_with_no_sm(
    _gpu_runtime: None, packages: Dict[str, Path],
) -> None:
    spec = _spec()
    meta = _meta_for(_export_lifted(), packages["code_only"], spec)
    meta["sm"] = ""
    with pytest.raises(aot_mint.MintRefused, match="compute capability"):
        aot_mint.cell_identity(meta, spec)


def test_key_scheme_is_unchanged_by_the_new_kind(
    _gpu_runtime: None, packages: Dict[str, Path],
) -> None:
    """A new artifact KIND rides the existing axis set — ``kind`` is the
    discriminator, so no KEY_SCHEME bump strands the dynamo cells."""
    spec = _spec()
    meta = _meta_for(_export_lifted(), packages["code_only"], spec)
    key = aot_mint.cell_identity(meta, spec)
    assert key.axes_dict()["kind"] == aot_serve.ARTIFACT_KIND
    assert key.digest.startswith(cell_key.KEY_SCHEME + "-")
    assert cell_key.is_key(key.digest)


# ---------------------------------------------------------------------------
# Publish + CLI
# ---------------------------------------------------------------------------


class _RecordingPublisher:
    def __init__(self) -> None:
        self.calls: list = []

    def publish(self, family: str, artifact: Path, meta: dict) -> str:
        self.calls.append((family, artifact, meta))
        return "checkpoint-1"


def test_publish_passes_the_keyed_metadata_through(
    tmp_path: Path, _gpu_runtime: None, packages: Dict[str, Path],
) -> None:
    """Receipts are HUB-added at publish-finalize (#709), so the producer's whole
    obligation is a keyed ``metadata.json`` in the tar."""
    meta = _meta_for(_export_lifted(), packages["code_only"], _spec())
    result = aot_mint.MintResult(
        artifact=tmp_path / "cell.tar.gz", metadata=meta, timings={})
    publisher = _RecordingPublisher()
    assert aot_mint.publish(result, publisher) == "checkpoint-1"
    family, _artifact, sent = publisher.calls[0]
    assert family == "tiny"
    assert sent["cell_key"] == meta["cell_key"]


def test_publish_refuses_an_unkeyed_artifact(tmp_path: Path) -> None:
    """An unaddressable cell would be stored under a flavor nothing requests."""
    result = aot_mint.MintResult(
        artifact=tmp_path / "cell.tar.gz",
        metadata={"family": "tiny"}, timings={})
    with pytest.raises(aot_mint.MintRefused, match="cell_key"):
        aot_mint.publish(result, _RecordingPublisher())


def test_cli_reports_a_bad_request(tmp_path: Path, capsys: Any) -> None:
    request = tmp_path / "req.json"
    request.write_text(json.dumps({"family": "sdxl"}))  # no target
    assert aot_mint.main([str(request), "--out", str(tmp_path)]) == 3
    assert "BAD REQUEST" in capsys.readouterr().err


def test_cli_reports_a_missing_request(tmp_path: Path) -> None:
    assert aot_mint.main(
        [str(tmp_path / "nope.json"), "--out", str(tmp_path)]) == 3


def test_cli_loads_a_full_request(tmp_path: Path) -> None:
    request = tmp_path / "req.json"
    request.write_text(json.dumps({
        "family": "sdxl", "target": "unet", "weight_lane": "w8a8",
        "shapes": [[1024, 1024], [1152, 896]], "text_lens": [77],
        "guidance_scales": [6.0], "lora_bucket": 64,
        "dynamic": [{"input": "sample", "axis": 2, "min": 64, "max": 160,
                     "multiple_of": 8}],
        "lifted_inputs": ["lora_a", "lora_b"],
        "specialization": {"gemm_mode": "pertensor"},
    }))
    spec, body = aot_mint._load_spec(request)
    assert spec.family == "sdxl"
    assert spec.lane_label() == "w8a8-lora64"
    assert spec.shapes == ((1024, 1024), (1152, 896))
    assert spec.dynamic[0].multiple_of == 8
    assert spec.strict is True
    assert body["specialization"] == {"gemm_mode": "pertensor"}


# ---------------------------------------------------------------------------
# The SDXL input contract
# ---------------------------------------------------------------------------


def test_sdxl_latent_dims_span_the_declared_aspect_set() -> None:
    """Derived from the declared aspects so the admissible range cannot drift
    from "these are the ratios we serve"."""
    from gen_worker import aot_inputs

    spec = aot_mint.ExportSpec(
        family="sdxl", target="unet",
        shapes=((1024, 1024), (1152, 896), (832, 1216)))
    dims = aot_inputs.latent_dims(spec)
    assert [d.axis for d in dims] == [2, 3]
    for d in dims:
        assert d.multiple_of == 8
        assert d.min <= 104 and d.max >= 152
        assert d.min % 8 == 0 and d.max % 8 == 0


def test_sdxl_latent_dims_round_bounds_OUT_not_in() -> None:
    """Rounding in would exclude a declared aspect from the artifact's own
    contract, and B2 means nothing at runtime would tell us."""
    from gen_worker import aot_inputs

    spec = aot_mint.ExportSpec(
        family="sdxl", target="unet", shapes=((1000, 1000),))
    dim = aot_inputs.latent_dims(spec)[0]
    assert dim.max >= 125  # 1000/8 = 125 -> rounded out to 128
    assert dim.max % 8 == 0


def test_sdxl_lora_bucket_zero_takes_no_adapter_arguments() -> None:
    """The branchless pipeline is its own graph class and carries no adapter
    arguments at all."""
    from gen_worker import aot_inputs

    spec = aot_mint.ExportSpec(family="sdxl", target="unet", lora_bucket=0)
    assert aot_inputs.lifted_lora_kwargs(LiftedModule(), spec) == {}


def test_unknown_family_has_no_input_contract() -> None:
    from gen_worker import aot_inputs

    with pytest.raises(aot_inputs.InputContractError, match="no AOT export"):
        aot_inputs.builder_for("madeup")
