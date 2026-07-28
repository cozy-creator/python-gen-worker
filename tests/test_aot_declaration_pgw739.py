"""pgw#739 — derivation over real declarations (the engine half).

Red-verified against the accumulated worked examples:

- a wan-a14b-shaped declaration derives EXACTLY the measured ie#566 §4
  dynamic contract (latent H/W [90, 160] multiple-of-2, on the denoiser AND
  the decoder's ``z``) from class rows alone — no endpoint hand-math;
- the RELATIONAL axis (wan ti2v's per-token timestep) rides the measured
  solver-unification path: a free ranged dim with SDK-derived bounds, a real
  ``torch.export`` run, and torch unifying the relation into the shape env.
  The composite-range check in ``declared_range_gaps`` is proven
  load-bearing: every per-symbol range fails to cover the declared bounds,
  so the pre-#739 per-symbol comparison would have refused a sound artifact;
- two families with DIFFERENT declarations derive correct export inputs
  through ONE code path (``declared_inputs``);
- the fork gate reads ``(pipeline|module, field)`` sources off composed
  objects and refuses a wrong or unstated arm by name;
- fork coordinates and class rows reach the cell identity.

Real torch throughout; the only fabricated things are tiny modules, per the
pgw#723 test doctrine.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from gen_worker import aot_declaration, aot_mint  # noqa: E402
from gen_worker.aot_mint import ExportSpec, MintRefused  # noqa: E402
from gen_worker.api.decorators import Compile, DynamicDim  # noqa: E402
from gen_worker.api.export_contract import (  # noqa: E402
    Arg,
    Dim,
    Fork,
    GraphClass,
    Input,
    register_export_declaration,
    reset_export_declarations,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    reset_export_declarations()
    yield
    reset_export_declarations()


# ---------------------------------------------------------------------------
# wan a14b — dynamic-collapse derivation from rows (ie#566 §4, measured)
# ---------------------------------------------------------------------------


def _wan_a14b_decl() -> Compile:
    """The wan t2v-a14b declaration shape: two transposed rows, latent H/W
    each spanning [90, 160], patch_size (1,2,2) => multiple_of=2."""
    dims = (
        Dim("B", carried_by=(("hidden_states", 0), ("z", 0))),
        Dim("F_lat", carried_by=(("hidden_states", 2), ("z", 2))),
        Dim("H_lat", carried_by=(("hidden_states", 3), ("z", 3)), multiple_of=2),
        Dim("W_lat", carried_by=(("hidden_states", 4), ("z", 4)), multiple_of=2),
        Dim("T_txt", carried_by=(("encoder_hidden_states", 1),)),
    )
    fork = {"expand_timesteps": False, "use_tiling": False}
    return Compile(
        family="wan-2.2-t2v-a14b", targets=("transformer", "transformer_2", "vae.decode"),
        text_len=512,
        dims=dims,
        forks=(
            Fork("expand_timesteps", served=(False,), unserved=(True,),
                 source=("pipeline", "expand_timesteps"),
                 why="ti2v-5b's per-token timestep is a DIFFERENT graph class"),
            Fork("use_tiling", served=(False,), unserved=(True,),
                 source=("module", "use_tiling"), targets=("vae.decode",),
                 why="tiled decode forks AutoencoderKLWan._decode"),
        ),
        classes=(
            GraphClass(dims={"B": 1, "F_lat": 21, "H_lat": 90, "W_lat": 160,
                             "T_txt": 512}, fork=fork),
            GraphClass(dims={"B": 1, "F_lat": 21, "H_lat": 160, "W_lat": 90,
                             "T_txt": 512}, fork=fork),
        ),
        inputs=(
            Input("hidden_states",
                  shape=("B", ("config", "in_channels"), "F_lat", "H_lat", "W_lat"),
                  targets=("transformer", "transformer_2")),
            Input("timestep", shape=("B",), dtype="int64",
                  targets=("transformer", "transformer_2")),
            Input("encoder_hidden_states",
                  shape=("B", "T_txt", ("config", "text_dim")),
                  targets=("transformer", "transformer_2")),
            Input("z", shape=("B", ("config", "z_dim"), "F_lat", "H_lat", "W_lat"),
                  targets=("vae.decode",)),
        ),
        args=(Arg("return_dict", False),),
        shape_strategy="dynamic-collapse",
        warm_changes_key=False,
    )


def test_wan_a14b_bounds_derive_from_rows_exactly() -> None:
    decl = _wan_a14b_decl()
    plans = aot_declaration.mint_plans(decl, "transformer")
    assert len(plans) == 1  # one fork coordinate -> ONE collapsed artifact
    rows = {(d.input_name, d.axis): d for d in plans[0].dynamic}
    # The measured §4 contract, derived — never hand-written:
    for axis in (3, 4):
        d = rows[("hidden_states", axis)]
        assert (d.min, d.max, d.multiple_of) == (90, 160, 2)
    # B / F_lat / T_txt are constant across rows -> static, no row emitted.
    assert set(rows) == {("hidden_states", 3), ("hidden_states", 4)}


def test_wan_a14b_decoder_gets_the_same_axes_on_z() -> None:
    decl = _wan_a14b_decl()
    plan = aot_declaration.mint_plans(decl, "vae.decode")[0]
    rows = {(d.input_name, d.axis) for d in plan.dynamic}
    assert rows == {("z", 3), ("z", 4)}  # target-filtered via Input.targets


def test_static_rows_strategy_mints_one_plan_per_row() -> None:
    decl = _wan_a14b_decl()
    static = Compile(**{**_fields(decl), "shape_strategy": "static-rows"})
    plans = aot_declaration.mint_plans(static, "transformer")
    assert len(plans) == 2 and all(p.dynamic == () for p in plans)


def _fields(c: Compile) -> dict:
    import msgspec

    return {f.name: getattr(c, f.name) for f in msgspec.structs.fields(c)}


def test_collapse_refuses_a_dim_reaching_one() -> None:
    decl = Compile(
        family="zero-one", targets=("t",), text_len=0,
        dims=(Dim("B", carried_by=(("x", 0),)),),
        classes=(GraphClass(dims={"B": 1}), GraphClass(dims={"B": 2})),
        shape_strategy="dynamic-collapse", warm_changes_key=False)
    with pytest.raises(MintRefused, match="FORK the class"):
        aot_declaration.mint_plans(decl, "t")


def test_select_plan_refuses_unknown_coordinates_by_name() -> None:
    decl = _wan_a14b_decl()
    with pytest.raises(MintRefused, match="expand_timesteps.*True"):
        aot_declaration.select_plan(
            decl, "transformer",
            fork={"expand_timesteps": True, "use_tiling": False})


# ---------------------------------------------------------------------------
# The relational axis — SDK-derived bounds + solver unification (real export)
# ---------------------------------------------------------------------------


class RelatedTokens(torch.nn.Module):
    """The wan ti2v shape of fact: input ``t``'s axis 1 IS ``H*W`` — a
    product of two roots, inexpressible in the Dim API (measured verbatim:
    "Only increasing linear operations with integer coefficients are
    supported"), carried instead by solver unification."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(3, 3)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        flat = x.reshape(b, c, h * w) + t.unsqueeze(1)
        return self.proj(flat.transpose(1, 2))


def _relational_decl() -> Compile:
    fork: dict = {}
    return Compile(
        family="rel", targets=("t",), text_len=0,
        dims=(
            Dim("B", carried_by=(("x", 0),)),
            Dim("H", carried_by=(("x", 2),)),
            Dim("W", carried_by=(("x", 3),)),
            Dim("N", carried_by=(("t", 1),), relates_to=("H", "W")),
        ),
        classes=(
            GraphClass(dims={"B": 2, "H": 4, "W": 4, "N": 16}, fork=fork),
            GraphClass(dims={"B": 2, "H": 8, "W": 8, "N": 64}, fork=fork),
        ),
        inputs=(
            Input("x", shape=("B", 3, "H", "W")),
            Input("t", shape=("B", "N")),
        ),
        shape_strategy="dynamic-collapse",
        warm_changes_key=False,
    )


def test_relational_bounds_are_derived_not_hand_written() -> None:
    plan = aot_declaration.mint_plans(_relational_decl(), "t")[0]
    rows = {(d.input_name, d.axis): (d.min, d.max) for d in plan.dynamic}
    assert rows[("t", 1)] == (16, 64)      # hull over the rows: 4*4 .. 8*8
    assert rows[("x", 2)] == (4, 8) and rows[("x", 3)] == (4, 8)
    assert plan.seed.dim_map == {"B": 2, "H": 8, "W": 8, "N": 64}


def test_relational_axis_unifies_and_gates_on_solved_ranges() -> None:
    """The measured ie#566 §5 mechanism on a real export: the free derived
    range unifies into the shape env, and ``declared_range_gaps`` reads the
    SOLVED composite range — while every per-symbol range fails to cover the
    declared bounds, which is exactly why the pre-#739 per-symbol comparison
    refused a sound artifact."""
    decl = _relational_decl()
    plan = aot_declaration.mint_plans(decl, "t")[0]
    module = RelatedTokens().eval()
    args = (torch.randn(2, 3, 8, 8), torch.randn(2, 64))
    shapes = aot_mint.dynamic_shapes_spec(plan.dynamic, ("x", "t"))
    program = aot_mint.export_program(module, args, {}, dynamic_shapes=shapes,
                                      strict=True)

    gaps = aot_mint.declared_range_gaps(program, plan.dynamic)
    assert gaps == [], gaps

    # The red half: the relational axis's placeholder carries ONLY the
    # governing symbols (no free N symbol survives unification), and no
    # single symbol's range covers [16, 64].
    t_dim = aot_mint._placeholder_shapes(program)["t"][1]
    syms = aot_mint._free_symbols(t_dim)
    assert len(syms) >= 2, "N must be unified away into the governing symbols"
    ranges = getattr(program, "range_constraints", {})
    for sym in syms:
        interval = ranges.get(sym)
        assert interval is not None
        assert int(interval.upper) < 64, (
            "per-symbol ranges must NOT cover the declared bounds — the "
            "composite solved range is the load-bearing check")


def test_pinned_relational_axis_is_still_refused() -> None:
    """Leaving the related input STATIC while declaring H/W dynamic is the
    ie#566 G3 false pass; the guard-evidence gate refuses it by name."""
    module = RelatedTokens().eval()
    dims = (
        aot_mint.DynamicDim("x", 2, 4, 8),
        aot_mint.DynamicDim("x", 3, 4, 8),
    )
    args = (torch.randn(2, 3, 8, 8), torch.randn(2, 64))
    shapes = aot_mint.dynamic_shapes_spec(dims, ("x", "t"))
    program = aot_mint.export_program(module, args, {}, dynamic_shapes=shapes,
                                      strict=True)
    gaps = aot_mint.declared_range_gaps(program, dims)
    assert gaps and any("PINS" in g or "solved to" in g for g in gaps), gaps


# ---------------------------------------------------------------------------
# One code path, two families — generic declared inputs
# ---------------------------------------------------------------------------


class FakeUnet(torch.nn.Module):
    """sdxl-shaped: nested added_cond_kwargs container, config widths. The
    signature mirrors UNet2DConditionModel's positional order — the builder
    binds declared names to these slots and feeds ALL-POSITIONAL."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(4, 4)
        self.config = SimpleNamespace(in_channels=4, cross_attention_dim=2048)

    def forward(self, sample, timestep, encoder_hidden_states,
                added_cond_kwargs=None, return_dict=True):  # type: ignore[no-untyped-def]
        return sample


class FakeWanDit(torch.nn.Module):
    """wan-ti2v-shaped: per-token float32 timestep, config widths, and an
    undeclared in-between parameter (encoder_hidden_states_image) whose
    signature default must fill its positional slot."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(8, 8)
        self.config = SimpleNamespace(in_channels=48, text_dim=4096)

    def forward(self, hidden_states, timestep, encoder_hidden_states,
                encoder_hidden_states_image=None, return_dict=True):  # type: ignore[no-untyped-def]
        return hidden_states


def _sdxl_shaped_decl() -> Compile:
    """The real sdxl contract, as a declaration: aspect rows x CFG arms,
    pooled width 1280 and 6 micro-conditioning ids as literal axes."""
    return Compile(
        family="sdxl-shaped", targets=("unet",), text_len=77,
        dims=(
            Dim("B", carried_by=(("sample", 0), ("encoder_hidden_states", 0))),
            Dim("H_lat", carried_by=(("sample", 2),), multiple_of=8),
            Dim("W_lat", carried_by=(("sample", 3),), multiple_of=8),
            Dim("T_txt", carried_by=(("encoder_hidden_states", 1),)),
        ),
        forks=(Fork("cfg", served=(True, False),
                    why="CFG is ONE batch-2 forward (ie#345); turbo is batch-1"),),
        classes=tuple(
            GraphClass(dims={"B": b, "H_lat": h // 8, "W_lat": w // 8,
                             "T_txt": 77}, fork={"cfg": cfg})
            for (w, h) in ((1024, 1024), (1152, 896))
            for (cfg, b) in ((True, 2), (False, 1))
        ),
        inputs=(
            Input("sample", shape=("B", ("config", "in_channels"),
                                   "H_lat", "W_lat")),
            Input("timestep", shape=(), value=1.0),
            Input("encoder_hidden_states",
                  shape=("B", "T_txt", ("config", "cross_attention_dim"))),
            Input("added_cond_kwargs.text_embeds", shape=("B", 1280)),
            Input("added_cond_kwargs.time_ids", shape=("B", 6)),
        ),
        args=(Arg("return_dict", False),),
        shape_strategy="static-rows",   # #730 ratified: conv-bearing -> static
        warm_changes_key=False,          # measured mint-order canon
    )


def _ti2v_shaped_decl() -> Compile:
    fork = {"expand_timesteps": True}
    return Compile(
        family="wan-ti2v-shaped", targets=("transformer",), text_len=512,
        dims=(
            Dim("B", carried_by=(("hidden_states", 0), ("timestep", 0))),
            Dim("F_lat", carried_by=(("hidden_states", 2),)),
            Dim("H_lat", carried_by=(("hidden_states", 3),), multiple_of=2),
            Dim("W_lat", carried_by=(("hidden_states", 4),), multiple_of=2),
            Dim("T_txt", carried_by=(("encoder_hidden_states", 1),)),
            Dim("N_tok", carried_by=(("timestep", 1),),
                relates_to=("F_lat", "H_lat", "W_lat")),
        ),
        forks=(Fork("expand_timesteps", served=(True,), unserved=(False,),
                    source=("pipeline", "expand_timesteps"),
                    why="per-token timestep IS the ti2v graph class"),),
        classes=(GraphClass(
            dims={"B": 1, "F_lat": 31, "H_lat": 44, "W_lat": 80,
                  "T_txt": 512, "N_tok": 27280}, fork=fork),),
        inputs=(
            Input("hidden_states",
                  shape=("B", ("config", "in_channels"), "F_lat", "H_lat",
                         "W_lat")),
            Input("timestep", shape=("B", "N_tok"), dtype="float32"),
            Input("encoder_hidden_states",
                  shape=("B", "T_txt", ("config", "text_dim"))),
        ),
        args=(Arg("return_dict", False),),
        shape_strategy="static-rows",   # decision (c): ONE row, ONE artifact
        warm_changes_key=False,          # §7: wan needs no pre-warm
    )


def test_two_families_one_code_path() -> None:
    """The #739 red test: different declarations, correct example inputs,
    ONE generic function — zero per-family SDK code. Both feeds come out
    ALL-POSITIONAL (the pod-9 mint obligation): named rows bind to their
    signature slots, containers ride as one argument, and undeclared
    in-between parameters get their signature defaults."""
    sdxl_spec = ExportSpec(
        family="sdxl-shaped", target="unet",
        fork=(("cfg", True),),
        class_dims=(("B", 2), ("H_lat", 128), ("W_lat", 128), ("T_txt", 77)))
    args, kwargs = aot_declaration.declared_inputs(
        FakeUnet(), sdxl_spec, _sdxl_shaped_decl())
    assert kwargs == {}
    assert args[0].shape == (2, 4, 128, 128)
    assert args[1].ndim == 0 and float(args[1]) == 1.0
    assert args[2].shape == (2, 77, 2048)
    assert args[3]["text_embeds"].shape == (2, 1280)
    assert args[3]["time_ids"].shape == (2, 6)
    assert args[4] is False  # return_dict, positionally

    wan_spec = ExportSpec(
        family="wan-ti2v-shaped", target="transformer",
        fork=(("expand_timesteps", True),))
    args2, kwargs2 = aot_declaration.declared_inputs(
        FakeWanDit(), wan_spec, _ti2v_shaped_decl())
    assert kwargs2 == {}
    assert args2[0].shape == (1, 48, 31, 44, 80)
    assert args2[1].shape == (1, 27280)
    assert args2[1].dtype == torch.float32
    assert args2[2].shape == (1, 512, 4096)
    assert args2[3] is None   # encoder_hidden_states_image: default fills the slot
    assert args2[4] is False  # return_dict, positionally


def test_declared_inputs_refuse_missing_config_fields_by_name() -> None:
    class NoConfig(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.p = torch.nn.Linear(2, 2)
            self.config = SimpleNamespace()

    spec = ExportSpec(family="wan-ti2v-shaped", target="transformer",
                      fork=(("expand_timesteps", True),))
    with pytest.raises(MintRefused, match="in_channels"):
        aot_declaration.declared_inputs(NoConfig(), spec, _ti2v_shaped_decl())


def test_builder_for_prefers_the_registered_declaration() -> None:
    from gen_worker import aot_inputs

    register_export_declaration(_sdxl_shaped_decl())
    builder = aot_inputs.builder_for("sdxl-shaped", "unet")
    spec = ExportSpec(
        family="sdxl-shaped", target="unet", fork=(("cfg", False),),
        class_dims=(("B", 1), ("H_lat", 128), ("W_lat", 128), ("T_txt", 77)))
    args, kwargs = builder(FakeUnet(), spec)
    assert kwargs == {}
    assert args[0].shape == (1, 4, 128, 128)
    assert args[-1] is False  # return_dict rides positionally


def test_declared_inputs_actually_export() -> None:
    """The one code path produces inputs a REAL strict export accepts."""

    class TinyUnet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = torch.nn.Linear(2048, 4)
            self.config = SimpleNamespace(in_channels=4, cross_attention_dim=2048)

        def forward(self, sample: torch.Tensor, timestep: torch.Tensor,
                    encoder_hidden_states: torch.Tensor,
                    added_cond_kwargs: dict | None = None,
                    return_dict: bool = True) -> Any:
            cond = self.proj(encoder_hidden_states).mean(dim=(1, 2))
            out = sample * timestep + cond.reshape(-1, 1, 1, 1)
            if added_cond_kwargs:
                out = out + added_cond_kwargs["text_embeds"].mean()
            return (out,) if not return_dict else out

    spec = ExportSpec(
        family="sdxl-shaped", target="unet", fork=(("cfg", True),),
        class_dims=(("B", 2), ("H_lat", 128), ("W_lat", 128), ("T_txt", 77)))
    args, kwargs = aot_declaration.declared_inputs(
        TinyUnet(), spec, _sdxl_shaped_decl())
    assert kwargs == {}  # the pod-9 obligation: the trace is all-positional
    program = aot_mint.export_program(TinyUnet().eval(), args, kwargs,
                                      strict=True)
    assert program.graph_module is not None


# ---------------------------------------------------------------------------
# The two pgw#723-residuals mint gates (pods 8/9)
# ---------------------------------------------------------------------------


def test_mint_refuses_kwarg_example_feeds_by_name(tmp_path) -> None:
    """The pod-9 shape, refused at the mint instead of discovered at first
    serve: the flat forward traced the lifted pair as kwargs, the package
    demanded kwargs, the positional serve marshal fed none — 'Ran into a
    kwarg keyword mismatch' swallowed on first call, every "armed" call
    silently eager (splat_served=false, B_vs_eagerB_maxdiff=0.0)."""

    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = torch.nn.Linear(4, 4)

        def forward(self, x, lora_a=None, lora_b=None):  # type: ignore[no-untyped-def]
            return self.proj(x)

    spec = ExportSpec(family="f", target="t", weight_lane="w8a8")
    with pytest.raises(MintRefused, match="kwarg"):
        aot_mint.mint(
            Tiny().eval(), spec, tmp_path,
            example_inputs=lambda: (
                (torch.randn(2, 4),),
                {"lora_a": torch.randn(2, 4), "lora_b": torch.randn(4, 2)}))


def test_positionalize_refuses_keyword_only_declared_inputs() -> None:
    class KwOnly(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.p = torch.nn.Linear(2, 2)
            self.config = SimpleNamespace(in_channels=48, text_dim=4096)

        def forward(self, hidden_states, timestep, *,
                    encoder_hidden_states=None):  # type: ignore[no-untyped-def]
            return hidden_states

    spec = ExportSpec(family="wan-ti2v-shaped", target="transformer",
                      fork=(("expand_timesteps", True),))
    with pytest.raises(MintRefused, match="KEYWORD-ONLY"):
        aot_declaration.declared_inputs(KwOnly(), spec, _ti2v_shaped_decl())


def test_lifted_lora_mint_requires_the_torch_213_floor(monkeypatch) -> None:
    """Pod 8 (pgw#723 residuals): torch 2.9 strict export refuses
    bind_views' in-trace setattr ('Mutating module attribute lora_a during
    export') that 2.13 traces fine — a mint PRECONDITION when the lifted
    fork is declared, recorded as a named refusal instead of a deep-trace
    AssertionError."""
    lifted = ExportSpec(family="f", target="t", lora_bucket=64)
    plain = ExportSpec(family="f", target="t")

    assert aot_mint.lifted_torch_gap(plain) == ""
    # This box runs the 2.13 prod floor, so the lifted spec passes here...
    assert aot_mint.lifted_torch_gap(lifted) == ""
    # ...and refuses by name on the measured pod-8 version.
    monkeypatch.setattr(torch, "__version__", "2.9.1+cu126")
    gap = aot_mint.lifted_torch_gap(lifted)
    assert "2.13" in gap and "bind_views" in gap
    assert aot_mint.lifted_torch_gap(plain) == ""
    monkeypatch.setattr(torch, "__version__", "nonsense")
    assert "cannot parse" in aot_mint.lifted_torch_gap(lifted)


# ---------------------------------------------------------------------------
# The fork gate — sources read off composed objects
# ---------------------------------------------------------------------------


def test_fork_gate_refuses_wrong_pipeline_arm_by_name() -> None:
    decl = _ti2v_shaped_decl()
    pipeline = SimpleNamespace(config=SimpleNamespace(expand_timesteps=None))
    gaps = aot_declaration.fork_gaps(
        decl, {"expand_timesteps": True}, target="transformer",
        pipeline=pipeline, module=None)
    assert len(gaps) == 1 and "expand_timesteps" in gaps[0]
    assert "wrong class" in gaps[0]

    honest = SimpleNamespace(config=SimpleNamespace(expand_timesteps=True))
    assert aot_declaration.fork_gaps(
        decl, {"expand_timesteps": True}, target="transformer",
        pipeline=honest, module=None) == []


def test_fork_gate_reads_module_attributes_for_module_sources() -> None:
    decl = _wan_a14b_decl()
    vae = SimpleNamespace(use_tiling=True, config=SimpleNamespace())
    gaps = aot_declaration.fork_gaps(
        decl, {"expand_timesteps": False, "use_tiling": False},
        target="vae.decode",
        pipeline=SimpleNamespace(config=SimpleNamespace(expand_timesteps=None)),
        module=vae)
    assert any("use_tiling" in g and "module" in g for g in gaps), gaps


def test_fork_gate_refuses_omitted_and_undeclared_forks() -> None:
    decl = _wan_a14b_decl()
    gaps = aot_declaration.fork_gaps(decl, {}, target="transformer")
    assert any("omits declared fork 'expand_timesteps'" in g for g in gaps)
    gaps2 = aot_declaration.fork_gaps(
        decl, {"expand_timesteps": False, "made_up": 1}, target="transformer")
    assert any("undeclared fork" in g and "made_up" in g for g in gaps2)


def test_fork_gate_refuses_missing_source_field_by_name() -> None:
    decl = _ti2v_shaped_decl()
    bare = SimpleNamespace()
    gaps = aot_declaration.fork_gaps(
        decl, {"expand_timesteps": True}, target="transformer",
        pipeline=bare, module=None)
    assert any("neither a config field nor an attribute" in g for g in gaps)


# ---------------------------------------------------------------------------
# Request resolution + identity
# ---------------------------------------------------------------------------


def test_apply_declaration_refuses_hand_dynamic_rows() -> None:
    register_export_declaration(_wan_a14b_decl())
    spec = ExportSpec(
        family="wan-2.2-t2v-a14b", target="transformer",
        fork=(("expand_timesteps", False), ("use_tiling", False)),
        dynamic=(aot_mint.DynamicDim("hidden_states", 3, 90, 160),))
    with pytest.raises(MintRefused, match="hand-written"):
        aot_declaration.apply_declaration(spec)


def test_apply_declaration_requires_the_warm_canon() -> None:
    decl = _wan_a14b_decl()
    unmeasured = Compile(**{**_fields(decl), "warm_changes_key": None})
    register_export_declaration(unmeasured)
    spec = ExportSpec(
        family="wan-2.2-t2v-a14b", target="transformer",
        fork=(("expand_timesteps", False), ("use_tiling", False)))
    with pytest.raises(MintRefused, match="warm_changes_key"):
        aot_declaration.apply_declaration(spec)


def test_apply_declaration_derives_the_contract() -> None:
    register_export_declaration(_wan_a14b_decl())
    spec = ExportSpec(
        family="wan-2.2-t2v-a14b", target="transformer",
        shapes=((1280, 720, 81), (720, 1280, 81)),
        fork=(("expand_timesteps", False), ("use_tiling", False)))
    out = aot_declaration.apply_declaration(spec)
    assert {(d.input_name, d.axis, d.min, d.max) for d in out.dynamic} == {
        ("hidden_states", 3, 90, 160), ("hidden_states", 4, 90, 160)}
    assert out.specialization["shape_strategy"] == "dynamic-collapse"
    assert out.specialization["warm_changes_key"] is False
    assert out.specialization["fork.expand_timesteps"] is False


def test_unregistered_family_passes_through_untouched() -> None:
    spec = ExportSpec(family="plain", target="unet")
    assert aot_declaration.apply_declaration(spec) is spec


def test_fork_and_row_reach_the_cell_identity() -> None:
    """A fork is a DISTINCT graph class in #716's hash; a static row is the
    artifact's identity."""
    meta = {"sm": "sm_89", "range_digest": "r1", "format": "pt2",
            "family": "sdxl-shaped", "graph": {"v": 1}}
    base = dict(family="sdxl-shaped", target="unet")
    a = aot_mint.cell_identity(meta, ExportSpec(**base, fork=(("cfg", True),)))
    b = aot_mint.cell_identity(meta, ExportSpec(**base, fork=(("cfg", False),)))
    c = aot_mint.cell_identity(meta, ExportSpec(**base))
    assert len({a.digest, b.digest, c.digest}) == 3
    d1 = aot_mint.cell_identity(
        meta, ExportSpec(**base, class_dims=(("H_lat", 128), ("W_lat", 128))))
    d2 = aot_mint.cell_identity(
        meta, ExportSpec(**base, class_dims=(("H_lat", 112), ("W_lat", 144))))
    assert d1.digest != d2.digest


def test_load_spec_parses_the_coordinate() -> None:
    import json

    body = {
        "family": "sdxl-shaped", "target": "unet",
        "fork": {"cfg": True},
        "class_dims": {"H_lat": 128, "W_lat": 128, "B": 2, "T_txt": 77},
    }
    path = None
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "req.json"
        path.write_text(json.dumps(body))
        spec, _raw = aot_mint._load_spec(path)
    assert spec.fork == (("cfg", True),)
    assert dict(spec.class_dims)["H_lat"] == 128
