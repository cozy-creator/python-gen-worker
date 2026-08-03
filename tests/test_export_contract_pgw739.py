"""pgw#739 — the export-declaration vocabulary (schema half, no torch).

The vocabulary must accept the accumulated rulings' worked examples and
refuse the accumulated failure modes BY NAME:

- named dims with ``(input, axis)`` bindings (wan's latent-spatial axis is
  the red case the old ``DynamicDim`` two-literal wall refused);
- ``Fork(served=, unserved=)`` with ``(pipeline|module, field)`` source
  bindings (wan's ``expand_timesteps`` is a PIPELINE field);
- rows-as-coordinates with NO formula DSL;
- per-family shape strategy (#730) and mint-warm canon (``warm_changes_key``)
  as declared facts;
- the LTX-AOT-DESIGN.md §2.3 draft is the spec this vocabulary must accept —
  it parses here as the third test case; its family stays unminted.
"""

from __future__ import annotations

import pytest

from gen_worker.api.decorators import Compile, DynamicDim
from gen_worker.api.export_contract import (
    DeclarationError,
    Dim,
    Fork,
    GraphClass,
    Input,
    Arg,
    export_declaration,
    register_export_declaration,
    reset_export_declarations,
)


# ---------------------------------------------------------------------------
# Dim — named axes with (input, axis) bindings
# ---------------------------------------------------------------------------


def test_dim_refusals_are_named() -> None:
    with pytest.raises(DeclarationError, match="no .*bindings"):
        Dim("H_lat", carried_by=())
    with pytest.raises(DeclarationError, match="negative axis"):
        Dim("H_lat", carried_by=(("sample", -1),))
    with pytest.raises(DeclarationError, match="repeats binding"):
        Dim("H_lat", carried_by=(("sample", 2), ("sample", 2)))
    with pytest.raises(DeclarationError, match="relates_to itself"):
        Dim("N", carried_by=(("t", 1),), relates_to=("N",))
    with pytest.raises(DeclarationError, match="multiple_of"):
        Dim("H_lat", carried_by=(("sample", 2),), multiple_of=0)


# ---------------------------------------------------------------------------
# Fork — graph-class forks with source bindings
# ---------------------------------------------------------------------------


def test_fork_served_unserved_partition() -> None:
    f = Fork("isolate_modalities", served=(False,), unserved=(True,),
             why="20,509 vs 30,877 nodes")
    assert f.served == (False,) and f.unserved == (True,)
    with pytest.raises(DeclarationError, match="no served arm"):
        Fork("dead", served=())
    with pytest.raises(DeclarationError, match="BOTH served and unserved"):
        Fork("both", served=(True,), unserved=(True,))


def test_fork_source_binds_pipeline_or_module() -> None:
    f = Fork("expand_timesteps", served=(True,), unserved=(False,),
             source=("pipeline", "expand_timesteps"))
    assert f.source == ("pipeline", "expand_timesteps")
    Fork("use_tiling", served=(False,), unserved=(True,),
         source=("module", "use_tiling"))
    with pytest.raises(DeclarationError, match="pipeline"):
        Fork("bad", served=(True,), source=("config", "x"))


# ---------------------------------------------------------------------------
# GraphClass — resolved rows, hashable for endpoint-side dedupe
# ---------------------------------------------------------------------------


def test_graph_class_accepts_mappings_and_hashes() -> None:
    a = GraphClass(fork={"cfg": False}, dims={"T_v": 130560, "B": 1})
    b = GraphClass(dims={"B": 1, "T_v": 130560}, fork={"cfg": False})
    assert a == b and len(dict.fromkeys([a, b])) == 1  # the LTX generator idiom
    assert a.dim_map == {"T_v": 130560, "B": 1}
    with pytest.raises(DeclarationError, match="non-positive"):
        GraphClass(dims={"T_v": 0})
    with pytest.raises(DeclarationError, match="no dim values"):
        GraphClass(dims={})


# ---------------------------------------------------------------------------
# Compile cross-validation — refusals by name
# ---------------------------------------------------------------------------


def _decl(**overrides) -> Compile:
    base = dict(
        family="fam", targets=("transformer",), text_len=512,
        dims=(Dim("H", carried_by=(("hidden_states", 2),), multiple_of=2),
              Dim("B", carried_by=(("hidden_states", 0),))),
        forks=(Fork("cfg", served=(False,), unserved=(True,)),),
        classes=(GraphClass(dims={"H": 90, "B": 1}, fork={"cfg": False}),
                 GraphClass(dims={"H": 160, "B": 1}, fork={"cfg": False})),
        shape_strategy="dynamic-collapse",
        warm_changes_key=False,
    )
    base.update(overrides)
    return Compile(**base)  # type: ignore[arg-type]


def test_valid_declaration_constructs_and_reaches_contract_axes() -> None:
    c = _decl()
    axes = c.contract_axes()
    assert axes["shape_strategy"] == "dynamic-collapse"
    assert axes["warm_changes_key"] is False
    assert [row["name"] for row in axes["dims"]] == ["H", "B"]
    assert len(axes["classes"]) == 2


def test_undeclared_facts_do_not_change_existing_digests() -> None:
    """A family that has not adopted the vocabulary must keep its digest."""
    c = Compile(family="old", shapes=((768, 768),), text_len=77)
    axes = c.contract_axes()
    for key in ("dims", "forks", "classes", "inputs", "args",
                "shape_strategy", "warm_changes_key", "eager"):
        assert key not in axes


def test_class_omitting_declared_fork_is_refused_by_name() -> None:
    with pytest.raises(DeclarationError, match="omits declared fork.*cfg"):
        _decl(classes=(GraphClass(dims={"H": 90, "B": 1}),))


def test_class_carrying_undeclared_fork_is_refused_by_name() -> None:
    """The #739 red case: a coordinate forking on a flag the vocabulary does
    not declare would silently export two classes as one."""
    with pytest.raises(DeclarationError, match="expand_timesteps.*does not declare"):
        _decl(classes=(GraphClass(
            dims={"H": 90, "B": 1},
            fork={"cfg": False, "expand_timesteps": True}),))


def test_class_on_unserved_arm_is_refused() -> None:
    with pytest.raises(DeclarationError, match="UNSERVED arm cfg=True"):
        _decl(classes=(GraphClass(dims={"H": 90, "B": 1}, fork={"cfg": True}),))


def test_class_dims_must_match_declared_dims_exactly() -> None:
    with pytest.raises(DeclarationError, match="omits declared dim.*B"):
        _decl(classes=(GraphClass(dims={"H": 90}, fork={"cfg": False}),))
    with pytest.raises(DeclarationError, match="undeclared dim.*W"):
        _decl(classes=(GraphClass(
            dims={"H": 90, "B": 1, "W": 3}, fork={"cfg": False}),))


def test_duplicate_class_rows_are_refused() -> None:
    row = GraphClass(dims={"H": 90, "B": 1}, fork={"cfg": False})
    with pytest.raises(DeclarationError, match="repeats row"):
        _decl(classes=(row, row))


def test_relates_to_must_name_declared_dims() -> None:
    with pytest.raises(DeclarationError, match="relates_to undeclared"):
        _decl(dims=(Dim("H", carried_by=(("hidden_states", 2),)),
                    Dim("B", carried_by=(("hidden_states", 0),)),
                    Dim("N", carried_by=(("t", 1),), relates_to=("W_lat",))),
              classes=(GraphClass(dims={"H": 90, "B": 1, "N": 10},
                                  fork={"cfg": False}),))


def test_shape_strategy_is_a_declared_choice() -> None:
    with pytest.raises(DeclarationError, match="shape_strategy"):
        _decl(shape_strategy="adaptive")
    # "" parses (the LTX draft predates #730's ratification); the mint-plan
    # derivation refuses it — see test_aot_declaration_pgw739.
    assert _decl(shape_strategy="").shape_strategy == ""


def test_hand_range_on_named_dim_is_refused_when_classes_exist() -> None:
    """Rows are coordinates: a hand-written range beside class rows is the
    endpoint hand-math the vocabulary exists to prevent."""
    with pytest.raises(ValueError, match="delete the hand range"):
        _decl(dynamic=(DynamicDim("H", min=90, max=160),))


def test_dynamic_naming_nothing_declared_is_refused() -> None:
    with pytest.raises(ValueError, match="neither"):
        Compile(family="f", shapes=((768, 768),), text_len=77,
                dynamic=(DynamicDim("latent_h", min=90, max=160),))


def test_wan_latent_spatial_axis_is_nameable() -> None:
    """The red case the old two-literal DynamicDim validation refused
    (ie#550 hit it from the endpoint side, ie#566 from the mint side)."""
    c = Compile(
        family="wan-shaped", shapes=((1280, 720, 81),), text_len=512,
        dims=(Dim("H_lat", carried_by=(("hidden_states", 3),), multiple_of=2),),
        dynamic=(DynamicDim("H_lat", min=90, max=160),),
    )
    assert c.dynamic[0].dim == "H_lat"


def test_fork_and_input_targets_must_exist() -> None:
    with pytest.raises(DeclarationError, match="vae.decode"):
        _decl(forks=(Fork("cfg", served=(False,), targets=("vae.decode",)),))
    with pytest.raises(DeclarationError, match="not in Compile.targets"):
        _decl(inputs=(Input("hidden_states", shape=("B",), targets=("vae",)),))


def test_dim_bindings_must_name_declared_inputs_when_inputs_exist() -> None:
    # pgw#853 widened the rule: a TEMPLATED Arg is a legal carrier too (an
    # extent may enter as a python int inside a container), so the refusal
    # now names both kinds of row it looked for.
    with pytest.raises(DeclarationError,
                       match="binds 'hidden_states'.*no declared Arg templates"):
        _decl(inputs=(Input("other", shape=("B", "H")),))


def test_shapes_omittable_only_with_classes() -> None:
    with pytest.raises(ValueError, match="shapes"):
        Compile(family="f", targets=("transformer",))
    assert _decl().shapes == ()  # classes carry the coordinates


# ---------------------------------------------------------------------------
# Input / Arg templates
# ---------------------------------------------------------------------------


def test_input_axis_specs() -> None:
    i = Input("hidden_states",
              shape=("B", ("config", "in_channels"), "H", "W"))
    assert i.shape[1] == ("config", "in_channels")
    with pytest.raises(DeclarationError, match="axis spec"):
        Input("x", shape=(("cfg", "in_channels"),))
    with pytest.raises(DeclarationError, match="positive"):
        Input("x", shape=(0,))
    assert Arg("return_dict", False).value is False


# ---------------------------------------------------------------------------
# The registry (the #740 load-to-register shape)
# ---------------------------------------------------------------------------


def test_registry_round_trip_and_conflict() -> None:
    reset_export_declarations()
    try:
        decl = _decl(family="regfam")
        register_export_declaration(decl)
        register_export_declaration(decl)  # idempotent for an identical one
        assert export_declaration("regfam") is decl
        different = _decl(family="regfam", shape_strategy="static-rows")
        with pytest.raises(DeclarationError, match="DIFFERENT"):
            register_export_declaration(different)
        register_export_declaration(different, replace=True)
        assert export_declaration("regfam") is different
    finally:
        reset_export_declarations()


def test_registry_refuses_classless_declarations() -> None:
    with pytest.raises(DeclarationError, match="no graph classes"):
        register_export_declaration(
            Compile(family="x", shapes=((768, 768),), text_len=77))


# ---------------------------------------------------------------------------
# The LTX-AOT-DESIGN.md §2.3 draft — the third test case. It must PARSE;
# the family stays unminted (no shape_strategy declared yet, and the plan
# derivation refuses that by name — the honest unminted state).
# ---------------------------------------------------------------------------

# The doc's tables, abbreviated to three size buckets x two duration buckets;
# the generator mirrors _ltx_graph_classes' structure (including the
# (10s,24) vs (5s,48) collision on frame count 241 that iterating
# _LTX_FRAMES.items() — not set(values()) — keeps visible via T_a).
_LTX_SIZES = {
    ("1216p", "16:9"): (1216, 704, True),
    ("704p", "1:1"): (704, 704, False),
    ("1080p", "9:16"): (608, 1088, False),
}
_LTX_FRAMES = {(10, 24): 241, (5, 48): 241, (5, 24): 121}
_AUDIO_TOKENS_PER_SECOND = 25.0


def _ltx_graph_classes() -> tuple[GraphClass, ...]:
    out: list[GraphClass] = []
    for (_res, _aspect), (width, height, two_stage) in _LTX_SIZES.items():
        for (duration_s, fps), frames in _LTX_FRAMES.items():
            t_a = round(frames / fps * _AUDIO_TOKENS_PER_SECOND)
            sizes = ((width // 2, height // 2), (width, height)) if two_stage \
                else ((width, height),)
            for w, h in sizes:
                f_lat, h_lat, w_lat = (frames - 1) // 8 + 1, h // 32, w // 32
                base = f_lat * h_lat * w_lat
                fork = {"isolate_modalities": False, "cfg": False, "stg": False}
                for t_kf in (0, h_lat * w_lat):
                    out.append(GraphClass(
                        fork=fork,
                        dims={"T_v": base + t_kf, "T_a": t_a, "T_at": 1,
                              "T_txt": 1024, "B": 1}))
                out.append(GraphClass(
                    fork=fork,
                    dims={"T_v": base, "T_a": t_a, "T_at": t_a,
                          "T_txt": 1024, "B": 1}))
    return tuple(dict.fromkeys(out))


def test_ltx_section_2_3_draft_parses() -> None:
    classes = _ltx_graph_classes()
    assert len(classes) > len({c.dim_map["T_v"] for c in classes})
    draft = Compile(
        family="ltx-2.3",
        targets=("transformer",),
        text_len=1024,
        regional=False,
        dims=(
            Dim("T_v", carried_by=(("hidden_states", 1), ("timestep", 1),
                                   ("video_coords", 2))),
            Dim("T_a", carried_by=(("audio_hidden_states", 1),
                                   ("audio_coords", 1))),
            Dim("T_at", carried_by=(("audio_timestep", 1),)),
            Dim("T_txt", carried_by=(("encoder_hidden_states", 1),
                                     ("audio_encoder_hidden_states", 1),
                                     ("encoder_attention_mask", 1),
                                     ("audio_encoder_attention_mask", 1))),
            Dim("B", carried_by=(("hidden_states", 0),)),
        ),
        forks=(
            Fork("isolate_modalities", served=(False,), unserved=(True,),
                 why="text_to_audio drops both AV cross-attentions: 20,509 vs "
                     "30,877 nodes. UNSERVED today (ie#383 phase 2)."),
            Fork("cfg", served=(False,), unserved=(True,),
                 why="distilled recipe pins guidance 1.0"),
            Fork("stg", served=(False,), unserved=(True,),
                 why="the only path arming perturbation_mask"),
        ),
        classes=classes,
        dynamic=(),
        eager=("video_to_video", "audio_reactive"),
    )
    assert draft.eager == ("video_to_video", "audio_reactive")
    # T_a differs across duration buckets sharing frame count 241 — the axis
    # the (w, h, frames) row could not carry is now a declared coordinate
    # (241f @ 24fps -> 251 tokens; 241f @ 48fps -> 126).
    t_a_values = {c.dim_map["T_a"] for c in classes}
    assert {251, 126} <= t_a_values
    # Unminted is a recorded state, not an accident: no strategy declared.
    from gen_worker import aot_declaration
    from gen_worker.aot_mint import MintRefused

    with pytest.raises(MintRefused, match="shape_strategy"):
        aot_declaration.mint_plans(draft, "transformer")
