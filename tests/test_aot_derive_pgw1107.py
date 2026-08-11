"""pgw#1107: the classes deriver reproduces sdxl/z-image byte-identically, and
the migration safety gate catches a re-key.

The two hand-written ``_..._graph_classes()`` helpers are transcribed here as
the STANDING oracle. If :func:`gen_worker.cfg_image_classes` ever
stops reproducing them the SDK deriver has drifted from the file it is meant to
replace — which is exactly the mismatch the gate exists to stop, caught in the
SDK's own suite instead of on a pod mid-mint.
"""

from __future__ import annotations

from typing import Tuple

from gen_worker import (
    Compile,
    DeclarationMismatch,
    Dim,
    Fork,
    GraphClass,
    Input,
    assert_faithful,
    cfg_image_classes,
    class_set_delta,
    contract_delta,
    override_delta,
)

# --- standing oracles (verbatim structure from the endpoint files) ---------

_SDXL_ASPECT_ROWS: Tuple[Tuple[int, int], ...] = (
    (1536, 640), (1344, 768), (1216, 832), (1152, 896), (1024, 1024),
    (896, 1152), (832, 1216), (768, 1344), (640, 1536),
)
_Z_IMAGE_ASPECT_ROWS: Tuple[Tuple[int, int], ...] = (
    (1024, 1024), (1344, 768), (768, 1344), (1536, 640), (640, 1536),
)


def _sdxl_graph_classes() -> Tuple[GraphClass, ...]:
    out = []
    for w, h in _SDXL_ASPECT_ROWS:
        for cfg, batch in ((True, 2), (False, 1)):
            out.append(GraphClass(
                dims={"B": batch, "H_lat": h // 8, "W_lat": w // 8,
                      "T_txt": 77},
                fork={"cfg": cfg}))
    return tuple(dict.fromkeys(out))


def _z_image_graph_classes() -> Tuple[GraphClass, ...]:
    out = []
    for w, h in _Z_IMAGE_ASPECT_ROWS:
        for cfg, arity in ((True, 2), (False, 1)):
            out.append(GraphClass(
                dims={"N": arity, "H_lat": h // 8, "W_lat": w // 8,
                      "T_cap": 512},
                fork={"cfg": cfg}))
    return tuple(dict.fromkeys(out))


# --- the deriver reproduces both helpers byte-identically ------------------

def test_sdxl_classes_derive_byte_identical() -> None:
    derived = cfg_image_classes(
        shapes=_SDXL_ASPECT_ROWS, latent_scale=8, text_len=77)
    standing = _sdxl_graph_classes()
    assert class_set_delta(standing, derived) == {}
    # order-sensitive equality, the exact thing contract_axes serialises
    assert [c.as_row() for c in derived] == [c.as_row() for c in standing]


def test_z_image_classes_derive_byte_identical() -> None:
    derived = cfg_image_classes(
        shapes=_Z_IMAGE_ASPECT_ROWS, latent_scale=8, text_len=512,
        batch_dim="N", text_dim="T_cap")
    standing = _z_image_graph_classes()
    assert class_set_delta(standing, derived) == {}
    assert [c.as_row() for c in derived] == [c.as_row() for c in standing]


def test_transposed_rows_are_kept_not_deduped() -> None:
    # (1536,640) and (640,1536) swap H_lat/W_lat -> distinct classes; only
    # exact duplicates collapse. sdxl has 9 rows x 2 regimes and no collision.
    derived = cfg_image_classes(
        shapes=_SDXL_ASPECT_ROWS, latent_scale=8, text_len=77)
    assert len(derived) == len(_SDXL_ASPECT_ROWS) * 2


# --- the migration gate: identical contract passes, a re-key STOPS ---------

def _sdxl_standing_compile(classes: Tuple[GraphClass, ...]) -> Compile:
    return Compile(
        family="sdxl",
        targets=("unet",),
        text_len=77,
        shapes=_SDXL_ASPECT_ROWS,
        dims=(
            Dim("B", carried_by=(("sample", 0),
                                 ("encoder_hidden_states", 0),
                                 ("added_cond_kwargs.text_embeds", 0),
                                 ("added_cond_kwargs.time_ids", 0))),
            Dim("H_lat", carried_by=(("sample", 2),), multiple_of=8),
            Dim("W_lat", carried_by=(("sample", 3),), multiple_of=8),
            Dim("T_txt", carried_by=(("encoder_hidden_states", 1),)),
        ),
        forks=(Fork("cfg", served=(True, False), why="CFG batch-2 forward"),),
        classes=classes,
        inputs=(
            Input("sample",
                  shape=("B", ("config", "in_channels"), "H_lat", "W_lat"),
                  dtype="model"),
            Input("timestep", shape=(), dtype="float32", value=1.0),
            Input("encoder_hidden_states",
                  shape=("B", "T_txt", ("config", "cross_attention_dim")),
                  dtype="model"),
            Input("added_cond_kwargs.text_embeds", shape=("B", 1280),
                  dtype="model"),
            Input("added_cond_kwargs.time_ids", shape=("B", 6), dtype="model"),
        ),
        shape_strategy="static-rows",
        warm_changes_key=False,
        numerics_floor=0.995,
        numerics_warn=0.999,
    )


def test_gate_passes_when_only_classes_source_changes() -> None:
    # The migration: hand-written helper -> derived cross-product, everything
    # else identical. The contract MUST be byte-identical.
    standing = _sdxl_standing_compile(_sdxl_graph_classes())
    migrated = _sdxl_standing_compile(
        cfg_image_classes(shapes=_SDXL_ASPECT_ROWS, latent_scale=8, text_len=77))
    assert contract_delta(standing, migrated) == {}
    assert_faithful(standing, migrated, family="sdxl")


def test_gate_stops_a_dropped_timestep_dtype() -> None:
    # attempt-32 scar: deriving timestep dtype instead of forcing float32.
    standing = _sdxl_standing_compile(_sdxl_graph_classes())
    bad = Compile(
        **{**_struct_kwargs(standing),
           "inputs": tuple(
               Input("timestep", shape=(), dtype="model", value=1.0)
               if i.name == "timestep" else i
               for i in standing.inputs)})
    assert "inputs" in contract_delta(standing, bad)
    try:
        assert_faithful(standing, bad, family="sdxl")
    except DeclarationMismatch as exc:
        assert "inputs" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("gate failed to STOP a dropped timestep dtype")


def test_gate_stops_a_loosened_numerics_floor() -> None:
    # numerics_floor is NOT a contract_axes field (loosening it does not
    # re-key), so contract_delta alone would wave it through — but it IS a
    # must-survive override, so override_delta + assert_faithful STOP it.
    standing = _sdxl_standing_compile(_sdxl_graph_classes())
    loosened = Compile(**{**_struct_kwargs(standing), "numerics_floor": 0.98})
    assert contract_delta(standing, loosened) == {}
    assert override_delta(standing, loosened) == {"numerics_floor": (0.995, 0.98)}
    try:
        assert_faithful(standing, loosened, family="sdxl")
    except DeclarationMismatch as exc:
        assert "numerics_floor" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("gate failed to STOP a loosened numerics floor")


def _struct_kwargs(c: Compile) -> dict:
    import msgspec
    return {fi.name: getattr(c, fi.name) for fi in msgspec.structs.fields(c)}
