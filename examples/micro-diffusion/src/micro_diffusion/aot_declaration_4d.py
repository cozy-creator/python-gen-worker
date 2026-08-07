"""The pgw#998 declaration: a 4-D latent grid, both spatial axes dynamic.

This is the shape z-image declares (`H_lat` and `W_lat` on a 4-D latent under
`dynamic-collapse`) and the shape pgw#998 measured as unlowerable across the
mint's `torch.export.save`/`load` hand-off to its compile child:

    InductorError: LoweringException: RuntimeError:
      ('unexpected None!', 512*s18*s57)

pgw#998's fix (the export handoff carries the ShapeEnv's symbol values) landed
on master. This declaration exists so the gauntlet re-proves that on every run
instead of trusting the changelog — and so that if it ever regresses, it does
so on a 1.1 MB toy in 20 seconds rather than on a rented A100.

Deliberately kept to ONE target and ONE fork-free class set: the point is the
nonlinear EXTENT, not entry count, and a second target would only add wall
clock to a variant whose whole job is a single lowering question.
"""

from __future__ import annotations

from gen_worker import (
    Compile,
    Dim,
    GraphClass,
    Input,
    register_export_declaration,
)

FAMILY = "micro-4d"

#: The two latent grids, as GRID extents (not token counts).
LATENT_ROWS = (32, 48)
VAE_SCALE = 8
PIXEL_ROWS = tuple((n * VAE_SCALE, n * VAE_SCALE) for n in LATENT_ROWS)
COND_LEN = 16
#: Batch is fixed here: the variable under test is the SPATIAL product, and a
#: fork would just double the compile for no extra coverage of it.
ARITY = 1


def build_declaration() -> Compile:
    return Compile(
        family=FAMILY,
        targets=("transformer",),
        shapes=PIXEL_ROWS,
        text_len=COND_LEN,
        dims=(
            Dim("N", carried_by=(("t", 0),)),
            # BOTH spatial axes dynamic on ONE input — this is the whole
            # point. Their product is the M extent of every matmul inside.
            Dim("H_lat", carried_by=(("x", 1),), multiple_of=2),
            Dim("W_lat", carried_by=(("x", 2),), multiple_of=2),
        ),
        classes=tuple(
            GraphClass(dims={"N": ARITY, "H_lat": n, "W_lat": n})
            for n in LATENT_ROWS),
        inputs=(
            Input("x", shape=(("config", "in_channels"), "H_lat", "W_lat"),
                  repeat="N", dtype="float32"),
            Input("t", shape=("N",), dtype="float32"),
            Input("cond", shape=(COND_LEN, ("config", "cond_dim")),
                  repeat="N", dtype="float32"),
        ),
        shape_strategy="dynamic-collapse",
        warm_changes_key=False,
    )


DECLARATION = build_declaration()

register_export_declaration(DECLARATION, replace=True)

__all__ = ["ARITY", "COND_LEN", "DECLARATION", "FAMILY", "LATENT_ROWS",
           "PIXEL_ROWS", "VAE_SCALE", "build_declaration"]
