"""ie#637's declaration, at micro scale: two rows that pad to DIFFERENT lengths.

The rows are chosen so that a graph which decided the pad once at trace cannot
serve both of them — which is the property that made z-image's refusal correct
and is the property this member has to keep:

    grid 12 -> L = 144 -> pad 16 -> padded 160
    grid 16 -> L = 256 -> pad  0 -> padded 256

and the served hull between them contains a third class (grid 14 -> L = 196 ->
pad 28 -> padded 224), which the parity leg exercises precisely because it is
NOT a declared row: the artifact must serve it from the derived range.

Same 4-D nonlinear-extent shape as `aot_declaration_4d` (both spatial axes
dynamic, so the token count is a PRODUCT of two symbols) — this adds the pad on
top, which is the one thing z-image has that micro-4d does not.
"""

from __future__ import annotations

from gen_worker import (
    Compile,
    Dim,
    GraphClass,
    Input,
    register_export_declaration,
)

FAMILY = "micro-pad32-branchy"

#: Grid extents whose token counts pad to DIFFERENT multiples of 32.
LATENT_ROWS = (12, 16)
#: Inside the declared hull, and a THIRD pad class. Served, never traced.
UNDECLARED_ROW = 14
VAE_SCALE = 8
PIXEL_ROWS = tuple((n * VAE_SCALE, n * VAE_SCALE) for n in LATENT_ROWS)
COND_LEN = 16
ARITY = 1


def build_declaration() -> Compile:
    return Compile(
        family=FAMILY,
        targets=("transformer",),
        shapes=PIXEL_ROWS,
        text_len=COND_LEN,
        dims=(
            Dim("N", carried_by=(("t", 0),)),
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
           "PIXEL_ROWS", "UNDECLARED_ROW", "VAE_SCALE", "build_declaration"]
