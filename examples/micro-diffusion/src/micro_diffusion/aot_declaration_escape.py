"""The pgw#1062 declaration: the escape-hatch graph, one entry.

Deliberately the smallest declaration that still puts every author-defined-op
surface inside a minted cell: ONE target, fork-free, batch fixed, two token
rows collapsing under ``dynamic-collapse`` into a single entry. The variable
under test is the OPS in the graph — a custom op with a fake kernel, a
``triton_op`` kernel, a raw ``@triton.jit`` call — not entry count, so
anything that would multiply entries is pinned.
"""

from __future__ import annotations

from gen_worker import (
    Compile,
    Dim,
    GraphClass,
    Input,
    register_export_declaration,
)

FAMILY = "micro-escape"

LATENT_ROWS = (32, 48)
VAE_SCALE = 8
PIXEL_ROWS = tuple((n * VAE_SCALE, n * VAE_SCALE) for n in LATENT_ROWS)
#: Token-shaped like the base family: the dynamic extent stays LINEAR in one
#: symbol (pgw#998), so the only new facts in this cell are the ops.
TOKEN_ROWS = tuple(n * n for n in LATENT_ROWS)
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
            Dim("T", carried_by=(("x", 0),)),
        ),
        classes=tuple(
            GraphClass(dims={"N": ARITY, "T": tokens})
            for tokens in TOKEN_ROWS),
        inputs=(
            Input("x", shape=("T", ("config", "in_channels")),
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
           "PIXEL_ROWS", "TOKEN_ROWS", "VAE_SCALE", "build_declaration"]
