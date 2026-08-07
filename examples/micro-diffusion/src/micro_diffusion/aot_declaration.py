"""The micro family's export declaration — THREE entries, on purpose.

sdxl declares 36 (10 aspect rows x static-rows x targets) and a mint cycle
against it costs ~95 minutes on a pod. Nothing about the MACHINERY needs 36:
the mint path's decisions — plan selection, container flattening, the derived
dynamic range, the fork coordinate, the seal, the publish wire, the adopt
filter — are all exercised by three. This declaration is the smallest set that
still exercises every one of them.

The three entries
-----------------

    transformer/cfg=true     the guided arm; the container arity is 2
    transformer/cfg=false    the turbo arm; the container arity is 1
    decoder               one arm, no fork, its own dims

Why each declared fact is what it is
------------------------------------

* ``shape_strategy="dynamic-collapse"`` — legitimate here rather than
  convenient: #730 ratified the strategy for conv-free graphs, and
  :mod:`micro_diffusion.model` contains zero ``nn.Conv*`` layers by
  construction. It is also the strategy that makes the two declared latent
  rows COLLAPSE into one artifact per arm instead of multiplying entries,
  which is the difference between 3 entries and 6.

* ``cfg`` is a FORK, not a dim. The pipeline batches CFG into ONE forward by
  building a two-element list, so the pytree ARITY changes and N=1 / N=2 are
  two graphs rather than one graph at two sizes. That is z-image's measured
  shape and it is why the container inputs exist at all.

* ``T`` (the token count) is carried by BOTH a container element axis
  (``("x", 0)`` — the axis of a list ELEMENT) and a plain tensor axis
  (``("latent", 1)``). One dim, two carriers, two flattening regimes: that is
  precisely the seam pgw#993 fixed on the declaration side (``carried_by``
  must resolve through the same expansion ``dynamic_shapes`` mirrors) and
  pgw#994 on the serve side (a contract position among FLATTENED leaves,
  replayed against a caller's PRE-flattening args).

* The compiled targets take TOKEN SEQUENCES, not latent grids, so the one
  dynamic extent is LINEAR in a single symbol. That is flux/qwen's real
  interface, and it is also forced: with H and W both dynamic every matmul's
  M extent is a PRODUCT of two symbols, and the mint's
  ``torch.export.save``/``load`` hand-off to its compile child loses the
  facts inductor needs to lower one (pgw#998, found by the rig's first
  micro cycle).

* ``t`` is a PLAIN input declared immediately AFTER a container input. That
  ordering is deliberate and is the cheapest possible reproduction of the
  pgw#994 defect shape: with ``x`` expanding to N leaves, every leaf position
  after it shifts, so a serve side that matched contract positions against
  pre-flattening args binds the whole list to element 0 and shifts ``t`` and
  ``cond``. Every micro cycle re-proves that it does not.

* ``warm_changes_key=False`` — MEASURED, not defaulted. The model builds no
  tables lazily: the one table it has is a registered buffer created in
  ``__init__``, so a pre-warmed export and a cold export trace the same graph.
  (z-image's True comes from ``RopeEmbedder`` building its rope tables inside
  the first ``__call__``.)
"""

from __future__ import annotations

from gen_worker import (
    Compile,
    Dim,
    Fork,
    GraphClass,
    Input,
    register_export_declaration,
)

FAMILY = "micro-diffusion"

#: The two latent grids the family serves, and the pixel rows they are.
#: Two, not one: a single row would leave ``dynamic-collapse`` with nothing to
#: collapse and the export fully static, which is the one path where the
#: pgw#993 dynamic-shapes mirroring never runs.
LATENT_ROWS = (32, 48)
VAE_SCALE = 8
PIXEL_ROWS = tuple((n * VAE_SCALE, n * VAE_SCALE) for n in LATENT_ROWS)

#: The TOKEN count each latent grid becomes. The compiled targets take token
#: sequences, so the one dynamic extent is linear in a single symbol — see
#: pgw#998 for what a nonlinear extent costs across the mint's export
#: save/load hand-off.
TOKEN_ROWS = tuple(n * n for n in LATENT_ROWS)

#: The pinned condition length (the ie#544 axis: declared, never discovered).
COND_LEN = 16

#: The CFG batch. Two on the guided arm.
CFG_ARITY = 2


def _classes() -> tuple:
    rows = []
    for tokens in TOKEN_ROWS:
        for cfg, arity in ((True, CFG_ARITY), (False, 1)):
            rows.append(GraphClass(
                dims={"N": arity, "T": tokens},
                fork={"cfg": cfg}, targets=("transformer",)))
        rows.append(GraphClass(
            dims={"N": 1, "T": tokens}, targets=("decoder",)))
    return tuple(dict.fromkeys(rows))


def build_declaration() -> Compile:
    """Construct and validate the declaration without registering it."""
    return Compile(
        family=FAMILY,
        targets=("transformer", "decoder"),
        shapes=PIXEL_ROWS,
        text_len=COND_LEN,
        dims=(
            # N is the container ARITY and, simultaneously, the only axis of
            # `t` — which is what makes it nameable. A list length has no
            # (input, axis) binding of its own to give.
            Dim("N", carried_by=(("t", 0), ("latent", 0))),
            # `x` elements are (T, C) -> element axis 0.
            # `latent` is a plain (N, T, C) tensor -> axis 1.
            # ONE dim, one carrier INSIDE a container and one outside it.
            Dim("T", carried_by=(("x", 0), ("latent", 1))),
        ),
        forks=(
            Fork("cfg", served=(True, False), targets=("transformer",),
                 why="the pipeline batches classifier-free guidance into ONE "
                     "forward by building a two-element latent list, so the "
                     "pytree ARITY doubles: N=1 and N=2 are two graphs, not "
                     "one graph at two sizes. The turbo lane pins guidance to "
                     "0.0 and takes the arity-1 branch."),
        ),
        classes=_classes(),
        inputs=(
            # CONTAINER, then a PLAIN input, then a CONTAINER again — the
            # ordering that makes flattened leaf positions diverge from
            # argument positions (pgw#993/pgw#994).
            Input("x", shape=("T", ("config", "in_channels")),
                  repeat="N", dtype="float32", targets=("transformer",)),
            Input("t", shape=("N",), dtype="float32", targets=("transformer",)),
            Input("cond", shape=(COND_LEN, ("config", "cond_dim")),
                  repeat="N", dtype="float32", targets=("transformer",)),
            Input("latent", shape=("N", "T", ("config", "in_channels")),
                  dtype="float32", targets=("decoder",)),
        ),
        args=(),
        shape_strategy="dynamic-collapse",
        warm_changes_key=False,
    )


DECLARATION = build_declaration()

register_export_declaration(DECLARATION, replace=True)


__all__ = [
    "CFG_ARITY",
    "COND_LEN",
    "DECLARATION",
    "FAMILY",
    "LATENT_ROWS",
    "PIXEL_ROWS",
    "TOKEN_ROWS",
    "VAE_SCALE",
    "build_declaration",
]
