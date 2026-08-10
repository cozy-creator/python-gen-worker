"""The pgw#1073 conv declaration — STATIC-ROWS, the sdxl class at micro scale.

Four static entries: 2 latent rows x the 2 CFG regimes, one target. That is
sdxl's exact declaration shape (aspect rows x cfg x static-rows) with the row
count cut to the smallest set that still produces MULTIPLE static entries per
cell — the artifact-identity surface pgw#1058 broke on (entry labels are
per-row static facts under this strategy, and label/ask drift admits nothing).

What this member declares that no other micro member does:

* ``shape_strategy="static-rows"`` — forced, not chosen: the denoiser is
  conv-bearing, and #730 ratified static-rows for conv-bearing graphs
  (symbolic latent H/W turns off inductor's channels-last layout opt on the
  convs, +7.2% measured on sdxl).
* ``dtype="int64"`` on ``timestep`` — the mixed-dtype signature (wan-2.2's
  shape). dtype is a REQUIRED per-input declared fact since pgw#1058; this
  keeps the integer half of that axis load-bearing on every cycle.
* Plain 4-D inputs, no containers — the container seam is ``micro``'s job;
  one seam per member keeps a red diagnosable.
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

FAMILY = "micro-conv"

#: Two latent rows -> with the cfg fork, FOUR static entries. Enough to make
#: "multiple entries per cell" real; few enough that the cycle stays seconds.
LATENT_ROWS = (24, 32)
VAE_SCALE = 8
PIXEL_ROWS = tuple((n * VAE_SCALE, n * VAE_SCALE) for n in LATENT_ROWS)
COND_LEN = 16
CFG_ARITY = 2


def _classes() -> tuple:
    rows = []
    for n in LATENT_ROWS:
        for cfg, batch in ((True, CFG_ARITY), (False, 1)):
            rows.append(GraphClass(
                dims={"B": batch, "H_lat": n, "W_lat": n},
                fork={"cfg": cfg}))
    return tuple(dict.fromkeys(rows))


def build_declaration() -> Compile:
    return Compile(
        family=FAMILY,
        targets=("unet",),
        shapes=PIXEL_ROWS,
        text_len=COND_LEN,
        dims=(
            Dim("B", carried_by=(("sample", 0), ("timestep", 0),
                                 ("cond", 0))),
            # One downsampling stage => latent H/W are multiples of 2.
            Dim("H_lat", carried_by=(("sample", 2),), multiple_of=2),
            Dim("W_lat", carried_by=(("sample", 3),), multiple_of=2),
        ),
        forks=(
            Fork("cfg", served=(True, False),
                 why="CFG is ONE batch-2 forward, turbo pins the batch-1 "
                     "graph — sdxl's fork, kept so static-rows x fork "
                     "(the 36-entry generator) stays under test at 4"),
        ),
        classes=_classes(),
        inputs=(
            Input("sample",
                  shape=("B", ("config", "in_channels"), "H_lat", "W_lat"),
                  dtype="float32"),
            # INT64, and structural: it indexes an nn.Embedding. The
            # mixed-dtype signature is this member's reason to exist.
            Input("timestep", shape=("B",), dtype="int64"),
            Input("cond", shape=("B", COND_LEN, ("config", "cond_dim")),
                  dtype="float32"),
        ),
        args=(),
        shape_strategy="static-rows",
        warm_changes_key=False,
    )


DECLARATION = build_declaration()

register_export_declaration(DECLARATION, replace=True)

__all__ = ["CFG_ARITY", "COND_LEN", "DECLARATION", "FAMILY", "LATENT_ROWS",
           "PIXEL_ROWS", "VAE_SCALE", "build_declaration"]
