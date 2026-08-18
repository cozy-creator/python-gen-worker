"""The release fixtures' lane contract — a REAL tensorfs contract document.

A lane IS a tensorfs layout contract: an imported OBJECT carrying the layout,
its stamp and its load dtype; never a bare string, and never a handle-shaped
stand-in (Paul's contract-objects ruling).

pgw#1391 REPLACED THE STAND-IN THIS FILE USED TO BE. It was
``LaneRef("tiny.diffusers-fp32@1", dtype=torch.float32)`` — a handle plus a
dtype and NO LAYOUT, which is the exact shape the release path now refuses:
the whole point of contract objects is that the layout TRAVELS in the release
metadata, and a stamp with nothing behind it is a claim the hub cannot intern.
The fixture asserted the release document's lane rows, so it was asserting the
bug.

It is a NAMED document rather than an anonymous custom because these tests pin
the ``"contract": "tiny.diffusers-fp32@1"`` spelling in the derive document —
the named-stamp path is what they exist to cover. The name is not in the
curated library and makes no promise; the digest is real either way.
"""

from __future__ import annotations

import json

from gen_worker._vendor.tensorfs.contract import Contract

#: A minimal but HONEST layout for the tiny fixture pipeline: real roles, real
#: patterns, a real declared load dtype.
TINY_DIFFUSERS_FP32 = Contract.from_document(
    json.dumps(
        {
            "format": "tensorfs-contract-v1",
            "name": "tiny.diffusers-fp32",
            "version": 1,
            "description": "release-fixture layout: the tiny diffusers pipeline in fp32",
            "dtype": "float32",
            "tensors": [
                {
                    "role": "unet.down_blocks.{i}.resnets.{i}.conv1",
                    "pattern": "unet.down_blocks.{i}.resnets.{i}.conv1.weight",
                    "rank": 4,
                },
                {
                    "role": "unet.conv_out",
                    "pattern": "unet.conv_out.weight",
                    "rank": 4,
                },
                {
                    "role": "vae.decoder.conv_out",
                    "pattern": "vae.decoder.conv_out.weight",
                    "rank": 4,
                    "required": False,
                },
            ],
        }
    )
)
