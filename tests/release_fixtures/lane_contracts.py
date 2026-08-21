"""The release fixtures' lane — a REAL tensor-layout **v2** stamp pair.

pgw#1621: a lane is no longer an imported tensorfs v1 ``Contract`` OBJECT. It
is the ``(topology, quant)`` PAIR, both halves ratified documents in the
vendored ``spec/v2`` corpus, rendered ``"<topology>+<quant>"``.

This file used to author its OWN document — a hand-written
``tiny.diffusers-fp32@1`` contract, because v1 let anyone mint one from a JSON
string. **That is deliberately impossible now**, and the impossibility is the
point of the cut: a topology is EXTRACTED MECHANICALLY from a reference
checkpoint's headers and a quant rule is a RATIFIED DOCUMENT, so a fixture
cannot invent either. It names a real pair instead.

``plain.f32@1`` is the honest rule for these fixtures — the tiny pipeline
really is fp32 on CPU — and it is the one ratified rule whose
``capability_floor_sm`` is 0, so a fixture lane still derives NO placement row
(``model_requires() == {}``), which several release tests assert.
"""

from __future__ import annotations

#: The pair, in the author's spelling. `parse_lane_stamp` accepts the tuple and
#: the rendered string alike; the tuple is what a real header writes.
TINY_DIFFUSERS_FP32 = ("sd15.diffusers@1", "plain.f32@1")

#: The wire rendering, for tests that assert the derived document's lane key.
TINY_DIFFUSERS_FP32_ID = "sd15.diffusers@1+plain.f32@1"
