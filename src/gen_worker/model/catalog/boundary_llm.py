"""The LLM/VLM boundary models: four EAGER declarations, permanently eager.

pgw#1346 B5. Each of these four endpoints serves through something that is not
a traceable PyTorch composition:

======================  ==========================================  ===========
model                   what actually serves it                     tier
======================  ==========================================  ===========
``qwen36_35b_a3b``      ``vllm serve`` — a separate OS process       external
``qwen36_27b_mtp``      ``llama-server`` — llama.cpp, a C++ binary   external
``internvl_u``          a vendored ``internvlu`` runtime, self-loading
``joycaption``          ``JoyCaptionPipeline``, non-introspectable
======================  ==========================================  ===========

Paul's F3 ruling (DESIGN-RULINGS, 2026-08-17) settles what pgw#1326 left
ambiguous for exactly these four: *"for non-pytorch-based runtimes obviously we
cannot compile them using torchcg lol. And that's perfectly fine."* They are
eager-PERMANENT citizens of the SAME ``models={...}`` surface — no graph
classes, no recipe, no compile rows, and no obligation to acquire any.

So each declaration below states precisely what the retired ``Slot`` stated and
nothing more: the family handle, the tuned-value SCHEMA where the endpoint has
an inference vocabulary, and the layout axes — including the two A19
``layouts_undeclarable`` reasons, which are the entire reason this migration
cannot be done by dropping the field. ``ModelSpec``'s only alternative is
``DEFAULT_LAYOUT = "bf16"``, and declaring bf16 for vLLM's compressed-tensors
fp8 or for llama.cpp's GGUF bytes would be a promise the loader cannot keep.

**No ``build=``.** A ``build`` callable is what the mint traces; these runtimes
load themselves from a snapshot path, which is a SERVING fact and belongs to
the endpoint's ``setup()``. Declaring a constructor here that nothing traces
would be a graph pretense in a tier defined by not having one.

This module is import-cheap by construction — msgspec structs and the SDK's own
declaration types, no model library on any path.
"""

from __future__ import annotations

from typing import Final

from ..spec import ModelSpec, TunedValues


class Qwen36A3bTuned(TunedValues, frozen=True):
    """The vLLM sampling recipe, migrated from ``qwen3.6-35b-a3b``'s
    ``@family("qwen3.6-35b-a3b") QwenA3bDefaults``.

    Field names are the WIRE names (pgw#654 gap #4): ``RuntimeFormula``
    resolves its terms by same-named lookup, payload over the stamped values,
    so ``max_tokens`` must keep this spelling for ``a + b*max_tokens`` to
    evaluate.
    """

    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95


class Qwen36MtpTuned(TunedValues, frozen=True):
    """``qwen3.6-27b-mtp-gguf``'s sampling recipe. The schema defaults are
    Qwen3.6's own recommended sampling — the hub-less fallback."""

    max_tokens: int = 256
    temperature: float = 0.6
    top_p: float = 0.95


class InternvlUTuned(TunedValues, frozen=True):
    """InternVL-U's one tunable. ``num_inference_steps`` carries the WIRE
    spelling so ``RuntimeFormula("a + b*num_inference_steps")`` resolves it."""

    num_inference_steps: int = 20


#: Qwen3.6-35B-A3B, served by vLLM.
#:
#: ``layouts_undeclarable`` migrated BY VALUE from the endpoint's ``Slot``: the
#: A19 gate refuses a model that names neither a contract nor a reason, and the
#: reason here is a real one rather than a formality.
QWEN36_35B_A3B: Final = ModelSpec(
    name="qwen36_35b_a3b",
    tuned=Qwen36A3bTuned,
    layouts_undeclarable=(
        "vLLM compressed-tensors fp8 has no registered quant descriptor; "
        "inventing one is the failure mode, not the fix"
    ),
)

#: Qwen3.6-27B-MTP, served by llama-server from GGUF files.
#:
#: The K2 finding this row exists for: ``gguf.native@1`` is a TOPOLOGY handle
#: and this axis is QUANT, which has no GGUF entry at all (th#1809 T3). Naming
#: the topology handle would answer a different question than the one the axis
#: asks, and ``DEFAULT_LAYOUT`` would answer it wrongly.
QWEN36_27B_MTP: Final = ModelSpec(
    name="qwen36_27b_mtp",
    tuned=Qwen36MtpTuned,
    layouts_undeclarable=(
        "GGUF: `gguf.native@1` is a TOPOLOGY handle and this axis is QUANT, "
        "which has no GGUF entry (th#1809 T3)"
    ),
)

#: InternVL-U. The runtime is vendored into the worker image and loads itself
#: from the snapshot path, so there is no importable class to derive a tree
#: from — but the BYTES are ordinary bf16 and the endpoint says so.
INTERNVL_U: Final = ModelSpec(
    name="internvl_u",
    tuned=InternvlUTuned,
    layouts={"*": ("plain.bf16@1",)},
)

#: JoyCaption. No tuned schema, and that is the honest state rather than an
#: omission: the endpoint declares no ``@family`` and reads no ``ctx.defaults``,
#: so there are no catalog-stamped values for a schema to decode. K8 therefore
#: does not apply — ``_register()`` publishes nothing, so no tensorhub PR is
#: owed for this name.
JOYCAPTION: Final = ModelSpec(
    name="joycaption",
    layouts={"*": ("plain.bf16@1",)},
    # ie#740's floor, BY VALUE from the endpoint's `Slot`: bf16 is the only
    # lane it serves, and 24 GB is the endpoint's own measured scalar.
    layout_requirements={"plain.bf16@1": "vram24g"},
)


__all__ = [
    "INTERNVL_U",
    "JOYCAPTION",
    "QWEN36_27B_MTP",
    "QWEN36_35B_A3B",
    "InternvlUTuned",
    "Qwen36A3bTuned",
    "Qwen36MtpTuned",
]
