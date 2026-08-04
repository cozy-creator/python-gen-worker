"""Mint CONTRACT types — the vocabulary, with none of the mint driver.

Extracted from :mod:`gen_worker.aot_mint` (pgw#868) for one measured reason:
``cell_key.compute`` keys a dynamo cell on ``code_closure``, the CONTENT digest
of the static import-graph closure of the compile entrypoints — and that walk
is an AST walk, so it follows function-level imports too. ``models.provision``
is one of those entrypoints, so anything the ARM reaches statically enters the
identity of every dynamo cell in the fleet.

The numerics gate (pgw#868) has to build a probe feed the way the MINT builds
one — that is its whole parity argument — which means the arm now reaches
``aot_declaration.declared_inputs``. Through ``aot_mint`` that dragged
``aot_wrapper_split`` and ``aot_run_impl_split`` into the closure, which
pgw#793 and pgw#811 both forbid in as many words: a compile-time transform must
re-key NOTHING. Their guards caught it, red, on the first run.

So the DECLARATION layer no longer depends on the mint DRIVER. These four names
are the whole of what ``aot_declaration`` and ``aot_inputs`` needed from
``aot_mint``; everything here is types, constants and a label, with no compile,
no export and no pool. ``aot_mint`` re-exports all four, so every existing
``from .aot_mint import ...`` keeps working.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

from .compile_cache import execution_lane_bucket, execution_lane_token


class MintRefused(RuntimeError):
    """A named, terminal refusal to produce or publish an artifact.

    Every mint failure is one of these with a reason that names the offending
    thing — a lane, a tensor, a missing declaration field. A mint that cannot
    say what went wrong is the silent-failure path the doctrine forbids.

    ``mint_phases`` carries the PARTIAL phase table of the mint that refused
    (pgw#825): the entries that did export and compile before the refusal
    spent real minutes on a real pod, and a terminus that reports only a
    wall-clock total is a measurement lost to a pod that no longer exists.
    Populated by :func:`mint`; empty for a refusal raised before any entry.
    """

    def __init__(self, *args: Any, mint_phases: Optional[Mapping[str, Any]] = None) -> None:
        super().__init__(*args)
        self.mint_phases: Dict[str, Any] = dict(mint_phases or {})


#: pgw#790: the SDK-owned fork coordinate of a LoRA-bucket family — one graph
#: class WITH the lifted adapter inputs and one WITHOUT.
#:
#: gw#627 gave every branch-capable leaf a canonical zeroed rank-bucket branch
#: so a curated attach is a buffer copy instead of a recompile. Measured
#: (WARM-INFERENCE-MATRIX §2b, 4090+5090, n=28 warm): those zeroed branches
#: cost **+31.8% / +44.9% of the compiled per-forward** and roughly DOUBLE the
#: kernel-launch count, and adapter-free traffic pays all of it to compute
#: zeros. On the hub's own record 95% of sdxl denoiser forwards name no
#: adapter, so the branch-bearing graph was the minority case wearing the
#: majority's clothes.
#:
#: This is a FORK, not a flag: both classes are exported, compiled, keyed and
#: shipped in the same cell (Paul: "worst case compile 2x more graphs, one with
#: LoRAs and one without"), and the serve path picks between them by the
#: DECLARED ingress contract. Nothing about a program varies with Python state
#: — the arm is a mint coordinate that lands in the class hash, exactly like
#: any other fork.
ADAPTER_FORK = "adapter"


@dataclass(frozen=True)
class DynamicDim:
    """One declared symbolic dimension of one input.

    ``multiple_of`` expresses a divisibility the graph genuinely requires (SDXL
    latents downsample 3x, so latent H/W must be multiples of 8) as a Dim
    factor rather than a comment. Without it export 0/1-specializes or produces
    a guard it cannot express, and the "one artifact serves every aspect ratio"
    headline quietly stops holding.

    ``dim`` is the DECLARED ``Compile.Dim`` this row came from, and it is what
    makes a multi-carrier dim expressible (pgw#812 D1). A declaration may bind
    ONE logical axis to several inputs — flux2 declares
    ``Dim("T_img", carried_by=(("hidden_states", 1), ("img_ids", 1)))``
    precisely so the edit lane cannot let ``img_ids`` specialize while
    ``hidden_states`` stays free. Every carrier of one declared dim must
    therefore share ONE torch symbol; rows that carry no declared name (the
    hand-registered builder path, where latent H and W are genuinely
    independent axes of one input) keep a symbol each.
    """

    input_name: str
    axis: int
    min: int
    max: int
    multiple_of: int = 1
    dim: str = ""

    def as_row(self) -> Dict[str, Any]:
        return {
            "input": self.input_name, "axis": self.axis,
            "min": self.min, "max": self.max, "multiple_of": self.multiple_of,
        }


@dataclass
class ExportSpec:
    """Everything the mint needs to produce one artifact.

    ``example_inputs`` is a zero-arg factory rather than tensors so a caller
    can build them on meta (tests, control-plane) or on cuda (a real mint)
    without this module deciding. ``lora_fqns`` names the adapter tensors that
    must stay dynamic; ``lifted_inputs`` names what was promoted to a graph
    input to keep them that way.
    """

    family: str
    target: str
    weight_lane: str = ""
    precision: str = "bf16"
    lora_bucket: int = 0
    shapes: Tuple[Tuple[int, ...], ...] = ()
    #: Traced batch. 0 = the family's input builder decides. Declared rather
    #: than inferred from ``guidance_scales`` because CFG batching is a FAMILY
    #: fact: sdxl runs CFG as one batch-2 forward, wan as two sequential batch-1
    #: forwards, so guidance changes wan's call COUNT and not its shape
    #: (ie#566 G2).
    batch: int = 0
    text_lens: Tuple[int, ...] = ()
    guidance_scales: Tuple[float, ...] = ()
    dynamic: Tuple[DynamicDim, ...] = ()
    #: pgw#739 declaration coordinate: the fork arm values and (for a
    #: static-rows family) the class row this artifact is minted at. Both
    #: KEY (a fork is a distinct graph class in #716's hash) — see
    #: :func:`cell_identity`. Sorted (name, value) pairs.
    fork: Tuple[Tuple[str, Any], ...] = ()
    class_dims: Tuple[Tuple[str, int], ...] = ()
    specialization: Dict[str, Any] = field(default_factory=dict)
    lora_fqns: Tuple[str, ...] = ()
    lifted_inputs: Tuple[str, ...] = ()
    strict: bool = True
    source_ref: str = ""
    source_digest: str = ""
    closure_roots: Tuple[str, ...] = ()

    def execution_lane_label(self) -> str:
        base, observed = execution_lane_bucket(self.weight_lane)
        bucket = observed or self.lora_bucket
        token = execution_lane_token(base)
        if bucket:
            return f"{token}-lora{bucket}" if token else f"lora{bucket}"
        return token


__all__ = ["ADAPTER_FORK", "DynamicDim", "ExportSpec", "MintRefused"]
