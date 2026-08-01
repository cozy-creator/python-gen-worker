"""Regional cells: a cell whose entries are BLOCK CLASSES, not shape rows.

pgw#817 (Paul-approved), implementing pgw#812's S1-S7 spec.

A DiT — and SDXL's UNet — is one block repeated N times, and a whole-graph
mint traces, lowers, codegens and g++-compiles all N. Compiling ONE block per
class and reusing it across every instance is upstream's own AoT recipe, and
pgw#812 measured it on our path: **19.4 s vs 274.7 s on the real sdxl w8a8
mint (14.2x)**, serve parity +0.24%, artifact 4.7 MB vs 18.2 MB, and numerics
CLEANER than whole-graph on fp8 (cos 0.989-0.993 against the pgw#814
whole-graph degradation).

What this module owns:

* **Block discovery** (:func:`repeated_block_groups`) — generic, from the
  module's own ``_repeated_blocks`` plus a parameter-shape fingerprint. No
  per-family knowledge, and two blocks of one class at different widths are
  correctly different artifacts.
* **The shell digest** (:func:`shell_digest`) — S3's load-bearing identity
  change. A regional artifact describes a PART of the model, so two models
  with identical blocks and different shells would otherwise collide.
* **Per-instance, bind-by-reference arming** (:class:`RegionalArm`) — S4.
  All-or-nothing per target: a model with 24 of 25 blocks armed is a silently
  half-eager model.
* **The adoption numerics gate** — S6, on the shared ladder
  (:mod:`gen_worker.numerics_ladder`). The assembled model must reproduce the
  eager forward it replaces, or the cell REFUSES to arm, typed, with the
  verdict on the wire.

What it deliberately does NOT own: the shell stays EAGER (S2). Exporting the
shell with the blocks elided is not expressible in ``torch.export`` today —
blocks are inlined at trace time — so the compiled fraction of the model
equals the repeated-block fraction, and that is the honest bound on the win.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from . import activity as activity_mod
from . import aot_serve
from . import numerics_ladder
from .activity import KIND_CELL_NUMERICS

logger = logging.getLogger(__name__)

#: The entry-coordinate key a regional entry is named by. S1: the cell is
#: still ONE ``.pt2`` and the entry grammar is unchanged — only the AXIS
#: inverts, from shape coordinates of one whole target to block classes of it.
#: ``aot_declaration.entry_name`` renders this as
#: ``unet/block=BasicTransformerBlock#0``.
BLOCK_FORK = "block"

#: ``cell_key`` mode axis value for a regional cell. The axis already exists
#: and already feeds the digest; pgw#812 D4 found ``cell_identity`` hardcoding
#: ``""`` under the comment "an exported cell is always whole-graph: regional
#: is a dynamo partitioning strategy with no export counterpart". That comment
#: is falsified by measurement — an exported regional cell exists, is 1.37 MB,
#: and serves.
MODE_REGIONAL = "regional"


# ---------------------------------------------------------------------------
# The per-family numerics tolerance (S6 / pgw#814's owed calibration)
# ---------------------------------------------------------------------------

#: Cosine floor for an ASSEMBLED-vs-EAGER comparison — below it the cell is
#: DESTROYED and refuses to arm.
#:
#: Derived from pgw#814's measured band on the production toolchain (torch
#: 2.13.0+cu130, L4/sm_89), NOT inherited from pgw#800's adapter floors —
#: pgw#814 says in as many words that those are calibrated for adapter deltas
#: and must not be assumed here:
#:
#:   worst configuration we ACCEPT   0.9890  flux2 w8a8 pertensor REGIONAL vs
#:                                           eager at T_img=4096 (0.9926 at
#:                                           8160). pgw#812's good arm — the
#:                                           one whose numerics are 4-8x
#:                                           closer than whole-graph's.
#:   best configuration we REFUSE    0.9730  flux2 w8a8 ROWWISE whole-graph vs
#:                                           eager — pgw#814's "do not adopt a
#:                                           flux2 w8a8 cell until this
#:                                           closes", i.e. the artifact the
#:                                           platform decided is not servable.
#:
#: 0.98 is that band's geometric midpoint (sqrt(0.9890 * 0.9730) = 0.98097),
#: 1.0092x of headroom below the worst accepted case and 1.0072x above the
#: best refused one. The healthy population sits far above it: bf16 control
#: 0.99979, sdxl w8a8 whole-graph 0.99984.
NUMERICS_FLOOR = 0.98

#: Gray-band ceiling — at or above this the arm is silent, below it the cell
#: arms and confesses ``cell_numerics phase=degraded``.
#:
#: Every configuration anyone has called healthy measures 0.9998+ (bf16
#: control 0.99979, sdxl w8a8 whole-graph 0.99984), so an artifact that has
#: lost more than 0.1% of the output's DIRECTION has lost it to something —
#: fp8 accumulation drift, a fused reassociation — and that is worth counting
#: fleet-wide even when it is served. flux2 w8a8 regional (0.989-0.993) lands
#: here: served, known, and on the wire.
NUMERICS_WARN = 0.999

#: Magnitude band. Cosine is scale-invariant, so an artifact that reproduces
#: eager's direction exactly at 0.9x the magnitude scores a PERFECT cosine
#: while serving a systematically dimmer image; nothing in pgw#800's ladder
#: could see that, because an adapter's retention is evidence rather than a
#: bound (a destroyed one measures 15.3).
#:
#: Derived from the same measured band: worst accepted 0.997 (bf16 control),
#: best refused 0.905 (flux2 w8a8 pertensor whole-graph; rowwise 0.902).
#: sqrt(0.997 * 0.905) = 0.9500. Applied symmetrically in the log, so
#: retention outside [0.95, 1.0526] is at least DEGRADED.
NUMERICS_RETENTION_FLOOR = 0.95

DEFAULT_THRESHOLDS = numerics_ladder.Thresholds(
    floor=NUMERICS_FLOOR, warn=NUMERICS_WARN,
    retention_floor=NUMERICS_RETENTION_FLOOR,
    label="assembled-vs-eager (pgw#814 §VERDICT)")


def declared_thresholds(cfg: Any) -> numerics_ladder.Thresholds:
    """The tolerance THIS family declares, falling back to the SDK default.

    S6 requires a declared per-family tolerance because bf16 attention
    reassociation makes exact equality the wrong bar and the right bar is not
    the same for a 25-block fp8 DiT and a conv-bearing UNet. A family that
    declares nothing gets :data:`DEFAULT_THRESHOLDS`, whose derivation is the
    measured band above — a default with evidence, not a guess.
    """
    floor = getattr(cfg, "numerics_floor", None)
    warn = getattr(cfg, "numerics_warn", None)
    if floor is None and warn is None:
        return DEFAULT_THRESHOLDS
    return numerics_ladder.Thresholds(
        floor=float(NUMERICS_FLOOR if floor is None else floor),
        warn=float(NUMERICS_WARN if warn is None else warn),
        retention_floor=NUMERICS_RETENTION_FLOOR,
        label=f"assembled-vs-eager (declared by {getattr(cfg, 'family', '?')})")


class RegionalArmRefused(RuntimeError):
    """A regional cell must not arm on this pipeline.

    Carries ``comparison`` when the refusal came from the numerics gate, so
    the whole verdict — aggregate cosine, magnitude ratio, and the worst
    per-output rows — is reachable from the exception rather than only from
    the activity record.
    """

    def __init__(
        self, message: str, *,
        comparison: Optional[numerics_ladder.Comparison] = None,
        reason: str = "",
    ) -> None:
        super().__init__(message)
        self.comparison = comparison
        self.reason = reason or "regional_arm_refused"


# ---------------------------------------------------------------------------
# Block discovery
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BlockGroup:
    """Every instance of one repeated class that can SHARE one artifact."""

    #: ``<ClassName>#<ordinal>`` — the entry coordinate value.
    key: str
    class_name: str
    #: The live modules, in ``model.modules()`` order (which is declaration
    #: order for a ``ModuleList``), so instance ``i`` is stable across runs.
    instances: Tuple[Any, ...]
    #: Ordered ``(name, shape)`` fingerprint that decided the grouping.
    fingerprint: Tuple[Tuple[str, Tuple[int, ...]], ...] = ()

    @property
    def count(self) -> int:
        return len(self.instances)

    @property
    def prototype(self) -> Any:
        return self.instances[0]


def declared_repeated_classes(module: Any) -> Tuple[str, ...]:
    """The class names the MODULE ITSELF declares as repeated.

    ``_repeated_blocks`` is diffusers' own class attribute — the same fact
    ``compile_repeated_blocks`` reads. Taking it from the module keeps this
    declaration-free and R8-clean: nothing here is per-family knowledge and no
    endpoint has to restate what the model already says about itself.
    """
    return tuple(str(n) for n in (
        getattr(type(module), "_repeated_blocks", None) or ()))


def repeated_block_groups(module: Any) -> Tuple[BlockGroup, ...]:
    """Group every declared repeated block by what lets them share an artifact.

    The grouping key is the ordered parameter-SHAPE fingerprint, not the class
    name: two blocks of one class at different widths compile to different
    kernels and must be different entries. Ordinals are assigned in first-seen
    order so the entry names are stable across mints of the same model.
    """
    names = set(declared_repeated_classes(module))
    if not names:
        return ()
    order: List[Tuple[str, Tuple[Tuple[str, Tuple[int, ...]], ...]]] = []
    buckets: Dict[Tuple[str, Tuple[Tuple[str, Tuple[int, ...]], ...]], List[Any]] = {}
    counters: Dict[str, int] = {}
    for mod in module.modules():
        cls = type(mod).__name__
        if cls not in names:
            continue
        fp = (cls, tuple(
            (str(n), tuple(int(v) for v in p.shape))
            for n, p in sorted(mod.named_parameters(), key=lambda kv: kv[0])))
        if fp not in buckets:
            buckets[fp] = []
            order.append(fp)
        buckets[fp].append(mod)
    groups: List[BlockGroup] = []
    for fp in order:
        cls = fp[0]
        ordinal = counters.get(cls, 0)
        counters[cls] = ordinal + 1
        groups.append(BlockGroup(
            key=f"{cls}#{ordinal}", class_name=cls,
            instances=tuple(buckets[fp]), fingerprint=fp[1]))
    return tuple(groups)


def block_entry_fork(group_key: str) -> Tuple[Tuple[str, Any], ...]:
    """The fork coordinate one block entry is named by (S1)."""
    return ((BLOCK_FORK, str(group_key)),)


def capture_block_feeds(
    groups: Sequence[BlockGroup], run: Any,
) -> Dict[str, Tuple[Tuple[Any, ...], Dict[str, Any]]]:
    """Record the real ``(args, kwargs)`` each block class is CALLED with.

    A block's inputs are internal and captured, never declared (S5): they are
    whatever the shell hands it. ``run`` is a zero-arg callable that performs
    one forward of the assembled model on the declared seed feed; hooks on the
    first instance of each group record its arguments and nothing else runs.
    """
    seen: Dict[str, Tuple[Tuple[Any, ...], Dict[str, Any]]] = {}
    handles = []

    def hook_for(key: str) -> Any:
        def hook(_mod: Any, args: Any, kwargs: Any) -> None:
            seen.setdefault(key, (tuple(args), dict(kwargs)))
        return hook

    try:
        for group in groups:
            handles.append(group.prototype.register_forward_pre_hook(
                hook_for(group.key), with_kwargs=True))
        run()
    finally:
        for handle in handles:
            handle.remove()
    missing = [g.key for g in groups if g.key not in seen]
    if missing:
        raise RegionalArmRefused(
            f"block class(es) {missing!r} were never called during the seed "
            f"forward — a class the shell does not reach cannot be exported "
            f"from a capture, and minting it from a guessed feed is the "
            f"failure this refuses",
            reason="block_never_called")
    return seen


#: One captured feed reduced to shapes: the block's own positional parameter
#: names, and per slot either the tensor's shape or ``None`` for a slot that
#: carries no shape (a scalar, a flag, a default).
BlockFeedShape = Tuple[Tuple[str, ...], Tuple[Optional[Tuple[int, ...]], ...]]


def capture_block_feed_shapes(
    groups: Sequence[BlockGroup], run: Any,
) -> Dict[str, BlockFeedShape]:
    """The SHAPES each block class is called with, retaining no tensors.

    pgw#829's derivation input. A block's inputs are internal (S5), so the
    only honest way to learn which of its axes VARY across a plan's class
    rows is to run the shell on each row and look. This is the cheap half of
    :func:`capture_block_feeds`: it reduces each captured argument to its
    shape inside the hook and lets the activations die with the forward, so
    observing nine aspect rows costs nine eager forwards and nine small
    tuples rather than nine retained activation sets.

    Slots are in the block's OWN signature order (:func:`positional_feed`),
    which is the order the export feed and the serve marshal both use — a
    shape vector indexed any other way could not be compared against them.
    """
    seen: Dict[str, BlockFeedShape] = {}
    handles = []

    def hook_for(key: str) -> Any:
        def hook(mod: Any, args: Any, kwargs: Any) -> None:
            if key in seen:
                return
            values, names = positional_feed(mod, tuple(args), dict(kwargs))
            shapes: List[Optional[Tuple[int, ...]]] = []
            for value in values:
                shape = getattr(value, "shape", None)
                shapes.append(
                    None if shape is None else tuple(int(d) for d in shape))
            seen[key] = (tuple(names), tuple(shapes))
        return hook

    try:
        for group in groups:
            handles.append(group.prototype.register_forward_pre_hook(
                hook_for(group.key), with_kwargs=True))
        run()
    finally:
        for handle in handles:
            handle.remove()
    missing = [g.key for g in groups if g.key not in seen]
    if missing:
        raise RegionalArmRefused(
            f"block class(es) {missing!r} were never called during the shape "
            f"probe — a class the shell does not reach on every declared "
            f"class row cannot have its varying axes derived",
            reason="block_never_called")
    return seen


def block_has_conv(block: Any) -> bool:
    """True when this block class carries a convolution (pgw#829's premise
    guard).

    #730's ``static-rows`` verdict is a statement about CONVS:
    ``decide_layout_opt`` bails out on a graph that has a conv AND a free
    symbol, so a dynamic dim costs a conv-bearing graph the channels-last
    layout optimization (+7.8% measured whole-graph). A conv-free region has
    no layout opt to lose, which is why pgw#812 measured dynamic inner dims
    at **0.0%** on exactly the region sdxl regionalizes.

    So the collapse is decided PER BLOCK CLASS off the live module, never
    fleet-wide and never from the family name: a conv-bearing block class
    keeps one static entry per class row, and only the conv-free ones
    collapse. sdxl's ``BasicTransformerBlock`` is conv-free; the convs live
    in the shell, which stays eager.
    """
    from torch.nn.modules.conv import _ConvNd

    return any(isinstance(mod, _ConvNd) for mod in block.modules())


# ---------------------------------------------------------------------------
# S3.3 — the shell digest
# ---------------------------------------------------------------------------


def _config_facts(module: Any) -> Dict[str, Any]:
    """The module's own resolved config, JSON-canonicalisable.

    Read off the RESOLVED module (``.config``, diffusers' ``FrozenDict``),
    never from a declaration — the shell digest must describe what is loaded,
    not what an endpoint said it would load.
    """
    cfg = getattr(module, "config", None)
    if cfg is None:
        return {}
    try:
        items = dict(cfg)
    except Exception:  # noqa: BLE001 — a non-mapping config contributes nothing
        return {}
    out: Dict[str, Any] = {}
    for key in sorted(items, key=str):
        name = str(key)
        if name.startswith("_"):
            continue
        value = items[key]
        if isinstance(value, (str, int, float, bool)) or value is None:
            out[name] = value
        elif isinstance(value, (list, tuple)):
            out[name] = [
                v if isinstance(v, (str, int, float, bool)) or v is None
                else str(v) for v in value]
        else:
            out[name] = str(value)
    return out


def _diffusers_version() -> str:
    try:
        from importlib.metadata import version

        return str(version("diffusers"))
    except Exception:  # noqa: BLE001
        return ""


def shell_facts(module: Any) -> Dict[str, Any]:
    """The facts a regional cell's SHELL is identified by (pgw#812 S3.3).

    Today ``combined_graph_hash`` is a proxy for "the graph the fleet serves".
    Regionally it describes a PART, so two models with identical blocks and a
    different shell — a different ``num_layers``, a different rope
    construction, a diffusers minor that rewrites the outer forward — would
    produce the SAME key while serving different math. Without this, regional
    trades compile time for a cache-poisoning class the platform does not have
    today.

    Deliberately NOT the shell's FX graph: ``graph_hash.py`` (the module that
    canonicalises FX structure) is unwired, and building this on it would make
    regional depend on closing ck6 first. These four facts are cheap, read off
    the resolved module, and cover every drift anyone has named.
    """
    groups = repeated_block_groups(module)
    return {
        "module": f"{type(module).__module__}.{type(module).__name__}",
        "blocks": [[g.class_name, g.count] for g in groups],
        "config": _config_facts(module),
        "diffusers": _diffusers_version(),
    }


def shell_digest(module: Any) -> str:
    """16-hex digest of :func:`shell_facts` — MANDATORY in a regional cell's
    contract facts."""
    encoded = json.dumps(
        shell_facts(module), sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


# ---------------------------------------------------------------------------
# S4 — per-instance, bind-by-reference, all-or-nothing arming
# ---------------------------------------------------------------------------


def positional_feed(
    block: Any, args: Sequence[Any], kwargs: Mapping[str, Any],
) -> Tuple[List[Any], List[str]]:
    """``(values, names)`` for one block call, in the block's OWN signature
    order, with trailing ``None`` defaults dropped.

    All-positional feeds are a mint obligation (pgw#723 pod 9): an AOTI
    package's call convention mirrors the traced args/kwargs split and the
    serve marshal is positional, so a kwarg-traced package arms and then
    silently revokes to eager on its first call. This is the one place the
    order is decided, so the mint's feed and the shim's call cannot drift.
    """
    import inspect

    signature = inspect.signature(type(block).forward)
    names = [n for n in signature.parameters if n != "self"]
    values = list(args)
    for name in names[len(args):]:
        param = signature.parameters[name]
        values.append(kwargs[name] if name in kwargs else param.default)
    while values and values[-1] is None:
        values.pop()
    return values, names[:len(values)]


class BlockShim:
    """``instance.forward = BlockShim(runner, instance)`` — the direct form.

    pytorch#156206 measured the export/save/load/UNFLATTEN path 1.95x SLOWER
    than direct submodule replacement, so the artifact is called in place of
    the block's forward and nothing re-enters torch's unflattener at serve
    time.

    Kwargs are marshalled into the TRACED positional order, which is the same
    obligation ``aot_serve.marshal_positional`` carries for a whole-graph
    entry: a package traced from positional feeds is uncallable by keyword and
    fails only at first serve.
    """

    def __init__(
        self, runner: Any, block: Any, outputs: int = 1,
        *, family: str = "", target: str = "",
    ) -> None:
        self.runner = runner
        self._block = block
        self.outputs = int(outputs)
        self.calls = 0
        #: The same state dict ``aot_serve.wrap_module`` writes, so the
        #: pipeline marker a regional arm publishes is read by the SAME
        #: ``is_armed``/``execution_count``/``proven_since``/``unwrap``
        #: helpers the whole-graph arm's marker is. A second reporting shape
        #: would mean the scheduler learns about one lane and not the other.
        self.state: Dict[str, Any] = {
            "failed": False,
            "successful_calls": 0,
            "ingress_refusals": 0,
            "last_refusal": "",
            "original": type(block).forward.__get__(block),
            "attr": "forward",
            "target": target or family,
            "family": family,
            "failure_callback": None,
            "refusal_callback": None,
            "revocation_error": "",
            "runner": runner,
        }

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        state = self.state
        eager = state["original"]
        if state["revocation_error"]:
            raise aot_serve.CompiledLaneUnavailableError(
                state["revocation_error"])
        if state["failed"]:
            return eager(*args, **kwargs)
        try:
            values, _names = positional_feed(self._block, args, kwargs)
            out = self.runner(*values)
        except aot_serve.IngressContractError as exc:
            # A per-request contract refusal, exactly as on the whole-graph
            # lane: named, counted, served eager, artifact stays armed.
            state["ingress_refusals"] = int(state["ingress_refusals"]) + 1
            state["last_refusal"] = f"{exc.reason}: {exc}"
            logger.warning(
                "aot-regional: %s REFUSED out-of-contract block input "
                "(%s: %s); serving this call eager, artifact stays armed",
                state["target"], exc.reason, exc)
            activity_mod.emit_event(
                "aot_ingress_refused",
                f"family={state['family']} target={state['target']} "
                f"mode={MODE_REGIONAL}: {exc}",
                phase=exc.reason,
            )
            aot_serve.report_ingress_refusal(state, exc.reason, str(exc))
            return eager(*args, **kwargs)
        except aot_serve.ConstantsUnboundError as exc:
            state["failed"] = True
            logger.error(
                "aot-regional: %s invoked with unbound constants (%s); eager "
                "for the rest of this process", state["target"], exc)
            activity_mod.emit_event(
                "aot_constants_unbound",
                f"family={state['family']} target={state['target']} "
                f"mode={MODE_REGIONAL}: {exc}",
                phase=exc.reason,
            )
            aot_serve._revoke(state, f"constants unbound: {exc}")
            return eager(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 — ANY artifact problem => eager
            state["failed"] = True
            aot_serve._revoke(
                state,
                f"AOTI regional block {state['target']} failed: "
                f"{type(exc).__name__}: {exc}")
            logger.warning(
                "aot-regional: %s failed (%s: %s); eager for the rest of this "
                "process", state["target"], type(exc).__name__, exc)
            return eager(*args, **kwargs)
        self.calls += 1
        state["successful_calls"] = int(getattr(self.runner, "calls", 0))
        if self.outputs == 1 and isinstance(out, (list, tuple)) and len(out) == 1:
            return out[0]
        return tuple(out) if isinstance(out, list) else out


@dataclass
class RegionalArm:
    """One target's regional arm — every instance of every block class.

    Arming is ALL-OR-NOTHING per target (S4). A model with 24 of 25 blocks
    armed is a silently half-eager model, which is the failure class the
    fail-soft doctrine already forbids: it serves, it is slower than either
    pure lane, and nothing reports it. Any per-instance failure reverts every
    instance and refuses by name.
    """

    target: str
    family: str = ""
    cell_key: str = ""
    shims: List[Tuple[Any, BlockShim]] = field(default_factory=list)
    bound_instances: int = 0
    weight_copies: int = 0
    armed: bool = False
    comparison: Optional[numerics_ladder.Comparison] = None

    def revert(self) -> None:
        """Restore every instance's eager forward. Idempotent."""
        for module, _shim in self.shims:
            try:
                del module.forward
            except AttributeError:
                pass
        self.shims = []
        self.armed = False

    def instances(self) -> int:
        return len(self.shims)


def arm_blocks(
    groups: Sequence[BlockGroup],
    runner_for: Any,
    *,
    target: str,
    family: str = "",
    cell_key: str = "",
    outputs_for: Optional[Any] = None,
    literals: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> RegionalArm:
    """Bind and install ONE runner per block INSTANCE, by reference.

    ``runner_for(group_key)`` returns a fresh, unbound
    :class:`aot_serve.ArtifactRunner` for that entry — one per instance,
    because each instance binds its own weights.

    Two things are load-bearing here and both come from pgw#812 D3/S4:

    * **``user_managed=True``.** With the copying bind, N instances mean N
      copies of the block weights in VRAM — for flux2, a second whole model.
      The values come from the instance's own resident ``state_dict``, so the
      pipeline keeps them alive by construction.
    * **The gate runs before the FIRST call of EVERY instance.** The
      unbound-call segfault surface multiplies by N; ``assert_ready`` is what
      keeps it unreachable, so it is per instance, not per cell.
    """
    arm = RegionalArm(target=str(target), family=str(family),
                      cell_key=str(cell_key))
    try:
        for group in groups:
            for module in group.instances:
                runner = runner_for(group.key)
                # pgw#825: NOT `state_dict()` — it omits the non-persistent
                # branch buffers the block's own constant table declares.
                # pgw#827: and the module is the BLOCK INSTANCE, never the
                # target. A regional entry's declared FQNs are block-relative
                # (`attn1.to_k.weight`), so a table built at denoiser scope
                # resolves NONE of them — which is the whole defect.
                state = aot_serve.resident_constants(module)
                runner.bind(state, literals or {}, user_managed=True)
                # S4: per INSTANCE, before its first call.
                runner.assert_ready()
                if not getattr(runner, "user_managed", False):
                    raise RegionalArmRefused(
                        f"entry {group.key!r} bound by COPY on instance "
                        f"{arm.bound_instances}: {len(group.instances)} "
                        f"instances would mean {len(group.instances)} copies "
                        f"of this block's weights in VRAM",
                        reason="bind_copied")
                shim = BlockShim(
                    runner, module,
                    outputs=1 if outputs_for is None else int(
                        outputs_for(group.key)),
                    family=family, target=f"{target}[{group.key}]")
                module.forward = shim
                arm.shims.append((module, shim))
                arm.bound_instances += 1
    except RegionalArmRefused:
        arm.revert()
        raise
    except Exception as exc:  # noqa: BLE001 — partial arm is never served
        arm.revert()
        raise RegionalArmRefused(
            f"target {target!r}: instance {arm.bound_instances} of "
            f"{sum(g.count for g in groups)} failed to arm "
            f"({type(exc).__name__}: {exc}) — every instance reverted to "
            f"eager, because a partly-armed model is a silently half-eager "
            f"model",
            reason="partial_arm") from exc
    expected = sum(g.count for g in groups)
    if arm.bound_instances != expected:
        arm.revert()
        raise RegionalArmRefused(
            f"target {target!r}: armed {arm.bound_instances} of {expected} "
            f"instances", reason="partial_arm")
    arm.armed = True
    return arm


# ---------------------------------------------------------------------------
# S6 — the adoption numerics gate
# ---------------------------------------------------------------------------


def gate_assembled(
    reference: Any,
    assembled: Any,
    *,
    thresholds: numerics_ladder.Thresholds,
    family: str = "",
    cell_key: str = "",
    mode: str = MODE_REGIONAL,
) -> numerics_ladder.Comparison:
    """Judge an ASSEMBLED forward against the EAGER forward it replaces.

    Whole-graph verify proves the artifact answers on the declared
    coordinates. Regionally, a correct block plus wrong wiring is a failure
    mode that could not previously exist — and pgw#814 showed the gate is
    needed for whole-graph cells too, where a flux2 w8a8 artifact reproduced
    eager at cosine 0.931 and nothing in the worker noticed.

    Below the floor this raises :class:`RegionalArmRefused` AND emits
    ``cell_numerics phase=refused``; in the gray band it returns and emits
    ``phase=degraded``. This is Paul's "no degradation in quality output"
    requirement made structural rather than procedural: a cell that degrades
    output cannot arm.
    """
    comparison = numerics_ladder.compare_outputs(
        reference, assembled, thresholds=thresholds,
        reference_label="eager",
        subject_label=f"{mode}:{cell_key or family or '<cell>'}")

    def _refuse(message: str, cmp_: numerics_ladder.Comparison) -> BaseException:
        return RegionalArmRefused(
            message, comparison=cmp_, reason="numerics_destroyed")

    result = numerics_ladder.gate(
        comparison, kind=KIND_CELL_NUMERICS, refuse=_refuse,
        context=f"family={family} mode={mode} key={cell_key}:")
    assert result is not None  # gate() returns None only for a None input
    return result


def arm_and_verify(
    module: Any,
    groups: Sequence[BlockGroup],
    runner_for: Any,
    run_forward: Any,
    *,
    target: str,
    thresholds: numerics_ladder.Thresholds,
    family: str = "",
    cell_key: str = "",
    outputs_for: Optional[Any] = None,
) -> RegionalArm:
    """The whole adoption sequence, in the only order that is safe.

    1. Run the EAGER forward and keep its output — the reference has to be
       taken before anything is installed, on this pod, with these weights.
    2. Arm every instance (all-or-nothing, bind-by-reference).
    3. Run the SAME feed through the assembled model.
    4. Judge it on the declared ladder. A DESTROYED verdict reverts every
       instance and refuses, so the pod serves eager and the hub is told.

    ``run_forward`` is a zero-arg callable producing the model's output on the
    declared seed coordinate; it is called exactly twice.
    """
    import torch

    with torch.no_grad():
        reference = _detach(run_forward())
    arm = arm_blocks(
        groups, runner_for, target=target, family=family, cell_key=cell_key,
        outputs_for=outputs_for)
    try:
        with torch.no_grad():
            assembled = _detach(run_forward())
        arm.comparison = gate_assembled(
            reference, assembled, thresholds=thresholds, family=family,
            cell_key=cell_key)
    except Exception:
        arm.revert()
        raise
    logger.info(
        "aot-regional: ARMED family=%s target=%s key=%s — %d instance(s) "
        "across %d block class(es), bound by reference (0 weight copies), "
        "assembled-vs-eager %s",
        family, target, cell_key, arm.bound_instances, len(groups),
        arm.comparison.evidence() if arm.comparison else "unjudged")
    activity_mod.emit_event(
        KIND_CELL_NUMERICS,
        f"family={family} target={target} key={cell_key} "
        f"instances={arm.bound_instances} classes={len(groups)}: "
        f"{arm.comparison.evidence() if arm.comparison else ''}",
        phase="armed")
    return arm


# ---------------------------------------------------------------------------
# S4, wired — the regional ADOPT path (pgw#827)
# ---------------------------------------------------------------------------


def entry_block_key(name: str, block: Mapping[str, Any]) -> str:
    """The block class one packaged entry belongs to, from its own fork.

    Read off the recorded coordinate rather than parsed out of the entry
    NAME: the name is a rendering of the fork, and re-deriving a fact from
    its rendering is how the two drift.
    """
    fork = {str(n): v for n, v in (
        tuple(row) for row in (block.get("fork") or ()))}
    return str(fork.get(BLOCK_FORK) or "")


def entry_output_count(block: Mapping[str, Any]) -> int:
    """How many tensors this block entry's traced forward returned.

    ``1`` means a BARE tensor, which the shim unwraps out of AOTI's output
    list. Taken from the program's own recorded ``out_spec`` — a guessed
    envelope is the pgw#758 "``return_dict=False`` means a 1-tuple" bug one
    level down, and it surfaces only at first serve.
    """
    spec_text = str(((block.get("graph") or {}).get("pytree") or {}).get(
        "out_spec") or "")
    if not spec_text:
        return 1
    try:
        import torch.utils._pytree as pytree

        spec = pytree.treespec_loads(spec_text)
    except Exception:  # noqa: BLE001 — an unparseable spec keeps the default
        return 1
    return 1 if spec.is_leaf() else max(1, int(spec.num_leaves))


def entry_adapter_arm(block: Mapping[str, Any]) -> Optional[bool]:
    """The adapter arm one entry's fork states, or ``None`` (pgw#790)."""
    fork = {str(n): v for n, v in (
        tuple(row) for row in (block.get("fork") or ()))}
    value = fork.get("adapter", None)
    return None if value is None else bool(value)


@dataclass
class BlockDispatch:
    """One block instance's entries, routed by the ADAPTER ARM first.

    pgw#790 discriminates the adapter fork by INGRESS: the branch-bearing
    class declares ``lora_a``/``lora_b`` and the branchless one refuses them,
    so a call either carries the pair or it does not, and exactly one class
    admits it. That works because the lift wraps the DENOISER, and the pair
    arrives in the denoiser's call.

    A regional entry is exported one block deep from the block's OWN
    signature, which never carries the pair (pgw#825) — the branch is
    module-resident. So both arms of a regional block declare the SAME
    ingress contract, both admit every call, and :class:`EntryDispatch`
    correctly refuses ``entry_ambiguous`` on every forward. The cell arms,
    reports armed, and serves 100 % eager.

    The discriminator regional actually has is the one the whole-graph lane
    reads for the same decision (:func:`aot_serve.adapter_call_kwargs` ->
    ``lora_lifted.adapter_active``): whether an adapter is CURRENTLY placed
    on the denoiser this block belongs to. It is asked of the target module,
    not the block, because that is the scope an attach has.
    """

    by_arm: Mapping[Optional[bool], aot_serve.EntryDispatch]
    adapter_owner: Any = None

    def _arms(self) -> Tuple[aot_serve.EntryDispatch, ...]:
        return tuple(self.by_arm.values())

    def bind(
        self, state_dict: Mapping[str, Any],
        literals: Mapping[str, Mapping[str, Any]],
        *, user_managed: bool = False,
    ) -> None:
        for dispatch in self._arms():
            dispatch.bind(state_dict, literals, user_managed=user_managed)

    def assert_ready(self) -> None:
        for dispatch in self._arms():
            dispatch.assert_ready()

    @property
    def user_managed(self) -> bool:
        arms = self._arms()
        return bool(arms) and all(d.user_managed for d in arms)

    def select_arm(self) -> aot_serve.EntryDispatch:
        if len(self.by_arm) == 1:
            return next(iter(self.by_arm.values()))
        from .models import lora_lifted

        active = bool(
            self.adapter_owner is not None
            and lora_lifted.adapter_active(self.adapter_owner))
        chosen = self.by_arm.get(active)
        if chosen is None:
            chosen = self.by_arm.get(None) or next(iter(self.by_arm.values()))
        return chosen

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.select_arm()(*args, **kwargs)

    @property
    def calls(self) -> int:
        return sum(d.calls for d in self._arms())

    def entry_calls(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for dispatch in self._arms():
            out.update(dispatch.entry_calls())
        return out

    def refusal_counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for dispatch in self._arms():
            for reason, count in dispatch.refusal_counts().items():
                out[reason] = out.get(reason, 0) + count
        return out

    def realignment_counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for dispatch in self._arms():
            for key, count in dispatch.realignment_counts().items():
                out[key] = out.get(key, 0) + count
        return out

    def excludes(self, names: Sequence[str]) -> bool:
        return any(d.excludes(names) for d in self._arms())


def _grouped_entries(
    entries: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Dict[str, List[Tuple[str, Mapping[str, Any]]]]]:
    """``{target: {block key: [(entry name, entry block), ...]}}``."""
    out: Dict[str, Dict[str, List[Tuple[str, Mapping[str, Any]]]]] = {}
    for name in sorted(entries):
        block = entries[name]
        key = entry_block_key(name, block)
        if not key:
            raise aot_serve.AdoptError(
                "regional_entry_unkeyed",
                f"entry {name!r} carries no {BLOCK_FORK!r} fork coordinate, "
                f"so this runtime cannot tell which block class it serves — "
                f"a regional cell whose entries are unattributable can only "
                f"be armed by guessing")
        out.setdefault(str(block.get("target") or ""), {}).setdefault(
            key, []).append((name, block))
    return out


def load_and_arm(
    pipeline: Any, cfg: Any, artifact: Path,
    cache_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Stage + verify + load + bind + install a REGIONAL cell (pgw#827).

    The regional twin of :func:`aot_serve.load_and_wrap`, and deliberately
    the same sequence through the same gates — stage, runtime-key verify,
    ISA, sm, per-entry contract, B1 bind gate, all-or-nothing before the
    first live mutation. It differs in exactly one thing, which is the
    entire point: the bind table is built **per block INSTANCE**
    (``resident_constants(block)``), not once per target
    (``resident_constants(denoiser)``).

    That one difference is what pgw#827 measured: a regional entry's
    declared FQNs are block-relative (``attn1.to_k.weight``), the denoiser
    carries them under their full path
    (``down_blocks.…transformer_blocks.0.attn1.to_k.weight``), and
    :func:`aot_serve.resolve_constants` is a direct FQN lookup by design —
    so a denoiser-scope table resolves NONE of a regional cell's constants.
    On a real L4 that discarded a complete 72-entry, 354 s artifact at
    ``constants_constant_unresolved``, and because the MINT's own self-adopt
    verification runs this same path, it meant a regional cell could not be
    PUBLISHED at all, not merely that it could not serve.

    Raises :class:`compile_cache.AdoptError` with a classified reason on any
    failure. The pipeline is untouched on every rejection.
    """
    family = str(getattr(cfg, "family", "") or "")
    staged = aot_serve.stage_artifact(
        Path(artifact), family, cache_dir=cache_dir)
    arms: List[RegionalArm] = []
    try:
        meta = staged.metadata
        try:
            entries = aot_serve.entries_from_meta(meta)
        except ValueError as exc:
            raise aot_serve.AdoptError("contract_invalid", str(exc)) from exc
        grouped = _grouped_entries(entries)

        # Resolve every target and prove the DECLARED block classes are the
        # LIVE ones before a single package is dlopen'd.
        owners: Dict[str, Any] = {}
        live: Dict[str, Dict[str, BlockGroup]] = {}
        for target, by_block in grouped.items():
            module, attr = aot_serve._target_owner(pipeline, target)
            if attr != "forward":
                raise aot_serve.AdoptError(
                    "regional_target_not_a_module",
                    f"target {target!r} resolves to a callable attribute "
                    f"({attr!r}), which has no block structure to arm")
            groups = {g.key: g for g in repeated_block_groups(module)}
            missing = sorted(set(by_block) - set(groups))
            if missing:
                raise aot_serve.AdoptError(
                    "regional_block_absent",
                    f"target {target!r}: the cell declares block class(es) "
                    f"{missing!r} that this resident "
                    f"{type(module).__name__} does not have")
            undeclared = sorted(set(groups) - set(by_block))
            if undeclared:
                # All-or-nothing (S4): a target whose blocks are only partly
                # covered would arm into a silently half-eager model.
                raise aot_serve.AdoptError(
                    "regional_block_undeclared",
                    f"target {target!r}: resident block class(es) "
                    f"{undeclared!r} have no entry in this cell — arming it "
                    f"would leave a silently half-eager model")
            owners[target] = module
            live[target] = groups

        literals_by_entry: Dict[str, Dict[str, Any]] = {}
        literals_path = staged.root / aot_serve.LITERALS_NAME
        if literals_path.exists():
            first = next(iter(owners.values()))
            device = str(getattr(first, "device", "cuda"))
            try:
                literals_by_entry = aot_serve.split_literals(
                    aot_serve._load_literals(literals_path, device))
            except ValueError as exc:
                raise aot_serve.AdoptError(
                    "contract_invalid", str(exc)) from exc

        package_path = staged.root / aot_serve.PACKAGE_NAME
        cell_key = str(meta.get("cell_key") or "")
        t0 = time.monotonic()
        loads = 0
        rows: Dict[str, Dict[str, Any]] = {}
        for target, by_block in grouped.items():
            module = owners[target]

            def runner_for(key: str, _t: str = target) -> BlockDispatch:
                nonlocal loads
                built: Dict[
                    Optional[bool],
                    List[Tuple[str, aot_serve.ArtifactRunner]]] = {}
                for name, block in grouped[_t][key]:
                    try:
                        contract = aot_serve.contract_from_meta(block)
                        constants = aot_serve.constants_from_meta(block)
                    except ValueError as exc:
                        raise aot_serve.AdoptError(
                            "contract_invalid",
                            f"entry {name!r}: {exc}") from exc
                    package = aot_serve._load_package(package_path, name)
                    loads += 1
                    built.setdefault(entry_adapter_arm(block), []).append(
                        (name, aot_serve.ArtifactRunner(
                            package=package, contract=contract,
                            constants=constants, module_name=_t, entry=name,
                            family=family)))
                return BlockDispatch(
                    by_arm={arm: aot_serve.EntryDispatch(tuple(rows))
                            for arm, rows in built.items()},
                    adapter_owner=owners[_t])

            def outputs_for(key: str, _t: str = target) -> int:
                return max(entry_output_count(block)
                           for _n, block in grouped[_t][key])

            try:
                arm = arm_blocks(
                    [live[target][key] for key in sorted(by_block)],
                    runner_for, target=target, family=family,
                    cell_key=cell_key, outputs_for=outputs_for,
                    literals=literals_by_entry)
            except aot_serve.ConstantsUnboundError as exc:
                raise aot_serve.AdoptError(
                    f"constants_{exc.reason}",
                    f"target {target!r}: {exc}") from exc
            except RegionalArmRefused as exc:
                raise aot_serve.AdoptError(
                    f"regional_{exc.reason}", str(exc)) from exc
            arms.append(arm)
            for index, (instance, shim) in enumerate(arm.shims):
                rows[f"{target}#{index}"] = {
                    "module": instance, "attr": "forward",
                    "state": shim.state}

        # The pipeline marker is the SAME format-2 shape the whole-graph arm
        # publishes, so `is_armed`, `execution_count`, `proven_since`,
        # `set_guard_failure_callback` and `unwrap` need no regional variant.
        # A second reporting shape would mean the executor's adoption PROOF
        # (pgw#735: it executed, and it is still armed) could never pass for
        # a regional cell — the mint would publish and then be rolled back as
        # unproven.
        setattr(pipeline, aot_serve._MARKER_ATTR,
                {"meta": meta, "targets": rows})
        instances = sum(arm.bound_instances for arm in arms)
        logger.info(
            "aot-regional: armed %s key=%s — %d entr%s across %d block "
            "class(es) bound BY REFERENCE onto %d instance(s) in %.1fs "
            "(%d package loads, 0 weight copies)",
            family, cell_key, len(entries),
            "y" if len(entries) == 1 else "ies",
            sum(len(v) for v in grouped.values()), instances,
            time.monotonic() - t0, loads)
        return meta
    except Exception:
        for arm in arms:
            arm.revert()
        try:
            delattr(pipeline, aot_serve._MARKER_ATTR)
        except AttributeError:
            pass
        raise
    finally:
        staged.close()


def enable(
    pipeline: Any, cfg: Any, cache_dir: Optional[Path] = None,
    artifact: Optional[Path] = None,
) -> bool:
    """Consumer entry point for a REGIONAL cell — the twin of
    :func:`aot_serve.enable`, with the identical fail-soft contract: any miss
    returns False and the caller's ordinary miss policy takes over.

    Emits the same ``aot_adopt`` event kind and phases, so a regional
    refusal is legible to the hub through the records that already exist.
    """
    if artifact is None:
        return False
    try:
        meta = load_and_arm(pipeline, cfg, Path(artifact), cache_dir=cache_dir)
    except Exception as exc:  # noqa: BLE001 — any miss => eager
        reason = str(getattr(exc, "reason", "") or "") or type(exc).__name__
        logger.warning(
            "aot-regional: artifact unusable (%s: %s); staying eager",
            reason, exc)
        activity_mod.emit_event(
            aot_serve.ADOPT_EVENT,
            f"{aot_serve._adopt_identity(Path(artifact))} "
            f"mode={MODE_REGIONAL}: {type(exc).__name__}: {exc}",
            phase=reason,
        )
        return False
    activity_mod.emit_event(
        aot_serve.ADOPT_EVENT,
        f"family={meta.get('family')} key={meta.get('cell_key')} "
        f"mode={MODE_REGIONAL} entries={len(meta.get('entries') or {})} "
        f"sku={meta.get('sku')} torch={meta.get('torch')} "
        f"precision={meta.get('precision')}",
        phase="armed",
    )
    return True


def _detach(value: Any) -> Any:
    """Clone every tensor out of a forward's return.

    The reference must survive the arm — a live activation buffer the second
    forward overwrites would make the comparison compare a tensor with itself.
    """
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, (list, tuple)):
        return type(value)(_detach(v) for v in value) \
            if not hasattr(value, "_fields") else tuple(_detach(v) for v in value)
    if isinstance(value, dict):
        return {k: _detach(v) for k, v in value.items()}
    fields = getattr(value, "__dict__", None)
    if isinstance(fields, dict) and fields:
        for key in list(fields):
            if not str(key).startswith("_"):
                fields[key] = _detach(fields[key])
    return value


__all__ = [
    "BLOCK_FORK", "DEFAULT_THRESHOLDS", "MODE_REGIONAL",
    "NUMERICS_FLOOR", "NUMERICS_RETENTION_FLOOR", "NUMERICS_WARN",
    "BlockGroup", "BlockShim", "RegionalArm", "RegionalArmRefused",
    "arm_and_verify", "arm_blocks", "block_entry_fork", "block_has_conv",
    "capture_block_feeds", "capture_block_feed_shapes",
    "positional_feed",
    "BlockDispatch",
    "declared_repeated_classes", "declared_thresholds", "enable",
    "entry_adapter_arm", "entry_block_key", "entry_output_count",
    "gate_assembled", "load_and_arm",
    "repeated_block_groups", "shell_digest", "shell_facts",
]
