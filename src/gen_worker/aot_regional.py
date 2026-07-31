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
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from . import activity as activity_mod
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

    def __init__(self, runner: Any, block: Any, outputs: int = 1) -> None:
        self.runner = runner
        self._block = block
        self.outputs = int(outputs)
        self.calls = 0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        values, _names = positional_feed(self._block, args, kwargs)
        out = self.runner(*values)
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
                state = dict(module.state_dict())
                runner.bind(state, {}, user_managed=True)
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
                        outputs_for(group.key)))
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
    "arm_and_verify", "arm_blocks", "block_entry_fork", "capture_block_feeds",
    "positional_feed",
    "declared_repeated_classes", "declared_thresholds", "gate_assembled",
    "repeated_block_groups", "shell_digest", "shell_facts",
]
