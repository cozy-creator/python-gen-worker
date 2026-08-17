"""The MEASURE-ONLY child — ``python -m gen_worker.measure_child``.

    python -m gen_worker.measure_child <request>.mint.json [<report>.json]

Runs the mint's own load and export loop — optionally with the INDUCTOR compile
— against ONE declared class set, records what it cost on the card, and
produces **nothing else**: no cell, no artifact, no package, no hub call, no
advertisement.

``<request>.mint.json`` is the DECLARATION payload an endpoint repo commits
(``family``, ``shapes``, ``text_lens``, ``specialization``,
``declaration_module``, ``source_ref``), not the runtime ``MintRequest`` the
hub-driven parent builds. Both decode through :func:`load_document` ->
:func:`resolve_job`, so there is ONE decoder. From the committed shape:
``modules`` <- ``declaration_module``; ``function`` <- the endpoint whose
``Compile(family=)`` is the payload's ``family`` (``--function`` disambiguates);
``targets`` <- that endpoint's own ``Compile(targets=)`` when the payload names
none; ``slots`` <- ``source_ref`` plus ``--slot NAME=...``. Anything the payload
cannot supply is refused BY NAME before a weight is read, naming the flag that
supplies it.

**Slots resolve OFFLINE** — a ref is looked up in the local store and never
downloaded, inheriting ``mint_process``'s rule that a mint process which could
download is one that can stall on a lemon host.

Why a second child exists: both other front doors are shut against a
measurement run — ``mint_child._assert_family_mintable`` refuses while any
blocker is open, and ``boot_trace_child`` composes structure-only, whose
``_refuse_artifact_lanes`` refuses a w8a8/w4a4/svdq tree by name.

The three properties that make that safe:

1. **Explicit invocation, never an ambient bypass.** Nothing spawns this child.
   ``_assert_family_mintable`` is untouched and every real mint still fails
   closed.
2. **It cannot produce an artifact, structurally.** :class:`MeasureJob` is a
   DIFFERENT wire struct from ``MintRequest`` and declares none of the
   output-side fields (:data:`WITHHELD_FIELDS` — ``target``, ``work_root``,
   ``report``, ``arm_token``). msgspec drops what a struct does not declare,
   so the artifact destination and report path never
   enter this process even when handed the same ``*.mint.json``.
3. **The real-weight fallback is scoped to HERE**, and the run REPORTS which
   lane it measured (:attr:`MeasureReport.weights`), read off the composed
   pipeline, so a weightless claim can never be implied by a run that was not.

``export_peak_device_bytes`` / ``export_peak_device_reserved_bytes`` are the
mint's own names for the same two counters (``aot_mint._mint_cell``) on the
same allocator, so numbers are comparable without translation. The per-entry
figure is the RUNNING high-water after that entry — the counter is reset once
before the first row, exactly as the mint resets it once before its export
phase — so the row that raised the water line is the row named beside it.

``--export-only`` skips the inductor half: cheaper, and a WEAKER answer, since
an export-only trace never exercises the whole-graph planner an OOM blocker is
usually about.
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import msgspec

from . import activity
from .child_contract import CompileSpec, MintSlot
from .serving_facts import FactsUnavailable
from .hostfacts import cuda_ready
from .child_preflight import (
    PreflightRefused,
    assert_slots_resolvable,
    bind_slots,
    pick_compile_target,
    select_specs,
)

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_REFUSED = 2
EXIT_BAD_JOB = 4

#: Every refusal token this child can write. Exhaustive and fenced by test —
#: the pgw#1116 rule, applied at a new refusal surface on the day it is added
#: rather than after the first unattributable run: a refusal whose token no
#: vocabulary carries is an event nobody can enumerate, count or alert on.
REASONS: Tuple[str, ...] = (
    # A declared slot the request cannot resolve — refused by name, before a
    # weight is read.
    "slots_unresolvable",
    # pgw#1153. The request file parsed, but it is not a request: it names no
    # `declaration_module` and no `modules`, so there is no image to collect
    # endpoints from and nothing to measure.
    "no_declaration_module",
    # pgw#1153. `declaration_module` named a module this image cannot import.
    # The commonest shape of it is running the command outside the endpoint's
    # own image, which is the one thing the runbook line says not to do.
    "declaration_module_unimportable",
    # pgw#1153. The payload's `family` selects no endpoint function, or selects
    # several that do not share an instance — `--function` says which.
    "function_underivable",
    # The endpoint's own `setup()` raised. Never re-classified here: a load
    # that fails under measurement fails under a mint too.
    "load_failed",
    # The composed pipeline carries no target the cell names.
    "no_compile_target",
    # The family ships no export declaration, so there is no class set to
    # measure.
    "no_declaration",
    # An entry refused (a mint refusal, or the device saying no). THE ANSWER
    # for an OOM blocker, not an error: the peaks measured up to that row are
    # in the report, and the refusing entry is named.
    "export_refused",
    # Anything else, classified rather than lost at the process boundary.
    "child_error",
)

#: The ``MintRequest`` fields :class:`MeasureJob` deliberately does NOT
#: declare. Every one of them is an output destination; a measure run has
#: none, and the cheapest way to prove it cannot
#: write one is for the path never to arrive.
WITHHELD_FIELDS: Tuple[str, ...] = (
    "target", "work_root", "report", "arm_token", "phases_snapshot",
)


class MeasureJob(msgspec.Struct, frozen=True, kw_only=True):
    """What a measure run needs: the INPUT half of a mint request, and nothing
    else.

    Decoded straight from a committed ``*.mint.json`` — an operator measures
    the request they would mint, not a hand-copied approximation of it — and
    msgspec drops every field this struct does not name. See
    :data:`WITHHELD_FIELDS`.
    """

    function: str
    modules: Tuple[str, ...]
    cfg: CompileSpec
    family: str = ""
    slots: Dict[str, MintSlot] = {}
    device: int = -1
    execution_lane: str = ""


class MeasureDocument(msgspec.Struct, frozen=True, kw_only=True):
    """pgw#1153: what a ``*.mint.json`` FILE may carry, either shape.

    There are exactly two request documents in the fleet and this decodes both,
    because a tool that decodes one of them is the defect this struct closes:

    * an endpoint repo's committed ``aot/*.mint.json`` — a flattened
      declaration payload. ``family`` / ``declaration_module`` / ``source_ref``
      are top-level and there is no ``cfg``, no ``function``, no ``slots``;
    * a mint work root's ``request.json`` — the runtime ``MintRequest``, whose
      input half is ``function`` / ``modules`` / ``cfg`` / ``slots``.

    Every field is optional and every field is INPUT-SIDE. :data:`WITHHELD_FIELDS`
    still holds here for the same structural reason it holds on
    :class:`MeasureJob`: msgspec drops what a struct does not declare, so an
    artifact destination cannot enter this process through the widened door
    either.
    """

    # The runtime envelope's input half.
    function: str = ""
    modules: Tuple[str, ...] = ()
    cfg: Optional[CompileSpec] = None
    slots: Dict[str, MintSlot] = {}
    device: int = -1
    execution_lane: str = ""
    # The committed declaration payload's half. `family` is common to both.
    family: str = ""
    #: The module whose IMPORT registers the family's export declaration
    #:. Committed by every endpoint repo and fenced by their own
    #: declaration suites, which is what makes it a safe default for `modules`.
    declaration_module: str = ""
    #: The compile target's checkpoint, as the endpoint repo records it. Bound
    #: to the slot that owns the declared targets — see :func:`resolve_job`.
    source_ref: str = ""


class MeasureRefused(Exception):
    """A typed refusal raised while BUILDING the job.

    It carries a :data:`REASONS` token so that a request that cannot start is
    reported in the same vocabulary as a run that started and stopped. The
    whole point of this issue is that "the command did not work" must be a row
    somebody can count, not a stack trace on a pod.
    """

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


class EntryMeasurement(msgspec.Struct, frozen=True, kw_only=True):
    """One declared graph class, and what exporting (and compiling) it cost."""

    entry: str
    ok: bool = False
    nodes: int = 0
    export_ms: int = 0
    compile_ms: int = 0
    #: This process's device high-water AFTER this entry — cumulative, because
    #: the counter is reset once before the first row. The row where it JUMPS
    #: is the row that sizes the mint.
    running_peak_device_bytes: int = 0
    running_peak_device_reserved_bytes: int = 0
    #: Loose inductor files this entry compiled. Counted, then DELETED — see
    #: :func:`_discard`. A measure run leaves no code behind.
    compiled_files: int = 0
    refusal: str = ""


class MeasureReport(msgspec.Struct, frozen=True, kw_only=True):
    """The typed evidence a blocker's ``resolution=`` can cite.

    It carries no artifact, no digest and no cell key, and it never will: this
    is a measurement of a mint, not a mint.
    """

    ok: bool = False
    reason: str = ""
    detail: str = ""
    family: str = ""
    function: str = ""
    #: Where the compile target was built — ``cuda:N`` or ``cpu``, stated
    #: rather than assumed. A cardless run measures zeroes HONESTLY, and a
    #: measured zero and an unmeasured one are different facts (``cuda``).
    device: str = ""
    cuda: bool = False
    #: ``virtual`` (the structure-only build held) or ``real`` (this run loaded
    #: the checkpoint). Read off the COMPOSED pipeline, never off the request.
    weights: str = ""
    #: The typed structure-only refusal that sent this run to real weights, and
    #: which of pgw#1123's two tokens it was.
    structure_refusal: str = ""
    structure_refusal_token: str = ""
    structure_only_components: Tuple[str, ...] = ()
    weight_lane: str = ""
    precision: str = ""
    entries: Tuple[EntryMeasurement, ...] = ()
    declared_classes: int = 0
    #: The mint's own two names for the phase high-water, on the mint's own
    #: counters (``aot_mint._mint_cell``), so these numbers are comparable
    #: with a real mint's without translation.
    export_peak_device_bytes: int = 0
    export_peak_device_reserved_bytes: int = 0
    #: False under ``--export-only``: the peaks then cover the export alone,
    #: which is a weaker answer than the whole-graph planner question.
    compiled: bool = False
    setup_ms: int = 0
    wall_ms: int = 0


# ---------------------------------------------------------------------------
# Probes. Each reads a counter and decides nothing.
# ---------------------------------------------------------------------------


def _cuda() -> Any:
    try:
        import torch

        return torch if cuda_ready() else None
    except Exception:  # noqa: BLE001 — torch-less: nothing to measure
        return None


def _reset_peak() -> None:
    torch = _cuda()
    if torch is not None:
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:  # noqa: BLE001 — a probe never fails a run
            logger.debug("measure: reset_peak_memory_stats failed",
                         exc_info=True)


def _peaks() -> Tuple[int, int]:
    """``(allocated, reserved)`` high-water since the last reset."""
    torch = _cuda()
    if torch is None:
        return 0, 0
    try:
        return (int(torch.cuda.max_memory_allocated()),
                int(torch.cuda.max_memory_reserved()))
    except Exception:  # noqa: BLE001 — a probe never fails a run
        return 0, 0


def _device_label(job: MeasureJob) -> str:
    torch = _cuda()
    if torch is None:
        return "cpu"
    ordinal = int(job.device or -1)
    return f"cuda:{ordinal}" if ordinal >= 0 else "cuda"


# ---------------------------------------------------------------------------
# The envelope: a committed declaration payload -> a MeasureJob.
#
# Nothing here decides anything about the measurement. It answers the four
# questions the payload does not spell out — which image, which function,
# which targets, which checkpoints — from the declaration the payload NAMES,
# and refuses by name when it cannot.
# ---------------------------------------------------------------------------


def load_document(raw: bytes) -> Tuple[MeasureDocument, CompileSpec]:
    """Decode one request file into ``(document, flattened compile spec)``.

    The committed payload IS a ``CompileSpec`` at top level, so the same
    bytes are read twice against two structs rather than sniffed for a
    discriminator: msgspec drops what each struct does not declare, and a
    document that carries neither half decodes to two empty structs and is
    refused by :func:`resolve_job` with its emptiness named.

    THE one decoder. ``main`` had a second (``type=MeasureJob``) and that is
    the whole of pgw#1153: it accepted a shape nothing in the fleet commits.
    """
    return (
        msgspec.json.decode(raw, type=MeasureDocument),
        msgspec.json.decode(raw, type=CompileSpec),
    )


def _parse_slot_flag(raw: str) -> Tuple[str, str, str]:
    """``NAME=VALUE[,ref=REF]`` -> ``(name, value, ref)``.

    ``VALUE`` is a local tree when it exists on this filesystem and a model ref
    otherwise, so an operator on a pod can name either the bytes or the
    catalog row without a second flag to remember.
    """
    name, sep, rest = raw.partition("=")
    name = name.strip()
    if not sep or not name or not rest.strip():
        raise MeasureRefused(
            "slots_unresolvable",
            f"--slot {raw!r} is not NAME=VALUE[,ref=REF]")
    value, _, tail = rest.partition(",ref=")
    return name, value.strip(), tail.strip()


#: pgw#1333: a measure run names a tree on this machine; no catalog is
#: consulted and none is available offline. Saying so by name is what keeps a
#: declared serving contract from being checked against a fabricated blank.
_OPERATOR_FACTS = FactsUnavailable(
    owed_by="a --slot operator argument (a measure run consults no catalog)")


def _slot_from_value(name: str, value: str, ref_text: str) -> MintSlot:
    """One resolved slot from ``VALUE`` — a local tree, or a ref already here.

    A ref with no provider prefix is read as a tensorhub ref, which is what a
    pod serves from. Nothing here guesses harder than that: a miss is a typed
    refusal naming the flag that settles it, and a wrong guess would be a
    measurement of the wrong checkpoint.

    Identity is deliberately weak on the path form. No cell key, digest or
    artifact leaves this process (:class:`MeasureReport` carries none of the
    three), so a slot's ``ref`` exists only to satisfy ``ctx.slots``; it is the
    operator's when ``,ref=`` gives one and a slot-shaped placeholder when not.
    """
    from .api.binding import ModelRef
    from .models.provision import resolve_local_path

    if Path(value).is_dir():
        return MintSlot(
            ref=ModelRef(source="tensorhub", path=ref_text or f"local/{name}"),
            path=str(Path(value)),
            facts=_OPERATOR_FACTS)
    ref = ModelRef(source="tensorhub", path=value)
    try:
        path = resolve_local_path(
            ref=value, provider="tensorhub", offline=True, emit=lambda _e: None)
    except Exception as exc:  # noqa: BLE001 — every miss is one refusal
        raise MeasureRefused(
            "slots_unresolvable",
            f"slot {name!r}: {value!r} is neither a directory on this machine "
            f"nor a tensorhub ref already materialized in its local store "
            f"({type(exc).__name__}: {exc}). A measure run never downloads — "
            f"it is compute, exactly as a mint is — so name a tree this pod "
            f"already fetched: --slot {name}=/path/to/tree") from exc
    return MintSlot(ref=ref, path=str(path), facts=_OPERATOR_FACTS)


def _target_owner(spec: Any, targets: Sequence[str]) -> str:
    """The ONE declared slot that owns every compile target, or ``""``.

    Read off the declaration alone (``slot_components``, derived at decoration
    from each slot's pipeline class), because this runs BEFORE the load: the
    question "which checkpoint does `source_ref` name" has to be answered
    before a checkpoint is opened.
    """
    roots = {str(t).split(".", 1)[0] for t in targets if str(t)}
    owners = [
        name for name, tree in (getattr(spec, "slot_components", {}) or {}).items()
        if roots and roots.issubset(set(tree))
    ]
    if len(owners) != 1:
        family = str(getattr(getattr(spec, "compile", None), "family", "") or "")
        owners = [
            name for name, fam in (getattr(spec, "slot_family", {}) or {}).items()
            if family and str(fam) == family
        ]
    return owners[0] if len(owners) == 1 else ""


def _declares(specs: Sequence[Any], family: str) -> bool:
    return any(
        str(getattr(getattr(s, "compile", None), "family", "") or "") == family
        for s in specs)


def _function_for_family(specs: Sequence[Any], family: str) -> str:
    """The endpoint function the payload's ``family`` names.

    A committed payload names a FAMILY because that is what a cell is scoped to
    (pgw#758: one mint -> one cell for the family's whole declared class set);
    it does not name a function, and it should not have to. Functions sharing a
    class are interchangeable here — ``select_specs`` pulls the whole sibling
    set from any of them — so ambiguity is only real when the candidates span
    classes, and then it is `--function`'s to settle.
    """
    candidates = [
        s for s in specs
        if str(getattr(getattr(s, "compile", None), "family", "") or "") == family
    ]
    if not candidates:
        declared = sorted({
            str(getattr(getattr(s, "compile", None), "family", "") or "")
            for s in specs} - {""})
        raise MeasureRefused(
            "function_underivable",
            f"no endpoint in this image declares Compile(family={family!r}) — "
            f"this image declares {declared or '(no compiling family)'}. Either "
            f"the request belongs to a different endpoint or the payload's "
            f"family is stale; --function names one explicitly.")
    classes = {id(s.cls) for s in candidates}
    if len(classes) > 1:
        raise MeasureRefused(
            "function_underivable",
            f"family {family!r} is declared by functions on more than one "
            f"class ({sorted(s.name for s in candidates)}), which are "
            f"different instances and therefore different loads — name one "
            f"with --function")
    return sorted(s.name for s in candidates)[0]


def resolve_job(
    doc: MeasureDocument, flat: CompileSpec, *,
    function: str = "", slot_flags: Sequence[str] = (),
) -> MeasureJob:
    """Build the job the run needs from whichever document arrived.

    Imports the endpoint's declaring module — the same walk ``mint_child`` and
    ``boot_trace_child`` do — because the three answers the committed payload
    omits all live on the declaration, and reading them from anywhere else
    would be a second declaration.
    """
    from .registry import collect_endpoints

    modules = tuple(str(m) for m in doc.modules if str(m).strip())
    if not modules and doc.declaration_module.strip():
        modules = (doc.declaration_module.strip(),)
    if not modules:
        raise MeasureRefused(
            "no_declaration_module",
            "the request names neither `modules` nor `declaration_module`, so "
            "there is no image to collect endpoints from. Every committed "
            "aot/*.mint.json carries `declaration_module` (pgw#1107); a file "
            "without one is not a request this child can measure.")

    cfg = doc.cfg if doc.cfg is not None else flat
    family = (doc.family or cfg.family).strip()
    if not family:
        raise MeasureRefused(
            "no_declaration_module",
            "the request names no `family`, so nothing selects a declaration "
            "or a class set")

    def _collect(names: Sequence[str]) -> List[Any]:
        try:
            return list(collect_endpoints(list(names)))
        except BaseException as exc:  # noqa: BLE001 — an unimportable image is one
            raise MeasureRefused(
                "declaration_module_unimportable",
                f"cannot import {list(names)} ({type(exc).__name__}: {exc}). "
                f"Run this in the ENDPOINT'S OWN IMAGE, where the module that "
                f"registers the declaration is on sys.path.") from exc

    specs = _collect(modules)
    if not _declares(specs, family) and len(modules) == 1 and "." in modules[0]:
        # sdxl's shape: `declaration_module` names a declaration-ONLY module
        # (`sdxl.aot_declaration`), which registers the export declaration and
        # decorates nothing. `find_endpoints` walks submodules, never parents,
        # so the family's functions live one level up. Widen ONCE, to the
        # package that module belongs to — the same tree the pod imports.
        package = modules[0].split(".", 1)[0]
        widened = _collect((package,))
        if _declares(widened, family):
            modules, specs = (package,), widened

    name = (function or doc.function).strip() or _function_for_family(
        specs, family)
    try:
        chosen, _siblings = select_specs(specs, name)
    except PreflightRefused as exc:
        raise MeasureRefused("function_underivable", str(exc)) from exc

    if not cfg.targets:
        # targets DERIVE from the declaration, so no committed
        # payload carries them — and `compile_cache.resolve_targets` returns
        # nothing for an empty tuple, which is the same defect one step later.
        declared = tuple(
            str(t) for t in
            (getattr(getattr(chosen, "compile", None), "targets", ()) or ()))
        cfg = msgspec.structs.replace(cfg, targets=declared)
    if not cfg.family:
        cfg = msgspec.structs.replace(cfg, family=family)

    # Flags FIRST, then the payload's own ref for whatever they left unnamed —
    # a `--slot` that names the target slot must not have to survive an
    # eager refusal from the ref it was passed to replace.
    slots: Dict[str, MintSlot] = dict(doc.slots)
    for raw in slot_flags:
        slot_name, value, ref_text = _parse_slot_flag(str(raw))
        slots[slot_name] = _slot_from_value(slot_name, value, ref_text)
    owner = _target_owner(chosen, cfg.targets)
    if doc.source_ref.strip() and owner and owner not in slots:
        slots[owner] = _slot_from_value(owner, doc.source_ref.strip(), "")

    return MeasureJob(
        function=chosen.name, modules=modules, cfg=cfg, family=family,
        slots=slots, device=int(doc.device), execution_lane=doc.execution_lane)


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def _write(report_path: Path, report: MeasureReport) -> None:
    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = report_path.with_suffix(".tmp")
        tmp.write_bytes(msgspec.json.encode(report))
        tmp.replace(report_path)
    except OSError:
        logger.exception("measure: could not write %s", report_path)


def _fail(
    report_path: Path, reason: str, detail: str, *,
    partial: Optional[MeasureReport] = None,
) -> int:
    report = msgspec.structs.replace(
        partial or MeasureReport(), ok=False, reason=reason,
        detail=str(detail)[:4000])
    _write(report_path, report)
    activity.emit_event(
        activity.KIND_MEASURE_ONLY, phase=reason,
        detail=f"family={report.family or '-'} function={report.function or '-'} "
               f"{str(detail)[:600]}")
    logger.error("measure: %s: %s", reason, detail)
    return EXIT_REFUSED


def run(
    job: MeasureJob, report_path: Path, *, compile_entries: bool = True,
) -> int:
    """Measure this job's declared class set. Never raises, never publishes."""
    from . import (
        aot_compile_child,
        aot_declaration,
        aot_mint,
        compile_cache as cc,
        fleet_cells,
    )
    from .cli.run import run_setup
    from .models import structure_only
    from .registry import collect_endpoints

    started = time.monotonic()
    cfg = job.cfg
    family = str(job.family or getattr(cfg, "family", "") or "")
    partial = MeasureReport(family=family, function=job.function)

    t_setup = time.monotonic()
    try:
        specs = collect_endpoints(list(job.modules))
        chosen, siblings = select_specs(specs, job.function)
        bind_slots(siblings, job.slots)
        assert_slots_resolvable(
            siblings, job.slots,
            what=f"measure-only run of {job.function!r} — a committed "
                 f"aot/*.mint.json names ONE checkpoint (`source_ref`), so "
                 f"every other slot the endpoint's setup() requires is named "
                 f"with `--slot NAME=/path/to/tree`")
    except PreflightRefused as exc:
        return _fail(report_path, "slots_unresolvable", str(exc),
                     partial=partial)

    paths = {name: slot.path for name, slot in job.slots.items()}
    device_label = _device_label(job)
    refusal = ""
    refusal_token = ""

    def _load(structure: Sequence[str]) -> Dict[str, Any]:
        return run_setup(
            chosen.cls(), dict(paths), arm_compile=False, return_loaded=True,
            structure_only=tuple(structure)) or {}

    try:
        try:
            loaded = _load(tuple(cfg.targets))
        except structure_only.StructureOnlyUnsupported as exc:
            # THE pgw#1134 fallback, and it lives only here. A stranded family
            # (`_refuse_artifact_lanes` on a w8a8 tree) and a family whose
            # composition discarded the injected structure are both reasons a
            # MINT declines to claim weightlessness — and both are reasons a
            # MEASUREMENT must load the real checkpoint, because the graph the
            # pod serves is the graph made of those bytes. The distinction is
            # recorded, not collapsed: the report says which one it was.
            refusal = str(exc)
            refusal_token = (
                "structure_not_honored"
                if isinstance(exc, structure_only.StructureNotHonored)
                else structure_only.refusal_token(exc))
            logger.warning(
                "measure: structure-only refused (%s) — measuring the REAL "
                "weight lane, which is the lane the pod serves: %s",
                refusal_token, exc)
            loaded = _load(())
    except Exception as exc:  # noqa: BLE001 — a load failure is classified
        logger.exception("measure: the endpoint's setup() failed")
        return _fail(report_path, "load_failed", f"{type(exc).__name__}: {exc}",
                     partial=partial)

    try:
        _slot, pipeline = pick_compile_target(loaded, cfg)
    except PreflightRefused as exc:
        return _fail(report_path, "no_compile_target", str(exc),
                     partial=partial)

    if cfg.lora_bucket:
        # The CONTAINER half and the lane stamp, exactly as `mint_child` and
        # `boot_trace_child` arm the pipeline they hand the export. The LIFTED
        # half belongs to the loop that needs it.
        cc.apply_lora_execution_lane(pipeline, int(cfg.lora_bucket))
    spec = fleet_cells.aot_export_spec(pipeline, cfg)
    virtual = structure_only.structure_only_components(pipeline)
    partial = msgspec.structs.replace(
        partial,
        family=str(spec.family or family), device=device_label,
        cuda=_cuda() is not None,
        weights="virtual" if virtual else "real",
        structure_refusal=refusal[:2000],
        structure_refusal_token=refusal_token,
        structure_only_components=tuple(virtual),
        weight_lane=str(spec.weight_lane or ""),
        precision=str(spec.precision or ""),
        compiled=bool(compile_entries),
        setup_ms=int((time.monotonic() - t_setup) * 1000))

    from .api.export_contract import export_declaration

    decl = export_declaration(str(spec.family or family))
    if decl is None:
        return _fail(
            report_path, "no_declaration",
            f"family {spec.family!r} has no registered export declaration — "
            f"a measure run has no class set without one", partial=partial)

    # The row order the loop below will export, enumerated FIRST and from the
    # loop's own function — so a row that dies (the OOM an export blocker is
    # usually about) can be named. An exception escaping a generator carries
    # no row identity, and "something in the export ran out of memory" is not
    # evidence anybody can act on.
    try:
        ordered = [
            aot_declaration.plan_entry_name(plan)
            for plan, _arm in aot_mint.declared_class_rows(pipeline, spec, decl)]
    except Exception as exc:  # noqa: BLE001 — an unreadable class set is one
        return _fail(
            report_path, "no_declaration",
            f"family {spec.family!r} declares a class set that will not "
            f"enumerate ({type(exc).__name__}): {exc}", partial=partial)

    entries: List[EntryMeasurement] = []
    declared = len(ordered)
    reason = ""
    detail = ""
    # ONE reset, before the first row — the mint resets once before its export
    # phase, and a per-row reset would report N unrelated windows where the
    # question is a single high-water.
    _reset_peak()
    t_entry = time.monotonic()
    try:
        with tempfile.TemporaryDirectory(
            prefix="measure-compiled-graphs-",
        ) as raw:
            work = Path(raw)
            engine_runtime = (
                aot_compile_child._tcg_runtime(work / "cas")
                if compile_entries else None
            )
            for traced in aot_mint.trace_for_key(pipeline, spec, decl):
                compile_ms = 0
                compiled_files = 0
                if engine_runtime is not None:
                    engine, runtime = engine_runtime
                    compiled = aot_compile_child.compile_traced_class(
                        traced,
                        spec,
                        engine,
                        runtime,
                        work=work,
                        out_dir=work / "exports",
                    )
                    compile_ms = int(
                        (compiled.compile_s + compiled.reuse_s) * 1000
                    )
                    compiled_files = 1
                else:
                    traced.release()
                allocated, reserved = _peaks()
                timings = dict(traced.timings or {})
                declared = int(traced.declared) or declared
                entries.append(EntryMeasurement(
                    entry=traced.name,
                    ok=True,
                    nodes=int(traced.nodes),
                    export_ms=int(float(timings.get("export_s", 0.0)) * 1000),
                    compile_ms=compile_ms,
                    running_peak_device_bytes=allocated,
                    running_peak_device_reserved_bytes=reserved,
                    compiled_files=compiled_files,
                ))
                t_entry = time.monotonic()
    except BaseException as exc:  # noqa: BLE001 — an OOM here IS the answer
        allocated, reserved = _peaks()
        in_flight = ordered[len(entries)] if len(entries) < len(ordered) else ""
        entries.append(EntryMeasurement(
            entry=in_flight, ok=False,
            export_ms=int((time.monotonic() - t_entry) * 1000),
            running_peak_device_bytes=allocated,
            running_peak_device_reserved_bytes=reserved,
            refusal=f"{type(exc).__name__}: {exc}"[:2000]))
        reason, detail = "export_refused", f"{type(exc).__name__}: {exc}"
        if not isinstance(exc, Exception):
            raise

    allocated, reserved = _peaks()
    report = msgspec.structs.replace(
        partial, ok=not reason, reason=reason, detail=detail[:4000],
        entries=tuple(entries), declared_classes=declared,
        export_peak_device_bytes=allocated,
        export_peak_device_reserved_bytes=reserved,
        wall_ms=int((time.monotonic() - started) * 1000))
    _write(report_path, report)
    activity.emit_event(
        activity.KIND_MEASURE_ONLY,
        phase=reason or ("measured" if compile_entries else "measured_export"),
        duration_ms=report.wall_ms,
        detail=(
            f"family={report.family} function={report.function} "
            f"weights={report.weights} device={report.device} "
            f"entries={len(entries)}/{declared} "
            f"export_peak_device_bytes={report.export_peak_device_bytes} "
            f"export_peak_device_reserved_bytes="
            f"{report.export_peak_device_reserved_bytes} "
            f"compiled={report.compiled}"))
    print(
        f"MEASURED {report.family} {len(entries)}/{declared} entr(ies) "
        f"weights={report.weights} compiled={report.compiled} "
        f"export_peak_device_bytes={report.export_peak_device_bytes} "
        f"export_peak_device_reserved_bytes="
        f"{report.export_peak_device_reserved_bytes} -> {report_path}",
        file=sys.stderr)
    return EXIT_OK if not reason else EXIT_REFUSED


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s %(message)s",
        stream=sys.stderr)
    parser = argparse.ArgumentParser(
        prog="python -m gen_worker.measure_child",
        description=(
            "Export (and compile) one family's declared graph classes and "
            "report what it cost on the card. Publishes nothing, ever."))
    parser.add_argument(
        "request",
        help="an endpoint repo's committed aot/*.mint.json (a declaration "
             "payload) or a mint work root's request.json; either way the "
             "output-side fields are dropped by construction")
    parser.add_argument("report", nargs="?", default="",
                        help="where to write the typed measurement "
                             "(default: <request>.measure.json)")
    parser.add_argument(
        "--export-only", action="store_true",
        help="skip the inductor compile — cheaper, and a weaker answer")
    parser.add_argument(
        "--function", default="",
        help="the endpoint function to measure. Derived from the payload's "
             "Compile(family=) unless that is ambiguous across classes")
    parser.add_argument(
        "--slot", action="append", default=[], metavar="NAME=VALUE[,ref=REF]",
        help="a setup slot the payload does not name: VALUE is a local tree, "
             "or a ref already materialized in this machine's store (a "
             "measure run never downloads). Repeatable")
    args = parser.parse_args(list(argv) if argv is not None else None)

    request = Path(args.request)
    report_path = Path(args.report) if args.report else request.with_suffix(
        request.suffix + ".measure.json")
    try:
        doc, flat = load_document(request.read_bytes())
    except (OSError, msgspec.DecodeError, msgspec.ValidationError) as exc:
        sys.stderr.write(f"measure: unreadable request {request}: {exc}\n")
        return EXIT_BAD_JOB
    try:
        job = resolve_job(
            doc, flat, function=str(args.function),
            slot_flags=[str(s) for s in args.slot])
    except MeasureRefused as exc:
        return _fail(report_path, exc.reason, exc.detail)
    try:
        return run(job, report_path, compile_entries=not args.export_only)
    except BaseException as exc:  # noqa: BLE001 — every terminus is reported
        logger.exception("measure: child failed")
        return _fail(report_path, "child_error", f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":  # pragma: no cover — process entrypoint
    raise SystemExit(main())


__all__ = ["EntryMeasurement", "MeasureDocument", "MeasureJob",
           "MeasureRefused", "MeasureReport", "REASONS", "WITHHELD_FIELDS",
           "load_document", "main", "resolve_job", "run"]
