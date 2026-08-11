"""pgw#1134: the MEASURE-ONLY child — ``python -m gen_worker.measure_child``.

    python -m gen_worker.measure_child <request.json> [<report.json>]

It runs the mint's own load and the mint's own export loop — optionally with
the INDUCTOR compile — against ONE declared class set, records what the run
cost on the card, and produces **nothing else**. No cell, no artifact, no
package, no hub call, no advertisement.

Why a second child exists at all
--------------------------------
A blocker (pgw#1115) is a declared refusal whose exit criterion is often a
MEASUREMENT, and ltx-video-2.3's OQ-3 is the worked example: *"is a whole-graph
export of the served w8a8 lane bigger than the card?"*. Both front doors were
shut against the run that answers it:

* ``mint_child._assert_family_mintable`` refuses while ANY blocker is open —
  correct, and deliberately so. The measurement that would close OQ-3 was
  refused BY OQ-3.
* ``boot_trace_child`` is ungated, but it composes structure-only, and
  ``structure_only._refuse_artifact_lanes`` refuses a w8a8 / w4a4 / svdq
  artifact tree BY NAME. The boot child treats that as a hard refusal (it must:
  §4.27 step 1 forbids weights for identity, and a boot that quietly downloaded
  42 GiB to state its key would satisfy the letter of the derivation and
  destroy its purpose).

So the blocker could never gather the evidence that resolves it, and the only
remaining moves were to guess or to resolve the blocker in order to unblock its
own run — the circularity pgw#1115 exists to prevent.

The three properties that make this safe
----------------------------------------
1. **It is an explicit invocation, never an ambient bypass.** Nothing spawns
   this child: an operator or a harness runs it, at a request file, on purpose.
   ``_assert_family_mintable`` is untouched and every real mint still fails
   closed — this module does not call it because it cannot mint, not because it
   is exempt.
2. **It cannot produce an artifact, structurally.** :class:`MeasureJob` is a
   DIFFERENT wire struct from ``MintRequest`` and it declares none of the
   output-side fields (:data:`WITHHELD_FIELDS` — ``target``, ``work_root``,
   ``resume``, ``report``, ``arm_token``). msgspec drops what a struct does not
   declare, so the artifact destination, the resume bank and the mint's report
   path never enter this process's memory even when the operator hands it the
   very same ``*.mint.json`` file. There is no publish call to audit because
   there is nothing here to publish TO.
3. **The real-weight fallback is scoped to HERE.** ``mint_child``'s pgw#1080
   invariants are untouched: ``StructureNotHonored`` still fails a mint closed,
   and the boot trace still refuses a stranded family rather than downloading
   its checkpoint. This child accepts real weights because a measurement of the
   served lane is exactly what it is for — and it REPORTS which lane it
   measured (:attr:`MeasureReport.weights`), read off the composed pipeline, so
   a weightless claim can never be implied by a run that was not.

What it measures, and against what vocabulary
---------------------------------------------
``export_peak_device_bytes`` / ``export_peak_device_reserved_bytes`` are the
mint's own names for the same two counters (``aot_mint._mint_cell``), read on
the same allocator, so a number from this child and a number from a real mint
are comparable without translation. The per-entry figure is the RUNNING
high-water after that entry — the counter is reset once, before the first row,
exactly as the mint resets it once before its export phase — so the row that
raised the water line is the row named beside it.

With ``--export-only`` the inductor half is skipped. That is a cheaper first
pass and a WEAKER answer: an export-only trace never exercises the whole-graph
planner an OOM blocker is usually about, which is why the compile runs by
default.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import msgspec

from . import activity
from .mint_process import CompileCellSpec, MintSlot

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
#: declare. Every one of them is an output destination or a cross-attempt
#: bank; a measure run has neither, and the cheapest way to prove it cannot
#: write one is for the path never to arrive.
WITHHELD_FIELDS: Tuple[str, ...] = (
    "target", "work_root", "report", "resume", "arm_token", "phases_snapshot",
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
    cfg: CompileCellSpec
    family: str = ""
    slots: Dict[str, MintSlot] = {}
    device: int = -1
    execution_lane: str = ""


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

        return torch if torch.cuda.is_available() else None
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


def _discard(files: Sequence[str]) -> int:
    """Delete an entry's compiled output and return how much there was.

    The compile is the half of the question that matters, and its OUTPUT is
    the half that must not survive: a measure run that left loose ``.so``
    files behind would be one packaging step away from the artifact this
    module may not produce.
    """
    count = 0
    for name in files:
        count += 1
        try:
            Path(name).unlink()
        except OSError:
            logger.debug("measure: could not remove %s", name, exc_info=True)
    return count


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
    from . import aot_declaration, aot_mint, compile_cache as cc, fleet_cells
    from .cli.run import run_setup
    from .mint_child import (
        MintChildRefused, assert_slots_resolvable, bind_slots,
        pick_compile_target, select_specs,
    )
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
            siblings, job.slots, what=f"measure-only run of {job.function!r}")
    except MintChildRefused as exc:
        return _fail(report_path, "slots_unresolvable", str(exc),
                     partial=partial)

    paths = {name: slot.path for name, slot in job.slots.items()}
    overrides = {
        name: dict(slot.component_paths or {})
        for name, slot in job.slots.items() if slot.component_paths}
    device_label = _device_label(job)
    refusal = ""
    refusal_token = ""

    def _load(structure: Sequence[str]) -> Dict[str, Any]:
        return run_setup(
            chosen.cls(), dict(paths), arm_compile=False, return_loaded=True,
            component_paths=overrides, structure_only=tuple(structure)) or {}

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
    except MintChildRefused as exc:
        return _fail(report_path, "no_compile_target", str(exc),
                     partial=partial)

    if cfg.lora_bucket:
        # The CONTAINER half and the lane stamp, exactly as `mint_child` and
        # `boot_trace_child` arm the pipeline they hand the export. The LIFTED
        # half belongs to the loop that needs it (pgw#1132).
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

    decl = aot_mint.export_declaration(str(spec.family or family))
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
        for traced in aot_mint.trace_for_key(
                pipeline, spec, decl, compile_now=bool(compile_entries)):
            allocated, reserved = _peaks()
            timings = dict(traced.timings or {})
            # ONE enumeration: `ordered` above and this count both come from
            # `declared_class_rows`, so they cannot disagree.
            declared = int(traced.declared) or declared
            entries.append(EntryMeasurement(
                entry=traced.name, ok=True, nodes=int(traced.nodes),
                export_ms=int(float(timings.get("export_s", 0.0)) * 1000),
                compile_ms=int(float(timings.get("compile_s", 0.0)) * 1000),
                running_peak_device_bytes=allocated,
                running_peak_device_reserved_bytes=reserved,
                compiled_files=_discard(traced.files)))
            # The program is the largest object this child holds and nothing
            # downstream reads it.
            traced.program = None
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
    parser.add_argument("request", help="a mint request JSON; its output-side "
                                        "fields are ignored by construction")
    parser.add_argument("report", nargs="?", default="",
                        help="where to write the typed measurement "
                             "(default: <request>.measure.json)")
    parser.add_argument(
        "--export-only", action="store_true",
        help="skip the inductor compile — cheaper, and a weaker answer")
    args = parser.parse_args(list(argv) if argv is not None else None)

    request = Path(args.request)
    report_path = Path(args.report) if args.report else request.with_suffix(
        request.suffix + ".measure.json")
    try:
        job = msgspec.json.decode(request.read_bytes(), type=MeasureJob)
    except (OSError, msgspec.DecodeError, msgspec.ValidationError) as exc:
        sys.stderr.write(f"measure: unreadable request {request}: {exc}\n")
        return EXIT_BAD_JOB
    try:
        return run(job, report_path, compile_entries=not args.export_only)
    except BaseException as exc:  # noqa: BLE001 — every terminus is reported
        logger.exception("measure: child failed")
        return _fail(report_path, "child_error", f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":  # pragma: no cover — process entrypoint
    raise SystemExit(main())


__all__ = ["EntryMeasurement", "MeasureJob", "MeasureReport", "REASONS",
           "WITHHELD_FIELDS", "main", "run"]
