"""The ONE model load+place core, plus the CLI's hub-less resolve (pgw#515).

Production (the executor's setup injection) and the local CLI
(``gen-worker run`` / ``serve``) drive the SAME code for turning a resolved
snapshot into a ready slot value: annotation-typed injection, binding
dtype / storage-dtype honoring, the pre-load cast gate (th#737), the
adaptive fit ladder outcome stamps (gw#491), worker-owned placement, and
compiled-artifact arming. Structural reporting (ServePlan / FnDegraded)
stays with the executor — :class:`SlotLoad` carries the outcomes so the
caller reports them however it reports.

Resolution differs by necessity: the executor's bytes come from
orchestrator-resolved snapshots (``ModelStore.ensure_local``); the CLI has
no orchestrator, so :func:`resolve_local_path` resolves standalone — local
CAS, tensorhub's public resolve route (th#560), direct HF / Civitai /
ModelScope downloads — through the same download layer.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping, Optional, Tuple

from .. import artifact_meta
from ..cell_adopt import AdoptOutcome
from ..component_vocab import denoiser_components
from ..api.binding import ModelRef, wire_ref
from ..config import Settings, current_or

_STANDALONE = Settings()
from . import disk_gc, load_progress
from .cache_paths import tensorhub_cas_dir
from .errors import UrlExpiredError
from .envelope import envelope_refusal
from .loading import (
    RUNG_NF4_UNLANDED,
    assert_uniform_compute_dtype,
    composition_compute_dtype,
    detect_diffusers_variant,
    load_from_pretrained,
    model_index_components,
    specialized_weight_layout,
)
from .memory import place_pipeline
from .refs import DEFAULT_REF_TAG, parse_model_ref
from .. import activity as activity_mod
from .. import mint_workers

if TYPE_CHECKING:
    from ..aot_identity import ExpectedIdentity

__all__ = ["model_index_components"]  # re-export: single source in loading.py (gw#521)

logger = logging.getLogger(__name__)

EmitFn = Callable[[Dict[str, Any]], None]


class ModelResolutionError(Exception):
    """A model binding cannot be resolved locally (CLI exit 3)."""


# ---------------------------------------------------------------------------
# Shared load + place + compile (executor and CLI)
# ---------------------------------------------------------------------------


@dataclass
class SlotLoad:
    """Outcome of loading one setup slot.

    ``obj`` is what the slot receives: the local path (``str``/``Path``
    annotations and unknown annotations), or a constructed + placed pipeline
    for a class annotation exposing ``from_pretrained``. The remaining fields
    are non-default only on the pipeline lane; detail fields are non-empty
    exactly when the caller should report that degradation (the decision
    logic lives here, once)."""

    obj: Any
    is_pipeline: bool = False
    ran: str = ""                # compute precision label ("bf16" default)
    # th#737 pre-load gate: the cast directive had no cast surface and was
    # dropped before the load.
    pre_drop_wanted: str = ""
    pre_drop_detail: str = ""
    # gw#491: the loader engaged an emergency fit rung ("fp8" / "nf4").
    rung: str = ""
    rung_detail: str = ""
    # th#737 backstop: the resolved cast was attempted at load and failed on
    # every component.
    cast_fail_wanted: str = ""
    cast_fail_detail: str = ""
    # place_pipeline outcome ({} when placement was skipped).
    placed: Dict[str, Any] = field(default_factory=dict)


def load_slot(
    annotation: Any,
    path: str,
    *,
    binding: Any = None,
    slot: str = "",
    ref: str = "",
    mode: str = "auto",
    components: Optional[Dict[str, Any]] = None,
    component_trees: Optional[Dict[str, str]] = None,
    device: str = "",
    place: bool = True,
    declared_vram_gb: float = 0.0,
    force_storage_dtype: str = "",
    strict_vram: bool = False,
    artifact_digest: str = "",
) -> SlotLoad:
    """Typed slot injection: the slot receives exactly what its ``setup``
    annotation says — a ``str``/``Path`` local path, or a constructed
    pipeline for a class annotation exposing ``from_pretrained`` (the
    binding's dtype/storage_dtype honored, the worker's placement/offload
    policy applied). Blocking; callers on an event loop run it via
    ``asyncio.to_thread``.

    ``mode`` is the placement mode (plan-time offload verdicts / learned
    degraded floors — the executor's knowledge; the CLI passes ``auto``).
    ``device="cpu"`` (CLI ``--device cpu``) skips placement entirely.
    ``place=False`` (pgw#1124) skips it too, WITHOUT claiming the composition
    is a CPU one: a boot-trace child builds its compile target virtually on
    the compute device and never runs a forward, so it needs the serving
    placement ladder for nothing — and running it there put a slot's real
    non-target weights (qwen-image's 15.5 GiB text encoder) onto the card the
    serving parent already occupies.
    ``components`` are preloaded shared modules (gw#479) forwarded to
    ``from_pretrained``. ``force_storage_dtype`` overrides the binding's own
    storage_dtype (th#1043): a joint multi-lane fit decision made BEFORE any
    lane in a shared-component group loads, so the first lane to load never
    greedily consumes free VRAM at native precision and starves a sibling
    lane into an offload placement the shared-component invariant refuses.
    """
    if annotation is None or annotation is str:
        return SlotLoad(obj=path)
    if annotation is Path:
        return SlotLoad(obj=Path(path))
    if not (isinstance(annotation, type)
            and callable(getattr(annotation, "from_pretrained", None))):
        return SlotLoad(obj=path)

    dtype = str(getattr(binding, "dtype", "") or "")
    storage_dtype = force_storage_dtype or str(getattr(binding, "storage_dtype", "") or "")
    out = SlotLoad(obj=None, is_pipeline=True, ran=(dtype or "bf16"))

    # th#737: a cast directive on a denoiser-less diffusers tree is a
    # load-time no-op that would silently serve bf16. Gate it up front when
    # the snapshot's model_index proves there is no cast surface (unknown
    # layouts pass through; the post-load outcome check below is the
    # backstop).
    if storage_dtype in ("fp8", "fp8+te"):
        comps = model_index_components(path)
        if comps and not (set(denoiser_components()) & comps):
            out.pre_drop_wanted = storage_dtype
            out.pre_drop_detail = (
                f"cast {storage_dtype!r} dropped for slot {slot!r}: pipeline "
                f"has no denoiser/cast surface (components: {sorted(comps)}); "
                "serving at base precision")
            storage_dtype = ""

    # pgw#1117 / th#1777: the artifact is weighed AS IT WILL LOAD and checked
    # against the declared envelope BEFORE a single byte is staged. ie#642
    # printed both numbers ("staged 0.67 GiB of 32.81 GiB" against
    # vram_gb=22), staged anyway, and OOMed on a billed card inside setup().
    # A clear breach is a typed refusal here; a marginal one still tries.
    trees = [path, *sorted((component_trees or {}).values())]
    refusal = envelope_refusal(
        trees,
        declared_vram_gb=declared_vram_gb,
        strict_vram=strict_vram,
        cast_dtype=dtype,
        storage_dtype=storage_dtype,
        variant=detect_diffusers_variant(Path(path)) or "",
        specialized_layout=specialized_weight_layout(path),
        slot=slot,
        ref=ref,
        artifact_digest=artifact_digest,
    )
    if refusal is not None:
        logger.error("pre-load envelope refusal: %s", refusal)
        try:
            activity_mod.emit_event(
                activity_mod.KIND_ENVELOPE_REFUSAL, str(refusal),
                phase="refused")
        except Exception:  # noqa: BLE001 - the refusal outranks its telemetry
            logger.debug("envelope-refusal event dropped", exc_info=True)
        raise refusal

    # pgw#1041: byte-level staging progress + death breadcrumb for the whole
    # load (hydration AND placement). The counter feeds the existing 10s
    # activity beat; the breadcrumb names the phase a SIGKILL lands in.
    staged_total = disk_gc.tree_bytes(Path(path))
    for tree in (component_trees or {}).values():
        staged_total += disk_gc.tree_bytes(Path(tree))
    # `clean=True` even on a raise: a Python-level failure reports itself, so
    # the breadcrumb clears. Only a kernel kill skips the finally — exactly
    # the death the surviving breadcrumb is for.
    reporter = load_progress.LoadProgressReporter(
        f"{slot or 'slot'}:{ref or path}", staged_total).start()
    try:
        pipe = load_from_pretrained(
            annotation, path, dtype=dtype, storage_dtype=storage_dtype,
            components=components or None,
            component_trees=component_trees or None,
            declared_vram_gb=declared_vram_gb,
            ref=ref,
            # pgw#1063: the loader's own host-RAM decisions depend on where
            # this pipeline will END UP. `place_pipeline(mode=...)` below is
            # the same knowledge arriving too late to inform them.
            placement_mode=mode,
        )
        out.obj = pipe

        # pgw#683: the composition must present ONE compute dtype to its GEMMs.
        # Fail HERE, naming the component and the tensor, instead of at warm unit
        # 4/18 with torch's `mat1 and mat2 must have the same dtype` — which names
        # neither, and which cost `generate` on a live prod release.
        assert_uniform_compute_dtype(
            pipe, composition_compute_dtype(path, dtype), label=f"slot {slot!r} ({ref})")

        rung = str(getattr(pipe, "_cozy_adaptive_rung", "") or "")
        cast_failed = getattr(
            pipe, "_cozy_fp8_storage_requested", False
        ) and not getattr(pipe, "_cozy_fp8_storage_ok", True)
        if rung == RUNG_NF4_UNLANDED:
            # pgw#824: the emergency rung ENGAGED and landed on nothing. Routed
            # through the same SlotLoad.rung path as every sibling rung, so it
            # reaches ServePlan/FnDegraded via `_record_adaptive_rung` instead of
            # dying in a log line. It used to clear the stamp, which suppressed the
            # only report the ladder had — the worst outcome was the silent one.
            out.rung = rung
            out.rung_detail = (
                f"adaptive fit rung 'nf4' engaged at load for slot {slot!r} "
                f"({type(pipe).__name__}) and landed on ZERO modules; this slot "
                f"serves FULL PRECISION over the VRAM it was budgeted, and only "
                f"the offload ladder carries it")
        elif rung == "nf4" or (rung == "fp8" and not cast_failed):
            # gw#491: the loader engaged an emergency rung because free VRAM at
            # load was tighter than planning assumed.
            out.rung = rung
            out.rung_detail = (
                f"adaptive fit rung {rung!r} engaged at load for slot {slot!r} "
                f"({type(pipe).__name__}); free VRAM below the stored-precision "
                "footprint")
        elif cast_failed and not rung:
            # th#737 backstop: the RESOLUTION cast was attempted at load and
            # failed on every target — structural report, not a silent bf16
            # fallback. (A failed adaptive fp8 is not a plan deviation: the plan
            # was base precision.)
            out.cast_fail_wanted = storage_dtype or "fp8"
            out.cast_fail_detail = (
                f"fp8 storage failed on every component of slot {slot!r} "
                f"({type(pipe).__name__}); serving at base precision")
        elif (force_storage_dtype and not rung
              and getattr(pipe, "_cozy_fp8_storage_requested", False)
              and getattr(pipe, "_cozy_fp8_storage_ok", True)):
            # th#1043: a joint shared-lane fit forced fp8 storage the binding
            # never asked for — report it structurally (FnDegraded) exactly like
            # an adaptive rung; a silent precision downgrade lies to placement.
            out.rung = "fp8"
            out.rung_detail = (
                f"joint shared-lane fit forced fp8 storage for slot {slot!r} "
                f"({type(pipe).__name__}); sibling lanes share the VRAM budget")

        # Worker-owned placement/offload policy: one decider for the whole
        # worker; endpoints never write device/offload code. A CUDA OOM inside
        # is a ladder transition, not a failure.
        if place and device.strip().lower() != "cpu":
            reporter.set_phase("place")
            out.placed = place_pipeline(
                pipe, mode=mode, ref=ref, strict_vram=strict_vram)
    finally:
        reporter.stop(clean=True)
    return out


def arm_route(mode: str) -> Optional[str]:
    """The name of the arm that serves cells of ``mode``, or ``None``.

    ONE registry, asked by :func:`arm_aot` when it arms and by the mint
    BEFORE it spends anything (``fleet_cells.mint_recipe``). pgw#827 is the
    fourth defect in the "a gate that models the arm differently from the
    arm" class (pgw#816, #822, #825): the regional recipe minted 72 entries
    in 354 s of L4 and only then discovered that this runtime had no arm
    that could adopt the kind of cell it had just built. "Can this runtime
    adopt the kind of cell I am about to mint?" is answerable at
    ``self_mint_started``, and it is answerable HERE, from the same table
    the arm dispatches on.

    pgw#846: regional cells are RETIRED, so the whole-graph arm is the only
    row. A cell whose metadata still says ``mode='regional'`` is declined BY
    NAME (``arm_aot`` stays eager and says why) — never handed to the
    whole-graph arm, whose denoiser-scope bind table it cannot use (pgw#827).
    """
    return {
        "": "aot_serve.enable",
    }.get(str(mode or ""))


def arm_aot(
    pipe: Any, cfg: Any, cache_dir: Optional[Path], artifact: Path,
    bucket: int, meta: Optional[Dict[str, Any]] = None,
    *, expected: "Optional[ExpectedIdentity]" = None,
    verify_numerics: bool = False,
) -> AdoptOutcome:
    """Arm ONE exported ``.pt2`` cell on ``pipe``. The whole AOT arm, in one
    place, for every source of such an artifact.

    ``verify_numerics`` (DESIGN-RULINGS §4.32, pgw#1141) runs the parity gate,
    and exactly ONE caller sets it: the pod that MINTED these bytes, before it
    publishes them. Adoption runs no quality gate at all — see
    :func:`gate_cell_numerics` for why re-measuring an adopted cell was taxing
    every adopter forever for an author's one-time mistake.

    pgw#805 extracted this from :func:`enable_compiled`'s kind dispatch: a
    cell this pod MINTED ITSELF has to arm through exactly the same gates a
    hub-delivered one does (that is the point of the delegated split — a
    child-built cell EARNS adoption), but it must not re-enter
    ``enable_compiled``, whose pgw#709 receipts gate would drop an artifact
    that by construction carries no hub signature yet.

    pgw#721: an exported cell rides the branch-bearing lane too — LoRA
    adapters are lifted to graph INPUTS, so one artifact serves the whole
    bucket. A lifted cell refuses an unlifted module
    (``lifted_inputs_unbindable``), so for a bucket-bearing endpoint the
    lifted binding is installed on the artifact's target module NOW — after
    ``apply_lora_lane`` allocated the canonical branch containers, BEFORE
    ``aot_serve.enable`` runs ``assert_lifted_contract`` (the exact C2 pod-10
    proven order). Rolled back on a failed arm so a dynamo fallthrough never
    traces a lifted forward it did not ask for.
    """
    # Deferred: hoisting drags aot_serve onto the `import gen_worker` path
    # (+39 modules).
    from .. import aot_serve

    if meta is None:
        meta = artifact_meta.try_read_metadata(artifact)
    lifted_target: Any = None
    lifted_installed = False
    #: pgw#999: why the lifted-binding install failed, if it did. Carried into
    #: the refusal instead of dying in a logger no pod exposes.
    lifted_install_error = ""
    mode = str((meta or {}).get("mode") or "")
    if arm_route(mode) is None:
        # A cell whose mode this runtime has no arm for must decline BY NAME
        # rather than be handed to whichever arm happens to be the default —
        # pgw#827 was exactly that: a regional cell routed into the
        # whole-graph arm, which built ONE bind table at denoiser scope.
        # Since pgw#846 retired regional, `mode='regional'` cells land here
        # and stay eager, which is the correct retirement semantics.
        logger.warning(
            "aot arm: artifact declares mode=%r, which this runtime has no "
            "arm for; staying eager", mode)
        return AdoptOutcome.miss(
            "no_arm_for_mode",
            f"artifact declares mode={mode!r}, which this runtime has no arm for")
    if bucket:
        from . import lora_lifted

        # The target module comes from the ARTIFACT's own recorded facts
        # ("module", else its first compile target) — never a hardcoded
        # component name (pgw#740: the vocabulary is not repeated in live
        # code; a guessed name on a non-UNet family would silently skip the
        # install and waste the arm).
        targets = [str(t) for t in ((meta or {}).get("targets") or ())]
        if not targets:
            # pgw#1001: a packed multi-entry cell records its targets PER
            # ENTRY and carries no top-level `targets`/`module` (measured:
            # both None on a real 5-entry lora64 cell). Without this the name
            # resolved to "" and the lifted install was silently skipped.
            seen: List[str] = []
            for entry in ((meta or {}).get("entries") or {}).values():
                name = str((entry or {}).get("target") or "").strip()
                if name and name not in seen:
                    seen.append(name)
            targets = seen
        # ...and among them the BRANCH-CAPABLE one: `decoder` sorts first
        # among entry names, and a lifted forward on a module with no branch
        # container fails by name. `branch_targets` is the authority.
        branch_capable = lora_lifted.branch_targets(pipe)
        module_name = str((meta or {}).get("module") or "")
        if not module_name:
            module_name = next(
                (t for t in targets if t in branch_capable),
                targets[0] if targets else "")
        lifted_target = (
            getattr(pipe, module_name, None) if module_name else None)
        if lifted_target is None:
            # pgw#1098: a declared bucket whose lifted target cannot be
            # RESOLVED is a refusal, never a skip. Falling through leaves the
            # module unlifted and hands `assert_lifted_contract` a guaranteed
            # `lifted_inputs_unbindable` — the gate then names itself and the
            # real cause (an envelope with no readable targets, a pipeline
            # missing the named component) is nowhere on the wire. That is
            # exactly how row 7 read as a LoRA-contract defect when the
            # envelope had simply not been read. pgw#1001 closed this hole for
            # its two known causes; this closes it for every future one, by
            # refusing to leave the branch silently.
            lifted_install_error = (
                f"no lifted target resolved: metadata names module="
                f"{str((meta or {}).get('module') or '') or '<absent>'} "
                f"targets={targets or '<absent>'}, branch-capable="
                f"{sorted(branch_capable) or '<none>'}"
                + ("; the cell envelope was unreadable" if meta is None else ""))
            logger.warning(
                "aot arm: bucket=%d declared but no lifted target resolved "
                "(%s); a lifted artifact will refuse at "
                "assert_lifted_contract", bucket, lifted_install_error)
        elif lora_lifted.lifted_binding(lifted_target) is None:
            try:
                lora_lifted.install_lifted_lora_forward(lifted_target, bucket)
                lifted_installed = True
            except Exception as exc:  # noqa: BLE001 — arm decides
                # pgw#999: KEPT, not merely logged. This branch predicted its
                # own downstream symptom ("will refuse at
                # assert_lifted_contract") and then discarded the cause, so
                # the refusal that follows names the gate that noticed rather
                # than the install that failed. Same discard as the one this
                # issue is closing, one frame deeper, on exactly the
                # bucket-bearing path a w8a8-lora64 family takes.
                lifted_install_error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "aot arm: lifted-binding install failed on %r (%s); a "
                    "lifted artifact will refuse at assert_lifted_contract",
                    module_name, exc)
    # pgw#1168: THE ADOPT'S DEVICE COST, MEASURED AT THE ONE SEAM EVERY ARM
    # ROUTE PASSES. pgw#1164 measured this in `fleet_cells.adopt_delegated_mint`
    # — the SELF-MINT adopt only — so the boot adopt, the local-store adopt and
    # the re-arm ran the identical `aot_serve.enable` -> `load_and_wrap` and
    # reported nothing. That is this program's most common defect shape (an
    # emitter wired on one of N paths), and the fix is one emitter here rather
    # than a second call site there.
    #
    # The two terms are measured SEPARATELY because they answer different
    # questions, and the split is what decides whether the CELL or the GATE is
    # the problem (th#1825):
    #   load   — every loaded entry runner, which EVERY serving pod pays for the
    #            life of the arm. This is the term that decides whether a cell
    #            fits on the fleet it was built for.
    #   verify — the §4.32 parity gate's two forwards, paid ONLY on the minting
    #            pod (`verify_numerics=True`), never by an adopter.
    # A boot adopt therefore reports `verify=0` by construction, and that row —
    # taken on the card the fleet actually serves on — is the empirical answer
    # to "does this cell fit", where before there was only arithmetic.
    _budget_device = mint_workers.device_of(pipe)
    _resident_before, _ = mint_workers.adopt_watermark(_budget_device)
    _load_bytes = 0
    _emitted = False

    def _emit_adopt_budget(verify_bytes: int, armed: bool) -> None:
        """One `cell_adopt_budget` row per arm attempt, whatever the outcome.

        Emitted even for a REFUSED arm: the device high-water was paid either
        way, and a refusal is exactly when the number is most worth having.
        """
        nonlocal _emitted
        if _emitted:
            return
        _emitted = True
        total = int(_load_bytes) + max(0, int(verify_bytes))
        if total <= 0:
            return
        family = str((meta or {}).get("family") or getattr(cfg, "family", "") or "")
        # The cell's OWN recorded lane.
        lane = str((meta or {}).get("weight_lane") or "")
        entries = len((meta or {}).get("entries") or {})
        gib = 1 << 30
        activity_mod.emit_event(
            "cell_adopt_budget",
            f"family={family} lane={lane or '(plain)'} entries={entries} "
            f"adopt_device_peak={total / gib:.3f}GiB "
            f"load={_load_bytes / gib:.3f}GiB "
            f"verify={max(0, int(verify_bytes)) / gib:.3f}GiB "
            f"resident_before={_resident_before / gib:.3f}GiB "
            f"verified={bool(verify_numerics)} armed={bool(armed)} "
            f"basis=measured — `load` is what EVERY adopting pod pays and is "
            f"the term that decides whether this cell fits its serving fleet; "
            f"`verify` is the §4.32 gate and is paid only by the minting pod.",
            phase="measured",
        )

    # §4.33 / pgw#1175: THE HEADROOM GATE IS GONE, and the ATTEMPT replaces it.
    # `mint_budget.adopt_headroom` refused an arm here on `2 * activation`,
    # where `activation` was a quarter of the RESIDENT SET whenever no forward
    # had run — a fraction its own docstring called unmeasured — and its
    # refusal was STICKY for the life of the process. Its own text conceded it
    # "CANNOT refuse a card that merely cannot hold 36 runners", i.e. it could
    # not refuse the failure it was written for (th#1825) and could refuse
    # cards that were fine. The honest gate is the bind itself:
    # `aot_serve.load_and_wrap` attempts each entry and returns a typed
    # `insufficient_adopt_vram` miss on a real device OOM, before any live
    # mutation, and this pod serves eager exactly as it did — on evidence.
    outcome = aot_serve.enable(pipe, cfg, cache_dir, artifact, expected=expected)
    _, _peak_after_load = mint_workers.adopt_watermark(_budget_device)
    _load_bytes = max(0, _peak_after_load - _resident_before)
    if not outcome.armed and lifted_install_error:
        # The refusal is real; its ROOT is one frame up. Both, in the order a
        # reader needs them: what refused, and what made it refuse.
        outcome = AdoptOutcome.miss(
            outcome.reason or "lifted_install_failed",
            f"{outcome.detail} [root: the lifted binding was never installed"
            + (f" on {module_name!r}" if module_name else "")
            + f" — {lifted_install_error}]".strip(),
            outcome.identity)
    if outcome.armed:
        # §4.32: quality is proven at MINT and in author CI, never at adoption.
        # An adopting pod materializes, arms and serves.
        gate_ok = not verify_numerics or gate_cell_numerics(pipe, cfg, strict=True)
        _, _peak_after_verify = mint_workers.adopt_watermark(_budget_device)
        if gate_ok:
            _emit_adopt_budget(_peak_after_verify - _peak_after_load, True)
            return outcome
        _emit_adopt_budget(_peak_after_verify - _peak_after_load, False)
        # A refused cell is UNARMED, not merely reported: the whole point is
        # that it must not serve. Staying eager is the ordinary miss policy
        # every other adopt gate uses, so the tenant keeps being served.
        #
        # pgw#923: and the ADOPT ledger agrees with it by CONSTRUCTION now.
        # `enable` used to announce `aot_adopt phase=armed` before the gate
        # ran, so a reader counting armed adoptions over-counted every numerics
        # refusal and a second closing row existed only to correct the first.
        # Nothing is announced until the arm is final, so the refusal is simply
        # what this function returns; the numbers still ride `cell_numerics`.
        meta = aot_serve.armed_metadata(pipe)
        aot_serve.unwrap(pipe)
        outcome = AdoptOutcome.miss(
            "numerics_refused",
            f"family={meta.get('family')} key={meta.get('cell_key')}: this pod "
            f"MINTED these bytes and they do not reproduce the eager forward "
            f"they were traced from — nothing is published and this pod serves "
            f"eager (pgw#868, §4.32)",
            outcome.identity)
    # An arm that never reached the gate (refused at `enable`) still paid the
    # load, so it still reports — `_emit_adopt_budget` dedupes, so the armed
    # paths above have already had their say.
    _emit_adopt_budget(0, bool(outcome.armed))
    if lifted_installed:
        from . import lora_lifted

        lora_lifted.remove_lifted_lora_forward(lifted_target)
    return outcome


def gate_cell_numerics(pipe: Any, cfg: Any, *, strict: bool = False) -> bool:
    """THE numerics gate (pgw#868): does this cell reproduce the eager forward
    it replaces? Returns False when it must not serve — and, on the only path
    that runs it, when it must not be PUBLISHED.

    **It runs on the MINTING pod, and nowhere else (DESIGN-RULINGS §4.32).**
    It used to run at every ADOPT, on the reasoning that a mint-only gate
    "cannot protect an adopting pod whose weights, lane and card differ". Paul
    overruled that: every failure this gate has ever caught (a baked
    `conv_out.bias`, timestep dtype scars) was an AUTHOR defect in endpoint
    code or config, and re-measuring on every adopter taxes the whole fleet
    forever for one author's one-time mistake. It is not the consumer's job to
    catch the author's bugs.

    What makes adoption safe without re-measuring is CONSTRUCTION, not identity
    — a ck1 key is graph x envelope x sm x toolchain and carries NO checkpoint
    hash, deliberately, so one cell serves every checkpoint of the
    architecture:

    * the cell is compiled CODE; weights flow through it as data (call inputs
      and arm-time-bound constants), so a mint-time parity proof proves the
      FUNCTION and transfers to any checkpoint that function accepts;
    * the one way that breaks — a weight VALUE baked into the artifact — is
      structurally fenced fail-closed by the constant-folding fence (0.100.0),
      not policed by measurement;
    * a checkpoint that changes the COMPUTATION (a config flag that alters the
      traced graph) hashes to a different graph, hence a different key, hence
      no match at all. The graph axis protects there, not a checkpoint digest.

    ``strict`` is what the mint path passes, and §4.32 requires it: identical
    or refuse, with no DEGRADED-publish band. A cell that lands in the gray
    band is one an ADOPTER can never re-check, so shipping it would export an
    unmeasured degradation to every pod that pulls it.

    Outcomes, all typed rows on the wire:

    * HEALTHY -> `cell_numerics phase=checked`; the mint arms and publishes.
      The pass is announced deliberately: an unannounced pass is
      indistinguishable from a gate that never ran, which is this program's
      signature failure.
    * DEGRADED -> `phase=degraded`, and under ``strict`` it REFUSES.
    * DESTROYED / unmeasurable -> refuse. `numerics_ladder.gate` raises the
      typed refusal below the floor; a probe that could not be TAKEN is
      refused on its own `phase=unmeasurable`, because "nobody could ask" is
      not "it passed".
    """
    # Deferred: +38 modules on the `import gen_worker` path if hoisted.
    from .. import aot_serve, numerics_ladder
    # Deferred: +2 modules on the `import gen_worker` path if hoisted.
    from .. import numerics_probe

    family = str(getattr(cfg, "family", "") or "")
    try:
        report = numerics_probe.probe_cell(
            pipe, cfg, aot_serve.armed_metadata(pipe))
    except numerics_probe.ProbeUnavailable as exc:
        return _refuse_unmeasurable(family, exc.reason, str(exc))
    except Exception as exc:  # noqa: BLE001 — an unexplained probe is a refusal
        # Deliberately NOT best-effort. The pgw#848 announcement could swallow
        # anything because it refused nothing; a GATE that swallowed an error
        # into an armed cell would be the exact hole this replaces.
        return _refuse_unmeasurable(
            family, "probe_error", f"{type(exc).__name__}: {exc}")
    if not report.measured:
        rows = "; ".join(
            f"{v.axis.name}: {v.reason} ({v.detail})"
            for v in report.unmeasured[:6])
        return _refuse_unmeasurable(
            family, (report.unmeasured[0].reason if report.unmeasured
                     else "no_axis_measured"),
            f"{report.context()} | unmeasured: {rows}")
    comparison = report.comparison()
    try:
        numerics_ladder.gate(
            comparison,
            kind=activity_mod.KIND_CELL_NUMERICS,
            refuse=lambda detail, worst_row: numerics_probe.CellNumericsRefused(
                detail, worst_row),
            context=report.context())
        if strict and comparison is not None and not comparison.healthy:
            # §4.32: identical or refuse. The ladder already emitted the
            # `degraded` row above, so the confession is on the wire; what
            # changes here is that the bytes do not ship.
            logger.error(
                "aot mint: REFUSING to publish %s — the cell it just compiled "
                "lands in the gray band (%s), and an adopter runs no gate that "
                "could re-check it (§4.32)", family or "cell", comparison.verdict)
            return False
        if comparison is not None and comparison.healthy:
            activity_mod.emit_event(
                activity_mod.KIND_CELL_NUMERICS,
                f"CHECKED against eager on every packaged entry — "
                f"{report.context()}",
                phase="checked", duration_ms=report.elapsed_ms)
    except numerics_probe.CellNumericsRefused as exc:
        logger.error(
            "aot arm: REFUSING to arm %s — the cell does not reproduce its "
            "eager reference: %s", family or "cell", exc)
        return False
    except Exception as exc:  # noqa: BLE001 — a gate that errored is a refusal
        # Includes a raising activity sink, and the announcement is INSIDE the
        # try on purpose: an arm nobody could record is an arm we do not make.
        # A telemetry failure cannot be told apart from a logic failure here,
        # and the two costs are not symmetric — refusing costs an un-armed cell
        # (the ordinary miss policy), passing costs a silently-wrong one.
        return _refuse_unmeasurable(
            family, "gate_error",
            f"{type(exc).__name__}: {exc} | {report.context()}")
    return True


def _refuse_unmeasurable(family: str, reason: str, detail: str) -> bool:
    """A cell that could not be measured does not arm — and says which half.

    Fail-closed by construction: the only way to reach an armed cell is
    through a comparison that exists. An exception swallowed into a `True`
    here would rebuild the exact hole pgw#848 CP12 refused to ship.
    """

    logger.error(
        "aot arm: REFUSING to arm %s — the numerics gate could not be run "
        "(%s): %s", family or "cell", reason, detail)
    try:
        activity_mod.emit_event(
            activity_mod.KIND_CELL_NUMERICS,
            f"family={family} REFUSED TO ARM: the compiled-vs-eager "
            f"comparison could not be taken ({reason}). This is not a pass — "
            f"an unmeasurable cell stays eager (pgw#868). {detail}",
            phase="unmeasurable")
    except Exception:  # noqa: BLE001 — the refusal stands even if the wire is down
        logger.debug("could not announce unmeasurable numerics", exc_info=True)
    return False


def enable_compiled(
    pipe: Any, cfg: Any, cache_dir: Optional[Path] = None,
    artifact: Optional[Path] = None,
) -> AdoptOutcome:
    """Arm the best available compiled path for a freshly loaded pipeline:
    an AOTI export swaps the module (fail-soft), anything else goes
    through the torch.compile cache policy (which also covers the no-
    artifact and ALLOW_COLD lanes).

    ``Compile.lora_bucket`` (gw#561) puts the pipeline on the branch-bearing
    graph family BEFORE arming, so only matching ``-lora<bucket>`` cells
    adopt. Staying eager rolls the branches back — canonical zeroed slots
    cost +21-32% eager (gw#547); the eager adapter path re-enables sparse
    placement per request."""
    from .. import aot_serve, compile_cache  # lazy: keeps `import gen_worker` off the compile/pb stack
    # Deferred: receipts pulls +151 modules onto the `import gen_worker` path.
    from .. import receipts

    # pgw#709: hub-delivered artifacts must carry a verifiable hub-signed
    # receipt (signature + blake3/size + key binding + revocation check).
    # A refused artifact is DROPPED — the ordinary miss policy (fleet
    # self-mint / eager / typed refusal) takes over. No-op when the gate is
    # unconfigured (cozy-local, CLI, unit rigs — local trust model).
    if artifact is not None and not receipts.gate_delivered_artifact(
        Path(artifact), family=str(getattr(cfg, "family", "") or "")
    ):
        artifact = None

    refused: Optional[AdoptOutcome] = None
    bucket = int(getattr(cfg, "lora_bucket", 0) or 0)
    if bucket and not compile_cache.has_compile_target(pipe, cfg):
        # gw#627 live find: this arming runs for EVERY worker-loaded setup
        # slot — a bare component slot (sdxl's AutoencoderKL vae) resolves
        # none of cfg.targets and must stay branchless-eager, not raise.
        # The loud no-denoiser error remains for real compile targets.
        bucket = 0
    if bucket:
        compile_cache.apply_lora_execution_lane(pipe, bucket)
    if artifact is not None:
        # ONE kind sniff for every non-inductor backend. `metadata.json` is
        # the shared envelope member across all artifact kinds (the pgw#709
        # receipts gate above reads it from the same place), so `kind` is the
        # dispatch key: absent/unknown falls through to the inductor lane.
        meta = artifact_meta.try_read_metadata(artifact)
        kind = str((meta or {}).get("kind") or "")
        if kind == aot_serve.ARTIFACT_KIND:
            aot = arm_aot(pipe, cfg, cache_dir, Path(artifact), bucket, meta)
            if aot.armed:
                return aot
            refused = aot
            artifact = None  # unusable artifact: fall through to eager policy
    # pgw#1181: `compile_cache.enable` no longer takes an artifact — the
    # `torch-inductor-cache` format it seeded has had no writer since pgw#1178
    # and is deleted. Everything delivered is dispatched above by
    # `metadata.json`'s `kind`; what reaches here is the JIT lane.
    armed = compile_cache.enable(pipe, cfg)
    if bucket and not armed:
        compile_cache.drop_lora_execution_lane(pipe)
    if armed:
        return AdoptOutcome.hit()
    # The inductor lane declines without a classified token of its own — "no
    # delivered cell for this identity" is the whole answer. A prior typed
    # refusal from the exported arm is the more specific one and survives.
    return refused if refused is not None else AdoptOutcome.miss("no_cell")


# ---------------------------------------------------------------------------
# pgw#517: the arming seam for SELF-loaded pipelines. `enable_compiled`
# above is what the executor calls automatically for a worker-loaded
# (pipeline-class-annotated) setup() slot; a str/Path-annotated slot never
# builds a `pipe` the executor can see (the endpoint's own setup() does),
# so that arming call is unreachable for it. `arm_compile` is the same
# policy exposed to the endpoint itself: an explicit, ctx-less call the
# author makes at the end of setup(), mirroring `place_pipeline`'s existing
# "worker-owned policy, endpoint invokes it directly" pattern for self-
# loaded pipelines (`gen_worker.models.memory.place_pipeline`).
#
# The (Compile, cache_dir, compile-artifact) triple `enable_compiled` needs
# are executor/CLI internals an endpoint has no business constructing
# itself — a `contextvars.ContextVar` carries them instead, scoped by the
# caller (executor/CLI) to exactly the `setup()` call, so `arm_compile(pipe)`
# needs no parameter beyond the pipeline and cannot leak past setup().
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ArmingContext:
    compile: Any
    cache_dir: Optional[Path]
    artifact: Optional[Path]
    # The executor owns this list and reads it only after setup() returns.
    # Capturing the exact objects passed to arm_compile() is what lets
    # self-loading endpoints participate in object-scoped compile targets;
    # inferring them later from class attributes would be ambiguous.
    objects: list[tuple[Any, bool]]
    # gw#587: the scope owner's arming policy. The executor routes the fleet
    # policy (delivered cell first, self-mint on miss) here so an endpoint's
    # own arm_compile() call gets the SAME behavior as a worker-loaded slot;
    # None keeps the bare delivered-artifact policy (CLI / unit rigs). The
    # callable may return a bool or an object with `.armed`/`.self_mint`
    # (fleet_cells.ArmOutcome) — provision cannot import fleet_cells (cycle).
    enable: Optional[Callable[[Any, Any, Optional[Path], Optional[Path]], Any]]
    # id(pipe) -> self-mint identity (fleet_cells.SelfMint) for pipes the
    # scope's policy armed from their OWN mint rather than a delivered cell.
    self_mints: dict[int, Any]
    # id(pipe) -> caught CellSelectionBugError (th#1031): the fleet policy
    # no longer raises this (it self-mints instead), so the executor reads
    # it here to still send the loud th#883 wire event after setup() returns.
    selection_bugs: dict[int, Any]


_ARMING_CTX: "contextvars.ContextVar[Optional[_ArmingContext]]" = contextvars.ContextVar(
    "gen_worker_compile_arming_ctx", default=None
)


class ArmingScope:
    """Context manager the executor/CLI holds open around one ``setup()``
    call so ``arm_compile()`` can reach the active ``Compile`` spec, compile
    cache dir, and any hub-attached artifact. Re-entrant-safe (a nested
    scope restores the outer one on exit); a no-op when ``compile`` is
    ``None`` so callers can open it unconditionally."""

    def __init__(
        self, compile: Any, cache_dir: Optional[Path] = None,
        artifact: Optional[Path] = None,
        enable: Optional[
            Callable[[Any, Any, Optional[Path], Optional[Path]], Any]
        ] = None,
    ) -> None:
        self._objects: list[tuple[Any, bool]] = []
        self._self_mints: dict[int, Any] = {}
        self._selection_bugs: dict[int, Any] = {}
        self._value = (
            _ArmingContext(
                compile=compile,
                cache_dir=cache_dir,
                artifact=artifact,
                objects=self._objects,
                enable=enable,
                self_mints=self._self_mints,
                selection_bugs=self._selection_bugs,
            )
            if compile is not None else None
        )
        self._token: Optional["contextvars.Token[Optional[_ArmingContext]]"] = None

    def __enter__(self) -> "ArmingScope":
        if self._value is not None:
            self._token = _ARMING_CTX.set(self._value)
        return self

    def __exit__(self, *_exc_info: Any) -> None:
        if self._token is not None:
            _ARMING_CTX.reset(self._token)
            self._token = None

    @property
    def objects(self) -> tuple[tuple[Any, bool], ...]:
        """Exact ``(pipeline, armed)`` observations from this setup scope."""
        return tuple(self._objects)

    @property
    def self_mints(self) -> dict[int, Any]:
        """``id(pipe) -> SelfMint`` for scope pipes armed from their own mint."""
        return dict(self._self_mints)

    @property
    def selection_bugs(self) -> dict[int, Any]:
        """``id(pipe) -> CellSelectionBugError`` caught (and recovered from
        via self-mint) for scope pipes, th#1031."""
        return dict(self._selection_bugs)


def arm_compile(pipe: Any) -> bool:
    """Arm ``@endpoint(compile=Compile(...))`` on a pipeline the endpoint
    loaded and placed itself (a str/Path-annotated ``setup()`` slot the
    executor never materializes — pgw#517). Call once per pipeline object,
    at the end of ``setup()``, after placement. Same cache-artifact-gated
    policy as the automatic worker-loaded path: arms only when a verified
    compiled artifact for (family, SKU, torch, triton) is seeded, otherwise
    stays eager. Returns whether a compiled path was armed.

    ie#522 (Paul's ruling, 2026-07-22): the endpoint's own ``setup()`` call
    is a fixed declaration of intent ("this pipeline is compile-eligible");
    whether that intent is ACTIVE is the release's decision (an eager
    registration declares no ``compile=Compile(...)`` at all, so the
    executor's ``ArmingScope`` opens as a no-op — see ``ArmingScope``'s own
    "no-op when compile is None" contract above). A no-op scope must mean a
    no-op ``arm_compile()``, not a crash: every ``@endpoint`` with a
    self-loading setup() calls this unconditionally (sdxl, wan-2.2, ...),
    and each of those must keep working under an eager registration exactly
    like the automatic worker-loaded path already does. No active scope ->
    log once at info and return False (never armed) — never raise."""
    ctx = _ARMING_CTX.get()
    if ctx is None:
        logger.info(
            "gen_worker.arm_compile(): no active compile-arming scope "
            "(this release is eager — no compile=Compile(...) declared); "
            "staying eager for %r", type(pipe).__name__,
        )
        return False
    enable = ctx.enable if ctx.enable is not None else enable_compiled
    outcome = enable(pipe, ctx.compile, ctx.cache_dir, ctx.artifact)
    armed = bool(getattr(outcome, "armed", outcome))
    mint = getattr(outcome, "self_mint", None)
    if mint is not None:
        ctx.self_mints[id(pipe)] = mint
    bug = getattr(outcome, "selection_bug", None)
    if bug is not None:
        ctx.selection_bugs[id(pipe)] = bug
    ctx.objects.append((pipe, armed))
    return armed


# ---------------------------------------------------------------------------
# pgw#1104: the APPLIED-LANE report. `metrics.lane` used to be a pure function
# of the binding, so a recipe that quantized in setup() served fp8 under a
# bf16 label — and the lane id is a KEY (th#935 verdicts, compile cells,
# pricing, the executed-lane proof). A static `handles=`-style declaration
# cannot fix it: the recipe is runtime-gated (sm89 for w8a8, the compile
# preflight), so a declaration would over-claim on the card that skips it.
# Only the code that converted the weights can report provably, so it does —
# through the same contextvar scope `arm_compile` uses, so the report is
# attributed to exactly the setup() that made it and cannot be forged later.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _AppliedLaneContext:
    applied: list[Any]  # list[execution_lanes.AppliedLane]; owned by the scope


_APPLIED_LANE_CTX: "contextvars.ContextVar[Optional[_AppliedLaneContext]]" = (
    contextvars.ContextVar("gen_worker_applied_lane_ctx", default=None)
)


class AppliedLaneScope:
    """Context manager the executor/CLI holds open around one ``setup()`` call
    so ``report_applied_lane()`` lands on that instance. Re-entrant-safe."""

    def __init__(self) -> None:
        self._applied: list[Any] = []
        self._value = _AppliedLaneContext(applied=self._applied)
        self._token: Optional["contextvars.Token[Optional[_AppliedLaneContext]]"] = None

    def __enter__(self) -> "AppliedLaneScope":
        self._token = _APPLIED_LANE_CTX.set(self._value)
        return self

    def __exit__(self, *_exc_info: Any) -> None:
        if self._token is not None:
            _APPLIED_LANE_CTX.reset(self._token)
            self._token = None

    @property
    def applied(self) -> tuple[Any, ...]:
        """Every ``AppliedLane`` reported inside this setup scope, in order."""
        return tuple(self._applied)


def report_applied_lane(
    component: str,
    lane_body: str,
    *,
    modules: int = 0,
    kept_bf16: int = 0,
) -> bool:
    """Report the lane a serve-time recipe just APPLIED to ``component``'s
    weights. Call it from ``setup()`` immediately after the conversion
    returns — the way ``arm_compile()`` is called after placement.

    ``lane_body`` is one of ``known_execution_lane_bodies()`` (the th#1050
    vocabulary, e.g. ``"fp8-w8a8-dynamic"``); an unknown token raises
    ``ValueError`` — the lane vocabulary is shared with the hub and is never
    extended from an endpoint. The execution axis is NOT the author's: the
    worker composes ``+compiled``/``+eager`` from live compile state.

    Returns whether the report was recorded. Outside a setup scope (hub-less
    ``cozy run``, a unit rig) it logs once and returns False — never raises,
    so every endpoint can call it unconditionally."""
    from . import execution_lanes

    body = str(lane_body or "").strip().lower()
    if not execution_lanes.valid_execution_lane_body(body):
        raise ValueError(
            f"report_applied_lane({component!r}, {lane_body!r}): not a known "
            "lane body (known: "
            f"{', '.join(execution_lanes.known_execution_lane_bodies())})")
    ctx = _APPLIED_LANE_CTX.get()
    if ctx is None:
        logger.info(
            "gen_worker.report_applied_lane(): no active setup scope; "
            "%s applied %s is not attributed to an instance", component, body)
        return False
    ctx.applied.append(execution_lanes.AppliedLane(
        component=str(component or "").strip() or "instance",
        body=body,
        modules=max(0, int(modules)),
        kept_bf16=max(0, int(kept_bf16)),
    ))
    return True


# ---------------------------------------------------------------------------
# The attention axis (pgw#1043 §PRODUCTIZATION) — same shape as the lane report
# above, deliberately: only the code that INSTALLED the attention path can prove
# what it installed, and a static declaration would over-claim on a card whose
# kernel gate refused (the exact reason pgw#1104 rejected position 2).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _AppliedAttentionContext:
    applied: list[Any]  # list[attention_modes.AppliedAttention]


_APPLIED_ATTENTION_CTX: (
    "contextvars.ContextVar[Optional[_AppliedAttentionContext]]"
) = contextvars.ContextVar("gen_worker_applied_attention_ctx", default=None)


class AppliedAttentionScope:
    """Held open by the executor around one ``setup()`` so a report lands on
    that instance and cannot be forged from a handler or a background thread."""

    def __init__(self) -> None:
        self._applied: list[Any] = []
        self._value = _AppliedAttentionContext(applied=self._applied)
        self._token: Optional[
            "contextvars.Token[Optional[_AppliedAttentionContext]]"] = None

    def __enter__(self) -> "AppliedAttentionScope":
        self._token = _APPLIED_ATTENTION_CTX.set(self._value)
        return self

    def __exit__(self, *_exc_info: Any) -> None:
        if self._token is not None:
            _APPLIED_ATTENTION_CTX.reset(self._token)
            self._token = None

    @property
    def applied(self) -> tuple[Any, ...]:
        return tuple(self._applied)


def report_applied_attention(
    component: str,
    mode: str,
    *,
    k_blocks: int = 0,
    block_size: int = 0,
    density: float = 0.0,
    selector: str = "",
    index_ref: str = "",
) -> bool:
    """Report the attention path that was actually INSTALLED on ``component``.

    Call it from ``setup()`` right after the processor/dispatch is patched —
    the way ``report_applied_lane()`` is called after ``quantize_()`` returns.
    ``mode`` is ``"dense"`` or ``"sparse-k<N>"``; an ungrammatical token raises
    ``ValueError``. Reporting nothing means dense, so no endpoint is obliged to
    call this.

    ``density`` is the MEASURED kept fraction, not the budget: ``k`` is what was
    asked for and the density is what the geometry produced, and the wall is a
    function of the second. Returns whether the report was recorded; outside a
    setup scope it logs once and returns False rather than raising."""
    from . import attention_modes

    tok = str(mode or "").strip().lower()
    if not attention_modes.valid_attention_mode(tok):
        raise ValueError(
            f"report_applied_attention({component!r}, {mode!r}): not a valid "
            "attention mode (expected 'dense' or 'sparse-k<N>')")
    k = attention_modes.sparse_k_of(tok)
    if k is not None and k_blocks and int(k_blocks) != k:
        raise ValueError(
            f"report_applied_attention({component!r}, {mode!r}): k_blocks="
            f"{k_blocks} contradicts the mode token")
    ctx = _APPLIED_ATTENTION_CTX.get()
    if ctx is None:
        logger.info(
            "gen_worker.report_applied_attention(): no active setup scope; "
            "%s applied %s is not attributed to an instance", component, tok)
        return False
    ctx.applied.append(attention_modes.AppliedAttention(
        component=str(component or "").strip() or "instance",
        mode=tok,
        k_blocks=int(k_blocks or k or 0),
        block_size=max(0, int(block_size)),
        density=max(0.0, float(density)),
        selector=str(selector or "").strip(),
        index_ref=str(index_ref or "").strip(),
    ))
    return True


# ---------------------------------------------------------------------------
# Standalone (hub-less) resolution — the CLI's half. The executor's bytes
# come from orchestrator-resolved snapshots via ModelStore.ensure_local.
# ---------------------------------------------------------------------------


def resolve_bindings(
    bindings: Mapping[str, Any],
    *,
    offline: bool,
    emit: EmitFn,
    slots: Optional[Mapping[str, Any]] = None,
    payload: Any = None,
) -> Dict[str, str]:
    """Resolve every binding to a local path / loader-ready string.

    ``slots``/``payload`` (pgw#520): when a binding's slot is Slot-declared
    with a ``selected_by`` field, and this hub-less run has no hub to
    resolve a curated/BYOM pick against, a payload that actually NAMES a
    model (a non-empty ``selected_by`` field value) is a clear usage error
    instead of silently running the slot's default — ``cozy run`` only ever
    runs a Slot's ``default_checkpoint`` ref locally.
    """

    out: Dict[str, str] = {}
    for param_name, binding in bindings.items():
        slot = (slots or {}).get(param_name)
        selected_by = str(getattr(slot, "selected_by", "") or "") if slot is not None else ""
        if selected_by and payload is not None:
            picked = str(getattr(payload, selected_by, "") or "").strip()
            if picked:
                raise ModelResolutionError(
                    f"slot {param_name!r}: payload names model {picked!r} via "
                    f"{selected_by!r}, but no hub is configured — "
                    "hub-less mode (`cozy run` / `gen-worker run`) only runs "
                    "a Slot's default_checkpoint= ref; configure HUB= (or "
                    f"drop the {selected_by!r} field) to run against a hub."
                )
        if not isinstance(binding, ModelRef):
            raise ModelResolutionError(
                f"unknown binding type for param {param_name!r}: "
                f"{type(binding).__name__}"
            )
        out[param_name] = resolve_local_path(
            ref=wire_ref(binding), provider=binding.source,
            offline=offline, emit=emit,
            allow_patterns=tuple(getattr(binding, "files", ()) or ()),
            components=tuple(getattr(binding, "components", ()) or ()),
            civitai_version_id=str(getattr(binding, "version", "") or ""),
        )
    return out


def _hub_ref_map_path(cache_dir: Path, thref: Any) -> Path:
    """CAS-local memory of tag->snapshot resolutions, so a previously-fetched
    tag ref keeps working offline: cas/refs/<owner>/<repo>/<tag>."""
    name = str(thref.tag or DEFAULT_REF_TAG)
    safe = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in name)
    return cache_dir / "refs" / str(thref.owner) / str(thref.repo) / safe


def _remember_hub_ref(cache_dir: Path, thref: Any, digest: str) -> None:
    try:
        p = _hub_ref_map_path(cache_dir, thref)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(digest)
    except OSError:
        pass


def _fetch_tensorhub_snapshot(
    thref: Any, *, cache_dir: Path, emit: EmitFn, components: Tuple[str, ...] = (),
) -> str:
    """Resolve a Hub ref via th#560 and download its snapshot into the CAS.

    One re-resolve retry on a presigned-URL expiry mid-download (the same
    contract the orchestrator honors on ``url_expired``).

    ``components`` (pgw#505): th#560's resolve route always returns the
    FULL repo manifest today (selective CAS resolve is the hub-side
    desired-snapshot scoping — a separate, not-yet-built platform change).
    Until then this narrows client-side: the worker fully owns this
    resolve+download+materialize loop (unlike the production executor path,
    which digest-verifies against an orchestrator-issued file list), so it
    can safely fetch only the declared components — ``ensure_snapshot_async``
    keys the materialized directory by ``(digest, components)`` so a partial
    fetch never collides with a full one of the same ref. NOTE: offline
    reuse (``--offline`` / the ``_hub_ref_map_path`` tag memory below) only
    covers the FULL-repo case — a components=-scoped ref must be fetched
    online at least once per component set.
    """
    # Deferred: cozy_snapshot pulls +305 modules onto the `import gen_worker`
    # path — the single largest boot-cost import in the SDK.
    from .cozy_snapshot import ensure_snapshot_async, snapshot_dir_key
    # Deferred: hub_client pulls +129 modules onto the `import gen_worker` path.
    from .hub_client import HubResolveError, resolve_repo

    canonical = thref.canonical()

    def _resolve() -> Any:
        try:
            return resolve_repo(thref)
        except HubResolveError as e:
            raise ModelResolutionError(str(e)) from e

    emit({"kind": "model_fetch.started", "ref": canonical, "provider": "tensorhub"})
    resolved = _resolve()

    # Already materialized under the resolved (digest, components) key? No download.
    key = snapshot_dir_key(resolved.snapshot_digest, components)
    snap_dir = cache_dir / "snapshots" / key
    if snap_dir.exists():
        if not components:
            _remember_hub_ref(cache_dir, thref, resolved.snapshot_digest)
        emit({"kind": "model_fetch.completed", "ref": canonical,
              "provider": "tensorhub", "local_dir": str(snap_dir)})
        return str(snap_dir)

    last_at = [0.0]

    def _progress(done: int, total: Optional[int]) -> None:
        now = time.monotonic()
        if now - last_at[0] < 1.0 and (total is None or done < total):
            return
        last_at[0] = now
        emit({"kind": "model_fetch.progress", "ref": canonical,
              "provider": "tensorhub", "done_bytes": int(done),
              "total_bytes": int(total) if total else None})

    async def _download(res: Any) -> Path:
        return await ensure_snapshot_async(
            base_dir=cache_dir, ref=thref, resolved=res, progress=_progress,
            components=components,
        )

    try:
        try:
            snap = asyncio.run(_download(resolved))
        except UrlExpiredError:
            emit({"kind": "model_fetch.reresolve", "ref": canonical,
                  "provider": "tensorhub", "reason": "url_expired"})
            snap = asyncio.run(_download(_resolve()))
    except ModelResolutionError:
        raise
    except Exception as e:
        raise ModelResolutionError(
            f"failed to download tensorhub snapshot for {canonical}: {e}"
        ) from e
    if not components:
        _remember_hub_ref(cache_dir, thref, resolved.snapshot_digest)
    emit({"kind": "model_fetch.completed", "ref": canonical,
          "provider": "tensorhub", "local_dir": str(snap)})
    return str(snap)


def resolve_local_path(
    *, ref: str, provider: str, offline: bool, emit: EmitFn,
    allow_patterns: Tuple[str, ...] = (),
    components: Tuple[str, ...] = (),
    civitai_version_id: str = "",
) -> str:
    """Resolve one model ref to a local snapshot dir / loader-ready string.

    Order matches the live worker:
      1. local CAS lookup (digest-pinned snapshot dirs).
      2. HF refs → ``download_hf`` (auto-fetches from HF).
      3. ModelScope refs → ``modelscope.snapshot_download``.
      4. Cozy refs missing from CAS: standalone resolve against tensorhub's
         public resolve route (th#560); ``--offline`` stays CAS-only (exit 3).
      5. Civitai refs → model → latest-version lookup (or the pinned
         version), then ``download_civitai``.

    ``components`` (pgw#505) narrows an HF/tensorhub fetch to the named
    pipeline component subfolders (+ root config files) — see
    ``download.select_component_paths`` / ``cozy_snapshot.snapshot_dir_key``.
    """

    configured_cas = current_or(_STANDALONE).tensorhub_cas_dir.strip()
    cache_dir = Path(configured_cas) if configured_cas else Path(tensorhub_cas_dir())

    # Decode the bare ref into typed parts using the explicit provider.
    # No string-prefix sniffing — provider is the source of truth.
    try:
        parsed = parse_model_ref(ref, provider=provider)
    except Exception as e:
        raise ModelResolutionError(
            f"failed to parse model ref {ref!r} (provider={provider!r}): {e}"
        ) from e

    if parsed.provider == "tensorhub" and parsed.tensorhub and parsed.tensorhub.digest:
        # Snapshot dirs are keyed by the bare hex digest (no algo prefix).
        digest = parsed.tensorhub.digest.split(":", 1)[-1]
        snap_dir = cache_dir / "snapshots" / digest
        if snap_dir.exists():
            return str(snap_dir)

    # HF refs: fall through to the shared HF downloader.
    if parsed.provider == "hf" and parsed.hf is not None:
        if offline:
            # Best-effort: check the HF cache (huggingface_hub manages this
            # itself; a cache hit returns a path, miss raises).
            patterns = list(allow_patterns)
            if components and not patterns:
                patterns = [f"{c}/" for c in components] + ["*.json"]
            try:
                from ..net import hf
                p = hf().snapshot_download(
                    repo_id=parsed.hf.repo_id,
                    revision=parsed.hf.revision,
                    local_files_only=True,
                    cache_dir=current_or(_STANDALONE).hf_home or None,
                    token=current_or(_STANDALONE).hf_token or None,
                    allow_patterns=patterns or None,
                )
                return str(p)
            except Exception as e:
                raise ModelResolutionError(
                    f"--offline: huggingface ref {parsed.hf.canonical()} not "
                    f"in local cache ({e}); warm the cache by running without "
                    "--offline first."
                ) from e

        emit({"kind": "model_fetch.started", "ref": parsed.hf.canonical()})
        try:
            from .download import download_hf

            local_dir = download_hf(
                parsed.hf,
                hf_home=current_or(_STANDALONE).hf_home or None,
                hf_token=current_or(_STANDALONE).hf_token or None,
                allow_patterns=tuple(allow_patterns),
                components=components,
            )
        except Exception as e:
            raise ModelResolutionError(
                f"failed to fetch huggingface ref {parsed.hf.canonical()}: {e}"
            ) from e
        emit({
            "kind": "model_fetch.completed",
            "ref": parsed.hf.canonical(),
            "local_dir": str(local_dir),
        })
        return str(local_dir)

    # ModelScope refs: fetch directly via modelscope.snapshot_download. This is
    # file-oriented (allow_patterns) and has NO diffusers-layout requirement, so
    # it handles ComfyUI/DiffSynth split checkpoints the HF resolver rejects.
    if parsed.provider == "modelscope" and parsed.modelscope is not None:
        try:
            from modelscope import snapshot_download as _ms_snap
        except Exception as e:
            raise ModelResolutionError(
                f"modelscope is required for modelscope refs ({parsed.modelscope.canonical()}): {e}"
            ) from e
        kwargs: Dict[str, Any] = {}
        if parsed.modelscope.revision:
            kwargs["revision"] = parsed.modelscope.revision
        if allow_patterns:
            kwargs["allow_patterns"] = list(allow_patterns)
        if offline:
            kwargs["local_files_only"] = True
        emit({"kind": "model_fetch.started", "ref": parsed.modelscope.canonical(), "provider": "modelscope"})
        try:
            local = _ms_snap(model_id=parsed.modelscope.repo_id, **kwargs)
        except Exception as e:
            raise ModelResolutionError(
                f"failed to fetch modelscope ref {parsed.modelscope.canonical()}: {e}"
            ) from e
        emit({"kind": "model_fetch.completed", "ref": parsed.modelscope.canonical(), "local_dir": str(local)})
        return str(local)

    # Cozy refs that miss the CAS (#379): resolve standalone against
    # tensorhub's public resolve route (th#560) and feed the shared
    # cozy_snapshot downloader. TENSORHUB_URL selects the hub; TENSORHUB_TOKEN
    # (optional) unlocks private repos. Offline stays CAS-only.
    if parsed.provider == "tensorhub" and parsed.tensorhub is not None:
        if offline:
            # Tag refs: a previous online resolve remembered tag->digest.
            ref_map = _hub_ref_map_path(cache_dir, parsed.tensorhub)
            if ref_map.exists():
                snap = cache_dir / "snapshots" / ref_map.read_text().strip()
                if snap.exists():
                    return str(snap)
            raise ModelResolutionError(
                f"--offline: tensorhub ref {parsed.tensorhub.canonical()} not in local "
                f"CAS ({cache_dir}); warm the cache by running without "
                "--offline once (or set TENSORHUB_CAS_DIR to a path with the "
                "snapshot pre-seeded)."
            )
        return _fetch_tensorhub_snapshot(
            parsed.tensorhub, cache_dir=cache_dir, emit=emit, components=components,
        )

    # Civitai refs: download the model-version files directly. Auth (for gated
    # creators) comes from CIVITAI_API_KEY; public models need none.
    if parsed.provider == "civitai" and parsed.civitai is not None:
        if offline:
            raise ModelResolutionError(
                f"--offline: civitai ref {ref!r} not available offline (no local "
                "civitai cache); run once online to fetch it."
            )
        from .download import (
            download_civitai,
            fetch_civitai_model,
            parse_civitai_version_id,
        )
        api_key = current_or(_STANDALONE).civitai_api_key

        if civitai_version_id:
            # Explicit version pin via Civitai(version="<id>"). The pinned id
            # IS a model-VERSION id, so use it directly — no model lookup.
            try:
                version_id = parse_civitai_version_id(civitai_version_id)
            except Exception as e:
                raise ModelResolutionError(
                    f"bad civitai version pin {civitai_version_id!r} on ref {ref!r}: {e}"
                ) from e
        else:
            # Civitai's ref is a MODEL id by convention; map it to its latest
            # version id. No silent fallback: if the lookup fails or the model
            # has no versions, the ref is wrong (e.g. a bare version id was
            # passed where a model id was expected) — surface it rather than
            # guessing and downloading an unrelated model.
            try:
                model_id = parse_civitai_version_id(parsed.civitai.model_id)
            except Exception as e:
                raise ModelResolutionError(f"bad civitai ref {ref!r}: {e}") from e
            try:
                model = fetch_civitai_model(model_id, api_key=api_key)
            except Exception as e:
                raise ModelResolutionError(
                    f"failed to resolve civitai model {model_id} for ref {ref!r}: {e}; "
                    "Civitai's ref must be a MODEL id (pin a specific version "
                    'with .version("<version_id>")).'
                ) from e
            versions = model.get("modelVersions") or []
            version_id = int(versions[0].get("id") or 0) if versions else 0
            if version_id <= 0:
                raise ModelResolutionError(
                    f"civitai model {model_id} (ref {ref!r}) has no published "
                    'version to download (pin one with .version("<version_id>")).'
                )
        out_dir = cache_dir / "civitai" / str(version_id)
        emit({"kind": "model_fetch.started", "ref": ref, "provider": "civitai"})
        try:
            local = download_civitai(version_id, out_dir, api_key=api_key)
        except Exception as e:
            raise ModelResolutionError(
                f"failed to fetch civitai ref {ref!r} (resolved version {version_id}): {e}"
            ) from e
        emit({"kind": "model_fetch.completed", "ref": ref, "local_dir": str(local)})
        return str(local)

    raise ModelResolutionError(
        f"unsupported model ref: {ref!r} (provider={provider!r})"
    )
