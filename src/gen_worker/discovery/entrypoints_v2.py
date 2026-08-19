"""Publish-time extraction for the pgw#1382 author surface.

The v1 walk looks for ``__gen_worker_endpoint__``; ``@entrypoint`` stamps
``__cozy_entrypoint__``. Nothing read the new attribute, so a v2 endpoint
discovered ZERO functions, the builder warned and exited 0, and hub admission
refused the empty manifest nine minutes later (pgw#1387).

``entrypoints[]`` IS ``functions[]``' SUCCESSOR SPELLING, NOT A SECOND
DOCUMENT SPACE (th#2146). The hub folds the key into ``Functions`` at its one
decode site (``builder.ParseManifest``) and nothing downstream learns a second
word — so the ITEM shape here must be the item shape ``_extract_entries``
already emits, key for key. Emitting a bespoke row shape (or wrapping the list
in a ``{"v": …}`` envelope) does not merely lose fields: ``entrypoints`` decodes
into ``[]manifestFunction``, so a JSON object where the hub expects an array
fails ``ParseManifest`` outright and the release does not admit.

A manifest carrying BOTH keys is refused by the hub rather than silently
merged, so this module emits ``entrypoints`` only for a package that HAS a v2
surface, and the v1 path is byte-identical when it does not.

**LANES ARE NOT HERE.** A v2 lane is a tensorfs layout contract, and it travels
as ``{stamp, document}`` on the release-derive document's ``lane_contracts``,
where the hub interns the document content-addressed and needs no prior
knowledge of the layout (th#2146's `docs/lane-vocabulary.md`). Making that
``document`` non-null is pgw#964's, not this module's.

**AND THEY WERE HERE ANYWAY, WHICH MADE EVERY v2 ENDPOINT UNPUBLISHABLE
(pgw#1394).** The paragraph above was written before the code and the code
disagreed with it: ``_model_slot`` emitted the lane stamps as the slot's
``layouts`` and keyed ``layout_requirements`` by them. Those are the hub's
ARTIFACT-layout fields — quant x topology, registered in
``internal/tensorlayout``, resolved to descriptor digests for the conversion
ladder — so the hub refused the publish with
``artifact_contract_unregistered`` after a full image build and registry push.
The two fields existed only to carry the ie#740 floor, because the hub refuses
a ``layout_requirements`` handle the slot does not accept; the accepted set was
manufactured to make the floor legal, and manufactured out of the one
vocabulary that could not be.

The floor now travels where a floor belongs: ``functions[].resources.requires``,
THE FUNCTION SCOPE of the one requirement vocabulary, parsed by the SAME
``machinereq`` parser. Same terms, same parser, correct field — and a
model-bearing entrypoint now also declares ``gpu: true``, which a v2 release
never did.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import pkgutil
import textwrap
from typing import Any, Dict, List, Set

from .schema import type_schema_and_hash

#: The DEFAULT kind: an entrypoint that declares nothing is an inference
#: handler, which is also what the hub's `functionkind.Normalize("")` returns.
#: pgw#1406 gave `@entrypoint` a `kind=` kwarg so a PRODUCER can say otherwise
#: — the sentence that used to sit here ("the v2 surface declares no other kind
#: today, and inventing one here would be a value space no hub reader names")
#: was right about the second half and is why `KINDS` is copied verbatim from
#: `internal/functionkind.All` rather than invented.
ENTRYPOINT_KIND = "inference"

class EntrypointDiscoveryError(ValueError):
    """A v2 endpoint package does not yield an extractable surface."""


def _entrypoint_specs(module: Any) -> List[Any]:
    from ..serving.entrypoints import ENTRYPOINT_ATTR

    out: List[Any] = []
    for value in vars(module).values():
        spec = getattr(value, ENTRYPOINT_ATTR, None)
        if spec is None:
            continue
        # Re-exports: only the DECLARING module owns the row, exactly as the
        # v1 walk skips re-exported @endpoint objects.
        if getattr(value, "__module__", None) != module.__name__:
            continue
        out.append(spec)
    return out


def _import_sites(tree: ast.AST) -> Dict[str, str]:
    """Local name -> dotted path, from every ``from x.y import Name`` in
    ``tree``. Imports NOTHING.

    pgw#1431. The runtime lookup recovers the pipeline class by importing and
    type-checking, and it fails in TWO measured shapes that have one cause —
    the AST already carried the answer:

    * **deferred (function-body) import.** An endpoint whose upstream runtime
      is source-built cannot import it at module top: the package is not on
      PyPI and compiles CUDA extensions, so a discovery runner has nothing to
      import. The SDK's own convention says to defer it (``trellis-3d/main.py``:
      *"deferring a source-built extra ... is exactly what keeps discovery (and
      the CPU test suite) importable"*). A deferred import binds no
      module-level name, so ``getattr(module, …)`` finds nothing.
    * **module-top import of a STUBBED package.** ``internvl-U`` writes
      ``from internvlu import InternVLUPipeline`` at module top with
      ``discovery_heavy_deps = ["internvlu"]``; off-image, the stub finder
      binds a MODULE OBJECT, so ``isinstance(target, type)`` is False even
      though the import is module-top and sanctioned. In-image publish is
      fine; every off-image gate broke — the caller is
      ``serverless-endpoints/scripts/lint_discovery.py:121``.

    So the fallback is scoped to the CAUSE, not to either symptom: whenever the
    runtime lookup fails to yield a type, resolve the name statically from any
    ``ImportFrom`` in the module OR in ``load`` itself.

    ``import a.b.c`` is deliberately not handled — it binds ``a``, so the
    pipeline would be spelled ``a.b.c.Name`` as an ``ast.Attribute`` rather
    than the ``ast.Name`` this parse accepts.
    """
    found: Dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.module is None:
            continue
        if node.level:  # relative import — no absolute dotted path to state
            continue
        for alias in node.names:
            if alias.name == "*":
                continue
            found[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return found


def _pipeline_class(model_cls: type) -> str:
    """The dotted pipeline class ``load()`` builds, read STATICALLY.

    The hub requires `pipeline_class` on a model slot. In v1 it was a `Slot`
    field; in v2 the author writes ``self.pipe = ctx.load(SomePipeline)`` inside
    ``load``, so it is recovered by parsing that call — no author code runs
    beyond the import that already happened, which is the same promise the
    class header makes. Unreadable (a dynamic class, no ``load``) returns "",
    and the caller decides what an absent one means; guessing a pipeline here
    would be a manifest that lies about what the worker will build.

    Two resolutions, in order of authority (pgw#1431):

    1. the name bound on the endpoint's MODULE — a real class object, so the
       answer carries its true ``__module__``/``__qualname__`` even when the
       import site spells an alias or a re-export;
    2. failing that, a ``from x.y import Name`` — at module top OR inside
       ``load`` — resolved statically off the AST, importing nothing.

    (2) is strictly a fallback keyed on (1) not yielding a TYPE, so it covers a
    deferred import (nothing bound) and a stubbed heavy dep (a module object
    bound where a class was expected) with one branch. When the class really is
    importable at discovery, (1) still wins, so this cannot move any answer
    that already resolved.
    """
    load = getattr(model_cls, "load", None)
    if load is None:
        return ""
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(load)))
    except (OSError, TypeError, SyntaxError):
        return ""
    module = inspect.getmodule(model_cls)
    # Module-top imports first, then load()'s own — the inner scope wins on a
    # name collision, exactly as Python binds it.
    static: Dict[str, str] = {}
    try:
        module_source = inspect.getsource(module) if module is not None else ""
    except (OSError, TypeError):
        module_source = ""
    if module_source:
        try:
            static.update(_import_sites(ast.parse(module_source)))
        except SyntaxError:
            pass
    static.update(_import_sites(tree))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if not (isinstance(fn, ast.Attribute) and fn.attr == "load"):
            continue
        if not node.args:
            continue
        arg = node.args[0]
        if not isinstance(arg, ast.Name):
            continue
        target = getattr(module, arg.id, None)
        if isinstance(target, type):
            return f"{target.__module__}.{target.__qualname__}"
        dotted = static.get(arg.id)
        if dotted:
            return dotted
    return ""


def _model_slot(slot: Any) -> Dict[str, Any]:
    """A model slot in the hub's `functions[].slots[]` vocabulary.

    NO `layouts` / `layout_requirements` KEY IS EMITTED HERE, and that is the
    whole of pgw#1394. Those two fields are the hub's ARTIFACT-layout
    vocabulary — the quant x topology axes registered in
    `internal/tensorlayout` — and `slot_layouts.go` resolves every handle in
    them to a descriptor digest so the conversion ladder can walk
    `(axis, from, to)` edges. A SERVING LANE is not a point on either axis and
    has no descriptor digest, so writing lane stamps here made the hub refuse
    the publish outright:

        slot_layout_demand_invalid: artifact_contract_unregistered —
        function "generate" slot "model" path "pipeline"
        demands "sdxl.diffusers-bf16@1"

    after the image build and the registry push. The two repos were never
    drifted: `models/tensor_layout_contract.py` already lists exactly the
    handles the hub registers. It was the wrong vocabulary in the right field.

    The lane itself already has its own channel and does not belong here at
    all — see this module's header, which said so before the code did.
    """
    from ..serving import model_type

    from ..serving.model import SELF_LOADING_ATTR

    model_cls = slot.annotation
    out: Dict[str, Any] = {
        "name": slot.name,
        # `kind` is omitted rather than spelled "model": empty IS model on the
        # hub side (th#2140 5c) and every pre-5c manifest means exactly that.
    }
    self_loading = str(getattr(model_cls, SELF_LOADING_ATTR, "") or "").strip()
    readable = _pipeline_class(model_cls)
    if self_loading:
        # pgw#1431 fix (b). A marked slot states WHY it has no class instead of
        # naming one, exactly as `layouts_undeclarable` does one level down for
        # the bytes. `pipeline_class` and `self_loading` are mutually exclusive
        # in the manifest: a slot either names its class or says why it has none.
        if readable:
            # Both would be true at once, and they contradict. Refuse naming
            # BOTH sites — otherwise the marker becomes a way to silence a
            # perfectly readable class, which is the se#757 silent-lie shape
            # wearing a new hat.
            raise EntrypointDiscoveryError(
                f"{model_cls.__qualname__}: declares self_loading= "
                f"({self_loading!r}) AND its load() calls "
                f"ctx.load({readable.rsplit('.', 1)[-1]}) — those contradict. "
                "self_loading= is for a pipeline ctx.load CANNOT drive; drop "
                "the marker if it can, or drop the ctx.load call if it cannot."
            )
        out["self_loading"] = self_loading
    else:
        out["pipeline_class"] = readable
    declared = model_type(model_cls)
    family = getattr(declared, "name", "") or ""
    if family:
        out["family"] = family
    return out


def _stricter(term: str, left: Any, right: Any) -> Any:
    """The tighter of two floors for one term, by THE evaluator.

    `term_meets` is the vocabulary's single comparator at both ends — the
    declaration check and the runtime machine-fit check — and picking the
    stricter floor is that same question asked once. A hand-rolled `max()`
    here would be a second implementation, and this file would be where the
    two ends start disagreeing about what "meets" means (`min_cuda` and
    `min_torch` are version tuples, not floats).
    """
    from ..models.tensor_layout_contract import term_meets

    return left if term_meets(term, left, right) else right


def _merge_floors(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Fold per-lane floors into the ONE floor the function must be placed on.

    A function is placed on a single machine before anything knows which lane
    it will serve, so the honest function-scope floor is the strictest across
    the lanes it may serve. Every endpoint in the repo declares one lane
    today, which makes this the identity; it is written as a fold anyway
    because collapsing several floors by taking the LAST one would be a silent
    under-declaration, and the buy path cannot detect one.

    `recommended` nests and folds by the same rule as the flat minimum.
    """
    merged: Dict[str, Any] = {}
    for row in rows:
        for term, value in row.items():
            if term == "recommended":
                continue
            merged[term] = _stricter(term, value, merged[term]) if term in merged else value
    recommended: Dict[str, Any] = {}
    for row in rows:
        for term, value in (row.get("recommended") or {}).items():
            recommended[term] = (
                _stricter(term, value, recommended[term]) if term in recommended else value
            )
    if recommended:
        merged["recommended"] = recommended
    return merged


def _resources(specs: List[Any]) -> Dict[str, Any] | None:
    """The function's `resources` block: THE FUNCTION SCOPE of the one
    requirement vocabulary (`internal/builder/function_requirements.go`).

    This is where a `requires=` floor travels now that it no longer rides the
    artifact-layout field. The hub parses `resources.requires` with the SAME
    `machinereq` parser it used for `layout_requirements`, so the terms cross
    the wire unchanged — only the field they arrive in is corrected.

    TWO THINGS THIS IS CAREFUL ABOUT:

    1. **An EMPTY block is an explicit CPU declaration** on the hub side
       ("a present-but-empty block is an explicit CPU declaration"), so a
       model-bearing entrypoint must never emit `{}`. It emits `gpu: true`.
    2. **Absent stays absent for a slotless entrypoint that declares
       nothing.** No model slot and no `resources=` means no claim, and the
       hub falls back to release-level resolution exactly as it does today.

    `gpu: true` is also a fix in its own right: a v2 release previously
    declared NO accelerator anywhere (its only machine signal was the
    `layout_requirements` the hub was refusing), which is the
    `[runpod] no GPU constraint declared; buying with NO VRAM bound` path.

    THE REST OF THE BLOCK — se#755/pgw#1396. This used to hand-build
    `{gpu, requires}` from the model slots alone, so five of `Resources`'
    eight fields could never appear in a v2 manifest no matter what an author
    wrote: `vcpus`, `gpu_count`, `libraries`, `max_gpu_count`,
    `max_gpus_per_execution_group`, `parallel`. Every one of them is read by
    the hub already — `vcpus` becomes `min_vcpus` and reaches the RunPod
    inventory query and the Vast offer filter; the last three are
    `extractStaffingEnvelope` — so they were declared-but-unreachable, not
    unread. `Resources.manifest_dict()` is the projection and this function
    stops second-guessing it.

    `requires` can now arrive from BOTH scopes: the model class header's
    lane-keyed `requires=` and the function's own
    `Resources(requires=)` (te#209's scope, and the only one a weightless
    function has). They FOLD by `_merge_floors` — the strictest of everything
    declared for this function — rather than one silently shadowing the other.
    """
    from ..serving import model_requires

    floors: List[Dict[str, Any]] = []
    has_model_slot = False
    has_declaration = False
    declared: Dict[str, Any] = {}
    for spec in specs:
        resources = getattr(spec, "resources", None)
        if resources is not None:
            has_declaration = True
            # `manifest_dict()` renders `requires` as a manifest ROW, the same
            # shape `_merge_floors` folds, so it joins the lane floors rather
            # than overwriting them.
            block = resources.manifest_dict()
            row = block.pop("requires", None)
            if row:
                floors.append(row)
            declared.update(block)
        for slot in spec.slots:
            if slot.kind != "model":
                continue
            has_model_slot = True
            floors.extend(
                row.manifest_row()
                for row in model_requires(slot.annotation).values()
                if row.declared()
            )
    if not has_model_slot and not has_declaration:
        return None
    out: Dict[str, Any] = dict(declared)
    if has_model_slot:
        # A model-bearing entrypoint always declares the accelerator, even
        # when the author's `Resources(...)` left `gpu` at its False default
        # and `omit_defaults` elided it.
        out["gpu"] = True
    merged = _merge_floors(floors)
    if merged:
        out["requires"] = merged
    return out


def _adapter_slot(slot: Any) -> Dict[str, Any]:
    from ..serving.context import DistillationAdapter

    return {
        "name": slot.name,
        "kind": "adapter",
        # The typed takeover guard as a WIRE fact: the hub refuses an envelope
        # pick whose adapter row is not distillation-marked for this slot.
        "adapter_kind": (
            "distillation" if slot.annotation is DistillationAdapter else "general"
        ),
        "multiple": slot.kind == "adapters",
    }


def _is_job_kind(kind: str) -> bool:
    """The hub's ``functionkind.IsJob``, mirrored: every non-inference kind is
    submitted rather than invoked.

    One line, and it reads off the same `KINDS` tuple `@entrypoint` validates
    against — which is itself copied verbatim from `internal/functionkind.All`
    — so this is a mirror of one authority, not a second vocabulary.
    """
    return str(kind or "").strip().lower() not in ("", ENTRYPOINT_KIND)


def _entrypoint_row(spec: Any) -> Dict[str, Any]:
    resources = _resources([spec])
    row_kind = getattr(spec, "kind", "") or ENTRYPOINT_KIND
    input_schema, input_sha = type_schema_and_hash(spec.payload_type)
    output_schema, output_sha = type_schema_and_hash(spec.return_type)
    slots: List[Dict[str, Any]] = []
    for slot in spec.slots:
        slots.append(_model_slot(slot) if slot.kind == "model" else _adapter_slot(slot))
    return {
        "name": spec.name,
        "python_name": spec.fn.__name__,
        "module": spec.fn.__module__,
        "declared_module": spec.fn.__module__,
        "class_name": "",
        # pgw#1406: the author's `kind=` when declared, else the inference
        # default. `manifestFunction.Kind` already decodes it and the hub
        # derives three PLACEMENT facts from it — reserved-ref resolution,
        # capability TTL, source-sized disk — none of which a conversion
        # producer can get from the `inference` default. Nothing is invented:
        # the value space is `internal/functionkind.All`, validated at
        # decoration against `serving.entrypoints.KINDS`.
        "kind": row_kind,
        "input_schema": input_schema,
        "payload_schema_sha256": input_sha,
        "output_schema": output_schema,
        "output_schema_sha256": output_sha,
        # A v2 entrypoint returns one struct; streaming is not part of the
        # ratified surface, so the cardinality fact is stated, never omitted.
        "incremental_output": False,
        "slots": slots,
        # `None` is ABSENT (no model slot, no `resources=`); a dict is
        # PRESENT, and a present-but-empty block is the hub's explicit CPU
        # declaration — so an author who writes a bare `Resources()` on a
        # weightless entrypoint gets that meaning rather than a silent
        # fall-back to release-level class resolution.
        **({"resources": resources} if resources is not None else {}),
        # pgw#1406, the PRODUCER declarations. Both keys are already decoded
        # by the hub ON A FUNCTION ROW — `manifestFunction.Publishes` /
        # `.Env` (`internal/builder/manifest_contract.go`) — and
        # `entrypoints[]` folds into `Functions` at ParseManifest's one decode
        # site, so `publishes` reaches `ReleaseFunctionSchema.Publishes` ->
        # the capability JWT's `publishes` claim -> `callerDeclaresHubWrite`
        # WITHOUT A HUB CHANGE. Emitted only when declared, so every existing
        # v2 row stays byte-identical (the hub's own tags are `omitempty`).
        #
        # `emits_media` rides ONLY a job-kind row, and the fence is narrowed
        # rather than reversed (th#2177's correction to this lane's first
        # ruling, accepted). Both halves are measured:
        #
        #   * on the REQUEST path the hub grants `upload_media`
        #     UNCONDITIONALLY (`capability_subject_th2068.go:124`,
        #     `UploadsMedia: true`) and `manifestFunction` has no field and
        #     `endpoint_function_schemas` no COLUMN for it, so emitting it on
        #     an inference row is exactly the mirror-with-no-reader th#2087's
        #     fence exists to catch;
        #   * on the JOB path it is READ and it GATES —
        #     `jobCapabilitySubject` sets `UploadsMedia: d.EmitsMedia` from
        #     `endpoint_release_jobs.emits_media`, a column that already
        #     exists. Withholding the key there would silently downgrade every
        #     producer that declared media to no media at all.
        #
        # So the first ruling was right about requests and did not survive a
        # row dispatched as a job. `_is_job_kind` mirrors the hub's
        # `functionkind.IsJob` off the one `KINDS` tuple this repo already
        # copies verbatim, so there is no second vocabulary to drift.
        **({"publishes": True} if spec.publishes else {}),
        **({"env": list(spec.env)} if spec.env else {}),
        **(
            {"emits_media": bool(spec.emits_media)}
            if spec.emits_media is not None and _is_job_kind(row_kind)
            else {}
        ),
    }


def _pipeline_class_or_refuse(rows: List[Dict[str, Any]]) -> None:
    """The hub hard-fails a model slot with no `pipeline_class`, after the
    build and push. Refuse here instead, naming the author's own line."""
    for row in rows:
        for slot in row["slots"]:
            if slot.get("kind") == "adapter" or slot.get("pipeline_class"):
                continue
            # pgw#1431 fix (b): a slot that DECLARED why it has no class is
            # answered, not silent. This is the ArmSelfLoading boundary from
            # fix (c) becoming the green path — for MARKED slots only.
            if slot.get("self_loading"):
                continue
            raise EntrypointDiscoveryError(
                f"@entrypoint {row['name']!r} slot {slot['name']!r}: could not "
                "read the pipeline class from the model's load(). Publish "
                "needs it, so write the ONE spelling with a plain class name: "
                "`self.pipe = ctx.load(StableDiffusionXLPipeline)`"
            )


def discover_entrypoints(main_module: str) -> List[Dict[str, Any]]:
    """Walk ``main_module``'s top-level package; return the manifest
    ``entrypoints`` rows — a FLAT LIST in ``functions[]``' item shape.

    An empty list means the package has no v2 surface, which is a legal answer
    for a v1 endpoint and not an error here; the empty-manifest refusal belongs
    to the caller that knows about jobs and v1 functions too
    (:func:`assert_manifest_advertises_something`).
    """
    top_level = main_module.split(".", 1)[0]
    try:
        top = importlib.import_module(top_level)
    except Exception as exc:
        raise EntrypointDiscoveryError(
            f"could not import endpoint package {top_level!r}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    modules: List[Any] = [top]
    if hasattr(top, "__path__"):
        for sub in pkgutil.walk_packages(top.__path__, prefix=top.__name__ + "."):
            try:
                modules.append(importlib.import_module(sub.name))
            except Exception as exc:
                # Same rule as the v1 walk (pgw#689): a module that fails to
                # import silently drops entrypoints from the release.
                raise EntrypointDiscoveryError(
                    f"could not import endpoint submodule {sub.name!r} "
                    f"(walking {top_level!r}): {type(exc).__name__}: {exc}"
                ) from exc

    rows: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for module in modules:
        for spec in _entrypoint_specs(module):
            row = _entrypoint_row(spec)
            key = f"{row['module']}.{row['python_name']}"
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)

    names: Dict[str, str] = {}
    for row in rows:
        prior = names.get(row["name"])
        if prior is not None:
            raise EntrypointDiscoveryError(
                f"two entrypoints both named {row['name']!r}: {prior} and "
                f"{row['module']}.{row['python_name']} — the request envelope "
                "routes by this name, so one would silently shadow the other"
            )
        names[row["name"]] = f"{row['module']}.{row['python_name']}"
    _pipeline_class_or_refuse(rows)
    rows.sort(key=lambda r: str(r["name"]))
    return rows


def assert_manifest_advertises_something(manifest: Dict[str, Any]) -> None:
    """A release that advertises NOTHING is a build failure, not a warning.

    pgw#1387, measured: the builder printed "no functions discovered", exited
    0, spent 9m20s baking, pushed, and only then did hub admission refuse. The
    refusal is correct; discovering it after the build is the defect.
    """
    if manifest.get("entrypoints") or []:
        return
    raise EntrypointDiscoveryError(
        "this endpoint advertises NOTHING: discovery found no @entrypoint "
        "functions. A release with an empty manifest is refused at hub "
        "admission, so the build stops here rather than after the image "
        "bake. Check that the module named by [tool.gen_worker] main is the "
        "one carrying the declarations, and that its decorator is "
        "gen_worker's @entrypoint (pgw#1373: @endpoint/@job are deleted)."
    )


def entrypoints_block(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The manifest value for ``entrypoints``. A FLAT LIST — the hub decodes
    this key into ``[]manifestFunction`` and folds it into ``Functions``."""
    return rows


__all__ = [
    "ENTRYPOINT_KIND",
    "EntrypointDiscoveryError",
    "assert_manifest_advertises_something",
    "discover_entrypoints",
    "entrypoints_block",
]
