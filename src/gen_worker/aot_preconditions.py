"""The AOT mint's STATIC preconditions — asked at IMAGE BUILD, once.

A precondition is static when the endpoint IMAGE decides it: the C++ toolchain
apt installed into it, the torch wheel pinned into it, the declaration module
baked into it. Asking those questions on a rented pod instead makes a "no" a
silent downgrade with a rental bill attached — the fleet serves eager forever
and the evidence is one ``self_mint_skipped`` event nobody queries.

The ruling: a release whose image cannot AOT-compile the endpoints it DECLARES
is a broken build, and must refuse to publish naming the precondition. An
endpoint that declares NO export declaration is not covered — JIT is intake
mode (ruled), and it keeps the dynamo lane.

This module is the ONE spelling of those rows. ``discovery.discover`` stamps
them into ``endpoint.lock`` and ``discovery.validation`` turns a refusal into a
build error. This is the ONLY place the torch floor is decided: there is no
second spelling, so a build cannot prove one thing and a pod discover another.

Four verdicts, and the difference between the last three is the whole point:

``ok``         the precondition holds on this image.
``refused``    it does not, and nothing on a pod can fix it — FAIL THE BUILD.
``blocked``    the declaration carries unresolved ``Compile.blockers``
. The family said why; it is a DECLARED block, not a
               broken image, and it does not need a pod to repeat the
               sentence.
``abstained``  this environment cannot decide (a torch-less manifest build).
               Recorded, never silent — an unrecorded abstention is how a gate
               becomes decorative.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

from .api.export_contract import (
    blocker_refusal,
    declaration_import_failures,
    export_declaration,
    has_export_declaration,
    open_blockers,
)

# Every heavier import in this module is DEFERRED into the function that needs
# it. Discovery imports this at build time, where torch may be absent and
# `compile_cache` (via `aot_contract`) would pull the whole model stack: a
# precondition gate that cannot be imported in the environment it gates is not
# a gate. The abstaining path below touches neither.

logger = logging.getLogger(__name__)

#: torch below this cannot trace the lifted-LoRA fork: 2.9 strict export
#: refuses ``bind_views``' in-trace setattr ("Mutating module attribute lora_a
#: during export") that 2.13 traces fine — measured. A mint PRECONDITION for
#: the bucket-bearing lane, not a taste.
LIFTED_LORA_TORCH_FLOOR: Tuple[int, int] = (2, 13)

CHECK_DECLARATION_IMPORT = "declaration_import"
CHECK_DECLARATION_EVALUATES = "declaration_evaluates"
CHECK_CXX_TOOLCHAIN = "cxx_toolchain"
CHECK_CUDA_ROOT = "cuda_root"
CHECK_TORCH_SINGLETON = "torch_singleton"
CHECK_LIFTED_LORA_TORCH_FLOOR = "lifted_lora_torch_floor"
CHECK_ADAPTER_BACKEND = "adapter_backend"

#: The distribution diffusers' ``load_lora_weights`` hard-requires, and the
#: one ``models/fp8_storage`` imports three layer modules from at overlay
#: time. gen-worker does NOT declare it: an endpoint that serves adapters
#: declares that in ITS image, and this check is what makes the declaration
#: PROVABLE at build instead of discoverable on a paid pod.
ADAPTER_BACKEND_DIST = "peft"

#: Distributions that must appear at most once. A second copy of any of them
#: silently doubles the image by ~7 GB and desyncs the CUDA runtime — the
#: synthesized Dockerfile asserts this in-image (`torchInvariantLine`); this
#: check is what asks it of a custom Dockerfile too.
TORCH_FAMILY: Tuple[str, ...] = ("torch", "torchvision", "torchaudio", "triton")

OK = "ok"
REFUSED = "refused"
BLOCKED = "blocked"
ABSTAINED = "abstained"


@dataclass(frozen=True)
class Precondition:
    """One named precondition verdict, addressed to the endpoint author."""

    check: str
    verdict: str
    detail: str
    family: str = ""

    def manifest_row(self) -> Dict[str, str]:
        row = {"check": self.check, "verdict": self.verdict,
               "detail": self.detail}
        if self.family:
            row["family"] = self.family
        return row


def torch_version_gap(version: str) -> str:
    """'' when ``version`` meets the lifted-LoRA floor, else the refusal.

    Pure string arithmetic, and the ONE implementation: the floor is decided
    at image build (``static_mint_preconditions`` -> ``endpoint.lock``), never
    again per mint. Two spellings of a floor is how a build proves one thing
    and a pod discovers another.
    """
    parts = version.split("+")[0].split(".")
    floor = ".".join(map(str, LIFTED_LORA_TORCH_FLOOR))
    try:
        found = (int(parts[0]), int(parts[1]))
    except (IndexError, ValueError):
        return (f"cannot parse torch version {version!r} to check the "
                f"lifted-LoRA mint floor (torch >= {floor})")
    if found < LIFTED_LORA_TORCH_FLOOR:
        return (
            f"lifted-LoRA mint requires torch >= {floor}, got {version}: "
            f"torch 2.9 strict export refuses bind_views' in-trace setattr "
            f"('Mutating module attribute lora_a during export') that 2.13 "
            f"traces fine — measured on pod 8 (pgw#723 residuals), so the "
            f"2.13 prod floor is a mint PRECONDITION for this lane, not a "
            f"preference")
    return ""


def _torch_version() -> str:
    import torch

    return str(getattr(torch, "__version__", "") or "")


def static_mint_preconditions(
    declared: Mapping[str, int], *, torch_available: bool,
    torch_version: Optional[str] = None,
) -> Tuple[Precondition, ...]:
    """Every static precondition verdict for the families this image declares.

    ``declared`` maps a family to the largest ``lora_bucket`` any of its
    endpoints declares (0 = no bucket-bearing lane). Families are the ones
    carrying a ``compile=`` block; a family with no registered export
    declaration produces NO row at all — it is intake-mode JIT and owes the
    AOT lane nothing.

    ``torch_available`` is False under discovery's heavy-dep stubs (a
    torch-less manifest build). Every row then abstains rather than guessing:
    a stubbed ``torch.__version__`` is absent, and a declaration that touches
    torch at build raises for a reason that has nothing to do with the image
    an endpoint will actually run.
    """

    rows: list[Precondition] = []

    for module, exc in declaration_import_failures():
        rows.append(Precondition(
            check=CHECK_DECLARATION_IMPORT, verdict=REFUSED,
            detail=(
                f"export declaration module {module!r} raised at import "
                f"({exc}). `import_export_declaration` swallows this so an "
                f"endpoint still BOOTS — correct there, fatal here: the image "
                f"would ship an endpoint that declares AOT and registers "
                f"nothing, and on a pod that is indistinguishable from an "
                f"endpoint which never declared one. Fix the import")))

    families = sorted(f for f in declared if f and has_export_declaration(f))
    if not families and not rows:
        return ()

    if not torch_available:
        detail = (
            "this discovery ran without torch (heavy-dep stubs armed), so no "
            "static AOT precondition is decidable here — the toolchain probe, "
            "the torch floor and every declaration that touches torch would "
            "answer about the BUILD HOST, not about the image. The in-image "
            "build decides them")
        return tuple(rows) + tuple(
            Precondition(check=CHECK_DECLARATION_EVALUATES,
                         verdict=ABSTAINED, detail=detail, family=f)
            for f in families)

    aot_declaring: list[str] = []
    for family in families:
        decl = export_declaration(family)
        if decl is None:
            rows.append(Precondition(
                check=CHECK_DECLARATION_EVALUATES, verdict=REFUSED,
                family=family, detail=(
                    "the registered export declaration evaluated to None — "
                    "there is no Compile to derive graph classes from")))
            continue
        # The declared block, read as DATA — never a thunk that says it by
        # raising.
        blocked = open_blockers(decl)
        if blocked:
            rows.append(Precondition(
                check=CHECK_DECLARATION_EVALUATES, verdict=BLOCKED,
                family=family, detail=blocker_refusal(family, blocked)))
            continue
        rows.append(Precondition(
            check=CHECK_DECLARATION_EVALUATES, verdict=OK, family=family,
            detail=(f"{len(getattr(decl, 'classes', ()) or ())} declared "
                    f"graph class(es) over targets "
                    f"{list(getattr(decl, 'targets', ()) or [])}")))
        aot_declaring.append(family)

    if not aot_declaring:
        # Nothing on this image will ever run an AOTI export, so the toolchain
        # and the torch floor are not this image's obligations.
        return tuple(rows)

    rows.append(_cxx_row(aot_declaring))
    rows.append(_cuda_root_row(aot_declaring))
    rows.append(_torch_singleton_row())

    version = torch_version if torch_version is not None else _torch_version()
    for family in aot_declaring:
        if int(declared.get(family, 0) or 0) <= 0:
            continue
        gap = torch_version_gap(version)
        rows.append(Precondition(
            check=CHECK_LIFTED_LORA_TORCH_FLOOR,
            verdict=REFUSED if gap else OK, family=family,
            detail=gap or (
                f"torch {version} traces the lifted-LoRA fork (bucket "
                f"{int(declared[family])})")))
    return tuple(rows)


def _cxx_row(aot_declaring: list[str]) -> Precondition:
    from . import compile_cache as cc

    found = cc.cxx_compiler()
    if found:
        return Precondition(
            check=CHECK_CXX_TOOLCHAIN, verdict=OK,
            detail=f"AOTInductor links with {found}")
    return Precondition(
        check=CHECK_CXX_TOOLCHAIN, verdict=REFUSED, detail=(
            f"no C++ compiler on this image, but {aot_declaring!r} declare an "
            f"AOT export. torch._inductor would raise InvalidCxxCompiler at "
            f"the LINK step — measured at 336 s of L4 time to reach it "
            f"(pgw#823). AOTInductor forces the C++ wrapper and links a "
            f"shared object, unlike the dynamo lane's Triton + Python "
            f"wrapper: install g++/build-essential in the endpoint image, or "
            f"drop the export declaration and serve the dynamo/JIT lane"))


def _torch_is_cuda_build() -> bool:
    import torch

    return bool(getattr(getattr(torch, "version", None), "cuda", None))


def _cuda_root_row(aot_declaring: list[str]) -> Precondition:
    """`g++` is necessary and NOT sufficient on a CUDA image.

    Unasked, this refuses NOWHERE: the pod boots, loads and EXPORTS — the
    expensive part — before `cpp_extension` says "CUDA_HOME environment
    variable is not set". Free refusal here, paid failure there.
    """
    from . import cuda_root as cr

    if not _torch_is_cuda_build():
        return Precondition(
            check=CHECK_CUDA_ROOT, verdict=OK,
            detail=("torch is a CPU build, so AOTInductor's host compile needs "
                    "no CUDA root"))
    home = cr.torch_cuda_home()
    gaps = cr.missing_parts(cr.Path(home)) if home else ["the root itself"]
    if home and not gaps:
        return Precondition(
            check=CHECK_CUDA_ROOT, verdict=OK,
            detail=f"AOTInductor host-compiles against {home}")
    return Precondition(
        check=CHECK_CUDA_ROOT, verdict=REFUSED, detail=(
            f"torch's cpp_extension cannot host-compile on this image and "
            f"{aot_declaring!r} declare an AOT export. Missing: "
            f"{'; '.join(gaps)}. The pytorch runtime bases ship CUDA as pip "
            f"wheels and never create /usr/local/cuda, so a C++ compiler alone "
            f"gets you to the LINK step and no further (pgw#823, pgw#1017). "
            f"Fix: add `RUN python -m gen_worker.cuda_root` to your Dockerfile "
            f"after your dependency install — it composes the root out of parts "
            f"the image already ships, writes nothing into site-packages, and "
            f"is a no-op on an image that already has one. Or drop the export "
            f"declaration and serve the dynamo/JIT lane"))


def _torch_singleton_row() -> Precondition:
    """One torch-family install, not two.

    `torchInvariantLine` fails the SYNTHESIZED build when a second copy of a
    torch-family distribution lands — *"silently doubles the image by ~7GB and
    desyncs the CUDA runtime"*. A custom Dockerfile installing its deps its own
    way (no `--no-emit-package` list, no base-image constraints file)
    reproduces exactly that, and the hub reads no distribution's version out of
    the image but `gen-worker`'s.

    Scope, stated rather than implied: this verifies the DUPLICATE half only.
    The synthesized file's other half — "torch has not drifted from the base
    image's pin" — reads `/tmp/torch-constraints.txt`, which is written and
    deleted inside the generated Dockerfile and does not exist here. That half
    is decidable only where the base image is known, so it belongs to the hub
 and is not faked here.
    """
    import importlib.metadata as md

    counts: Dict[str, int] = {}
    for dist in md.distributions():
        name = str((dist.metadata["Name"] or "")).lower().replace("_", "-")
        if name in TORCH_FAMILY:
            counts[name] = counts.get(name, 0) + 1
    dups = sorted(n for n, c in counts.items() if c > 1)
    if not dups:
        return Precondition(
            check=CHECK_TORCH_SINGLETON, verdict=OK,
            detail=(f"one install each of "
                    f"{sorted(n for n in counts) or list(TORCH_FAMILY)}"))
    return Precondition(
        check=CHECK_TORCH_SINGLETON, verdict=REFUSED, detail=(
            f"duplicate torch-family install: {dups}. A second copy shadows the "
            f"base image's, adds ~7 GB to every pod's image pull and desyncs "
            f"the CUDA runtime the compiled kernels were built against. Pin "
            f"your dependencies against the versions the base image already "
            f"provides, or exclude them from your install (the synthesized "
            f"Dockerfile passes --no-emit-package for the whole torch/CUDA "
            f"wheel set)"))


def adapter_backend_present() -> bool:
    """Whether this image can actually overlay an adapter.

    `find_spec` rather than `import peft`: discovery must not pay a heavy
    import to answer a yes/no, and `peft` is deliberately NOT in
    `heavy_deps.DEFAULT_HEAVY_ROOTS`, so the probe is honest even under the
    stubbing a torch-less manifest build arms.
    """
    import importlib.util

    try:
        return importlib.util.find_spec(ADAPTER_BACKEND_DIST) is not None
    except (ImportError, ValueError):  # a broken/namespace parent
        return False


def adapter_backend_preconditions(
    declared: Mapping[str, int], *, present: Optional[bool] = None,
) -> Tuple[Precondition, ...]:
    """One verdict per family that declares adapter serving.

    ``lora_bucket > 0`` is the author declaring the unknowable — *this
    endpoint serves adapters*. The platform's job is then to PROVE the
    capability exists, at build, by name. Until this check existed nothing
    did: `peft` is not a gen-worker dependency, `models/fp8_storage` imports
    `peft.tuners.{loha,lokr,lora}.layer` at call time and `utils/lora` routes
    overlays through diffusers' `load_lora_weights`, which hard-requires the
    package. So the failure was an `ImportError` at the FIRST OVERLAY — after
    acquisition, after the weights download, on a paid GPU pod, which is the
    most expensive place in the system to learn that a package is missing.
    Measured live on an A100 (qwen-image-edit BYOM serve): *"adapter failed to
    load onto base pipeline: PEFT backend is required for this method."*

    Unlike the AOT rows this sits beside, it does NOT depend on a registered
    export declaration — an adapter-serving endpoint owes this whether or not
    it compiles.
    """
    buckets = sorted(f for f, b in (declared or {}).items()
                     if f and int(b or 0) > 0)
    if not buckets:
        return ()
    ok = adapter_backend_present() if present is None else bool(present)
    if ok:
        return tuple(
            Precondition(
                check=CHECK_ADAPTER_BACKEND, verdict=OK, family=family,
                detail=(f"{ADAPTER_BACKEND_DIST} is importable in this image, "
                        f"so a declared adapter overlay can load"))
            for family in buckets)
    return tuple(
        Precondition(
            check=CHECK_ADAPTER_BACKEND, verdict=REFUSED, family=family,
            detail=(
                f"this endpoint declares `lora_bucket={int(declared[family])}` "
                f"for family {family!r} — it serves ADAPTERS — but "
                f"{ADAPTER_BACKEND_DIST!r} is not installed in this image. "
                f"diffusers' `load_lora_weights` hard-requires it and "
                f"`models/fp8_storage` imports "
                f"`{ADAPTER_BACKEND_DIST}.tuners.{{loha,lokr,lora}}.layer` at "
                f"overlay time, so this image boots, serves the base model "
                f"fine, and then fails the FIRST adapter request on a paid "
                f"pod. Add `{ADAPTER_BACKEND_DIST}` to this endpoint's "
                f"dependencies, or drop the `lora_bucket` declaration"))
        for family in buckets)


def declared_compile_families(functions: Any) -> Dict[str, int]:
    """family -> largest declared ``lora_bucket``, over manifest functions.

    Reads the same ``functions[].compile`` block the hub keys its family-cache
    lookups off, so "which families does this image declare" has one
    answer at build time and no second derivation.
    """
    out: Dict[str, int] = {}
    for fn in functions or ():
        block = (fn or {}).get("compile") or {}
        family = str(block.get("family") or "").strip()
        if not family:
            continue
        bucket = int(block.get("lora_bucket") or 0)
        out[family] = max(out.get(family, 0), bucket)
    return out


__all__ = [
    "ABSTAINED",
    "BLOCKED",
    "ADAPTER_BACKEND_DIST",
    "CHECK_ADAPTER_BACKEND",
    "CHECK_CUDA_ROOT",
    "CHECK_CXX_TOOLCHAIN",
    "CHECK_DECLARATION_EVALUATES",
    "CHECK_DECLARATION_IMPORT",
    "CHECK_LIFTED_LORA_TORCH_FLOOR",
    "CHECK_TORCH_SINGLETON",
    "LIFTED_LORA_TORCH_FLOOR",
    "OK",
    "REFUSED",
    "TORCH_FAMILY",
    "Precondition",
    "adapter_backend_preconditions",
    "adapter_backend_present",
    "declared_compile_families",
    "static_mint_preconditions",
    "torch_version_gap",
]
