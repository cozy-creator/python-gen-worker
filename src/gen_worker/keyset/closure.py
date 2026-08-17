"""The key set's ADDRESS: what a per-class graph hash is a pure function of.

Moved here verbatim from ``boot_key`` (pgw#1327). It was the local memo's key and
it is now the shipped document's key as well, and that is deliberate: **one
closure identity, one implementation.** A shipped key set and a pod-local cache
that disagreed about what "the same code" means would be two answers to the
question that decides whether a pod may skip its traces.

Why the binding is the whole safety story
-----------------------------------------
The document ships class hashes the mint lane derived. A pod folds them with its
own ``sm``/``toolchain`` and asks the hub by the result. Nothing about that
sequence proves the shipped hashes describe THIS pod's code — so the closure
digest does: it covers the worker's trace/translation closure, the vendored TCG
rev, the endpoint's module bytes, the declaration's shape ladder, the checkpoint
subjects, and torch/triton content identity. A pod whose closure digest is not in
the document has no shipped key set, full stop; it never reads "the nearest"
closure. That is checklist row 3 — *drift fails safe* — implemented as an address
rather than as a comparison somebody could forget to make.

Computing it costs milliseconds and imports no model code and no exporter: source
bytes, ``importlib.util.find_spec``, and ``compile_cache``'s already-cached
version probes.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .. import compile_cache as cc, graph_facts
from .._vendor import vendored_rev
from ..child_contract import CompileSpec, MintSlot, slot_subjects
from .identifiers import ClosureDigest, KeySetError, parse_closure_digest

__all__ = [
    "CONTRACT_MODULES",
    "closure_digest",
    "contract_code_digest",
    "endpoint_source_facts",
    "tcg_version",
]

#: The modules that define the parent/child TRACE contract. Their BYTES are read
#: — never imported — so a serve process that must not import a tracer can still
#: state which tracer contract its key set was derived under.
CONTRACT_MODULES: Tuple[str, ...] = (
    "boot_key.py",
    "boot_trace_child.py",
    "aot_mint.py",
    "aot_declaration.py",
    "aot_inputs.py",
    "child_preflight.py",
    "meta_instantiation.py",
)

#: The schema version rides the digest input, so a stale row is unaddressable as
#: well as unreadable — two independent reasons it cannot be misread, which is
#: what typed invalidation has to mean for a cache whose wrong answer selects a
#: wrong graph. Bumped from ``boot_key.MEMO_VERSION = 5`` when the memo became
#: this document (pgw#1327); the cost of the bump is one re-trace per pod.
CLOSURE_VERSION = 6

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent


def contract_code_digest() -> str:
    """Digest of the trace-contract source, read from disk.

    Raises when a contract module cannot be read. The old ``boot_key._code_digest``
    returned ``""`` there, which meant a frozen or trimmed image computed a
    DIFFERENT closure digest than the pod that emitted the key set and silently
    missed every time. An unreadable contract is now a stated refusal: this pod
    cannot say what it would trace, so it does not get to read a key set that
    claims to know.
    """
    digest = hashlib.sha256()
    for name in CONTRACT_MODULES:
        try:
            digest.update(
                hashlib.sha256((_PACKAGE_ROOT / name).read_bytes()).digest())
        except OSError as exc:
            raise KeySetError(
                "closure_unavailable",
                f"trace-contract module {name!r} is unreadable ({exc}), so this "
                f"process cannot state the closure a key set would be keyed on"
            ) from exc
    return digest.hexdigest()[:16]


def endpoint_source_facts(
    modules: Sequence[str],
) -> Tuple[Tuple[str, str], ...]:
    """Content identity of the endpoint modules a trace child would import.

    A shipped key set skips the trace entirely, so naming only the endpoint
    function is insufficient: editing that function or a discovered sibling must
    fail to address the document even when its declaration is unchanged. Package
    modules contribute every Python source below their import root because
    discovery walks submodules; a plain module contributes its one source file.
    Absolute paths never enter the digest.
    """
    out: Dict[str, str] = {}
    for module in sorted({str(item).strip() for item in modules if str(item).strip()}):
        try:
            spec = importlib.util.find_spec(module)
        except (ImportError, ModuleNotFoundError, ValueError):
            spec = None
        if spec is None:
            out[f"{module}:<unresolved>"] = ""
            continue
        roots = tuple(Path(root) for root in (spec.submodule_search_locations or ()))
        paths: List[Tuple[str, Path]] = []
        if roots:
            for root in roots:
                try:
                    children = sorted(root.rglob("*.py"))
                except OSError:
                    children = []
                paths.extend(
                    (f"{module}/{path.relative_to(root).as_posix()}", path)
                    for path in children
                )
        elif spec.origin and spec.origin.endswith(".py"):
            paths.append((module.replace(".", "/") + ".py", Path(spec.origin)))
        if not paths:
            out[f"{module}:<no-python-source>"] = ""
            continue
        for logical, path in paths:
            try:
                out[logical] = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
            except OSError:
                out[logical] = ""
    return tuple(sorted(out.items()))


def tcg_version() -> str:
    """Vendored TCG rev whose declaration semantics produced the class hashes.

    pgw#1310: TCG is vendored, so it has no distribution metadata to read. The
    old ``importlib.metadata.version(...)`` swallowed ``PackageNotFoundError``
    into ``""`` — which, once the distribution was gone, would have silently
    dropped TCG from the address and stopped a TCG change from invalidating
    anything.
    """
    return vendored_rev("torchcg")


def closure_digest(
    family: str,
    cfg: CompileSpec,
    *,
    function: str = "",
    slots: Optional[Mapping[str, MintSlot]] = None,
    modules: Sequence[str] = (),
) -> ClosureDigest:
    """The address of this code's key set.

    The worker trace/translation closure, TCG release, and endpoint module bytes
    are the code half. Declaration facts are the other half — two builds of the
    same code that declare different shape ladders trace different graphs.
    ``cc.content_keys()`` covers the Torch/Triton implementation the trace would
    execute; it is not a substitute for endpoint or worker code.

    ``slots`` is the third half. The traced graph is a function of the
    CHECKPOINT's own config: ``zero_cond_t`` exists on ``Qwen-Image-Edit-2511``
    and not on ``Qwen-Image``, and block counts, head counts and quantization ops
    are the general case. Without it, a redeploy that rebinds a slot to a
    different checkpoint would address the PREVIOUS checkpoint's key set.

    Deliberately NOT included: sm, toolchain, env seal. They are key AXES, they
    re-derive every boot in milliseconds, and folding them in here would make the
    document miss on facts whose whole point is that they are cheap to restate —
    i.e. would make one shipped key set serve exactly one SKU.

    Nor the slots' local PATHS — see ``child_contract.slot_subjects``: where the
    bytes landed on this disk is not what was traced.
    """
    facts = {
        "v": CLOSURE_VERSION,
        "family": str(family or ""),
        "function": str(function or ""),
        "content_keys": dict(cc.content_keys()),
        "worker_code": {
            "boot_contract": contract_code_digest(),
            "static_closure": dict(cc.static_code_closure()),
        },
        "endpoint_code": dict(endpoint_source_facts(modules)),
        "tcg_version": tcg_version(),
        "declaration": {
            "targets": sorted(str(t) for t in cfg.targets),
            "shapes": sorted([int(v) for v in row] for row in cfg.shapes),
            "text_lens": sorted({int(v) for v in cfg.text_lens}),
            "guidance": sorted(float(v) for v in cfg.guidance_scales),
            "lora_bucket": int(cfg.lora_bucket or 0),
        },
        "subject": graph_facts.subject_facts(slot_subjects(dict(slots or {}))),
    }
    blob = json.dumps(facts, sort_keys=True, separators=(",", ":")).encode()
    return parse_closure_digest(hashlib.sha256(blob).hexdigest()[:32])
