"""``.pt2`` introspection and the B1 code-only gate.

``aot_serve`` owns the ENVELOPE — the metadata contract, ``pack``/``unpack``,
``verify``, and the serve-time gates that read the declared manifest. This
module owns the other direction: reading facts back out of a freshly compiled
package so the mint can PRODUCE that manifest, and proving the package is
publishable before it becomes a file anything could upload.

Why the split. The consumer reads its manifest from ``metadata.json`` — cheap,
no ``dlopen`` — and only the producer ever looks inside the package. Keeping the
two derivations apart is exactly what lets ``aot_serve.assert_bindable``
cross-check the artifact's own table against the declared manifest: two
independent readings that must agree, rather than one restated.

How the facts are read. AOTInductor emits its constant table AS SOURCE into
``*.wrapper.cpp`` inside the ``.pt2`` zip, and renders the
``package_constants_in_so`` flag we passed into its own model constructor as
``load_constants_from_blob``. So every fact here comes from the artifact's own
generated declaration, parsed with no CUDA, no ``dlopen``, and no load — which
is what lets the gate run on a mint pod before publish AND in a unit test on a
control-plane box.

What is NOT here: the no-baked-adapter gate. ``lora_lifted.assert_no_baked_adapter``
owns it and must be called on the **ExportedProgram**, never on a package —
packing renames a plain-``__dict__`` adapter to ``_tensor_constant0``, so a
package-side FQN scan is a false PASS on the plain-bf16 and fp8-hooks lanes
(measured). ``constant_names`` below is the advisory package-side read only.
"""

from __future__ import annotations

import logging
import re
import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from . import aot_flatten
from .aot_serve import SOURCE_COMPUTED, SOURCE_LITERAL, SOURCE_STATE_DICT
import hashlib

logger = logging.getLogger(__name__)


class PackageIntrospectionError(RuntimeError):
    """A ``.pt2`` package does not declare what it must, or is malformed."""


# ---------------------------------------------------------------------------
# The generated constant table
# ---------------------------------------------------------------------------

_CONSTANT_FIELD = re.compile(
    r"constants_info_\[(\d+)\]\.(name|data_size|from_folded|original_fqn)\s*=\s*"
    r"(?:\"([^\"]*)\"|(\d+)|(true|false))\s*;"
)
_CONSTANT_TYPE = re.compile(
    r"constants_info_\[(\d+)\]\.type\s*=\s*static_cast<int32_t>\s*\(\s*"
    r"torch::aot_inductor::ConstantType::(\w+)\s*\)\s*;"
)
_CONSTANT_DTYPE = re.compile(
    r"constants_info_\[(\d+)\]\.dtype\s*=\s*cached_torch_dtype_(\w+)\s*;"
)
_CONSTANT_SHAPE = re.compile(
    r"constants_info_\[(\d+)\]\.shape\s*=\s*\{([^}]*)\}\s*;"
)
_MODEL_BASE = "AOTInductorModelBase("

#: ``ConstantType`` values sourced from the module's ``state_dict``. Anything
#: else is a graph literal with no state_dict counterpart, whose bytes must ship
#: in ``aot_serve.LITERALS_NAME`` or it can never be bound.
_STATE_DICT_TYPES = ("Parameter", "Buffer")

#: ``ConstantType`` values AOTInductor COMPUTES at load from the constants that
#: were bound — the runtime constant-folding pass's outputs. They are in the
#: package's table and in no exported program, by construction.
_COMPUTED_TYPES = ("FoldedConstant",)


@dataclass(frozen=True)
class DeclaredConstant:
    """One entry of a package's own constant table.

    ``fqn`` is the ORIGINAL fully-qualified name where AOTInductor recorded one
    — it mangles ``lin.weight`` into the C++ identifier ``lin_weight``, so
    ``fqn`` is what a ``state_dict`` lookup must use and ``name`` is what the
    package's C API answers with. Conflating them is a silent bind failure.
    """

    index: int
    name: str
    fqn: str
    data_size: int
    kind: str
    dtype: str
    shape: Tuple[int, ...]
    from_folded: bool

    @property
    def source(self) -> str:
        """``aot_serve`` source class: from the state_dict, or a packed literal."""
        if self.kind in _STATE_DICT_TYPES:
            return SOURCE_STATE_DICT
        return SOURCE_COMPUTED if self.kind in _COMPUTED_TYPES \
            else SOURCE_LITERAL

    def as_manifest_row(self) -> Dict[str, Any]:
        """The row ``aot_serve.constants_from_meta`` parses."""
        return {
            "fqn": self.fqn,
            "source": self.source,
            "dtype": self.dtype,
            "shape": list(self.shape),
        }


def _entry_member(name: str, entry: str) -> bool:
    """Whether a zip member belongs to one named model of a multi-graph
    package. AOTI packages every model under ``data/aotinductor/<name>/``
    (verified on the pin); entry names may carry ``/`` so the
    whole segment run is matched, not a single path component."""
    marker = f"data/aotinductor/{entry}/"
    return f"/{marker}" in name or name.startswith(marker)


def _members(package: Path, suffix: str, entry: str) -> List[str]:
    with zipfile.ZipFile(package) as zf:
        names = [n for n in zf.namelist() if n.endswith(suffix)]
    if entry:
        names = [n for n in names if _entry_member(n, entry)]
    return names


def _wrapper_source(package: Path, entry: str = "") -> str:
    names = _members(package, ".wrapper.cpp", entry)
    if len(names) != 1:
        where = f" for entry {entry!r}" if entry else ""
        raise PackageIntrospectionError(
            f"{package}: expected exactly one *.wrapper.cpp in the "
            f"package{where}, found {len(names)}")
    with zipfile.ZipFile(package) as zf:
        return zf.read(names[0]).decode("utf-8", "replace")


def package_entry_names(package: Path) -> Tuple[str, ...]:
    """The named models a ``.pt2`` carries, read from its own layout —
    every ``data/aotinductor/<name>/`` directory holding a wrapper."""
    names = set()
    for member in _members(Path(package), ".wrapper.cpp", ""):
        _, _, rest = member.rpartition("data/aotinductor/")
        if rest and "/" in rest:
            names.add(rest.rsplit("/", 1)[0])
    return tuple(sorted(names))


def _last_call_argument(source: str, opening: str) -> str:
    """The last top-level argument of the first ``opening`` call.

    Paren-balanced rather than regex'd: the argument list spans lines and
    contains nested calls (``std::move(cubin_dir)``), so a regex would either
    stop early or run away.
    """
    start = source.find(opening)
    if start < 0:
        raise PackageIntrospectionError(
            f"generated wrapper has no {opening!r} call")
    i = start + len(opening)
    depth, arg_start = 1, i
    while i < len(source):
        ch = source[i]
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth == 0:
                return source[arg_start:i].strip()
        elif ch == "," and depth == 1:
            arg_start = i + 1
        i += 1
    raise PackageIntrospectionError(f"unbalanced {opening!r} argument list")


def constants_in_so(package: Path, entry: str = "") -> bool:
    """Whether the package BAKED its constants into the ``.so`` blob.

    Not an inference: this is the ``load_constants_from_blob`` argument
    AOTInductor rendered into its own generated constructor from the
    ``aot_inductor.package_constants_in_so`` config the mint passed.
    ``entry`` scopes the read to one named model of a multi-graph package.
    """
    arg = _last_call_argument(_wrapper_source(Path(package), entry), _MODEL_BASE)
    if arg not in ("true", "false"):
        raise PackageIntrospectionError(
            f"{package}: could not read load_constants_from_blob from the "
            f"generated wrapper (last {_MODEL_BASE[:-1]} argument was {arg!r})")
    return arg == "true"


def declared_constants(package: Path, entry: str = "") -> Tuple[DeclaredConstant, ...]:
    """The package's declared constant table, in declaration order.

    Present and identical whether or not the constants were baked — baking
    changes where the BYTES live, never the declaration. ``entry`` scopes
    the read to one named model of a multi-graph package.
    """
    source = _wrapper_source(Path(package), entry)
    fields: Dict[int, Dict[str, Any]] = {}
    for idx, field_name, text, number, boolean in _CONSTANT_FIELD.findall(source):
        row = fields.setdefault(int(idx), {})
        if field_name == "data_size":
            row[field_name] = int(number)
        elif field_name == "from_folded":
            row[field_name] = boolean == "true"
        else:
            row[field_name] = text
    for idx, kind in _CONSTANT_TYPE.findall(source):
        fields.setdefault(int(idx), {})["kind"] = kind
    for idx, dtype in _CONSTANT_DTYPE.findall(source):
        fields.setdefault(int(idx), {})["dtype"] = dtype
    for idx, dims in _CONSTANT_SHAPE.findall(source):
        fields.setdefault(int(idx), {})["shape"] = tuple(
            int(d.strip()) for d in dims.split(",") if d.strip())

    out: List[DeclaredConstant] = []
    for idx in sorted(fields):
        row = fields[idx]
        name = str(row.get("name") or "")
        if not name:
            raise PackageIntrospectionError(
                f"{package}: constants_info_[{idx}] declares no name")
        out.append(DeclaredConstant(
            index=idx,
            name=name,
            fqn=str(row.get("original_fqn") or name),
            data_size=int(row.get("data_size") or 0),
            kind=str(row.get("kind") or ""),
            dtype=str(row.get("dtype") or ""),
            shape=tuple(row.get("shape") or ()),
            from_folded=bool(row.get("from_folded")),
        ))
    return tuple(out)


def constants_manifest(package: Path, entry: str = "") -> List[Dict[str, Any]]:
    """The declared constant manifest for ``aot_serve.artifact_metadata``.

    Derived from the package's OWN table rather than from the pipeline's
    ``state_dict``, so ``aot_serve.assert_bindable`` — which compares the
    manifest against the loaded artifact's table — is a genuine cross-check of
    two independent derivations and not a tautology.
    """
    return [c.as_manifest_row() for c in declared_constants(Path(package), entry)]


def literal_constants(package: Path, entry: str = "") -> Tuple[DeclaredConstant, ...]:
    """Declared constants with no ``state_dict`` counterpart.

    Graph literals — folded scalars, sinusoidal tables, shape vectors. They
    cannot be bound from resident weights, and under B1 the ``.so`` does not
    carry them either, which is exactly the unbound-constant precondition for
    the worker-killing segfault. So the mint packs their bytes in
    ``aot_serve.LITERALS_NAME``; that is not an optimization, it is what makes a
    code-only artifact loadable at all.
    """
    return tuple(c for c in declared_constants(Path(package), entry)
                 if c.source == SOURCE_LITERAL)


def constant_names(package: Path, entry: str = "") -> Tuple[str, ...]:
    """ADVISORY package-side FQN read. Never a gate.

    Packing erases the FQN of a plain-``__dict__`` tensor (it becomes
    ``_tensor_constant0``), so an absence here proves nothing — which is why
    the adapter gate takes the ExportedProgram instead. Present for
    diagnostics and for the cross-check against the declared manifest.
    """
    return tuple(c.fqn for c in declared_constants(Path(package), entry))


# ---------------------------------------------------------------------------
# B1 — the code-only structural gate
# ---------------------------------------------------------------------------
#
# TWO INDEPENDENT proofs, because this gate is all that stands between us and
# silently duplicating multi-GiB weights into every cell in the fleet:
#
#   1. the rendered ``load_constants_from_blob`` literal — authoritative, the
#      flag itself round-tripped through the artifact;
#   2. the packaged ``.so``'s ``.lrodata`` section size, which is where a baked
#      blob lands. Measured on this toolchain: 0x4470 baked vs 0x31 code-only
#      for a table declaring 17,472 bytes.
#
# Proof 2 exists because proof 1 depends on codegen we do not own: if a torch
# upgrade moved the literal, proof 2 still catches a 2.7 GiB regression, and
# vice versa. Note what proof 2 is NOT — a size threshold. The predicate is
# "``.lrodata`` is at least as large as the bytes the table declares", which has
# no tunable in it: below that figure the blob demonstrably is not in there.

_ELF_MAGIC = b"\x7fELF"
_LRODATA = ".lrodata"


def packaged_so(package: Path, entry: str = "") -> Tuple[str, bytes]:
    """``(archive name, bytes)`` of one model's ``.so`` inside a ``.pt2``.
    ``entry`` scopes to one named model of a multi-graph package."""
    names = _members(Path(package), ".so", entry)
    if len(names) != 1:
        where = f" for entry {entry!r}" if entry else ""
        raise PackageIntrospectionError(
            f"{package}: expected exactly one .so in the package{where}, "
            f"found {len(names)}")
    with zipfile.ZipFile(package) as zf:
        return names[0], zf.read(names[0])


def elf_section_sizes(blob: bytes) -> Dict[str, int]:
    """``{section name: size}`` from a 64-bit little-endian ELF image.

    Hand-parsed rather than shelling out to ``readelf``: this gate must hold on
    a mint pod whose image carries no binutils, and a missing tool must never be
    the reason a weight-baking regression ships.
    """
    if len(blob) < 64 or not blob.startswith(_ELF_MAGIC):
        raise PackageIntrospectionError("packaged .so is not an ELF image")
    if blob[4] != 2 or blob[5] != 1:
        raise PackageIntrospectionError(
            f"packaged .so is not 64-bit little-endian ELF "
            f"(EI_CLASS={blob[4]}, EI_DATA={blob[5]})")
    e_shoff, = struct.unpack_from("<Q", blob, 0x28)
    e_shentsize, e_shnum, e_shstrndx = struct.unpack_from("<HHH", blob, 0x3A)
    if not e_shoff or not e_shnum or e_shstrndx >= e_shnum:
        raise PackageIntrospectionError("packaged .so has no ELF section table")
    strtab_entry = e_shoff + e_shstrndx * e_shentsize
    strtab_off, strtab_size = struct.unpack_from("<QQ", blob, strtab_entry + 0x18)
    strtab = blob[strtab_off:strtab_off + strtab_size]
    sizes: Dict[str, int] = {}
    for i in range(e_shnum):
        off = e_shoff + i * e_shentsize
        sh_name, = struct.unpack_from("<I", blob, off)
        _sh_offset, sh_size = struct.unpack_from("<QQ", blob, off + 0x18)
        end = strtab.find(b"\0", sh_name)
        name = strtab[sh_name:end if end >= 0 else None].decode("utf-8", "replace")
        sizes[name] = sh_size
    return sizes


def code_only_violations(package: Path, entry: str = "") -> List[str]:
    """Named reasons ``package`` is NOT code-only; empty when it is.

    Every reason names the offending tensors: "constants were baked" is not
    actionable, "these 743 parameters totalling 2.73 GiB were baked, largest
    ``down_blocks.0...``" is. Callers turn a non-empty list into a red refusal,
    never a warning. ``entry`` scopes the gate to one named model; the multi-
    graph mint runs it once per entry so a refusal names BOTH the entry and
    the cause.
    """
    package = Path(package)
    reasons: List[str] = []
    constants = declared_constants(package, entry)
    declared_bytes = sum(c.data_size for c in constants)

    if constants_in_so(package, entry):
        worst = sorted(constants, key=lambda c: -c.data_size)[:5]
        shown = ", ".join(f"{c.fqn} ({c.data_size}B)" for c in worst)
        reasons.append(
            f"package declares load_constants_from_blob=true: "
            f"{len(constants)} constants totalling {declared_bytes}B are BAKED "
            f"into the .so — largest: {shown}. Compile with "
            f"aot_inductor.package_constants_in_so=False (pgw#704 B1)")

    so_name, blob = packaged_so(package, entry)
    lrodata = elf_section_sizes(blob).get(_LRODATA, 0)
    if declared_bytes and lrodata >= declared_bytes:
        reasons.append(
            f"{so_name}: {_LRODATA} is {lrodata}B, >= the {declared_bytes}B "
            f"the constant table declares across {len(constants)} tensors — "
            f"the constant blob is linked into the .so (pgw#704 B1)")
    return reasons


def program_constant_fqns(program: Any) -> Tuple[str, ...]:
    """The constant FQNs an ``ExportedProgram`` lifts: parameters, buffers, and
    lifted tensor constants."""
    signature = getattr(program, "graph_signature", None)
    names: set[str] = set()
    for attr in ("parameters", "buffers", "lifted_tensor_constants"):
        names.update(str(n) for n in getattr(signature, attr, ()) or ())
    names.update(str(k) for k in getattr(program, "constants", {}) or {})
    return tuple(sorted(names))


def program_literal_fqns(program: Any) -> Tuple[str, ...]:
    """The FQNs an ``ExportedProgram`` will declare ``source=literal``.

    The complement of :func:`program_state_dict_fqns`: everything the export
    lifted that has NO state_dict counterpart. Those bytes ship INSIDE the
    artifact (``aot_serve.LITERALS_NAME``) and are never rebound at load —
    *"nothing outside the artifact knows its value"*.
    """
    signature = getattr(program, "graph_signature", None)
    state_dict = set(program_state_dict_fqns(program))
    names = {
        str(n) for n in getattr(signature, "lifted_tensor_constants", ()) or ()
    }
    names.update(
        str(k) for k in (getattr(program, "constants", {}) or {})
        if str(k) not in state_dict
    )
    return tuple(sorted(names - state_dict))


def literal_values_digest(program: Any) -> str:
    """Digest of the VALUES of every ``source=literal`` constant — ``""`` when
    there are none.

    **Why values, and only for literals.** A cell's identity folds constant
    NAMES but deliberately not their bytes, because a weight is rebound from
    the resident ``state_dict`` at load — so two fine-tunes of one family
    SHOULD share a cell, and keying weight values would break exactly that.
    A LITERAL is different in kind: it ships inside the artifact and is never
    rebound, so **for a literal the value IS the artifact**. Two checkpoints
    that need different literals must not share a key, and before this they
    did.

    Measured instances, both rope frequency tables and both pure functions of
    config: z-image's ``RopeEmbedder.freqs_cis`` (~393 KB) and qwen-image's
    ``QwenEmbedRope.pos_freqs``/``neg_freqs`` (4.19 MB). **The discriminator
    is ASSIGNMENT STYLE, not class ancestry** — ``QwenEmbedRope`` IS an
    ``nn.Module`` and its ``state_dict()`` is still empty, because the tables
    are plain attributes rather than ``register_buffer``.

    Returns ``""`` when the program lifts no literals, so the caller OMITS the
    field and every family without literals (sdxl: measured zero across five
    real mints) keys byte-identically to before.

    Fails CLOSED: a literal whose bytes cannot be read raises rather than
    being skipped, because a silently-skipped constant is the hole this
    function exists to close.
    """

    names = program_literal_fqns(program)
    if not names:
        return ""
    constants = getattr(program, "constants", {}) or {}
    digest = hashlib.sha256()
    for name in names:                      # already sorted
        value = constants.get(name)
        if value is None:
            raise ValueError(
                f"literal constant {name!r} is declared by the exported "
                f"program but carries no value — its bytes ship inside the "
                f"cell, so an unreadable one cannot be keyed and must not be "
                f"published (pgw#857)")
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        try:
            digest.update(str(getattr(value, "dtype", "")).encode("utf-8"))
            digest.update(str(tuple(getattr(value, "shape", ()))).encode("utf-8"))
            buf = value.detach().cpu().contiguous().reshape(-1)
            digest.update(buf.view(_BYTE_VIEW_DTYPE()).numpy().tobytes())
        except Exception as exc:  # noqa: BLE001 — fail closed, never skip
            raise ValueError(
                f"literal constant {name!r} could not be digested "
                f"({type(exc).__name__}: {exc}); its value is part of the "
                f"artifact and must be keyed (pgw#857)") from exc
    return digest.hexdigest()[:32]


def _BYTE_VIEW_DTYPE() -> Any:
    """``torch.uint8`` — a raw byte view works for every dtype including
    complex, where ``.numpy()`` would need a copy and a dtype special case."""
    import torch

    return torch.uint8


def program_state_dict_fqns(program: Any) -> Tuple[str, ...]:
    """The FQNs an ``ExportedProgram`` will declare ``source=state_dict``.

    Parameters and buffers only: a lifted tensor constant has no resident
    counterpart and ships its bytes as a LITERAL instead. This is the
    program-side mirror of ``DeclaredConstant.source`` — AOTInductor records
    exactly these two input kinds as ``ConstantType::Parameter``/``Buffer``,
    which is what makes the check below predictive of the packed manifest.
    """
    signature = getattr(program, "graph_signature", None)
    names: set[str] = set()
    for attr in ("parameters", "buffers"):
        names.update(str(n) for n in getattr(signature, attr, ()) or ())
    return tuple(sorted(names))


def unbindable_program_constants(
    program: Any, state_dict_keys: Iterable[str],
) -> List[str]:
    """:func:`unbindable_constants`, asked BEFORE the compile is paid for.

    The packed gate fires per entry AFTER that entry's 4-6 minute AOTI
    compile, so a declaration/module mismatch costs a whole rental to learn.
    The exported program already names every parameter and buffer it lifted,
    and AOTInductor's constant table is a function of exactly that — so the
    same refusal is knowable in milliseconds, off the program, before a kernel
    is built.

    The packed gate is NOT replaced by this one: it reads the artifact's own
    generated wrapper, which is the only proof of what actually shipped.
    """
    available = set(state_dict_keys)
    if not available:
        return []
    missing = [
        fqn for fqn in program_state_dict_fqns(program) if fqn not in available
    ]
    if not missing:
        return []
    return [
        f"{len(missing)} constant(s) the exported program lifts from the "
        f"module's state_dict are absent from the resident module's bind "
        f"table, e.g. {missing[:6]!r} — the compiled cell could never bind "
        f"them, so the compile would be paid for an unpublishable entry"
    ]


def program_package_drift(
    program: Any, package: Path, entry: str = "",
) -> List[str]:
    """Named reasons the package's constant table cannot be served.

    **The check is ASYMMETRIC.** The two sets are derived independently — one
    from the ``ExportedProgram``, one from the compiled artifact's own
    generated wrapper — but they are not required to be EQUAL, because the two
    directions mean opposite things:

    * **package-only** (the artifact declares a constant the program did not
      lift) is FATAL. Nothing would bind it, and invoking a code-only package
      with an unbound constant segfaults inside ``AOTICompiledModel.__call__``,
      killing the worker rather than failing the request.
    * **program-only** (the program lifted a constant the compiled artifact does
      not want) is BENIGN and routine: the compiler folded or inlined it away,
      so the artifact genuinely has no use for it. Measured on a real SDXL w8a8
      mint — program 2423, package 2422. Demanding equality refuses EVERY real
      mint.

    The manifest is built FROM the package, so program-only drift never
    threatens the manifest's fidelity — it is recorded for observability by
    :func:`eliminated_constants` and is not an error.

    A strict/non-strict trace mix (the two lift different constant sets) is
    still covered: it shows up as package-only entries, because the package
    would want constants the recorded program never lifted.
    """
    want = set(program_constant_fqns(program))
    have = {c.fqn for c in declared_constants(Path(package), entry)
            if c.source != SOURCE_COMPUTED}
    # A COMPUTED constant is package-only by design — the runtime fold produces
    # it from the bound constants at load, so "the program never lifted it" is
    # the expected state and not the segfault precondition this gate exists for.
    package_only = sorted(have - want)
    if not package_only:
        return []
    return [
        f"the compiled package declares {len(package_only)} constant(s) the "
        f"exported program never lifted, e.g. {package_only[:6]!r} — nothing "
        f"would bind them and the first call would segfault (pgw#704 B1; a "
        f"strict/non-strict trace mix surfaces here, pgw#728)"
    ]


def eliminated_constants(
    program: Any, package: Path, entry: str = "",
) -> List[str]:
    """Constants the program lifted that the compiled artifact does not want.

    Anonymous graph literals the compiler folded away. Recorded as
    observability so a surprising JUMP in the count is visible, rather than
    silently discarded — the count is stable for a given recipe.

    **A WEIGHT here is never routine** — that is :func:`folded_weights`, which
    refuses. The mechanism is INLINING, not fusion: a small 1-D constant meets
    ``GraphLowering.can_inline_constant``, so inductor renders that
    checkpoint's own values straight into the kernel source.
    """
    want = set(program_constant_fqns(program))
    have = {c.fqn for c in declared_constants(Path(package), entry)}
    return sorted(want - have)


def folded_weights(
    program: Any, package: Path, state_dict_keys: Iterable[str],
    entry: str = "",
) -> List[str]:
    """Weights the program lifted that the artifact will NOT let anyone bind.

    THE folding fence, enforcing the tensor-binding contract. One cell serves
    every fine-tune of a family
    because weights rebind BY NAME at load — which is sound only while the
    compiled code holds no weight VALUE. Where a lifted weight is missing
    from the artifact's own constant table, its value is not missing: it was
    rendered into the kernel source, and it is the MINTING checkpoint's.
    Every other fine-tune then adopts a cell carrying somebody else's tensor.

    That failure is caught downstream by the adopt-side parity floor, so
    nothing corrupt serves — but the cell is refused per checkpoint, which
    turns fleet-wide cell sharing into a per-checkpoint compile without
    anything saying why. So it is refused HERE, once, on the mint pod.

    Mechanism, read off torch 2.13.0 and MEASURED: with
    ``aot_inductor.use_runtime_constant_folding`` and
    ``always_keep_tensor_constants`` both off, ``GraphLowering.get_attr``
    inlines a constant whose SHAPE meets either rule — 0-dim (rendered via
    ``.item()``) or ``len(shape) == 1 and shape[0] <= 8``
    (``can_inline_constant``). The selection is by shape alone, so it is
    deterministic and enumerable; whether it BITES is whether two fine-tunes
    differ in one of those tensors. ``aot_mint.CONSTANT_BINDING_CONFIGS``
    turns it off — and this gate is what makes that a proof rather than a
    setting, so a future inlining route fails closed instead of shipping.
    """
    available = set(state_dict_keys)
    if not available:
        return []
    lifted = set(program_constant_fqns(program))
    have = {c.fqn for c in declared_constants(Path(package), entry)}
    folded = sorted((lifted & available) - have)
    if not folded:
        return []
    return [
        f"{len(folded)} weight(s) the exported program lifted are ABSENT from "
        f"the compiled artifact's constant table: {folded[:6]!r} — their "
        f"values were compiled into the kernel, so this cell carries THIS "
        f"checkpoint's copy and no other fine-tune could rebind them "
        f"(pgw#1097 folding fence; pgw#857 tensor-binding contract)"
    ]


def unbindable_constants(
    package: Path, state_dict_keys: Iterable[str], entry: str = "",
) -> List[str]:
    """Declared state_dict-sourced constants no resident weight could bind.

    The mint-side mirror of ``aot_serve``'s bound gate. Catching it here means a
    cell that could only ever fail to arm is never published — the fleet learns
    on the mint pod instead of by every serving pod refusing it one at a time.
    """
    available = set(state_dict_keys)
    if not available:
        return []
    missing = [
        c.fqn for c in declared_constants(Path(package), entry)
        if c.source == SOURCE_STATE_DICT and c.fqn not in available
    ]
    if not missing:
        return []
    return [
        f"{len(missing)} declared state_dict constant(s) are absent from the "
        f"resident module's state_dict, e.g. {missing[:6]!r} — the cell could "
        f"never bind its constants and must not be published"
    ]


# ---------------------------------------------------------------------------
# The generated input guards — the artifact's OWN admission facts
# ---------------------------------------------------------------------------
#
# AOTInductor renders, per user input, a `check_input_<i>` function into the
# generated wrapper: an exact dtype check always, an exact equality check per
# STATIC dim, and range checks (no equality) per symbolic dim. Those checks
# are what the compiled object itself will enforce — so they are the one
# package-side reading of the ingress contract that cannot drift from the
# artifact. `input_guard_drift` compares them against the declared manifest
# rows, making the manifest a VERIFIED ECHO of the artifact rather than a
# carried-alongside label: a label can be honest about the PROGRAM while the
# artifact is specialized on a dtype real traffic never presents, so only a
# reading of the artifact's own guards — or a real call — can rule on what it
# admits.

_INPUT_GUARD_FN = re.compile(r"static\s+void\s+check_input_(\d+)\s*\(")
_GUARD_DTYPE = re.compile(r"expected_dtype\s*=\s*aoti_torch_dtype_(\w+)\s*\(")
_GUARD_STATIC_DIM = re.compile(r"if\s*\(\s*(\d+)\s*!=\s*\w+_size\[(\d+)\]\s*\)")


@dataclass(frozen=True)
class InputGuard:
    """One input's generated admission checks, read off the artifact.

    ``dims`` carries only the dims the artifact checks for EQUALITY — its
    statically specialized dims. A symbolic dim has range checks and no
    equality check, so its absence here is itself a fact: a dim present in
    ``dims`` that the manifest declares symbolic is a lying label.
    """

    index: int
    dtype: str
    dims: Tuple[Tuple[int, int], ...]  # (dim position, required value)

    def dim_map(self) -> Dict[int, int]:
        return dict(self.dims)


def input_guards(package: Path, entry: str = "") -> Tuple[InputGuard, ...]:
    """Every ``check_input_<i>`` of one named model, in input order.

    Parsed from the artifact's own generated wrapper — no CUDA, no dlopen —
    exactly like the constant table above. Refuses (never returns empty) when
    the wrapper declares no input checks: the pinned toolchain demonstrably
    emits them, so their absence means the parse went blind, and a gate that
    silently sees no guards is no gate.
    """
    source = _wrapper_source(Path(package), entry)
    return guards_from_wrapper_source(source, where=str(package))


def guards_from_wrapper_source(
    source: str, where: str = "",
) -> Tuple[InputGuard, ...]:
    """:func:`input_guards` on wrapper SOURCE — split out so the gate can be
    asserted against captured wrapper bytes without a whole ``.pt2``."""
    starts = [(int(m.group(1)), m.start()) for m in _INPUT_GUARD_FN.finditer(source)]
    if not starts:
        raise PackageIntrospectionError(
            f"{where or 'wrapper'}: generated wrapper declares no "
            f"check_input_<i> functions — the artifact's own admission "
            f"checks cannot be read, so nothing can prove the declared "
            f"manifest describes this program (pgw#1058)")
    starts.sort(key=lambda pair: pair[1])
    out: List[InputGuard] = []
    for pos, (index, begin) in enumerate(starts):
        end = starts[pos + 1][1] if pos + 1 < len(starts) else len(source)
        body = source[begin:end]
        dtype = _GUARD_DTYPE.search(body)
        if dtype is None:
            raise PackageIntrospectionError(
                f"{where or 'wrapper'}: check_input_{index} declares no "
                f"expected dtype — the guard cannot be compared (pgw#1058)")
        dims = tuple(
            (int(dim), int(value))
            for value, dim in _GUARD_STATIC_DIM.findall(body)
        )
        out.append(InputGuard(index=index, dtype=dtype.group(1), dims=dims))
    indices = [g.index for g in out]
    if indices != list(range(len(out))):
        raise PackageIntrospectionError(
            f"{where or 'wrapper'}: check_input indices {indices!r} are not "
            f"dense from 0 — the guard order cannot be matched to the "
            f"declared input positions (pgw#1058)")
    return tuple(out)


def input_guard_drift(
    inputs_rows: Sequence[Mapping[str, Any]],
    guards: Sequence[InputGuard],
) -> List[str]:
    """Named reasons the declared manifest rows do NOT describe the program
    whose guards these are; empty when they agree.

    The comparison that makes an entry's label a verified echo:
    rows come from the ``ExportedProgram`` (or a packed ``metadata.json``),
    guards from the artifact's own generated wrapper — two independent
    readings that must agree, the same doctrine as the constant manifest.
    Every reason names the input, the axis and both values. Callers turn a
    non-empty list into a refusal — at package time so a divergent cell is
    never published, and at arm time so a corrupted one is never served.
    """
    rows = sorted(inputs_rows, key=lambda r: int(r.get("position", 0)))
    reasons: List[str] = []
    if len(rows) != len(guards):
        return [
            f"the manifest declares {len(rows)} input(s) but the artifact "
            f"guards {len(guards)} — the two cannot describe one call "
            f"contract (pgw#1058)"
        ]
    for row, guard in zip(rows, guards):
        name = str(row.get("name") or f"input_{guard.index}")
        declared_dtype = str(row.get("dtype") or "")
        if declared_dtype != guard.dtype:
            reasons.append(
                f"input {name!r}: manifest declares dtype {declared_dtype!r} "
                f"but the artifact's own guard requires {guard.dtype!r}")
        shape = list(row.get("shape") or [])
        guarded = guard.dim_map()
        for pos, declared in enumerate(shape):
            if isinstance(declared, int):
                have = guarded.pop(pos, None)
                if have is None:
                    reasons.append(
                        f"input {name!r} dim {pos}: manifest declares static "
                        f"{declared} but the artifact states no equality "
                        f"check there — the program did not specialize the "
                        f"dim the label claims")
                elif have != declared:
                    reasons.append(
                        f"input {name!r} dim {pos}: manifest declares static "
                        f"{declared} but the artifact's guard requires {have}")
            else:
                have = guarded.pop(pos, None)
                if have is not None:
                    reasons.append(
                        f"input {name!r} dim {pos}: manifest declares "
                        f"symbolic {declared!r} but the artifact's guard "
                        f"requires exactly {have} — the program is more "
                        f"specialized than its label admits")
        for pos, have in sorted(guarded.items()):
            reasons.append(
                f"input {name!r} dim {pos}: the artifact guards {have} on a "
                f"dim the manifest does not declare (declared rank "
                f"{len(shape)})")
    return reasons


def admission_drift(
    package: Path, entry: str, inputs_rows: Sequence[Mapping[str, Any]],
) -> List[str]:
    """:func:`input_guard_drift` against one named model of a ``.pt2``.

    ONE function, two call sites: the mint's
    package gate (a divergent cell is never published) and the arm's
    per-entry verification (a corrupted one is never served) must be the
    same derivation, or the gate models the arm and drifts from it.
    """
    return input_guard_drift(inputs_rows, input_guards(Path(package), entry))


# ---------------------------------------------------------------------------
# The ingress contract — recorded so B2 CAN be asserted
# ---------------------------------------------------------------------------


def input_contract(
    program: Any, leaves: Sequence[aot_flatten.Leaf],
) -> Tuple[List[Dict[str, Any]], Dict[str, List[int]]]:
    """``(inputs, symbols)`` rows for ``aot_serve.artifact_metadata``.

    Static dims are recorded as ints and symbolic dims by SYMBOL NAME, with the
    symbol's inclusive bounds in ``symbols`` — the shape ``aot_serve``'s B2
    ingress assertion consumes. Symbol names are fine as keys because they are
    scoped to one artifact and only ever resolved through that same artifact's
    ``symbols`` map.

    Recording bounds at all is the whole point of B2: 2048x2048 runs clean
    through an artifact declaring max=160 latent units, because the exported
    graph carries ZERO symbolic range assertions.
    ``ep.range_constraints`` is metadata only, so unless the mint writes it
    down the consumer has nothing to assert against.

    Dtypes and shapes come off the exported program's placeholders rather than
    off the example inputs, so what is recorded is what export actually
    committed to — which is what the consumer will be checked against.

    ``leaves`` are ``aot_flatten`` leaves, not names: each row
    records WHERE IN THE CALL its input lives — parameter, that parameter's
    position, and the path into it — because a name cannot tell the serve side
    that ``x.0`` is element 0 of the single argument ``x`` rather than the
    argument in slot 0. ``aot_serve.bind_call_inputs`` replays exactly this.
    """
    ranges = _symbol_ranges(program)
    rows: List[Dict[str, Any]] = []
    symbols: Dict[str, List[int]] = {}
    for position, (leaf, val) in enumerate(
        zip(leaves, _user_input_vals(program))
    ):
        name = leaf.name
        if val is None:
            continue
        shape: List[Any] = []
        for dim in getattr(val, "shape", ()) or ():
            text = str(dim)
            if text.lstrip("-").isdigit():
                shape.append(int(text))
                continue
            bounds = ranges.get(text)
            if bounds is None:
                raise PackageIntrospectionError(
                    f"input {name!r} has symbolic dim {text!r} with no range "
                    f"constraint in the exported program — the consumer would "
                    f"have nothing to assert (pgw#704 B2)")
            symbols[text] = [bounds[0], bounds[1]]
            shape.append(text)
        row: Dict[str, Any] = {
            "name": str(name),
            "position": position,
            "dtype": _dtype_name(getattr(val, "dtype", None)),
            "shape": shape,
            "optional": False,
        }
        if not leaf.trivial or leaf.param_position != position:
            # Written only when the identity is NOT derivable from the row
            # itself. An absent field means `param=name, param_position=
            # position, path=()` — true exactly when the leaf IS its argument
            # AND no earlier argument flattened into more than one leaf. The
            # second half is the half that bites: a plain tensor sitting after
            # a container has a flat position its ARGUMENT does not have, and
            # a serve-side bind reading `position` would fetch the wrong one
            # (measured). A container-free declaration keeps every row
            # derivable, so its metadata and cell key are unchanged — see
            # `aot_serve.range_digest`.
            row["param"] = leaf.param
            row["param_position"] = leaf.param_position
            row["path"] = [
                step if isinstance(step, str) else int(step)
                for step in leaf.path
            ]
        rows.append(row)
    if not rows:
        raise PackageIntrospectionError(
            "exported program declares no user inputs; an artifact with no "
            "ingress contract cannot be asserted (pgw#704 B2)")
    return rows, symbols


def _symbol_ranges(program: Any) -> Dict[str, Tuple[int, int]]:
    out: Dict[str, Tuple[int, int]] = {}
    for symbol, interval in (getattr(program, "range_constraints", {}) or {}).items():
        lower = _bound_int(getattr(interval, "lower", None))
        upper = _bound_int(getattr(interval, "upper", None))
        if lower is None or upper is None:
            # An unbounded symbol is the B2 hole with extra steps: it would
            # admit any value. Refuse rather than record a fake bound.
            raise PackageIntrospectionError(
                f"exported program symbol {symbol} has an unbounded range "
                f"({getattr(interval, 'lower', None)}.."
                f"{getattr(interval, 'upper', None)}); an artifact must declare "
                f"a finite declared envelope (pgw#704 B2)")
        out[str(symbol)] = (lower, upper)
    return out


def _bound_int(value: Any) -> Any:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _user_input_vals(program: Any) -> List[Any]:
    signature = getattr(program, "graph_signature", None)
    user_inputs = [str(n) for n in getattr(signature, "user_inputs", ()) or ()]
    by_name: Dict[str, Any] = {}
    module = getattr(program, "graph_module", None)
    graph = getattr(module, "graph", None)
    for node in getattr(graph, "nodes", ()) or ():
        if getattr(node, "op", "") == "placeholder":
            by_name[str(node.name)] = node.meta.get("val")
    return [by_name.get(name) for name in user_inputs]


def _dtype_name(value: Any) -> str:
    text = str(value or "")
    return text.split(".")[-1] if text.startswith("torch.") else text


__all__ = [
    "DeclaredConstant",
    "InputGuard",
    "PackageIntrospectionError",
    "admission_drift",
    "code_only_violations",
    "guards_from_wrapper_source",
    "input_guard_drift",
    "input_guards",
    "constant_names",
    "constants_in_so",
    "constants_manifest",
    "declared_constants",
    "elf_section_sizes",
    "input_contract",
    "literal_constants",
    "eliminated_constants",
    "folded_weights",
    "package_entry_names",
    "program_constant_fqns",
    "program_package_drift",
    "packaged_so",
    "literal_values_digest",
    "program_literal_fqns",
    "unbindable_constants",
]
