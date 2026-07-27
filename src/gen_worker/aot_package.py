"""``.pt2`` introspection and the B1 code-only gate (pgw#704; #723 produce half).

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
(pgw#725, measured). ``constant_names`` below is the advisory package-side read
only.
"""

from __future__ import annotations

import logging
import re
import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .aot_serve import SOURCE_LITERAL, SOURCE_STATE_DICT

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
        return SOURCE_STATE_DICT if self.kind in _STATE_DICT_TYPES \
            else SOURCE_LITERAL

    def as_manifest_row(self) -> Dict[str, Any]:
        """The row ``aot_serve.constants_from_meta`` parses."""
        return {
            "fqn": self.fqn,
            "source": self.source,
            "dtype": self.dtype,
            "shape": list(self.shape),
        }


def _wrapper_source(package: Path) -> str:
    with zipfile.ZipFile(package) as zf:
        names = [n for n in zf.namelist() if n.endswith(".wrapper.cpp")]
        if len(names) != 1:
            raise PackageIntrospectionError(
                f"{package}: expected exactly one *.wrapper.cpp in the "
                f"package, found {len(names)}")
        return zf.read(names[0]).decode("utf-8", "replace")


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


def constants_in_so(package: Path) -> bool:
    """Whether the package BAKED its constants into the ``.so`` blob.

    Not an inference: this is the ``load_constants_from_blob`` argument
    AOTInductor rendered into its own generated constructor from the
    ``aot_inductor.package_constants_in_so`` config the mint passed.
    """
    arg = _last_call_argument(_wrapper_source(Path(package)), _MODEL_BASE)
    if arg not in ("true", "false"):
        raise PackageIntrospectionError(
            f"{package}: could not read load_constants_from_blob from the "
            f"generated wrapper (last {_MODEL_BASE[:-1]} argument was {arg!r})")
    return arg == "true"


def declared_constants(package: Path) -> Tuple[DeclaredConstant, ...]:
    """The package's declared constant table, in declaration order.

    Present and identical whether or not the constants were baked — baking
    changes where the BYTES live, never the declaration.
    """
    source = _wrapper_source(Path(package))
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


def constants_manifest(package: Path) -> List[Dict[str, Any]]:
    """The declared constant manifest for ``aot_serve.artifact_metadata``.

    Derived from the package's OWN table rather than from the pipeline's
    ``state_dict``, so ``aot_serve.assert_bindable`` — which compares the
    manifest against the loaded artifact's table — is a genuine cross-check of
    two independent derivations and not a tautology.
    """
    return [c.as_manifest_row() for c in declared_constants(Path(package))]


def literal_constants(package: Path) -> Tuple[DeclaredConstant, ...]:
    """Declared constants with no ``state_dict`` counterpart.

    Graph literals — folded scalars, sinusoidal tables, shape vectors. They
    cannot be bound from resident weights, and under B1 the ``.so`` does not
    carry them either, which is exactly the unbound-constant precondition for
    the worker-killing segfault. So the mint packs their bytes in
    ``aot_serve.LITERALS_NAME``; that is not an optimization, it is what makes a
    code-only artifact loadable at all.
    """
    return tuple(c for c in declared_constants(Path(package))
                 if c.source == SOURCE_LITERAL)


def constant_names(package: Path) -> Tuple[str, ...]:
    """ADVISORY package-side FQN read. Never a gate.

    Packing erases the FQN of a plain-``__dict__`` tensor (it becomes
    ``_tensor_constant0``), so an absence here proves nothing — which is why
    pgw#725's adapter gate takes the ExportedProgram instead. Present for
    diagnostics and for the cross-check against the declared manifest.
    """
    return tuple(c.fqn for c in declared_constants(Path(package)))


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


def packaged_so(package: Path) -> Tuple[str, bytes]:
    """``(archive name, bytes)`` of the single ``.so`` inside a ``.pt2``."""
    with zipfile.ZipFile(package) as zf:
        names = [n for n in zf.namelist() if n.endswith(".so")]
        if len(names) != 1:
            raise PackageIntrospectionError(
                f"{package}: expected exactly one .so in the package, "
                f"found {len(names)}")
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


def code_only_violations(package: Path) -> List[str]:
    """Named reasons ``package`` is NOT code-only; empty when it is.

    Every reason names the offending tensors: "constants were baked" is not
    actionable, "these 743 parameters totalling 2.73 GiB were baked, largest
    ``down_blocks.0...``" is. Callers turn a non-empty list into a red refusal,
    never a warning.
    """
    package = Path(package)
    reasons: List[str] = []
    constants = declared_constants(package)
    declared_bytes = sum(c.data_size for c in constants)

    if constants_in_so(package):
        worst = sorted(constants, key=lambda c: -c.data_size)[:5]
        shown = ", ".join(f"{c.fqn} ({c.data_size}B)" for c in worst)
        reasons.append(
            f"package declares load_constants_from_blob=true: "
            f"{len(constants)} constants totalling {declared_bytes}B are BAKED "
            f"into the .so — largest: {shown}. Compile with "
            f"aot_inductor.package_constants_in_so=False (pgw#704 B1)")

    so_name, blob = packaged_so(package)
    lrodata = elf_section_sizes(blob).get(_LRODATA, 0)
    if declared_bytes and lrodata >= declared_bytes:
        reasons.append(
            f"{so_name}: {_LRODATA} is {lrodata}B, >= the {declared_bytes}B "
            f"the constant table declares across {len(constants)} tensors — "
            f"the constant blob is linked into the .so (pgw#704 B1)")
    return reasons


def unbindable_constants(
    package: Path, state_dict_keys: Iterable[str],
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
        c.fqn for c in declared_constants(Path(package))
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
# The ingress contract — recorded so B2 CAN be asserted
# ---------------------------------------------------------------------------


def input_contract(
    program: Any, input_names: Sequence[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, List[int]]]:
    """``(inputs, symbols)`` rows for ``aot_serve.artifact_metadata``.

    Static dims are recorded as ints and symbolic dims by SYMBOL NAME, with the
    symbol's inclusive bounds in ``symbols`` — the shape ``aot_serve``'s B2
    ingress assertion consumes. Symbol names are fine as keys because they are
    scoped to one artifact and only ever resolved through that same artifact's
    ``symbols`` map.

    Recording bounds at all is the whole point of B2: pgw#704 measured
    2048x2048 running clean through an artifact declaring max=160 latent units,
    because the exported graph carries ZERO symbolic range assertions.
    ``ep.range_constraints`` is metadata only, so unless the mint writes it
    down the consumer has nothing to assert against.

    Dtypes and shapes come off the exported program's placeholders rather than
    off the example inputs, so what is recorded is what export actually
    committed to — which is what the consumer will be checked against.
    """
    ranges = _symbol_ranges(program)
    rows: List[Dict[str, Any]] = []
    symbols: Dict[str, List[int]] = {}
    for position, (name, val) in enumerate(
        zip(input_names, _user_input_vals(program))
    ):
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
        rows.append({
            "name": str(name),
            "position": position,
            "dtype": _dtype_name(getattr(val, "dtype", None)),
            "shape": shape,
            "optional": False,
        })
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
                f"finite admissible traffic (pgw#704 B2)")
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
    "PackageIntrospectionError",
    "code_only_violations",
    "constant_names",
    "constants_in_so",
    "constants_manifest",
    "declared_constants",
    "elf_section_sizes",
    "input_contract",
    "literal_constants",
    "packaged_so",
    "unbindable_constants",
]
