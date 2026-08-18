"""Layout contracts as objects: the actual layout, imported and passed around.

A lane declaration is not a name or an enum pointing at a layout — it IS the
layout: ``lanes=(tensorfs.contracts.SDXL_DIFFUSERS_BF16, ...)``. The
predefined constants (:mod:`tensorfs.contracts`) are typed helpers over the
curated library; an author who needs a layout the library does not ship
constructs one inline::

    my_layout = Contract(
        dtype="bfloat16",
        tensors=[
            TensorDecl(
                role="blocks.{i}.attn.qkv",
                pattern="blocks.{i}.attn.qkv_proj.weight",
                rank=2,
                fusion=Fusion(parts=[("q", 1), ("k", 1), ("v", 1)]),
            ),
        ],
    )

Custom contracts are ANONYMOUS: there is deliberately no ``name=`` — identity
is the content digest (``sha256:<hex>``), because a free-text name on an
inline object validates nothing and can lie or collide. The author's variable
name is their label; platform surfaces spell the digest. ``name@version``
survives only for the curated library, where CI digest-pinning makes the name
a real promise.

Construction serializes to a v1 document and runs the validator, so a
malformed contract refuses at author-module import time, never at deploy.

pgw#1391 REWRITE — WHY THIS FILE IS NOT UPSTREAM'S VERBATIM
-----------------------------------------------------------
Upstream's ``contract.py`` delegates parse/validate/digest to the compiled
Rust extension (``from .native import contract_info``), and pgw#1310 rules a
compiled extension out of a source-vendored wheel. So the validator and the
canonical rendering are restored here in pure Python, exactly as pgw#1365 did
for ``planner.py``, and for the same reason.

This is a PORT, not a reimplementation, and it is not asserted — it is proven.
The digest input is a line-oriented canonical rendering fully specified by
``crates/tensorfs-core/src/contract.rs`` (``Contract::canonical``) and
independently re-expressed by tensorfs#114's Go implementation
(``contract.go``). All three are pinned to ONE language-neutral corpus,
``spec/v1/contract-vectors``, vendored at ``tests/testdata/contract-vectors/``
and run by ``tests/test_lane_contracts.py``: every golden digest
and every typed refusal. That corpus, not discipline, is the sync mechanism.

The public surface is upstream's, attribute for attribute, so re-vendoring a
future pure-Python upstream is a deletion rather than a migration.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "CONTRACT_FORMAT",
    "Contract",
    "ContractError",
    "Fusion",
    "MissingDtype",
    "Permute",
    "TensorDecl",
]

#: The one document format tag.
CONTRACT_FORMAT = "tensorfs-contract-v1"

MAX_CONTRACT_NAME_BYTES = 64
MAX_PATTERN_BYTES = 512
MAX_DTYPE_BYTES = 32


class MissingDtype(ValueError):
    """A lane read asked for a load dtype the contract does not declare."""


class ContractError(ValueError):
    """A typed refusal, carrying the kebab-case ``reason`` label shared with
    the Rust validator's ``ContractError::reason`` and the Go one's."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(f"contract: {reason}: {message}")
        self.reason = reason
        self.message = message


def _refuse(reason: str, message: str) -> ContractError:
    return ContractError(reason, message)


# ── the author-facing declaration types ──────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Fusion:
    """A fusion along the outermost axis: ``groups`` repetitions of the
    ordered ``parts`` cycle, each part ``(role_suffix, share)``."""

    parts: Sequence[tuple[str, int]]
    groups: int = 1

    def _raw(self) -> dict[str, Any]:
        return {
            "axis": 0,
            **({"groups": self.groups} if self.groups != 1 else {}),
            "parts": [{"role": role, "share": share} for role, share in self.parts],
        }


@dataclass(frozen=True, slots=True)
class Permute:
    """A generalized permute: reshape to ``view``, permute those axes,
    reshape back. ``view`` entries are literals, ``"shape[k]"`` /
    ``"shape[k]/n"``, or ``"auto"``."""

    view: Sequence[int | str]
    axes: Sequence[int]

    def _raw(self) -> dict[str, Any]:
        return {"view": list(self.view), "axes": list(self.axes)}


@dataclass(frozen=True, slots=True)
class TensorDecl:
    """One declared tensor family: its spelling-independent role, its spelling
    in the file, and the constraints that make the contract falsifiable."""

    role: str
    pattern: str
    dtypes: Sequence[str] = ()
    rank: int | None = None
    required: bool = True
    fusion: Fusion | None = None
    permute: Permute | None = None

    def _raw(self) -> dict[str, Any]:
        raw: dict[str, Any] = {"role": self.role, "pattern": self.pattern}
        if self.dtypes:
            raw["dtypes"] = list(self.dtypes)
        if self.rank is not None:
            raw["rank"] = self.rank
        if not self.required:
            raw["required"] = False
        if self.fusion is not None:
            raw["fusion"] = self.fusion._raw()
        if self.permute is not None:
            raw["permute"] = self.permute._raw()
        return raw


def _decl_of(raw: Mapping[str, Any]) -> TensorDecl:
    fusion = None
    if "fusion" in raw:
        fusion = Fusion(
            parts=tuple((part["role"], part["share"]) for part in raw["fusion"]["parts"]),
            groups=raw["fusion"].get("groups", 1),
        )
    permute = None
    if "permute" in raw:
        permute = Permute(
            view=tuple(raw["permute"]["view"]),
            axes=tuple(raw["permute"]["axes"]),
        )
    return TensorDecl(
        role=raw["role"],
        pattern=raw["pattern"],
        dtypes=tuple(raw.get("dtypes", ())),
        rank=raw.get("rank"),
        required=raw.get("required", True),
        fusion=fusion,
        permute=permute,
    )


# ── the validator (the port) ─────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class _Pattern:
    """Literal text with ``{i}`` integer holes. Adjacent holes, a digit
    directly after a hole, and stray braces all refuse."""

    text: str
    holes: int


def _parse_pattern(text: object) -> _Pattern:
    if not isinstance(text, str) or not text or len(text.encode()) > MAX_PATTERN_BYTES:
        raise _refuse("pattern", json.dumps(text))
    holes = 0
    last_was_hole = False
    rest = text
    while rest:
        at = rest.find("{i}")
        if at < 0:
            if "{" in rest or "}" in rest:
                raise _refuse("pattern", json.dumps(text))
            break
        if at > 0:
            if "{" in rest[:at] or "}" in rest[:at]:
                raise _refuse("pattern", json.dumps(text))
        elif last_was_hole:
            # Adjacent holes cannot be separated by any input.
            raise _refuse("pattern", json.dumps(text))
        holes += 1
        last_was_hole = True
        rest = rest[at + 3 :]
        if rest and rest[0].isdigit() and rest[0].isascii():
            # A digit after a hole makes the split ambiguous.
            raise _refuse("pattern", json.dumps(text))
    return _Pattern(text=text, holes=holes)


def _is_lower(character: str) -> bool:
    return character.isascii() and (character.islower() or character.isdigit())


def _is_contract_name(name: object) -> bool:
    """``<producer>.<format>``, split on the FIRST dot.

    The producer segment carries a hyphen because gen-worker's model-type
    vocabulary does (``hidream-o1``, ``flux-2``, ``wan-2``) — tensorfs#121
    relaxed it for exactly that reason. A LEADING hyphen still refuses. The
    format segment is lowercase alphanumerics plus ``.``, ``-`` and ``_``,
    also not leading.
    """

    if not isinstance(name, str) or not name or len(name.encode()) > MAX_CONTRACT_NAME_BYTES:
        return False
    producer, separator, layout = name.partition(".")
    if not separator or not producer or not layout:
        return False
    if not _is_lower(producer[0]):
        return False
    if not all(_is_lower(character) or character == "-" for character in producer):
        return False
    if not _is_lower(layout[0]):
        return False
    return all(_is_lower(character) or character in ".-_" for character in layout)


def _is_dtype(dtype: object) -> bool:
    """Torch-style, lowercase, bounded — a shape, not an enum."""

    if not isinstance(dtype, str) or not dtype or len(dtype.encode()) > MAX_DTYPE_BYTES:
        return False
    return all(
        character.isascii() and (character.islower() or character.isdigit() or character == "_")
        for character in dtype
    )


def _uint(value: object) -> int | None:
    """A JSON non-negative integer, or ``None`` when the value is not one.
    ``bool`` is not an integer here, matching the typed deserializers."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


_CONTRACT_FIELDS = frozenset(
    {"format", "name", "version", "description", "dtype", "tensors", "sets"}
)
_TENSOR_FIELDS = frozenset({"role", "pattern", "dtypes", "rank", "required", "fusion", "permute"})
_FUSION_FIELDS = frozenset({"axis", "groups", "parts"})
_PART_FIELDS = frozenset({"role", "share"})
_PERMUTE_FIELDS = frozenset({"view", "axes"})


def _known_fields_only(raw: object, allowed: frozenset[str], what: str) -> Mapping[str, Any]:
    """The typed deserializers on both other sides refuse an unknown field as
    a JSON SHAPE error rather than a semantic one; so does this."""

    if not isinstance(raw, Mapping):
        raise _refuse("json", f"{what} is not an object")
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise _refuse("json", f"unknown field {unknown[0]!r} in {what}")
    return raw


@dataclass(frozen=True, slots=True)
class _FusionPart:
    role: str
    share: int


@dataclass(frozen=True, slots=True)
class _Fusion:
    groups: int
    parts: tuple[_FusionPart, ...]


@dataclass(frozen=True, slots=True)
class _Dim:
    kind: str  # "literal" | "axis" | "auto"
    literal: int = 0
    axis: int = 0
    divisor: int = 1


@dataclass(frozen=True, slots=True)
class _Permute:
    view: tuple[_Dim, ...]
    axes: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _Tensor:
    role: _Pattern
    pattern: _Pattern
    dtypes: tuple[str, ...]
    rank: int | None
    required: bool
    fusion: _Fusion | None
    permute: _Permute | None


def _validate_fusion(pattern: str, raw: object) -> _Fusion:
    raw = _known_fields_only(raw, _FUSION_FIELDS, "a fusion")
    refusal = _refuse("fusion", json.dumps(pattern))
    if "axis" not in raw or "parts" not in raw:
        raise _refuse("json", "a fusion declares axis and parts")
    axis = _uint(raw["axis"])
    groups = _uint(raw.get("groups", 1))
    if axis is None or groups is None or not isinstance(raw["parts"], list):
        raise _refuse("json", "a fusion declares axis and parts")
    # Only the outer axis concatenates; a single part in a single group is the
    # whole tensor, which is not a fusion.
    if axis != 0 or groups == 0 or not raw["parts"] or (groups == 1 and len(raw["parts"]) < 2):
        raise refusal
    seen: set[str] = set()
    parts: list[_FusionPart] = []
    for entry in raw["parts"]:
        entry = _known_fields_only(entry, _PART_FIELDS, "a fusion part")
        if "role" not in entry or "share" not in entry:
            raise _refuse("json", "a fusion part declares role and share")
        role, share = entry["role"], _uint(entry["share"])
        if not isinstance(role, str) or share is None:
            raise _refuse("json", "a fusion part declares role and share")
        if share == 0 or role in seen:
            raise refusal
        seen.add(role)
        # An unnamed part is only meaningful as the sole part of an
        # interleaved slice: it IS the declared role.
        if role == "" and len(seen) > 1:
            raise refusal
        parts.append(_FusionPart(role=role, share=share))
    if len(parts) > 1 and any(part.role == "" for part in parts):
        raise refusal
    return _Fusion(groups=groups, parts=tuple(parts))


def _parse_dim(value: object) -> _Dim | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return _Dim("literal", literal=value) if value > 0 else None
    if not isinstance(value, str):
        return None
    if value == "auto":
        return _Dim("auto")
    if value.isdigit() and value.isascii():
        literal = int(value)
        return _Dim("literal", literal=literal) if literal > 0 else None
    head, divided, divisor_text = value.partition("/")
    divisor = 1
    if divided:
        if not (divisor_text.isdigit() and divisor_text.isascii()) or int(divisor_text) == 0:
            return None
        divisor = int(divisor_text)
    if not head.startswith("shape[") or not head.endswith("]"):
        return None
    digits = head[len("shape[") : -1]
    if not (digits.isdigit() and digits.isascii()):
        return None
    return _Dim("axis", axis=int(digits), divisor=divisor)


def _validate_permute(pattern: str, fused: bool, raw: object) -> _Permute:
    raw = _known_fields_only(raw, _PERMUTE_FIELDS, "a permute")
    refusal = _refuse("permute", json.dumps(pattern))
    if "view" not in raw or "axes" not in raw:
        raise _refuse("json", "a permute declares view and axes")
    view_raw, axes_raw = raw["view"], raw["axes"]
    if not isinstance(view_raw, list) or not isinstance(axes_raw, list):
        raise _refuse("json", "a permute declares view and axes")
    axes: list[int] = []
    for value in axes_raw:
        axis = _uint(value)
        if axis is None:
            raise _refuse("json", "a permute axis is a non-negative integer")
        axes.append(axis)
    # A permute inside a fused tensor would mean its seam runs are NOT the
    # split packaging's bytes, which is the one thing a seam promises.
    if fused:
        raise refusal
    if len(view_raw) < 2 or len(axes) != len(view_raw):
        raise refusal
    seen = [False] * len(axes)
    identity = True
    for at, axis in enumerate(axes):
        if axis >= len(seen) or seen[axis]:
            raise refusal
        seen[axis] = True
        if at != axis:
            identity = False
    if identity:
        # The identity permute is the absence of one.
        raise refusal
    view: list[_Dim] = []
    for value in view_raw:
        dim = _parse_dim(value)
        if dim is None:
            raise refusal
        view.append(dim)
    return _Permute(view=tuple(view), axes=tuple(axes))


def _parse(raw: object) -> tuple[str | None, int | None, str | None, list[_Tensor], dict[str, list[_Pattern]]]:
    """Validate one decoded v1 document. Refusals mirror the Rust and Go
    validators reason for reason; the shared corpus proves it."""

    raw = _known_fields_only(raw, _CONTRACT_FIELDS, "a contract")
    if "format" not in raw or "tensors" not in raw:
        raise _refuse("json", "missing format or tensors")
    if raw["format"] != CONTRACT_FORMAT:
        raise _refuse("format", str(raw["format"]))

    name: str | None = None
    version: int | None = None
    has_name, has_version = "name" in raw, "version" in raw
    if has_name and has_version:
        if not _is_contract_name(raw["name"]):
            raise _refuse("name", json.dumps(raw["name"]))
        parsed = _uint(raw["version"])
        if parsed is None:
            raise _refuse("json", "version is a non-negative integer")
        if parsed == 0:
            raise _refuse("version", "must be at least 1")
        name, version = raw["name"], parsed
    elif has_name or has_version:
        raise _refuse("identity", "name and version travel together or not at all")

    dtype: str | None = None
    if "dtype" in raw:
        if not _is_dtype(raw["dtype"]):
            raise _refuse("dtype", json.dumps(raw["dtype"]))
        dtype = raw["dtype"]

    if not isinstance(raw["tensors"], list):
        raise _refuse("json", "tensors is a list")
    if not raw["tensors"]:
        raise _refuse("no-tensors", "a contract declares at least one tensor")

    patterns: set[str] = set()
    roles: set[str] = set()
    tensors: list[_Tensor] = []
    for entry in raw["tensors"]:
        entry = _known_fields_only(entry, _TENSOR_FIELDS, "a tensor")
        if "role" not in entry or "pattern" not in entry:
            raise _refuse("json", "a tensor declares role and pattern")
        pattern = _parse_pattern(entry["pattern"])
        role = _parse_pattern(entry["role"])
        if role.holes != pattern.holes:
            raise _refuse("role-holes", json.dumps(role.text))
        if pattern.text in patterns:
            raise _refuse("duplicate", json.dumps(pattern.text))
        patterns.add(pattern.text)
        if role.text in roles:
            raise _refuse("duplicate", json.dumps(role.text))
        roles.add(role.text)

        dtypes = entry.get("dtypes", [])
        if not isinstance(dtypes, list) or not all(isinstance(item, str) for item in dtypes):
            raise _refuse("json", "dtypes is a list of strings")
        rank = None
        if entry.get("rank") is not None:
            rank = _uint(entry["rank"])
            if rank is None:
                raise _refuse("json", "rank is a non-negative integer")
        required = entry.get("required", True)
        if not isinstance(required, bool):
            raise _refuse("json", "required is a boolean")

        fusion = _validate_fusion(pattern.text, entry["fusion"]) if "fusion" in entry else None
        permute = (
            _validate_permute(pattern.text, fusion is not None, entry["permute"])
            if "permute" in entry
            else None
        )
        tensors.append(
            _Tensor(
                role=role,
                pattern=pattern,
                dtypes=tuple(dtypes),
                rank=rank,
                required=required,
                fusion=fusion,
                permute=permute,
            )
        )

    sets: dict[str, list[_Pattern]] = {}
    raw_sets = raw.get("sets", {})
    if not isinstance(raw_sets, Mapping):
        raise _refuse("json", "sets is an object")
    for set_name, members in raw_sets.items():
        if not isinstance(members, list):
            raise _refuse("json", "a set is a list of patterns")
        if not members:
            raise _refuse("set", json.dumps(set_name))
        sets[set_name] = [_parse_pattern(member) for member in members]
    return name, version, dtype, tensors, sets


def _canonical(
    name: str | None,
    version: int | None,
    dtype: str | None,
    tensors: Sequence[_Tensor],
    sets: Mapping[str, Sequence[_Pattern]],
) -> str:
    """The canonical rendering, BYTE FOR BYTE what Rust and Go emit.

    OMISSION-PRESERVING: an absent field emits no line at all, so the digests
    of the pre-existing named library are byte-identical to what they were
    when ``name``/``version`` were mandatory.
    """

    out = [f"{CONTRACT_FORMAT}\n"]
    if name is not None:
        out.append(f"name={name}\nversion={version}\n")
    if dtype is not None:
        out.append(f"dtype={dtype}\n")
    for tensor in tensors:
        rank = "any" if tensor.rank is None else str(tensor.rank)
        out.append(
            f"tensor role={tensor.role.text} pattern={tensor.pattern.text} "
            f"rank={rank} required={'true' if tensor.required else 'false'} "
            f"dtypes={','.join(tensor.dtypes)}"
        )
        if tensor.permute is not None:
            out.append(" permute=")
            for dim in tensor.permute.view:
                if dim.kind == "literal":
                    out.append(str(dim.literal))
                elif dim.kind == "auto":
                    out.append("auto")
                else:
                    out.append(f"shape[{dim.axis}]/{dim.divisor}")
                out.append(",")
            # The Rust side renders the axes list with Debug formatting.
            out.append(":[" + ", ".join(str(axis) for axis in tensor.permute.axes) + "]")
        if tensor.fusion is not None:
            out.append(f" fusion=groups:{tensor.fusion.groups},")
            for part in tensor.fusion.parts:
                out.append(f"{part.role}:{part.share},")
        out.append("\n")
    for set_name in sorted(sets):
        out.append(f"set {set_name}=")
        for pattern in sets[set_name]:
            out.append(f"{pattern.text},")
        out.append("\n")
    return "".join(out)


# ── the object ───────────────────────────────────────────────────────────────


class Contract:
    """One frozen, validated layout contract.

    The predefined :mod:`tensorfs.contracts` constants are instances of this
    class; ``Contract(...)`` constructs an ANONYMOUS custom (no ``name=`` — a
    custom's identity is its digest). Equality and hashing are by digest.
    """

    _digest: str
    _document: str
    _dtype: str | None
    _name: str | None
    _raw: dict[str, Any]
    _stamp: str
    _version: int | None

    __slots__ = ("_digest", "_document", "_dtype", "_name", "_raw", "_stamp", "_version")

    def __init__(
        self,
        *,
        tensors: Sequence[TensorDecl],
        dtype: str | None = None,
        sets: Mapping[str, Sequence[str]] | None = None,
        description: str = "",
    ) -> None:
        raw: dict[str, Any] = {"format": CONTRACT_FORMAT}
        if description:
            raw["description"] = description
        if dtype is not None:
            raw["dtype"] = dtype
        raw["tensors"] = [decl._raw() for decl in tensors]
        if sets:
            raw["sets"] = {name: list(patterns) for name, patterns in sets.items()}
        self._seal(raw)

    def _seal(self, raw: dict[str, Any]) -> None:
        """Canonicalize, validate, and freeze."""

        document = json.dumps(raw, sort_keys=True, separators=(",", ":"))
        name, version, dtype, tensors, sets = _parse(raw)
        digest = hashlib.sha256(
            _canonical(name, version, dtype, tensors, sets).encode()
        ).hexdigest()
        object.__setattr__(self, "_raw", raw)
        object.__setattr__(self, "_document", document)
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_version", version)
        object.__setattr__(self, "_dtype", dtype)
        object.__setattr__(self, "_digest", digest)
        object.__setattr__(
            self, "_stamp", f"{name}@{version}" if name is not None else f"sha256:{digest}"
        )

    def __setattr__(self, attribute: str, value: object) -> None:
        raise AttributeError("a Contract is frozen")

    @classmethod
    def from_document(cls, document: str) -> Contract:
        """A contract from a raw v1 JSON document (library or custom)."""

        contract = cls.__new__(cls)
        try:
            raw = json.loads(document)
        except ValueError as exc:
            raise _refuse("json", str(exc)) from None
        contract._seal(raw)
        return contract

    @classmethod
    def from_file(cls, path: str) -> Contract:
        with open(path, encoding="utf-8") as handle:
            return cls.from_document(handle.read())

    # -- identity ---------------------------------------------------------

    @property
    def name(self) -> str | None:
        """The library name; ``None`` on customs, which have none."""

        return self._name

    @property
    def version(self) -> int | None:
        return self._version

    @property
    def digest(self) -> str:
        """Bare hex SHA-256 of the canonical rendering — identical to the
        Rust ``Contract::digest``, and a custom's entire identity."""

        return self._digest

    @property
    def stamp(self) -> str:
        """``name@version`` for library documents, ``sha256:<hex>`` for
        customs: what a snapshot records."""

        return self._stamp

    @property
    def label(self) -> str:
        """The human spelling: the name for library entries, a digest prefix
        like ``sha256:ab12cd34…`` for customs."""

        if self._name is not None:
            return self._name
        return f"sha256:{self._digest[:8]}…"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Contract):
            return NotImplemented
        return self._digest == other._digest

    def __hash__(self) -> int:
        return hash(self._digest)

    def __repr__(self) -> str:
        return f"Contract({self.label!r})"

    # -- the document -----------------------------------------------------

    @property
    def document(self) -> str:
        """The canonical JSON serialization: what travels in a release
        derive document, and what every ``_tensorfs`` entrypoint accepts."""

        return self._document

    @property
    def description(self) -> str:
        return str(self._raw.get("description", ""))

    @property
    def tensors(self) -> tuple[TensorDecl, ...]:
        return tuple(_decl_of(raw) for raw in self._raw["tensors"])

    @property
    def sets(self) -> dict[str, tuple[str, ...]]:
        return {
            name: tuple(patterns) for name, patterns in self._raw.get("sets", {}).items()
        }

    # -- the serve-side read ----------------------------------------------

    @property
    def dtype(self) -> str:
        """The declared load dtype (``ctx.lane.dtype`` reads this). Reading
        it on a contract that declares none is an author error, refused
        loudly rather than answered with a guess."""

        if self._dtype is None:
            raise MissingDtype(
                f"contract {self.label} declares no top-level dtype; "
                "declare one on the lane contract to read it"
            )
        return self._dtype

    @property
    def torch_dtype(self) -> Any:
        """``self.dtype`` resolved against torch, imported lazily."""

        import importlib

        torch = importlib.import_module("torch")
        name = self.dtype
        resolved = getattr(torch, name, None)
        if resolved is None:
            raise MissingDtype(f"torch has no dtype named {name!r}")
        return resolved
