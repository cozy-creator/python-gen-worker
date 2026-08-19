"""Transform passes: what MAKES a lane's format (tcg#52).

torchcg observes author code; it does not rewrite it. A transform pass is not
a counter-example to that -- it is the thing that produces the FORMAT a lane
declares. ``Lane`` already means "the tensor-layout contract this lane expects
checkpoints in"; an fp8 lane's weights are fp8 because something converted
them, and a precompute lane's blocks hold a side table because something
folded them. A pass gives that "something" a name, a version, an identity and
an artifact. The author's own code still runs untouched over the result.

**Ordering is PASS -> DISCOVERY -> EXPORT, and it is structural.** The graph
that gets identity must be the graph that serves, so:

* a pass runs inside a :class:`TransformSession`, which ``seal()``s ONCE; a
  ``run`` after the seal is refused;
* :func:`seal_for_export` stamps every module discovery or adoption touches,
  and :meth:`TransformSession.run` refuses a target holding a stamped
  submodule -- a pass after export is refused from the other side too;
* the sealed :class:`TransformSet` is what ``discover_modules`` accepts as
  evidence, it is a derivation input of ``cg-graph-v1`` (exactly as pins
  are -- two lanes differing only by a pass are DIFFERENT graphs), and
  ``AdoptSession`` refuses a boot whose ran-pass set does not equal the set
  the document's lane declares.

**The side table is NON-PERSISTENT BUFFERS, not a dict.** ie#615 measured the
dict version's failure mode: ``transformer.to("cuda")`` moved every parameter
and left the plain-dict cache on the host, and the first cuda denoise died
"Expected all tensors to be on the same device". The endpoint patched it with
an ``_apply`` override -- one override, in one module, that every future
side-table module would have to remember. Registering the table as buffers
kills the bug class instead: ``nn.Module._apply`` moves them because they ARE
module state, and ``torch.export`` LIFTS them as named inputs the arm binds by
name (tcg#42's static lifted inputs) rather than baking half a gigabyte of
constants into the ``.so``. ``persistent=False`` keeps them out of
``state_dict()``, so the lane contract's weight names are untouched.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .lane import LaneError, LaneRef
from .lane import require_pass_ref as _require_pass_ref

#: The side-table bucket format. A bucket states it, and a reader that does
#: not know the number refuses BY NAME rather than by a shape error 50
#: modules in.
SIDE_TABLE_FORMAT = 1

_KEY_HEX = 32

#: Marks a module as export-sealed. Discovery and adoption stamp it; a pass
#: refuses to touch a module tree holding it.
SEAL_ATTR = "_torchcg_export_sealed"

#: Where an installed pass parks its handle on the transformed root.
_HANDLE_ATTR = "_torchcg_transforms"


class TransformError(RuntimeError):
    """A transform pass cannot plan, apply, or state what it did."""


class TransformOrderError(TransformError):
    """PASS -> DISCOVERY -> EXPORT was violated."""


class OffDomain(TransformError):
    """A caller presented a domain point the side table does not hold.

    Raised rather than recomputed: the weights that would recompute it have
    been released. The domain a pass is built over must therefore be an
    API-BOUNDARY type on the consumer side (h3's ``StepPreset = Literal[20,
    30, 50]``), so an out-of-set request is a typed rejection at the wire and
    never a serve-time surprise.
    """


# -- pass identity and registration ---------------------------------------


def require_pass_ref(value: object) -> str:
    """Validate one pass spelling: ``<name>@<major>``, nothing else.

    The spelling itself is LANE vocabulary (``lane.require_pass_ref``) --
    there is one regex, because a lane binding and a pass declaring itself
    must agree by construction. This adapter only restates the refusal in the
    transform namespace.
    """

    try:
        return _require_pass_ref(value)
    except LaneError as exc:
        raise TransformError(str(exc)) from exc


_REGISTRY: dict[str, type] = {}


def register_pass(cls: type) -> type:
    """Register a pass class under its ``NAME``; usable as a decorator.

    Registration is what makes a lane able to NAME a pass: a contract that
    says ``passes=("precompute-and-free@1",)`` resolves through here, and an
    unknown name is a refusal that lists what this build knows -- never a
    silently skipped rewrite.
    """

    name = require_pass_ref(getattr(cls, "NAME", None))
    known = _REGISTRY.get(name)
    if known is not None and known is not cls:
        raise TransformError(
            f"pass {name!r} is already registered to {known.__module__}."
            f"{known.__qualname__}; bump the version rather than redefining it"
        )
    _REGISTRY[name] = cls
    return cls


def resolve_pass(name: str) -> type:
    """The class registered under ``name``, or a refusal that lists the set."""

    require_pass_ref(name)
    known = _REGISTRY.get(name)
    if known is None:
        raise TransformError(
            f"no pass named {name!r} is registered in this build; known passes: "
            f"{sorted(_REGISTRY) or ['<none>']}"
        )
    return known


def registered_passes() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


# -- domain keys: one address, both directions -----------------------------


def _plain(value: Any) -> Any:
    """Canonicalize one domain key into JSON-able parts, or refuse it.

    A key that cannot be canonicalized is refused rather than hashed through
    ``repr`` -- a bucket computed on a mint pod and a bucket read back from
    the store have to address identically, and ``repr`` of an arbitrary
    object carries process identity.
    """

    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        return repr(float(value))
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in sorted(value.items())}
    raise TransformError(
        f"a domain key must be built from ints, floats, strings, bools, None, "
        f"sequences and mappings; got {type(value).__name__}"
    )


def canonical_payload(key: Any) -> str:
    """The one canonical rendering of a domain key."""

    return json.dumps(
        _plain(key), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )


def canonical_key(key: Any) -> str:
    """The key's ADDRESS -- the same value locally and in the store.

    Deliberately one function for both directions (the ``CacheKey`` lesson
    from adaln_skip.py): a bucket computed here and a bucket downloaded from
    the catalog are interchangeable because the address is derived, not
    assigned.
    """

    payload = canonical_payload(key)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()[:_KEY_HEX]


# -- byte census -----------------------------------------------------------


def tensor_bytes(tensor: Any) -> int:
    """RESIDENT bytes of one tensor, seeing through tensor subclasses.

    A quantized tensor subclass reports the LOGICAL dtype it emulates, so
    ``numel * element_size`` prices per-row fp8 as bf16 -- the row a reader
    reads as "no fp8 here" (pgw#1400). Flattening the subclass into its real
    components is what makes a freed/cache byte count evidence instead of a
    restatement of the dtype label.
    """

    flatten = getattr(tensor, "__tensor_flatten__", None)
    if callable(flatten):
        try:
            names, _ = flatten()
        except Exception:  # pragma: no cover - subclass without the contract
            names = ()
        if names:
            return sum(tensor_bytes(getattr(tensor, name)) for name in names)
    return int(tensor.numel()) * int(tensor.element_size())


def dtype_label(tensor: Any) -> str:
    """The dtype a census should PRINT: logical, plus the real components."""

    logical = str(getattr(tensor, "dtype", "?")).removeprefix("torch.")
    flatten = getattr(tensor, "__tensor_flatten__", None)
    if callable(flatten):
        try:
            names, _ = flatten()
        except Exception:  # pragma: no cover - subclass without the contract
            names = ()
        if names:
            inner = ",".join(
                sorted({dtype_label(getattr(tensor, name)) for name in names})
            )
            return f"{logical}[{inner}]"
    return logical


def module_bytes(module: Any) -> int:
    """Bytes a module holds: parameters plus buffers, censused honestly."""

    seen: set[int] = set()
    total = 0
    for tensor in list(module.parameters()) + list(module.buffers()):
        if id(tensor) in seen:
            continue
        seen.add(id(tensor))
        total += tensor_bytes(tensor)
    return total


# -- the plan and the report ----------------------------------------------


class TransformMode(StrEnum):
    """How a pass got its side tables. Never a correctness difference."""

    COMPUTED = "computed"
    LOADED = "loaded"
    MIXED = "mixed"
    CONVERTED = "converted"


@dataclass(frozen=True, slots=True)
class Produced:
    """One side-table bucket this run minted."""

    key: str
    address: str
    bytes: int


@dataclass(frozen=True, slots=True)
class TransformPlan:
    """What a pass WILL do, stated before it does it.

    ``freed_bytes`` and ``cache_bytes`` are deliberately NOT here (the tcg#52
    sketch put both on the plan): what a pass actually releases and what its
    replacement costs are only knowable once it has run, and a planned-but-
    unmeasured byte count is exactly the kind of number that gets believed.
    What IS measurable up front is ``scope_bytes`` -- the bytes the targets
    hold RIGHT NOW -- and it means the same thing for every pass: a fold
    releases all of it, a quantization shrinks it.
    """

    pass_name: str
    #: module FQNs the pass rewrites
    rewrites: tuple[str, ...]
    #: the tensorfs REMOVABLE SET names this rewrite makes droppable
    removes: tuple[str, ...]
    #: canonical domain-key addresses, in the order the pass will walk them
    domain: tuple[str, ...]
    #: bytes the rewrite targets hold before the pass runs
    scope_bytes: int

    def __post_init__(self) -> None:
        require_pass_ref(self.pass_name)
        if not self.rewrites:
            raise TransformError(
                f"pass {self.pass_name}: a plan that rewrites nothing is not a "
                f"plan -- the module shape moved under this pass"
            )
        if len(set(self.rewrites)) != len(self.rewrites):
            raise TransformError(f"pass {self.pass_name}: plan repeats a rewrite target")
        if not self.domain:
            raise TransformError(
                f"pass {self.pass_name}: an empty domain refuses every request"
            )
        if len(set(self.domain)) != len(self.domain):
            raise TransformError(f"pass {self.pass_name}: plan repeats a domain key")
        if not self.removes:
            raise TransformError(
                f"pass {self.pass_name}: a plan must name the removable set it "
                f"frees, so the checkpoint side can stop downloading those bytes"
            )
        if self.scope_bytes < 0:
            raise TransformError(f"pass {self.pass_name}: scope_bytes is negative")

    def as_dict(self) -> dict[str, Any]:
        return {
            "pass": self.pass_name,
            "rewrites": list(self.rewrites),
            "removes": list(self.removes),
            "domain": list(self.domain),
            "scope_bytes": self.scope_bytes,
        }


@dataclass(frozen=True, slots=True)
class TransformReport:
    """What a pass ACTUALLY did. Every number is measured, not assumed.

    A typed return, which is also how a pass obeys the charter's rule 2:
    libraries return structured facts, only gen-worker emits.
    """

    pass_name: str
    plan: TransformPlan
    mode: TransformMode
    freed_bytes: int
    cache_bytes: int
    produced: tuple[Produced, ...] = ()
    loaded: tuple[str, ...] = ()
    #: Inherited honesty (te#171 Phase B, adaln_skip.py's ``validated``): no
    #: pass in this library has passed a numerical-equivalence gate on real
    #: weights yet, and a report that quietly dropped the field on its way
    #: into a library would be the library laundering an unvalidated
    #: optimisation. A pass sets it True only when a gate actually ran.
    validated: bool = False

    def __post_init__(self) -> None:
        require_pass_ref(self.pass_name)
        if self.freed_bytes < 0 or self.cache_bytes < 0:
            raise TransformError(f"pass {self.pass_name}: byte counts are negative")

    @property
    def net_bytes(self) -> int:
        return self.freed_bytes - self.cache_bytes

    def as_dict(self) -> dict[str, Any]:
        return {
            "pass": self.pass_name,
            "plan": self.plan.as_dict(),
            "mode": str(self.mode),
            "freed_bytes": self.freed_bytes,
            "cache_bytes": self.cache_bytes,
            "net_bytes": self.net_bytes,
            "produced": [
                {"key": row.key, "address": row.address, "bytes": row.bytes}
                for row in self.produced
            ],
            "loaded": list(self.loaded),
            "validated": self.validated,
        }


# -- the interfaces --------------------------------------------------------


@runtime_checkable
class TransformPass(Protocol):
    """The mechanism. Two instances live in this tree: tcg#52 and tcg#53."""

    #: ``<name>@<major>`` -- versioned, because it is a graph-identity input.
    NAME: str

    @property
    def name(self) -> str: ...

    def plan(self, target: Any, *, lane: LaneRef) -> TransformPlan: ...

    def apply(
        self,
        target: Any,
        plan: TransformPlan,
        *,
        artifacts: SideTableStore | None = None,
    ) -> TransformReport: ...


@runtime_checkable
class SideTableStore(Protocol):
    """Where a pass's minted side tables live.

    The same seam as ``GraphStore``: torchcg holds no opinion on WHERE bytes
    go -- bytes-at-rest is tensorfs's charter and the hub is gen-worker's, so
    the caller owns the sink. ``LocalGraphStore`` implements this.
    """

    def has_side_table(self, pass_name: str, key: str) -> bool: ...

    def fetch_side_table(
        self, pass_name: str, key: str, destination: str | Path
    ) -> Path | None: ...

    def publish_side_table(self, pass_name: str, key: str, path: str | Path) -> str: ...


# -- ordering --------------------------------------------------------------


def seal_for_export(module: Any) -> None:
    """Stamp a module tree as export-sealed. Discovery and adoption call this."""

    try:
        object.__setattr__(module, SEAL_ATTR, True)
    except AttributeError:  # pragma: no cover - exotic __slots__ module
        pass


def is_sealed(module: Any) -> bool:
    return bool(getattr(module, SEAL_ATTR, False))


def _first_sealed(root: Any) -> str | None:
    for fqn, module in root.named_modules():
        if is_sealed(module):
            return fqn or "<root>"
    return None


def _contains(outer: Any, inner: Any) -> bool:
    return any(module is inner for module in outer.modules())


def related(a: Any, b: Any) -> bool:
    """Whether two modules are on one tree -- either contains the other."""

    return a is b or _contains(a, b) or _contains(b, a)


@dataclass(frozen=True, slots=True)
class TransformSet:
    """The sealed record of a lane's passes: discovery's evidence.

    ``passes`` is what enters ``cg-graph-v1`` and what the release document
    carries; ``roots`` are the transformed module objects, kept only so
    discovery can prove the graph it is about to identify is the graph the
    pass actually rewrote.
    """

    contract: str
    passes: tuple[str, ...]
    reports: tuple[TransformReport, ...] = ()
    roots: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        for name in self.passes:
            require_pass_ref(name)
        if len(set(self.passes)) != len(self.passes):
            raise TransformError(f"lane {self.contract!r} names a pass twice")


class TransformSession:
    """One lane's PASS phase. Runs passes, then seals, once.

    The session is an ordinary object the HOST constructs and threads in --
    never ambient, exactly like ``AdoptSession`` (tcg#42). The derive host
    runs it before ``discover_modules``; the serve host runs it before
    ``AdoptSession``, and hands the same sealed set to both.
    """

    def __init__(self, lane: LaneRef | str, *, store: SideTableStore | None = None) -> None:
        self.lane = lane if isinstance(lane, LaneRef) else LaneRef(lane)
        self._store = store
        self._reports: list[TransformReport] = []
        self._roots: list[Any] = []
        self._sealed = False

    @property
    def contract(self) -> str:
        return self.lane.contract

    @property
    def sealed(self) -> bool:
        return self._sealed

    @property
    def reports(self) -> tuple[TransformReport, ...]:
        return tuple(self._reports)

    def run(self, target: Any, transform: TransformPass) -> TransformReport:
        """Plan and apply one pass. Refuses anything already export-bound."""

        if self._sealed:
            raise TransformOrderError(
                f"lane {self.contract!r}: this transform session is sealed; a pass "
                f"cannot run after DISCOVERY -- the graph that gets identity must "
                f"be the graph that serves (PASS -> DISCOVERY -> EXPORT)"
            )
        sealed_at = _first_sealed(target)
        if sealed_at is not None:
            raise TransformOrderError(
                f"lane {self.contract!r}: pass {transform.name!r} was handed a "
                f"module whose submodule {sealed_at!r} is already export-sealed "
                f"(discovered or adopted). A pass AFTER export would mint an "
                f"artifact for a module that no longer exists"
            )
        name = require_pass_ref(transform.name)
        if any(report.pass_name == name for report in self._reports):
            raise TransformError(
                f"lane {self.contract!r}: pass {name!r} already ran in this session; "
                f"a lane names a pass once"
            )
        plan = transform.plan(target, lane=self.lane)
        if plan.pass_name != name:
            raise TransformError(
                f"pass {name!r} produced a plan naming {plan.pass_name!r}"
            )
        report = transform.apply(target, plan, artifacts=self._store)
        if report.pass_name != name:
            raise TransformError(
                f"pass {name!r} produced a report naming {report.pass_name!r}"
            )
        self._reports.append(report)
        self._roots.append(target)
        return report

    def seal(self) -> TransformSet:
        """Freeze the phase. Idempotent; the returned set is discovery's input."""

        self._sealed = True
        return TransformSet(
            contract=self.contract,
            passes=tuple(report.pass_name for report in self._reports),
            reports=tuple(self._reports),
            roots=tuple(self._roots),
        )


def require_transform_set(
    contract: str, transforms: TransformSet | None, declared: Sequence[str] = ()
) -> tuple[str, ...]:
    """Reconcile a lane's DECLARED passes with the set that actually ran.

    Both directions are refusals: a lane that declares passes and was handed
    no sealed set never ran them, and a session that ran a pass the lane does
    not declare produced a graph nobody can address.
    """

    ran = tuple(transforms.passes) if transforms is not None else ()
    want = tuple(declared)
    if transforms is not None and transforms.contract != contract:
        raise TransformOrderError(
            f"transform set was sealed for lane {transforms.contract!r} but "
            f"discovery is running lane {contract!r}"
        )
    if set(ran) != set(want):
        raise TransformOrderError(
            f"lane {contract!r} declares passes {sorted(want) or ['<none>']} but "
            f"{sorted(ran) or ['<none>']} ran; a pass is a derivation input of "
            f"cg-graph-v1, so the two sets must be equal"
        )
    return ran


# -- the replacement module ------------------------------------------------

_SIDE_TABLE_CLASS: Any = None


def side_table_class() -> Any:
    """The side-table ``nn.Module`` class, built on first use.

    Built lazily because ``import torchcg`` must not drag torch in: torch is
    an optional extra and the serve-role import closure is fenced (pgw#1331).
    """

    global _SIDE_TABLE_CLASS
    if _SIDE_TABLE_CLASS is not None:
        return _SIDE_TABLE_CLASS

    from torch import nn

    class SideTable(nn.Module):
        """One tensor lookup, no matmul, no weights.

        Holds the folded submodule's OUTPUT for every domain point the table
        was built over. The forward IGNORES its inputs -- being a pure
        function of the bound domain point is the whole reason this is
        foldable -- but the binding is explicit, so a caller that quietly
        changed the schedule gets a refusal and never a wrong answer.
        """

        def __init__(
            self,
            *,
            name: str,
            pass_name: str,
            table: Mapping[str, Sequence[Any]],
            unwrap: bool,
        ) -> None:
            super().__init__()
            self._st_name = name
            self._st_pass = pass_name
            self._st_unwrap = unwrap
            self._st_key: str | None = None
            index: dict[str, tuple[str, ...]] = {}
            for position, (key, chunks) in enumerate(table.items()):
                names = []
                for chunk_index, chunk in enumerate(chunks):
                    buffer_name = f"_st_{position}_{chunk_index}"
                    # persistent=False: the table is DERIVED state, so it must
                    # not enter state_dict() and must not perturb the lane
                    # contract's weight names. It still moves with .to()/.cuda()
                    # and still lifts by name under torch.export.
                    self.register_buffer(buffer_name, chunk, persistent=False)
                    names.append(buffer_name)
                index[key] = tuple(names)
            self._st_index = index

        @property
        def domain(self) -> tuple[str, ...]:
            return tuple(self._st_index)

        @property
        def bound(self) -> str | None:
            return self._st_key

        def bind(self, key: str) -> None:
            if key not in self._st_index:
                raise OffDomain(
                    f"{self._st_name}: pass {self._st_pass} holds "
                    f"{len(self._st_index)} domain point(s) and none of them is "
                    f"{key!r}. Serve a declared domain point, or rebuild the side "
                    f"table -- the folded weights are RELEASED and cannot be "
                    f"recomputed at request time"
                )
            self._st_key = key

        def forward(self, *args: Any, **kwargs: Any) -> Any:
            if self._st_key is None:
                raise OffDomain(
                    f"{self._st_name}: pass {self._st_pass} has no domain point "
                    f"bound for this forward pass"
                )
            chunks = tuple(getattr(self, name) for name in self._st_index[self._st_key])
            return chunks[0] if self._st_unwrap else chunks

        def extra_repr(self) -> str:
            return (
                f"pass={self._st_pass}, domain={len(self._st_index)}, "
                f"bound={self._st_key!r}"
            )

    _SIDE_TABLE_CLASS = SideTable
    return SideTable


# -- the author's handle ---------------------------------------------------


class Precomputed:
    """What ``ctx.precompute`` hands back: bind one domain point per request."""

    __slots__ = ("root", "pass_name", "report", "_bindings", "_key")

    def __init__(
        self,
        root: Any,
        pass_name: str,
        report: TransformReport,
        bindings: Sequence[Any],
        key: Callable[[Any], Any],
    ) -> None:
        self.root = root
        self.pass_name = pass_name
        self.report = report
        self._bindings = tuple(bindings)
        self._key = key

    @property
    def domain(self) -> tuple[str, ...]:
        return self._bindings[0].domain if self._bindings else ()

    @property
    def bound(self) -> str | None:
        return self._bindings[0].bound if self._bindings else None

    def bind(self, point: Any) -> str:
        """Point every folded submodule at the domain point this call runs under.

        Off-domain is :class:`OffDomain`, raised before any compute happens.
        """

        key = canonical_key(self._key(point))
        for binding in self._bindings:
            binding.bind(key)
        return key

    def __repr__(self) -> str:  # pragma: no cover - diagnostics
        return (
            f"Precomputed({self.pass_name}, modules={len(self._bindings)}, "
            f"domain={len(self.domain)}, bound={self.bound!r})"
        )


def installed_passes(root: Any) -> tuple[str, ...]:
    return tuple(sorted(getattr(root, _HANDLE_ATTR, {})))


def handle(root: Any, pass_name: str) -> Precomputed:
    """The handle a pass installed on ``root``, or a refusal naming what is there."""

    handles: Mapping[str, Precomputed] = getattr(root, _HANDLE_ATTR, {})
    found = handles.get(pass_name)
    if found is None:
        raise TransformError(
            f"pass {pass_name!r} is not installed on this "
            f"{type(root).__name__}; installed: {installed_passes(root) or ('<none>',)}"
        )
    return found


def _install_handle(root: Any, installed: Precomputed) -> None:
    handles = dict(getattr(root, _HANDLE_ATTR, {}))
    handles[installed.pass_name] = installed
    object.__setattr__(root, _HANDLE_ATTR, handles)


# -- module selection ------------------------------------------------------


def select_modules(root: Any, pattern: str) -> tuple[tuple[str, Any], ...]:
    """Every submodule whose FQN matches ``pattern`` (an fnmatch glob).

    ``transformer_blocks.*.adaln_proj`` is the shape this exists for. Order
    is ``named_modules`` order -- definition order, deterministic.
    """

    if not isinstance(pattern, str) or not pattern.strip():
        raise TransformError("a selector must be a non-empty FQN glob")
    return tuple(
        (fqn, module)
        for fqn, module in root.named_modules()
        if fqn and fnmatch.fnmatchcase(fqn, pattern)
    )


def set_submodule(root: Any, fqn: str, replacement: Any) -> None:
    """Replace one submodule by FQN, releasing whatever was there."""

    parent_path, _, attribute = fqn.rpartition(".")
    parent = root.get_submodule(parent_path) if parent_path else root
    if attribute.isdigit() and hasattr(parent, "__setitem__"):
        parent[int(attribute)] = replacement
        return
    setattr(parent, attribute, replacement)


# -- the first instance: precompute-and-free -------------------------------


@dataclass(frozen=True, slots=True)
class CallInputs:
    """Multi-argument call spelling for :class:`PrecomputeAndFree`."""

    args: tuple[Any, ...] = ()
    kwargs: Mapping[str, Any] = field(default_factory=dict)


def _normalize_output(pass_name: str, fqn: str, output: Any) -> tuple[Any, ...]:
    import torch

    if isinstance(output, torch.Tensor):
        return (output.detach().clone().contiguous(),)
    if isinstance(output, (list, tuple)) and output:
        if all(isinstance(item, torch.Tensor) for item in output):
            return tuple(item.detach().clone().contiguous() for item in output)
    raise TransformError(
        f"pass {pass_name}: {fqn} returned {type(output).__name__}; a foldable "
        f"submodule must return a tensor or a non-empty tuple of tensors"
    )


@register_pass
class PrecomputeAndFree:
    """Evaluate a submodule over a declared domain, then release its weights.

    The generic form of minimax-h3's ``adaln_skip.install`` -- which traded
    the DiT's 26.02 GB ``adaln_proj`` stack for a 0.08-0.48 GB table, 39% of
    a 66 GB checkpoint, because that projection is a pure function of the
    denoising schedule and the schedule set an endpoint serves is small and
    DECLARED. What the endpoint keeps is its domain: what a point IS, how to
    call the submodule with it, and any row remapping its packing needs.

    ``select``  FQN glob naming the submodules to fold.
    ``domain``  the points the served lane may present. Off-domain is refused.
    ``key``     point -> a canonical hashable; addresses the table AND the bucket.
    ``call``    ``(root, submodule, point)`` -> the submodule's input(s).
    ``removes`` the tensorfs removable-set name this rewrite frees.
    """

    NAME = "precompute-and-free@1"

    def __init__(
        self,
        *,
        select: str,
        domain: Sequence[Any],
        key: Callable[[Any], Any],
        call: Callable[[Any, Any, Any], Any],
        removes: str,
    ) -> None:
        self.select = select
        self.domain = tuple(domain)
        self.key = key
        self.call = call
        self.removes = removes

    @property
    def name(self) -> str:
        return self.NAME

    # -- plan ---------------------------------------------------------------

    def plan(self, target: Any, *, lane: LaneRef) -> TransformPlan:
        selected = select_modules(target, self.select)
        if not selected:
            raise TransformError(
                f"pass {self.NAME}: selector {self.select!r} matched no submodule "
                f"of {type(target).__name__}. An empty plan means the module shape "
                f"moved under this pass -- refusing rather than freeing nothing"
            )
        if not self.domain:
            raise TransformError(
                f"pass {self.NAME}: an empty domain refuses every request; state "
                f"the points lane {lane.contract!r} serves"
            )
        keys: list[str] = []
        for point in self.domain:
            address = canonical_key(self.key(point))
            if address in keys:
                raise TransformError(
                    f"pass {self.NAME}: two domain points address the same bucket "
                    f"({address}); the key function does not separate them"
                )
            keys.append(address)
        return TransformPlan(
            pass_name=self.NAME,
            rewrites=tuple(fqn for fqn, _ in selected),
            removes=(self.removes,),
            domain=tuple(keys),
            scope_bytes=sum(module_bytes(module) for _, module in selected),
        )

    # -- apply --------------------------------------------------------------

    def apply(
        self,
        target: Any,
        plan: TransformPlan,
        *,
        artifacts: SideTableStore | None = None,
    ) -> TransformReport:
        """Fold, free, and state the result -- ``ensure`` semantics throughout.

        Every bucket the sink already holds is READ (the submodule is never
        evaluated for it); the rest are computed and, when there is a sink,
        published. All-present is the artifact path, partial self-mints the
        remainder. The SAME key addresses both, so the fallback is a
        performance difference and never a correctness one.
        """

        import torch

        selected = select_modules(target, self.select)
        found = tuple(fqn for fqn, _ in selected)
        if found != plan.rewrites:
            raise TransformError(
                f"pass {self.NAME}: the plan named {len(plan.rewrites)} module(s) "
                f"but {len(found)} match now; the module tree moved between plan "
                f"and apply"
            )
        points = {canonical_key(self.key(point)): point for point in self.domain}

        tables: list[dict[str, tuple[Any, ...]]] = [{} for _ in selected]
        loaded: list[str] = []
        produced: list[Produced] = []
        unwrap: bool | None = None

        with tempfile.TemporaryDirectory(prefix="torchcg-sidetable-") as scratch:
            work = Path(scratch)
            for address in plan.domain:
                bucket = None
                if artifacts is not None:
                    bucket = artifacts.fetch_side_table(
                        self.NAME, address, work / f"{address}.safetensors"
                    )
                if bucket is not None:
                    chunks_by_module, bucket_unwrap = _read_bucket(
                        bucket, self.NAME, address, plan.rewrites
                    )
                    unwrap = _agree_unwrap(self.NAME, unwrap, bucket_unwrap)
                    for index, chunks in enumerate(chunks_by_module):
                        tables[index][address] = chunks
                    loaded.append(address)
                    continue
                point = points[address]
                computed: list[tuple[Any, ...]] = []
                with torch.no_grad():
                    for fqn, module in selected:
                        output = _invoke(self.call(target, module, point), module)
                        chunks = _normalize_output(self.NAME, fqn, output)
                        unwrap = _agree_unwrap(
                            self.NAME, unwrap, not isinstance(output, (list, tuple))
                        )
                        computed.append(chunks)
                for index, chunks in enumerate(computed):
                    tables[index][address] = chunks
                if artifacts is not None:
                    path = work / f"{address}.safetensors"
                    size = _write_bucket(
                        path,
                        pass_name=self.NAME,
                        address=address,
                        payload=canonical_payload(self.key(point)),
                        modules=plan.rewrites,
                        chunks_by_module=computed,
                        unwrap=bool(unwrap),
                    )
                    stored = artifacts.publish_side_table(self.NAME, address, path)
                    produced.append(Produced(key=address, address=stored, bytes=size))

        # The install runs in its own frame ON PURPOSE: a loop variable holding
        # the last folded submodule would keep half a gigabyte alive until this
        # function returned, which is precisely the memory the pass exists to
        # release. `selected` is dropped here for the same reason.
        del selected
        freed, cache_bytes, bindings = self._install(
            target, plan, tables, unwrap=bool(unwrap)
        )
        for binding in bindings:
            binding.bind(plan.domain[0])

        if loaded and len(loaded) == len(plan.domain):
            mode = TransformMode.LOADED
        elif loaded:
            mode = TransformMode.MIXED
        else:
            mode = TransformMode.COMPUTED
        report = TransformReport(
            pass_name=self.NAME,
            plan=plan,
            mode=mode,
            freed_bytes=freed,
            cache_bytes=cache_bytes,
            produced=tuple(produced),
            loaded=tuple(loaded),
        )
        _install_handle(
            target, Precomputed(target, self.NAME, report, bindings, self.key)
        )
        return report

    def _install(
        self,
        target: Any,
        plan: TransformPlan,
        tables: Sequence[Mapping[str, tuple[Any, ...]]],
        *,
        unwrap: bool,
    ) -> tuple[int, int, list[Any]]:
        """Swap each folded submodule for its side table; measure both sides."""

        table_class = side_table_class()
        bindings: list[Any] = []
        freed = 0
        cache_bytes = 0
        for index, fqn in enumerate(plan.rewrites):
            module = target.get_submodule(fqn)
            freed += module_bytes(module)
            device = next(
                (parameter.device for parameter in module.parameters()),
                next((tensor.device for tensor in module.buffers()), None),
            )
            table = {
                address: tuple(
                    chunk if device is None else chunk.to(device=device)
                    for chunk in tables[index][address]
                )
                for address in plan.domain
            }
            del module
            replacement = table_class(
                name=fqn, pass_name=self.NAME, table=table, unwrap=unwrap
            )
            cache_bytes += module_bytes(replacement)
            set_submodule(target, fqn, replacement)
            bindings.append(replacement)
        return freed, cache_bytes, bindings


def _invoke(inputs: Any, module: Any) -> Any:
    if isinstance(inputs, CallInputs):
        return module(*inputs.args, **dict(inputs.kwargs))
    return module(inputs)


def _agree_unwrap(pass_name: str, known: bool | None, seen: bool) -> bool:
    if known is not None and known != seen:
        raise TransformError(
            f"pass {pass_name}: the folded submodules disagree about their output "
            f"shape (tensor vs tuple); one side table cannot answer both"
        )
    return seen


# -- the bucket format -----------------------------------------------------
#
# One safetensors file per domain point:
#   m.<i>.chunk.<j>  -> the j-th output of the i-th rewritten module
# plus metadata carrying the format version, the pass name, the key payload
# AND its address, the module FQN list and the per-module chunk counts -- so a
# bucket minted for another checkpoint, another pass or another domain point
# fails BY NAME instead of by a shape error 50 modules in.


def _write_bucket(
    path: Path,
    *,
    pass_name: str,
    address: str,
    payload: str,
    modules: Sequence[str],
    chunks_by_module: Sequence[Sequence[Any]],
    unwrap: bool,
) -> int:
    from safetensors.torch import save_file

    tensors: dict[str, Any] = {}
    for index, chunks in enumerate(chunks_by_module):
        for chunk_index, chunk in enumerate(chunks):
            tensors[f"m.{index}.chunk.{chunk_index}"] = chunk.detach().cpu().contiguous()
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors,
        str(path),
        metadata={
            "format_version": str(SIDE_TABLE_FORMAT),
            "pass": pass_name,
            "key": payload,
            "key_ref": address,
            "modules": json.dumps(list(modules), separators=(",", ":")),
            "chunks": json.dumps(
                [len(chunks) for chunks in chunks_by_module], separators=(",", ":")
            ),
            "unwrap": "1" if unwrap else "0",
        },
    )
    return path.stat().st_size


def _read_bucket(
    path: Path, pass_name: str, address: str, modules: Sequence[str]
) -> tuple[list[tuple[Any, ...]], bool]:
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as reader:
        meta = reader.metadata() or {}
        version = meta.get("format_version")
        if version != str(SIDE_TABLE_FORMAT):
            raise TransformError(
                f"side table {address}: bucket format {version!r}, this build "
                f"reads {SIDE_TABLE_FORMAT}"
            )
        if meta.get("pass") != pass_name:
            raise TransformError(
                f"side table {address}: bucket was minted by pass "
                f"{meta.get('pass')!r}, this pass is {pass_name!r}"
            )
        payload = meta.get("key")
        if payload is None or canonical_key(json.loads(payload)) != address:
            raise TransformError(
                f"side table {address}: the bucket's own key does not address it; "
                f"the bucket and its position disagree"
            )
        stored_modules = json.loads(meta.get("modules", "[]"))
        if list(stored_modules) != list(modules):
            raise TransformError(
                f"side table {address}: bucket was minted for {len(stored_modules)} "
                f"module(s) starting {stored_modules[:1]}, this plan rewrites "
                f"{len(modules)} starting {list(modules)[:1]} -- this artifact is "
                f"not this checkpoint's"
            )
        counts = json.loads(meta.get("chunks", "[]"))
        if len(counts) != len(modules):
            raise TransformError(
                f"side table {address}: bucket states chunk counts for "
                f"{len(counts)} module(s), not {len(modules)}"
            )
        chunks_by_module = [
            tuple(
                reader.get_tensor(f"m.{index}.chunk.{chunk_index}")
                for chunk_index in range(count)
            )
            for index, count in enumerate(counts)
        ]
        return chunks_by_module, meta.get("unwrap") == "1"


__all__ = [
    "SEAL_ATTR",
    "SIDE_TABLE_FORMAT",
    "CallInputs",
    "OffDomain",
    "Precomputed",
    "PrecomputeAndFree",
    "Produced",
    "SideTableStore",
    "TransformError",
    "TransformMode",
    "TransformOrderError",
    "TransformPass",
    "TransformPlan",
    "TransformReport",
    "TransformSession",
    "TransformSet",
    "canonical_key",
    "canonical_payload",
    "dtype_label",
    "handle",
    "installed_passes",
    "is_sealed",
    "module_bytes",
    "register_pass",
    "registered_passes",
    "related",
    "require_pass_ref",
    "require_transform_set",
    "resolve_pass",
    "seal_for_export",
    "select_modules",
    "set_submodule",
    "side_table_class",
    "tensor_bytes",
]
