"""Split AOTI's generated ``run_impl`` across K translation units.

Two independent compiler profiles of the real banked SDXL w8a8 wrapper
agree: parse is 3-5% of the compile and
``AOTInductorModel::run_impl`` ALONE is 68% of it (clang attributes 128.84 s
of a 189.6 s compile to one ``OptFunction``). The cost is superlinear in a
SINGLE function's size — measured n^1.57 — because both compilers burn in
the same two per-function algorithms: stack-slot conflict colouring and
CFG-wide block placement. ``run_impl`` carries 10,032 declarations and 4,231
dispatch calls in one body. No flag fixes that (the best is -19%, most are
negative) and no bigger machine fixes it (a 32-core i9 does the TU in
140.0 s against a 4-vCPU pod's 180.6 s — it is one process on one core
either way). Reducing per-function size is the only lever, and it wins even
at zero parallelism: K=8 cuts total CPU 140.0 s -> 32.5 s.

THE SHAPE: a continuation chain. ``run_impl`` keeps its prologue verbatim
and calls ``part_0``; ``part_k`` ends by calling ``part_{k+1}`` with the
live set by reference. Chosen over "K calls from run_impl" because it is
strictly more faithful: no declaration is rewritten (a flat split would have
to hoist every crossing variable into ``run_impl`` and turn its definition
into an assignment), and every local's lifetime and destruction order is
unchanged, because each part's frame stays live for the whole chain.

Each part is a member function of a generated ``_tcg_run_ctx`` that
carries ``constants_``, ``kernels_``, ``device_idx_`` and ``cubin_dir_``
under their original names. That is what buys ZERO body rewriting: the
emitted statements' ``this->device_idx_``, ``this->cubin_dir_``,
``constants_->at(n)`` and the verbatim ``kernels`` binding all resolve
against the ctx unmodified.

FAIL-CLOSED BY CONSTRUCTION. :func:`reconstruct` is a self-contained
mechanical inverse: reading ONLY the split output, it drops
every generated line, restores every displaced one, splices the part bodies
back in order, and must reproduce the input byte for byte. It consumes no
side table from the transform, so it cannot be fooled by the transform's own
bookkeeping. Anything unrecognised, unbalanced, or untypable declines, and a
decline compiles inductor's own source unmodified. A slow mint beats a wrong
artifact.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

#: Every line this module generates carries this suffix, and every line it
#: displaces is preserved behind :data:`WAS`. Together they are what makes
#: :func:`reconstruct` self-contained.
MARK = "  //@tcg"
#: Re-derivations carry their own marker because the inverse must VERIFY
#: them rather than drop them: they are copies of a declaration that the
#: reconstruction still contains, so an altered copy has no original and is
#: caught. Dropping them blindly lets a mutated ``constants_->at(n)`` binding
#: through the gate.
MARK_RD = "  //@tcg:rederived"
WAS = "//@tcg:was:"
PARTS = "//@tcg:parts"
CHUNK_BEGIN = "//@tcg:chunk:begin:"
CHUNK_END = "//@tcg:chunk:end:"

PREFIX = "_tcg_"
CTX = f"{PREFIX}run_ctx"

#: Default K. Swept on the real TU: K=8 is the knee — K=16 buys
#: 0.5 s of wall and costs ~4 s more CPU (8 extra TUs x the 0.8 s per-TU
#: header floor), and on a 4-vCPU pod parts past K~8 are pure overhead.
DEFAULT_PARTS = 8

#: Below this the function is not the pathological shape this exists for.
MIN_COMPUTE_LINES = 4000


class Declined(Exception):
    """The source is not a shape this transform will touch."""


# ---------------------------------------------------------------------------
# Statement recognition
# ---------------------------------------------------------------------------

#: Declaration forms, in priority order. ``n`` names the declared symbol.
#: ``kind`` decides how the name crosses a part boundary: ``const_bind`` and
#: ``array`` are RE-DERIVED (a verbatim copy of the original declaration —
#: one is a reference into a container we already pass, the other a const
#: array over parameters), everything else is THREADED by reference.
_DECLS: Sequence[tuple[str, re.Pattern[str]]] = (
    # Both bind a reference derived purely from a ctx member, so both are
    # re-derived verbatim in each part that uses them rather than threaded.
    (
        "const_bind",
        re.compile(r"^\[\[maybe_unused\]\]\s*auto&\s+(?P<n>\w+)\s*=\s*constants_->at\(\d+\);$"),
    ),
    (
        "kernels_bind",
        re.compile(
            r"^\[\[maybe_unused\]\]\s*auto&\s+(?P<n>\w+)\s*=\s*"
            r"static_cast<AOTInductorModelKernels&>\(\*this->kernels_\.get\(\)\);$"
        ),
    ),
    (
        "array",
        re.compile(r"^(?:static\s+)?(?:constexpr\s+|const\s+)+int64_t\s+(?P<n>\w+)\[\]\s*=.*;$"),
    ),
    ("handle", re.compile(r"^AtenTensorHandle\s+(?P<n>\w+);$")),
    ("raii", re.compile(r"^RAIIAtenTensorHandle\s+(?P<n>\w+)\(\w+\);$")),
    ("steal", re.compile(r"^auto\s+(?P<n>\w+)\s*=\s*steal_from_raw_handles_to_raii_handles\(.*;$")),
    (
        "move",
        re.compile(
            r"^auto\s+(?P<n>\w+)\s*=\s*std::move\((?P<src>\w+)(?P<sub>\[\d+\])?\);(?:\s*//.*)?$"
        ),
    ),
    (
        "scalar",
        re.compile(
            r"^(?:\[\[maybe_unused\]\]\s*)?(?P<t>int64_t|int32_t|uint32_t|size_t|double|float|bool)"
            r"\s+(?P<n>\w+)\s*=.*;$"
        ),
    ),
    # Only the `AtenTensorHandle` overload of wrap_with_raii_handle_if_needed
    # returns RAIIAtenTensorHandle; the ArrayRefTensor ones return references
    # and the ConstantHandle one is `= delete`. reinterpret_tensor_wrapper
    # returns AtenTensorHandle, so requiring it pins the overload.
    (
        "wrap",
        re.compile(
            r"^auto\s+(?P<n>\w+)\s*=\s*wrap_with_raii_handle_if_needed\("
            r"reinterpret_tensor_wrapper\(.*$"
        ),
    ),
    ("auto_int", re.compile(r"^auto\s+(?P<n>\w+)\s*=\s*(?P<rhs>[^;]*);$")),
)

#: C++ type used for a threaded name of each kind. ``scalar`` reads its own;
#: ``move`` follows its source.
_KIND_TYPE = {
    "handle": "AtenTensorHandle",
    "raii": "RAIIAtenTensorHandle",
    "wrap": "RAIIAtenTensorHandle",
    "steal": "std::vector<RAIIAtenTensorHandle>",
}

#: Kinds re-emitted verbatim in every part that uses them, instead of
#: threaded: a reference into a container the ctx already carries, or a
#: const array over parameters. Both are pure re-derivations.
_REDERIVED = frozenset({"const_bind", "kernels_bind", "array"})

#: A ``auto x = <expr>;`` whose RHS is pure integer arithmetic over int64_t
#: names is an int64_t. Anything else is untypable and declines.
_INT_RHS = re.compile(r"^[\s\d\w()*+/%,<>:.\-]*$")
_INT_CALLS = ("c10::div_floor_integer", "static_cast<int64_t>")

#: Control flow this transform refuses to cut across. The generated compute
#: region has none of it — that is verified per-TU, never assumed.
_CONTROL = re.compile(r"^(for|while|do|switch|goto|try|catch|return|case|default)\b")

#: The declaration table, by kind, for the resolver.
_RX = {k: rx for k, rx in _DECLS}

_IDENT = re.compile(r"\b[A-Za-z_]\w*\b")
_GUARD = re.compile(r"^\s*\w*(Guard|RecordFunctionHandle)\s+\w+[({]")


def _balanced(line: str) -> bool:
    code = line
    return (
        code.count("(") == code.count(")")
        and code.count("{") == code.count("}")
        and code.count("[") == code.count("]")
        and code.count('"') % 2 == 0
        and 'R"' not in code
    )


def _strip_comment(line: str) -> str:
    i = line.find("//")
    return line if i < 0 else line[:i]


# ---------------------------------------------------------------------------
# Structural anchors
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Anchors:
    """Half-open [begin, end) line ranges of the regions a part TU needs.

    Every one is found by an anchor inductor's codegen emits verbatim; none
    is a line number. A missing anchor declines.
    """

    include: tuple[int, int]
    driver: tuple[int, int]
    kernels_class: tuple[int, int]
    kernels_open: int  # the `namespace {` wrapping the class
    kernels_close: int  # its `}  // namespace`
    templates: tuple[int, int]
    env_fn: tuple[int, int]
    run_impl: tuple[int, int]  # signature line .. the `} // ...run_impl` line


def _find(lines: Sequence[str], pred: Callable[[str], bool], start: int = 0, what: str = "") -> int:
    for i in range(start, len(lines)):
        if pred(lines[i]):
            return i
    raise Declined(f"anchor not found: {what}")


def find_anchors(lines: Sequence[str]) -> Anchors:
    inc = _find(
        lines,
        lambda s: s.startswith("#include <torch/csrc/inductor/aoti_include/"),
        what="aoti_include",
    )
    drv = _find(
        lines,
        lambda s: s.startswith("#define CUDA_DRIVER_CHECK"),
        inc,
        what="CUDA_DRIVER_CHECK (this transform targets the CUDA wrapper)",
    )
    ns1 = _find(lines, lambda s: s == "namespace torch::aot_inductor {", drv, what="namespace #1")
    kopen = _find(lines, lambda s: s == "namespace {", ns1, what="anonymous namespace")
    kcls = _find(
        lines,
        lambda s: s.startswith("class AOTInductorModelKernels"),
        kopen,
        what="AOTInductorModelKernels",
    )
    kend = _find(lines, lambda s: s == "};", kcls, what="end of AOTInductorModelKernels")
    kclose = _find(
        lines,
        lambda s: s.startswith("}") and "namespace" in s,
        kend,
        what="close of anonymous namespace",
    )
    using = _find(
        lines, lambda s: s == "using namespace torch::aot_inductor;", kclose, what="using namespace"
    )
    ns2 = _find(lines, lambda s: s == "namespace torch::aot_inductor {", using, what="namespace #2")
    env = _find(
        lines,
        lambda s: s.startswith("static bool _check_aoti_runtime_check_inputs_env"),
        ns2,
        what="_check_aoti_runtime_check_inputs_env",
    )
    env_end = _find(lines, lambda s: s == "}", env, what="end of runtime-check helper")
    run = _find(
        lines, lambda s: s.startswith("void AOTInductorModel::run_impl("), ns2, what="run_impl"
    )
    run_end = _find(
        lines,
        lambda s: s.startswith("} // AOTInductorModel::run_impl"),
        run,
        what="end of run_impl",
    )
    return Anchors(
        include=(0, inc + 1),
        driver=(drv, ns1),
        kernels_class=(kcls, kend + 1),
        kernels_open=kopen,
        kernels_close=kclose,
        templates=(using, ns2),
        env_fn=(env, env_end + 1),
        run_impl=(run, run_end + 1),
    )


# ---------------------------------------------------------------------------
# run_impl: prologue / compute, declarations, liveness
# ---------------------------------------------------------------------------


@dataclass
class Decl:
    idx: int  # absolute line index of the declaration
    kind: str
    ctype: str = ""  # resolved C++ type; "" until the fixpoint runs


@dataclass
class Body:
    sig: list[str]  # run_impl signature lines, verbatim
    prologue: tuple[int, int]  # [begin, end) — stays in run_impl
    compute: tuple[int, int]  # [begin, end) — what gets split
    decls: dict[str, Decl]
    uses: dict[str, set[int]]
    params: dict[str, str] = field(default_factory=dict)  # run_impl params -> type


#: One run_impl parameter. The LAST one carries no trailing comma and
#: its `) {` sits on the next line, so end-of-line has to count too —
#: a parameter missed here is an undeclared identifier in every part.
_SIG_PARAM = re.compile(r"^\s*(?P<t>[\w:*]+(?:\s*\*)?)\s+(?P<n>\w+)\s*(?:[,)]|$)")


def _parse_signature(lines: Sequence[str], begin: int) -> tuple[int, dict[str, str]]:
    """Return (index of the line after `) {`, run_impl's parameters)."""
    params: dict[str, str] = {}
    i = begin
    while i < len(lines):
        stripped = _strip_comment(lines[i]).strip()
        if stripped.endswith(") {"):
            return i + 1, params
        m = _SIG_PARAM.match(_strip_comment(lines[i]))
        if m and i > begin:
            params[m.group("n")] = m.group("t")
        i += 1
    raise Declined("run_impl signature never closes")


def parse_body(lines: Sequence[str], anchors: Anchors) -> Body:
    begin, end = anchors.run_impl
    body_begin, params = _parse_signature(lines, begin)
    body_end = end - 1  # the `} // ...run_impl` line

    # A run_impl parameter is spelled across two lines by the emitter
    # (`AtenTensorHandle*` then the name); recover those.
    for i in range(begin + 1, body_begin):
        raw = _strip_comment(lines[i]).strip()
        if raw.endswith("*") or raw in ("AtenTensorHandle*",):
            nxt = _strip_comment(lines[i + 1]).strip().rstrip(",)")
            if nxt.isidentifier():
                params[nxt] = raw

    # The prologue ends at the LAST scope guard or the last nested block —
    # a guard must stay in run_impl, and everything after it must be flat.
    depth = 0
    prologue_end = body_begin
    for i in range(body_begin, body_end):
        if depth > 0 or _GUARD.match(lines[i]):
            prologue_end = i + 1
        depth += lines[i].count("{") - lines[i].count("}")
    if depth != 0:
        raise Declined("run_impl body braces do not balance")

    compute = (prologue_end, body_end)
    if compute[1] - compute[0] < MIN_COMPUTE_LINES:
        raise Declined(
            f"compute region is {compute[1] - compute[0]} lines "
            f"(< {MIN_COMPUTE_LINES}); not the pathological shape"
        )

    # The compute region must be provably flat straight-line code.
    for i in range(*compute):
        line = lines[i]
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        if not _balanced(line):
            raise Declined(f"compute line {i + 1} is not a self-contained statement")
        if _CONTROL.match(stripped):
            raise Declined(f"compute line {i + 1} carries control flow: {stripped[:60]}")
        if stripped.startswith("#"):
            raise Declined(f"compute line {i + 1} is a preprocessor directive")

    decls: dict[str, Decl] = {}
    for i in range(body_begin, body_end):
        stripped = lines[i].strip()
        if not stripped or stripped.startswith("//"):
            continue
        for kind, rx in _DECLS:
            m = rx.match(stripped)
            if not m:
                continue
            name = m.group("n")
            if name in decls or name in params:
                raise Declined(f"{name} is declared more than once (line {i + 1})")
            decls[name] = Decl(i, kind)
            break

    uses: dict[str, set[int]] = {n: set() for n in list(decls) + list(params)}
    for i in range(body_begin, body_end):
        for name in set(_IDENT.findall(_strip_comment(lines[i]))):
            if name in uses:
                uses[name].add(i)

    return Body(
        sig=list(lines[begin:body_begin]),
        prologue=(body_begin, prologue_end),
        compute=compute,
        decls=decls,
        uses=uses,
        params=params,
    )


_INT_WORDS = frozenset({"L", "int64_t", "c10", "div_floor_integer", "static_cast"})


def resolve_type(lines: Sequence[str], body: Body, name: str, _seen: set[str] | None = None) -> str:
    """The C++ type of ``name``, resolved on demand and memoised.

    Only names that actually cross a boundary are ever asked — a wrapper is
    full of prologue-local ``auto``\\ s (``auto inputs = steal_from_raw…``)
    that are untypable here and need not be typed, because they never leave
    ``run_impl``.

    A wrong answer is a HARD COMPILE ERROR at the reference-binding site in
    the part TU (non-const references admit no conversions), so this map is
    checked by the compiler as well as by the gate below.
    """
    if name in body.params:
        return body.params[name]
    d = body.decls.get(name)
    if d is None:
        raise Declined(f"{name} crosses a boundary but is never declared")
    if d.ctype:
        return d.ctype
    seen = _seen or set()
    if name in seen:
        raise Declined(f"circular type dependency on {name}")
    seen.add(name)

    stripped = lines[d.idx].strip()
    if d.kind in _KIND_TYPE:
        d.ctype = _KIND_TYPE[d.kind]
    elif d.kind == "move":
        m = _RX["move"].match(stripped)
        src = resolve_type(lines, body, m.group("src"), seen)  # type: ignore[union-attr]
        if m.group("sub"):  # type: ignore[union-attr]
            if src != "std::vector<RAIIAtenTensorHandle>":
                raise Declined(f"`auto {name}` subscripts a {src}")
            src = "RAIIAtenTensorHandle"
        d.ctype = src
    elif d.kind == "scalar":
        d.ctype = _RX["scalar"].match(stripped).group("t")  # type: ignore[union-attr]
    elif d.kind in _REDERIVED:
        d.ctype = "<re-derived>"
    elif d.kind == "auto_int":
        rhs = _RX["auto_int"].match(stripped).group("rhs")  # type: ignore[union-attr]
        probe = rhs
        for call in _INT_CALLS:
            probe = probe.replace(call, "")
        if not _INT_RHS.match(probe):
            raise Declined(f"cannot type `auto {name}` from `{rhs[:60]}`")
        for ident in _IDENT.findall(probe):
            if ident in _INT_WORDS:
                continue
            if resolve_type(lines, body, ident, seen) not in ("int64_t", "int32_t"):
                raise Declined(f"`auto {name}` depends on non-integer `{ident}`")
        d.ctype = "int64_t"
    else:  # pragma: no cover - _DECLS is exhaustive
        raise Declined(f"unhandled declaration kind {d.kind}")
    return d.ctype


# ---------------------------------------------------------------------------
# Cut points and the live set
# ---------------------------------------------------------------------------

_ALLOC_OPEN = re.compile(r"^AtenTensorHandle\s+(?P<n>\w+);$")


def cut_points(lines: Sequence[str], body: Body, parts: int) -> list[int]:
    """``parts+1`` boundaries over the compute region.

    A boundary never lands inside an allocation group — the
    ``AtenTensorHandle h;`` / ``...(&h)`` / ``RAIIAtenTensorHandle b(h);``
    triple — because splitting one would thread a raw, not-yet-owned handle.
    """
    begin, end = body.compute
    open_groups: set[int] = set()
    pending: str | None = None
    for i in range(begin, end):
        stripped = lines[i].strip()
        m = _ALLOC_OPEN.match(stripped)
        if m:
            pending, start = m.group("n"), i
        elif pending and stripped.startswith(f"RAIIAtenTensorHandle {pending}("):
            open_groups.update(range(start, i + 1))
            pending = None

    bounds = [begin]
    for k in range(1, parts):
        b = begin + round((end - begin) * k / parts)
        while b < end and (
            b in open_groups or not lines[b].strip() or lines[b].strip().startswith("//")
        ):
            b += 1
        if b <= bounds[-1]:
            raise Declined(f"cannot place cut {k} of {parts}")
        bounds.append(b)
    bounds.append(end)
    return bounds


@dataclass
class Plan:
    bounds: list[int]
    threaded: list[list[str]]  # per part, names arriving as by-ref parameters
    rederived: list[list[str]]  # per part, names re-emitted verbatim
    parts: int


def plan_split(lines: Sequence[str], body: Body, parts: int) -> Plan:
    bounds = cut_points(lines, body, parts)

    def part_of(idx: int) -> int:
        if idx < bounds[0]:
            return -1  # prologue / signature
        for k in range(parts):
            if bounds[k] <= idx < bounds[k + 1]:
                return k
        return parts - 1

    threaded: list[list[str]] = [[] for _ in range(parts)]
    rederived: list[list[str]] = [[] for _ in range(parts)]

    names = list(body.params) + list(body.decls)
    for name in names:
        decl = body.decls.get(name)
        used = {part_of(i) for i in body.uses[name]}
        used.discard(-1)
        if not used:
            continue
        born = part_of(decl.idx) if decl else -1
        if decl and decl.kind in _REDERIVED:
            for k in sorted(used):
                if k > born:
                    rederived[k].append(name)
            continue
        for k in range(born + 1, max(used) + 1):
            threaded[k].append(name)

    order = {n: (body.decls[n].idx if n in body.decls else -1) for n in names}
    for k in range(parts):
        threaded[k].sort(key=lambda n: (order[n], n))
        rederived[k].sort(key=lambda n: (order[n], n))
    return Plan(bounds=bounds, threaded=threaded, rederived=rederived, parts=parts)


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------

_CTX_MEMBERS = (
    "    const std::shared_ptr<std::vector<ConstantHandle>>& constants_;",
    "    const std::unique_ptr<AOTInductorModelKernelsBase>& kernels_;",
    "    int32_t device_idx_;",
    "    const std::optional<std::string>& cubin_dir_;",
)


def _param(name: str, body: Body) -> str:
    t = body.decls[name].ctype if name in body.decls else body.params[name]
    return f"{t} {name}" if t.endswith("*") else f"{t}& {name}"


def _signature(k: int, body: Body, plan: Plan, *, defn: bool) -> str:
    args = ", ".join(_param(n, body) for n in plan.threaded[k])
    scope = f"{CTX}::" if defn else ""
    return f"void {scope}{PREFIX}part_{k}({args})"


def _ctx_struct(body: Body, plan: Plan) -> list[str]:
    out = [f"struct {CTX} {{"]
    out.extend(_CTX_MEMBERS)
    for k in range(plan.parts):
        out.append(f"    {_signature(k, body, plan, defn=False)};")
    out.append("};")
    return [line + MARK for line in out]


def _chunk(lines: Sequence[str], k: int, body: Body, plan: Plan) -> list[str]:
    out = [CHUNK_BEGIN + str(k)]
    for name in plan.rederived[k]:
        out.append(lines[body.decls[name].idx] + MARK_RD)
    out.extend(lines[plan.bounds[k] : plan.bounds[k + 1]])
    if k + 1 < plan.parts:
        call = ", ".join(plan.threaded[k + 1])
        out.append(f"    this->{PREFIX}part_{k + 1}({call});" + MARK)
    out.append(CHUNK_END + str(k))
    return out


def _part_tu(lines: Sequence[str], a: Anchors, k: int, body: Body, plan: Plan) -> str:
    out: list[str] = list(lines[a.include[0] : a.include[1]])
    out += lines[a.driver[0] : a.driver[1]]
    out.append("namespace torch::aot_inductor {")
    out += lines[a.kernels_class[0] : a.kernels_class[1]]
    out.append("} // namespace torch::aot_inductor")
    out += lines[a.templates[0] : a.templates[1]]
    out.append("namespace torch::aot_inductor {")
    out += lines[a.env_fn[0] : a.env_fn[1]]
    out += _ctx_struct(body, plan)
    out.append(_signature(k, body, plan, defn=True) + " {")
    out += _chunk(lines, k, body, plan)
    out.append("}")
    out.append("} // namespace torch::aot_inductor")
    return "\n".join(out)


def _main_tu(lines: Sequence[str], a: Anchors, body: Body, plan: Plan) -> str:
    out: list[str] = []
    begin, end = a.run_impl
    ctx_ctor = (
        f"    {CTX} {PREFIX}ctx{{this->constants_, this->kernels_, "
        f"this->device_idx_, this->cubin_dir_}};"
    )
    call = ", ".join(plan.threaded[0])
    for i, line in enumerate(lines):
        if i == a.kernels_open or i == a.kernels_close:
            # The class must have EXTERNAL linkage: an anonymous-namespace
            # type is a different type in every TU, and a part casting the
            # kernels object to its own copy of it would be UB. (It is also
            # the trap that silently voided the research lane's first
            # measurement: gcc emitted nothing for the parts.)
            out.append(WAS + line)
            continue
        if begin <= i < end:
            if i == begin:
                out += body.sig
                out += lines[body.prologue[0] : body.prologue[1]]
                out.append(ctx_ctor + MARK)
                out.append(f"    {PREFIX}ctx.{PREFIX}part_0({call});" + " " + PARTS)
                out.append(lines[end - 1])
            continue
        out.append(line)
        if i == a.env_fn[1] - 1:
            out += _ctx_struct(body, plan)
    return "\n".join(out)


# ---------------------------------------------------------------------------
# The gate: a self-contained mechanical inverse
# ---------------------------------------------------------------------------

_PART_DEFINITION = re.compile(
    rf"^void {re.escape(CTX)}::{re.escape(PREFIX)}part_(?P<part>\d+)\((?P<params>.*)\) \{{$"
)
_PARAMETER_NAME = re.compile(r"(?P<name>[A-Za-z_]\w*)$")
_CTX_CONSTRUCTION = (
    f"    {CTX} {PREFIX}ctx{{this->constants_, this->kernels_, "
    f"this->device_idx_, this->cubin_dir_}};" + MARK
)


def _parameter_names(parameters: str) -> tuple[str, ...]:
    if not parameters:
        return ()
    names: list[str] = []
    for parameter in parameters.split(", "):
        match = _PARAMETER_NAME.search(parameter)
        if match is None:
            raise Declined(f"cannot read generated parameter {parameter!r}")
        names.append(match.group("name"))
    return tuple(names)


def _validate_dispatch(main: str, parts: Sequence[str]) -> None:
    """Prove the generated continuation chain executes every emitted chunk.

    Reconstruction proves statement preservation, but generated calls are
    intentionally absent from the reconstructed source. Validate those calls
    independently so a faulty emitter cannot preserve every statement while
    making some or all of them unreachable.
    """

    if len(parts) < 2:
        raise Declined("generated dispatch has fewer than two parts")

    definitions: list[tuple[str, ...]] = []
    part_lines: list[list[str]] = []
    for expected_part, text in enumerate(parts):
        lines = text.split("\n")
        matches = [match for line in lines if (match := _PART_DEFINITION.match(line))]
        if len(matches) != 1 or int(matches[0].group("part")) != expected_part:
            raise Declined(f"part {expected_part} has no unique matching definition")
        definitions.append(_parameter_names(matches[0].group("params")))
        part_lines.append(lines)

    main_lines = main.split("\n")
    if main_lines.count(_CTX_CONSTRUCTION) != 1:
        raise Declined("main dispatch has no unique exact context construction")
    expected_main = f"    {PREFIX}ctx.{PREFIX}part_0({', '.join(definitions[0])}); " + PARTS
    main_dispatches = [line for line in main_lines if line.endswith(PARTS)]
    if main_dispatches != [expected_main]:
        raise Declined("main dispatch does not call part 0 with its exact live set")

    for part, lines in enumerate(part_lines):
        begin = CHUNK_BEGIN + str(part)
        end = CHUNK_END + str(part)
        unique_ordered_markers = (
            lines.count(begin) == 1
            and lines.count(end) == 1
            and lines.index(begin) < lines.index(end)
        )
        if not unique_ordered_markers:
            raise Declined(f"part {part} has non-unique or misordered chunk markers")
        emitted_calls = [line for line in lines if f"this->{PREFIX}part_" in line]
        if part + 1 == len(parts):
            if emitted_calls:
                raise Declined(f"part {part} dispatches after the final chunk")
            continue
        expected_call = (
            f"    this->{PREFIX}part_{part + 1}({', '.join(definitions[part + 1])});" + MARK
        )
        if emitted_calls != [expected_call]:
            raise Declined(
                f"part {part} dispatch does not call part {part + 1} with its exact live set"
            )


def reconstruct(main: str, parts: Sequence[str]) -> str:
    """Rebuild the original source from the split output ALONE.

    Reads no side table produced by the transform, so it cannot be fooled by
    it. Every generated line ends in :data:`MARK` and is dropped; every
    displaced line is carried behind :data:`WAS` and restored; the single
    :data:`PARTS` line is replaced by the part bodies in order; and every
    :data:`MARK_RD` re-derivation must be found, verbatim, in the
    reconstruction it claims to copy from.

    Raises :class:`Declined` when the output is not a well-formed split.
    """
    _validate_dispatch(main, parts)

    rederived: list[str] = []
    chunks: list[list[str]] = []

    def sift(line: str, into: list[str]) -> None:
        if line.endswith(MARK_RD):
            rederived.append(line[: -len(MARK_RD)])
        elif not line.endswith(MARK):
            into.append(line)

    for k, text in enumerate(parts):
        lines = text.split("\n")
        try:
            i = lines.index(CHUNK_BEGIN + str(k))
            j = lines.index(CHUNK_END + str(k))
        except ValueError as exc:
            raise Declined(f"part {k} has no chunk markers") from exc
        chunk: list[str] = []
        for line in lines[i + 1 : j]:
            sift(line, chunk)
        chunks.append(chunk)

    out: list[str] = []
    for line in main.split("\n"):
        if line.startswith(WAS):
            out.append(line[len(WAS) :])
        elif line.endswith(PARTS):
            for chunk in chunks:
                out.extend(chunk)
        else:
            sift(line, out)

    original = set(out)
    for line in rederived:
        if line not in original:
            raise Declined(f"re-derived line copies nothing in the original: {line.strip()[:70]}")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunImplSplit:
    """What the transform did to one wrapper TU."""

    applied: bool
    reason: str = ""
    main: str = ""
    parts: tuple[str, ...] = ()
    compute_lines: int = 0
    declarations: int = 0
    threaded_max: int = 0
    rederived_max: int = 0

    def detail(self) -> str:
        if not self.applied:
            return f"run_impl split declined — {self.reason}"
        return (
            f"run_impl: {self.compute_lines} compute lines / {self.declarations} "
            f"declarations -> {len(self.parts)} parts "
            f"(max {self.threaded_max} threaded, {self.rederived_max} re-derived)"
        )


def split_run_impl(source: str, *, parts: int = DEFAULT_PARTS) -> RunImplSplit:
    """Split ``run_impl`` into ``parts`` chained translation units.

    Returns an unapplied :class:`RunImplSplit` — never raises — whenever the
    wrapper is not the recognised shape or the inverse does not reproduce
    ``source`` byte for byte. The caller compiles the original in that case.
    """
    if parts < 2:
        return RunImplSplit(False, f"parts={parts} is not a split")
    lines = source.split("\n")
    try:
        anchors = find_anchors(lines)
        body = parse_body(lines, anchors)
        plan = plan_split(lines, body, parts)
        for names in plan.threaded:
            for name in names:
                resolve_type(lines, body, name)

        # Every re-derived line must be a COPY of a line that exists in the
        # original, declaring the name it claims to. The inverse drops these
        # lines, so this is the forward gate that keeps them honest.
        for k in range(parts):
            for name in plan.rederived[k]:
                decl = lines[body.decls[name].idx]
                if re.search(rf"\b{re.escape(name)}\b", decl) is None:
                    raise Declined(f"re-derivation of {name} does not declare it")

        main = _main_tu(lines, anchors, body, plan)
        part_sources = [_part_tu(lines, anchors, k, body, plan) for k in range(parts)]
        if reconstruct(main, part_sources) != source:
            raise Declined(
                "self-check failed: reconstructing the split did not reproduce "
                "the original byte for byte"
            )
    except Declined as exc:
        return RunImplSplit(False, str(exc))

    return RunImplSplit(
        True,
        main=main,
        parts=tuple(part_sources),
        compute_lines=body.compute[1] - body.compute[0],
        declarations=len(body.decls),
        threaded_max=max(len(t) for t in plan.threaded),
        rederived_max=max(len(r) for r in plan.rederived),
    )


__all__ = [
    "DEFAULT_PARTS",
    "MIN_COMPUTE_LINES",
    "Declined",
    "MARK",
    "MARK_RD",
    "RunImplSplit",
    "reconstruct",
    "split_run_impl",
]
