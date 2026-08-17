"""Split AOTI's generated wrapper constructor so g++ stops paying a
superlinear price for a data table written as code.

Measured, per host invocation: an AOTI mint's entire host cost is ONE
``g++ -O1 -c wrapper.cpp`` — 180.6 s of a 390 s AOTI compile (46%); linking is
2.0 s (0.5%), so every classic build lever (mold/lld, ccache, PCH, parallel
host builds) is measured null or already exhausted. That one
translation unit is 6.3 MB / 61k lines and is dominated by TWO functions,
the larger of which is ``AOTInductorModel::AOTInductorModel`` — 26,642
``constants_info_[i].field = ...;`` statements (2,422 constants x 11
fields). It is a DATA TABLE spelled as straight-line executable code, it
runs exactly once at model construction, and it has no runtime significance
whatsoever. GCC's optimizer is superlinear in statements WITHIN one
function, so this is a single-function blowup, not a big-file problem.

This module first rewrites that one run of constructor statements into chunked
helper functions. The same sealed compiler funnel then gives the resulting TU
to :mod:`_run_impl_split`, whose separate continuation-chain gate may split
``run_impl`` across translation units. Each transform declines independently;
an unrecognised shape remains exactly as Inductor emitted it.

Why a mint-path transform and not an inductor config: torch 2.13 exposes no
knob for constants emission or per-function optimization, and the only
config that moves this cost (``compile_wrapper_opt_level='O0'``) is DEAD —
measured at +1.7% per forward, which exceeds AOT's whole 1.4% serving margin,
and it also re-keys every compiled graph. This transform re-keys nothing (see
:func:`install`).

FAIL-CLOSED BY CONSTRUCTION. The transform only ever fires when it can
prove, by textual reconstruction, that it is a pure statement-preserving
regrouping: :func:`split_constants_constructor` re-inlines its own output
and refuses unless the reconstruction is byte-identical to the input. Any
wrapper shape it does not recognise compiles unmodified. A silent no-op is
correct; a mangled TU is not.
"""

from __future__ import annotations

import logging
import re
import shlex
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from . import _run_impl_split as runimpl
from .host_isa import _assert_command_is_clamped

logger = logging.getLogger(__name__)

#: Statements per generated helper. 250 is the measured chunk; the win is flat
#: across a wide band (the point is "not one 26k-statement function", not any
#: particular size).
CHUNK = 250

#: Below this the TU is not the pathological shape this exists for, and the
#: transform declines rather than adding a mechanism nobody needs.
MIN_STATEMENTS = 2000

#: Name prefix for every symbol this module introduces.
PREFIX = "_tcg_constants_info_"

#: The template header line emitted above every helper.
_TEMPLATE_LINE = "template <typename ConstantsInfoT_>"

#: Banner emitted once, immediately above the first helper.
_BANNER = (
    "// torchcg: constants_info_ moved out of the constructor.",
    "// Same statements, same order; grouped so gcc's superlinear",
    "// per-function cost stops dominating this translation unit.",
)

#: The constructor signature inductor emits (``aoti_model_class_name`` is
#: configurable, so the class name is captured rather than assumed).
_CTOR_RE = re.compile(
    r"^(?P<cls>[A-Za-z_]\w*)::(?P=cls)"
    r"\(std::shared_ptr<ConstantMap> constants_map,\s*$"
)

#: One ``constants_info_`` field assignment, whole line, one statement.
_STMT_RE = re.compile(r"^(?P<indent>\s*)constants_info_\[\d+\]\.\w+ = .*;$")


class _Declined(Exception):
    """Internal: the source is not the shape this transform recognises."""


@dataclass(frozen=True)
class SplitOutcome:
    """What the transform did to one translation unit."""

    applied: bool
    reason: str  # "" when applied, else why it declined
    statements: int = 0
    chunks: int = 0
    bytes_in: int = 0
    bytes_out: int = 0
    source: str = ""
    #: Set by the run_impl path, whose shape the ctor-split wording cannot
    #: describe.
    summary: str = ""

    def detail(self) -> str:
        name = Path(self.source).name if self.source else "<source>"
        if self.applied:
            if self.summary:
                return f"{name}: {self.summary}"
            return (
                f"{name}: {self.statements} constants_info_ statements -> "
                f"{self.chunks} helper functions "
                f"({self.bytes_in}B -> {self.bytes_out}B)"
            )
        return f"{name}: declined — {self.reason}"


def _statement_line(line: str) -> bool:
    """True when ``line`` is exactly one self-contained ``constants_info_``
    assignment: balanced brackets, no raw string, terminated. Anything
    subtler than that is not something this transform will move."""
    if not _STMT_RE.match(line):
        return False
    if 'R"' in line or line.count('"') % 2:
        return False
    return (
        line.count("{") == line.count("}")
        and line.count("(") == line.count(")")
        and line.count("[") == line.count("]")
    )


def _find_block(lines: Sequence[str]) -> tuple[int, int, int]:
    """``(ctor_line, first, last)`` of the constructor and the maximal run
    of constants_info_ statements inside it. Raises :class:`_Declined`."""
    ctors = [i for i, line in enumerate(lines) if _CTOR_RE.match(line)]
    if len(ctors) != 1:
        raise _Declined(f"expected exactly 1 AOTInductorModel constructor, found {len(ctors)}")
    ctor = ctors[0]
    hits = [i for i, line in enumerate(lines) if _statement_line(line)]
    if not hits:
        raise _Declined("no constants_info_ assignment statements")
    first, last = hits[0], hits[-1]
    if first <= ctor:
        raise _Declined("constants_info_ statements precede the constructor")
    if last - first + 1 != len(hits):
        raise _Declined(
            f"constants_info_ statements are not contiguous "
            f"({len(hits)} statements spanning {last - first + 1} lines)"
        )
    if len(hits) < MIN_STATEMENTS:
        raise _Declined(
            f"only {len(hits)} statements (< {MIN_STATEMENTS}); not the pathological shape"
        )
    return ctor, first, last


def _helper_attrs(opt_none: bool) -> str:
    # ``noinline`` is what keeps gcc from folding the chunks straight back
    # into the constructor at -O1 (-finline-functions-called-once).
    #
    # ``optimize("O0")`` resets OPTIMIZATION options for the marked function,
    # which is the intent and is harmless twice over: these statements are
    # integer/pointer/vector assignments, so the wrapper command's math flags
    # (-ffp-contract, -fmath-errno, ...) have nothing to act on; and -fPIC is a
    # code-generation option the attribute does not touch.
    attrs = "__attribute__((noinline))"
    if opt_none:
        attrs += ' __attribute__((optimize("O0")))'
    return attrs


def split_constants_constructor(
    source: str, *, chunk: int = CHUNK, opt_none: bool = True
) -> tuple[str, SplitOutcome]:
    """Return ``(source', outcome)``.

    ``source'`` differs from ``source`` only in that the constructor's run
    of ``constants_info_`` assignments has been moved, in order, into
    ``chunk``-sized static helper functions called in that same order. When
    ``opt_none`` the helpers additionally carry ``optimize("O0")``: they run
    once at model construction, so their optimization level is provably
    without runtime consequence, and pgw#793 measured that as the larger
    half of the win.

    Declines (returning ``source`` unchanged) whenever the wrapper is not
    the recognised shape, or whenever re-inlining the result does not
    reproduce the input byte for byte.
    """
    lines = source.split("\n")
    try:
        ctor, first, last = _find_block(lines)
    except _Declined as exc:
        return source, SplitOutcome(False, str(exc), bytes_in=len(source))

    body = lines[first : last + 1]
    indent = _STMT_RE.match(body[0]).group("indent")  # type: ignore[union-attr]
    groups = [body[i : i + chunk] for i in range(0, len(body), chunk)]
    attrs = _helper_attrs(opt_none)

    helpers: list[str] = list(_BANNER)
    calls: list[str] = []
    for idx, group in enumerate(groups):
        name = f"{PREFIX}{idx}"
        helpers.append(_TEMPLATE_LINE)
        helpers.append(f"{attrs} static void {name}(ConstantsInfoT_& constants_info_) {{")
        helpers.extend(group)
        helpers.append("}")
        calls.append(f"{indent}{name}(constants_info_);")
    helpers.append("")

    out = lines[:ctor] + helpers + lines[ctor:first] + calls + lines[last + 1 :]
    result = "\n".join(out)

    reconstructed = _reinline(result)
    if reconstructed != source:
        return source, SplitOutcome(
            False,
            "self-check failed: re-inlining the split source did not "
            "reproduce the original byte for byte",
            bytes_in=len(source),
        )
    return result, SplitOutcome(
        True,
        "",
        statements=len(body),
        chunks=len(groups),
        bytes_in=len(source),
        bytes_out=len(result),
    )


_CALL_RE = re.compile(rf"^\s*{re.escape(PREFIX)}(\d+)\(constants_info_\);$")
_DEF_RE = re.compile(rf"^.*static void {re.escape(PREFIX)}(\d+)\(ConstantsInfoT_&")


def _reinline(source: str) -> str:
    """Inverse of the split: substitute each helper's body back at its call
    site and delete the helper definitions. Used ONLY to self-verify the
    transform — if this does not reproduce the input exactly, the transform
    is not a pure regrouping and must not ship."""
    lines = source.split("\n")
    bodies: dict[str, list[str]] = {}
    keep: list[str] = []
    i = 0
    start: int | None = None
    while i < len(lines):
        match = _DEF_RE.match(lines[i])
        if match is None:
            keep.append(lines[i])
            i += 1
            continue
        # A helper definition, preceded by its "template <...>" line; the
        # three banner comments precede the first one.
        if not keep or not keep[-1].startswith(_TEMPLATE_LINE):
            return source  # not our emission shape; force the self-check to fail
        keep.pop()
        if start is None:
            start = len(keep)
        body: list[str] = []
        i += 1
        while i < len(lines) and lines[i] != "}":
            body.append(lines[i])
            i += 1
        i += 1  # the closing brace
        bodies[match.group(1)] = body
    if start is not None:
        span = len(_BANNER)
        if tuple(keep[start - span : start]) == _BANNER:
            del keep[start - span : start]
            start -= span
        if start < len(keep) and keep[start] == "":
            del keep[start]

    out: list[str] = []
    for line in keep:
        match = _CALL_RE.match(line)
        if match is None:
            out.append(line)
            continue
        out.extend(bodies.pop(match.group(1), []))
    if bodies:
        return source  # unmatched helper: force the caller's self-check to fail
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Compiler-path installation
# ---------------------------------------------------------------------------

# The outer worker already schedules several graph-class children. Keeping the
# K+1 host fragments serial avoids a hidden second pool and still preserves the
# measured total-CPU win from splitting one superlinear function. This is a
# sealed library policy, not an environment or caller knob.
HOST_COMPILE_JOBS = 1

_installed = False


#: Phase-label prefix for the run_impl lever — a name, never a version number.
_RUNIMPL = "runimpl"


def _emit(outcome: SplitOutcome, lever: str = "") -> None:
    """Log the private transform outcome without importing product telemetry."""

    state = "applied" if outcome.applied else "declined"
    logger.debug(
        "aot-wrapper-split %s%s: %s",
        f"{lever}_" if lever else "",
        state,
        outcome.detail(),
    )


def _split_source_arg(argv: Sequence[str]) -> int | None:
    """Index of the single ``.cpp`` source in a compile-only command line,
    or None when this is not a one-source C++ compile (links, the PCH
    build, the ``.S`` constants object all land here too)."""
    if "-c" not in argv:
        return None
    sources = [i for i, tok in enumerate(argv) if tok.endswith(".cpp") and not tok.startswith("-")]
    if len(sources) != 1:
        return None
    return sources[0]


def transform_command(cmd_line: str) -> tuple[str, SplitOutcome | None]:
    """``(cmd_line', outcome)``. ``outcome`` is None when the command is not
    a single-source C++ compile at all (nothing to report). Every failure
    mode — unreadable source, unrecognised shape, write error — returns the
    ORIGINAL command line."""
    try:
        argv = shlex.split(cmd_line)
    except ValueError:
        return cmd_line, None
    index = _split_source_arg(argv)
    if index is None:
        return cmd_line, None
    source = Path(argv[index])
    try:
        text = source.read_text()
    except OSError as exc:
        return cmd_line, SplitOutcome(False, f"unreadable source: {exc}", source=str(source))
    split, outcome = split_constants_constructor(text)
    outcome = replace(outcome, source=str(source))
    if not outcome.applied:
        return cmd_line, outcome
    target = source.with_name(source.name[: -len(".cpp")] + ".tcg.cpp")
    try:
        target.write_text(split)
    except OSError as exc:
        return cmd_line, replace(outcome, applied=False, reason=f"unwritable transform: {exc}")
    argv = list(argv)
    argv[index] = str(target)
    return shlex.join(argv), outcome


# ---------------------------------------------------------------------------
# run_impl: the K-way split, driven over K+1 real compiles
# ---------------------------------------------------------------------------


def _object_arg(argv: Sequence[str]) -> int | None:
    """Index of the ``-o`` VALUE in a compile-only command line."""
    for i, tok in enumerate(argv[:-1]):
        if tok == "-o":
            return i + 1
    return None


def _retarget(argv: Sequence[str], src_at: int, obj_at: int, source: Path, obj: Path) -> str:
    out = list(argv)
    out[src_at] = str(source)
    out[obj_at] = str(obj)
    return shlex.join(out)


def _partial_link(argv: Sequence[str], objects: Sequence[Path], target: Path) -> str:
    """``ld -r`` the part objects into the ONE object torch's link step
    expects. Driven through the same compiler binary torch chose, so the
    toolchain stays exactly the one the compiled-graph identity names."""
    return shlex.join([argv[0], "-r", "-nostdlib", "-o", str(target), *(str(o) for o in objects)])


def split_and_compile(
    cmd_line: str, cwd: str, run: Callable[[str, str], None]
) -> SplitOutcome | None:
    """Build ``cmd_line``'s source as K+1 chained TUs, partial-linked into
    the one object torch's link step expects.

    Returns the outcome on success, or None when the caller should fall back
    to compiling ``cmd_line`` unmodified. Nothing here may leave a
    half-written object at the target path: the partial link is the last
    step and runs only once every part has built.
    """

    argv = shlex.split(cmd_line)
    src_at, obj_at = _split_source_arg(argv), _object_arg(argv)
    if src_at is None or obj_at is None:
        return None
    source = Path(argv[src_at])
    try:
        text = source.read_text()
    except OSError as exc:
        _emit(SplitOutcome(False, f"unreadable source: {exc}", source=str(source)), lever=_RUNIMPL)
        return None
    split = runimpl.split_run_impl(text)
    if not split.applied:
        _emit(SplitOutcome(False, split.reason, source=str(source)))
        return None

    stem = source.name[: -len(".cpp")]
    units: list[tuple[Path, Path]] = []
    for name, body in [("main", split.main), *((f"part{i}", p) for i, p in enumerate(split.parts))]:
        cpp = source.with_name(f"{stem}.tcg.{name}.cpp")
        cpp.write_text(body)
        units.append((cpp, cpp.with_suffix(".o")))

    def build(unit: tuple[Path, Path]) -> None:
        run(_retarget(argv, src_at, obj_at, unit[0], unit[1]), cwd)

    for unit in units:
        build(unit)

    run(_partial_link(argv, [o for _, o in units], Path(argv[obj_at])), cwd)
    return SplitOutcome(
        True,
        "",
        statements=split.compute_lines,
        chunks=len(units),
        bytes_in=len(text),
        bytes_out=sum(len(b) for b in (split.main, *split.parts)),
        source=str(source),
        summary=(
            f"run_impl {split.compute_lines} statements / {split.declarations} "
            f"declarations -> {len(split.parts)} chained TUs "
            f"(max {split.threaded_max} threaded, {split.rederived_max} "
            f"re-derived), {HOST_COMPILE_JOBS} concurrent compile, partial-linked"
        ),
    )


def install() -> bool:
    """Install the transform on torch's single host-compile funnel.

    ``CppBuilder.build()`` reaches ``run_compile_cmd`` through the module
    global, so wrapping that name covers every host compile inductor
    performs and nothing else.

    IDENTITY: this changes neither the compiler, its flags, any Inductor
    config, nor any loaded library. It only regroups generated statements
    after the compiled-graph key has been derived.

    Returns True when installed, False when already installed.
    """
    global _installed
    if _installed:
        return False
    try:
        from torch._inductor import cpp_builder
    except Exception:
        logger.debug("aot-wrapper-split: torch._inductor unavailable")
        return False
    original: Callable[..., None] = cpp_builder.run_compile_cmd

    def _patched(cmd_line: str, cwd: str) -> None:
        # The ISA clamp is asserted at the CONFIG level by host_isa.impose();
        # this is the argv-level assertion. `run_compile_cmd` is torch's single
        # host-compile funnel, so every object a mint produces passes through
        # here exactly once. It raises rather than degrading: an unclamped
        # object is not slower, it is unportable — a SIGILL-class defect.
        _assert_command_is_clamped(cmd_line)

        try:
            new_cmd, outcome = transform_command(cmd_line)
        except Exception:
            logger.warning(
                "aot-wrapper-split: transform raised; compiling unmodified",
                exc_info=True,
            )
            original(cmd_line, cwd)
            return
        if outcome is None:
            original(cmd_line, cwd)  # not a single-source C++ compile
            return
        _emit(outcome)

        # The run_impl split works on whatever source the ctor split left —
        # inductor's own when the ctor split declined, its regrouped one when it
        # did not. It subsumes the single compile: when it fires it drives every
        # compile itself and there is no monolith left to run.
        try:
            runimpl_outcome = split_and_compile(new_cmd, cwd, original)
        except Exception as exc:
            _emit(
                SplitOutcome(
                    False,
                    "split TUs failed to build, retried the whole wrapper: "
                    f"{type(exc).__name__}: {exc}",
                    source=outcome.source,
                ),
                lever=_RUNIMPL,
            )
            logger.warning(
                "aot-wrapper-split: split failed (%s); recompiling the whole wrapper",
                type(exc).__name__,
                exc_info=True,
            )
        else:
            if runimpl_outcome is not None:
                _emit(runimpl_outcome, lever=_RUNIMPL)
                return

        if not outcome.applied:
            original(cmd_line, cwd)
            return
        try:
            original(new_cmd, cwd)
        except Exception as exc:
            # The transformed TU did not build. The original command is
            # untouched and reproducible, so a mint degrades to the slow
            # path instead of failing. If THAT also fails, the wrapper
            # itself is broken and the real error propagates.
            _emit(
                replace(
                    outcome,
                    applied=False,
                    reason=(
                        "transformed TU failed to compile, retried the "
                        f"original: {type(exc).__name__}"
                    ),
                )
            )
            logger.warning(
                "aot-wrapper-split: transformed TU failed to compile (%s); "
                "recompiling inductor's own source",
                type(exc).__name__,
            )
            original(cmd_line, cwd)
            return
        _emit(outcome)

    _patched.__wrapped__ = original  # type: ignore[attr-defined]
    cpp_builder.run_compile_cmd = _patched
    _installed = True
    logger.info("aot-wrapper-split: installed on cpp_builder.run_compile_cmd")
    return True
