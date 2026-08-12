"""Split AOTI's generated wrapper constructor so g++ stops paying a
superlinear price for a data table written as code (pgw#793).

pgw#793 measured, per host invocation, that an AOTI mint's entire host cost
is ONE ``g++ -O1 -c wrapper.cpp`` — 180.6 s of a 390 s AOTI compile (46%);
linking is 2.0 s (0.5%), so every classic build lever (mold/lld, ccache,
PCH, parallel host builds) is measured null or already exhausted. That one
translation unit is 6.3 MB / 61k lines and is dominated by TWO functions,
the larger of which is ``AOTInductorModel::AOTInductorModel`` — 26,642
``constants_info_[i].field = ...;`` statements (2,422 constants x 11
fields). It is a DATA TABLE spelled as straight-line executable code, it
runs exactly once at model construction, and it has no runtime significance
whatsoever. GCC's optimizer is superlinear in statements WITHIN one
function, so this is a single-function blowup, not a big-file problem.

This module rewrites that one run of statements into chunked helper
functions before the compiler sees it. Nothing else in the TU is touched;
``run_impl`` — which does carry runtime significance — is left exactly as
inductor emitted it.

Why a mint-path transform and not an inductor config: torch 2.13 exposes no
knob for constants emission or per-function optimization, and the only
config that moves this cost (``compile_wrapper_opt_level='O0'``) is DEAD —
pgw#793 measured it at +1.7% per forward, which exceeds AOT's whole 1.4%
serving margin, and it also re-keys every cell. This transform re-keys
nothing (see :func:`install`).

FAIL-CLOSED BY CONSTRUCTION. The transform only ever fires when it can
prove, by textual reconstruction, that it is a pure statement-preserving
regrouping: :func:`split_constants_constructor` re-inlines its own output
and refuses unless the reconstruction is byte-identical to the input. Any
wrapper shape it does not recognise compiles unmodified. A silent no-op is
correct; a mangled TU is not.
"""

from __future__ import annotations

import concurrent.futures as cf
import logging
import os
import re
import shlex
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

from . import host_isa
from . import aot_run_impl_split as runimpl

logger = logging.getLogger(__name__)

#: Typed activity event kind for the transform's outcome. One event per
#: wrapper compile (18 per sdxl cell), each naming applied/declined and why.
EVENT = "aot_wrapper_split"

#: Statements per generated helper. 250 is pgw#793's measured chunk; the
#: win is flat across a wide band (the point is "not one 26k-statement
#: function", not any particular size).
CHUNK = 250

#: Below this the TU is not the pathological shape this exists for, and the
#: transform declines rather than adding a mechanism nobody needs.
MIN_STATEMENTS = 2000

#: Name prefix for every symbol this module introduces.
PREFIX = "_pgw793_constants_info_"

#: The template header line emitted above every helper.
_TEMPLATE_LINE = "template <typename ConstantsInfoT_>"

#: Banner emitted once, immediately above the first helper.
_BANNER = (
    "// pgw#793: the constants_info_ table, moved out of the constructor.",
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
    #: Set by the pgw#811 path, whose shape v1's wording does not describe.
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


def _find_block(lines: Sequence[str]) -> Tuple[int, int, int]:
    """``(ctor_line, first, last)`` of the constructor and the maximal run
    of constants_info_ statements inside it. Raises :class:`_Declined`."""
    ctors = [i for i, line in enumerate(lines) if _CTOR_RE.match(line)]
    if len(ctors) != 1:
        raise _Declined(
            f"expected exactly 1 AOTInductorModel constructor, found "
            f"{len(ctors)}"
        )
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
            f"only {len(hits)} statements (< {MIN_STATEMENTS}); not the "
            "pathological shape"
        )
    return ctor, first, last


def _helper_attrs(opt_none: bool) -> str:
    # ``noinline`` is what keeps gcc from folding the chunks straight back
    # into the constructor at -O1 (-finline-functions-called-once).
    #
    # ``optimize("O0")`` resets OPTIMIZATION options for the marked function,
    # which is exactly the intent and is harmless here twice over: these
    # statements are integer/pointer/vector assignments, so the wrapper
    # command's math flags (-ffp-contract, -fmath-errno, ...) have nothing to
    # act on; and -fPIC is a code-generation option the attribute does not
    # touch, which tests/test_aot_wrapper_split_pgw793.py proves by linking a
    # transformed object into a shared library.
    attrs = "__attribute__((noinline))"
    if opt_none:
        attrs += ' __attribute__((optimize("O0")))'
    return attrs


def split_constants_constructor(
    source: str, *, chunk: int = CHUNK, opt_none: bool = True
) -> Tuple[str, SplitOutcome]:
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

    helpers: List[str] = list(_BANNER)
    calls: List[str] = []
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
    bodies: dict[str, List[str]] = {}
    keep: List[str] = []
    i = 0
    start: Optional[int] = None
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
        body: List[str] = []
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

    out: List[str] = []
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
# Mint-path installation
# ---------------------------------------------------------------------------

#: pgw#995: ``GEN_WORKER_AOT_WRAPPER_SPLIT_OFF`` is GONE. It defaulted ON, no
#: release ever declared it and no endpoint ever set it, so deleting it makes
#: the shape every pod already ran unconditional. An env that selects which
#: program runs is the class that took `GEN_WORKER_PREFER_AOT` dark for three
#: pod attempts; a switch nobody has ever thrown carries none of the safety it
#: appears to and all of the silent-disarm risk.

#: Kill switch for the pgw#811 ``run_impl`` split alone, so the (much older,
#: much better travelled) ctor split can stay on if run_impl ever goes off.
#: pgw#995 KEEPS this one, and the asymmetry with its deleted sibling above is
#: the finding rather than an oversight: this name is LIVE — five SDXL releases
#: (0.2.111/112/113/116/117) declare it and one endpoint carries a non-deleted
#: `endpoint_env_entries` row (verified on the standing hub 2026-08-03). Deleting
#: a live switch changes a running endpoint's behaviour; deleting a never-thrown
#: one cannot. "Zero declarations" is the fact that licenses a deletion, and it
#: is a fact you MEASURE, not one you assume from the code.
DISABLE_V2_ENV = "GEN_WORKER_AOT_RUN_IMPL_SPLIT_OFF"

#: How many of the K+1 part compiles may run at once. pgw#809's pool owns
#: the pod's CPU budget ACROSS entries; it sets this to its per-entry share
#: so the two levers compose (split-within-entry x pool-across-entries)
#: instead of oversubscribing. Unset: derive from the pod's own budget.
JOBS_ENV = "GEN_WORKER_AOT_HOST_COMPILE_JOBS"

#: Left for serving even when nothing else has claimed the box. Matches
#: pgw#809's `SERVING_HEADROOM_CPUS`; a mint never starves the tenant.
#:
#: pgw#973 (§4.24) censused this as a verbatim duplicate of
#: `aot_compile_pool.SERVING_HEADROOM_CPUS` and DEFERRED the collapse
#: deliberately, rather than leaving the question open. This module is an
#: INDUCTOR HOOK — it is imported from inside torch's compile path and its only
#: in-repo imports are `host_isa` and `aot_run_impl_split`. Importing the pool
#: to share one number would drag `aot_resume`, `env_seal` and
#: `worker_goals` — the whole mint driver — into that hook, which is a far
#: worse defect than two copies of `2`. A third module owning the constant
#: would be the clean answer and is not worth a new module for one integer.
#: The duplication is therefore INTENTIONAL and stated; the numbers are pinned
#: together by `test_serving_headroom_is_one_number_pgw973`.
SERVING_HEADROOM_CPUS = 2

_installed = False


def host_compile_jobs(tus: int) -> int:
    """How many part TUs to compile concurrently, never more than ``tus``.

    The split wins on TOTAL CPU (measured 2.8x on the real TU), so jobs=1 is
    already a win and this is only ever an accelerator — which is why it
    fails safe to 1 rather than guessing.
    """
    raw = os.environ.get(JOBS_ENV, "").strip()
    if raw:
        try:
            return max(1, min(tus, int(raw)))
        except ValueError:
            logger.warning("aot-wrapper-split: ignoring non-numeric %s=%r", JOBS_ENV, raw)
    try:
        from .postmortem import effective_cpu_count

        budget = effective_cpu_count() - SERVING_HEADROOM_CPUS
    except Exception:
        return 1
    return max(1, min(tus, budget))


#: Phase-label prefix for the pgw#811 run_impl lever (pgw#958: a name,
#: never a version number).
_RUNIMPL = "runimpl"


def _emit(outcome: SplitOutcome, lever: str = "") -> None:
    """One typed row per wrapper compile. ``lever`` names WHICH split fired:
    the pgw#811 ``run_impl`` split (``runimpl_applied`` / ``runimpl_declined``)
    versus the original ctor split (bare ``applied`` / ``declined``). A phase
    label, deliberately non-numeric — it is not a version counter."""
    try:
        from . import activity as activity_mod

        state = "applied" if outcome.applied else "declined"
        activity_mod.emit_event(
            EVENT,
            outcome.detail(),
            phase=f"{lever}_{state}" if lever else state,
        )
    except Exception:  # pragma: no cover - telemetry must never break a mint
        logger.debug("aot-wrapper-split: activity emit failed", exc_info=True)


def _split_source_arg(argv: Sequence[str]) -> Optional[int]:
    """Index of the single ``.cpp`` source in a compile-only command line,
    or None when this is not a one-source C++ compile (links, the PCH
    build, the ``.S`` constants object all land here too)."""
    if "-c" not in argv:
        return None
    sources = [
        i
        for i, tok in enumerate(argv)
        if tok.endswith(".cpp") and not tok.startswith("-")
    ]
    if len(sources) != 1:
        return None
    return sources[0]


def transform_command(cmd_line: str) -> Tuple[str, Optional[SplitOutcome]]:
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
        return cmd_line, SplitOutcome(
            False, f"unreadable source: {exc}", source=str(source))
    split, outcome = split_constants_constructor(text)
    outcome = replace(outcome, source=str(source))
    if not outcome.applied:
        return cmd_line, outcome
    target = source.with_name(source.name[: -len(".cpp")] + ".pgw793.cpp")
    try:
        target.write_text(split)
    except OSError as exc:
        return cmd_line, replace(
            outcome, applied=False, reason=f"unwritable transform: {exc}")
    argv = list(argv)
    argv[index] = str(target)
    return shlex.join(argv), outcome


# ---------------------------------------------------------------------------
# run_impl: the K-way split, driven over K+1 real compiles
# ---------------------------------------------------------------------------

def _object_arg(argv: Sequence[str]) -> Optional[int]:
    """Index of the ``-o`` VALUE in a compile-only command line."""
    for i, tok in enumerate(argv[:-1]):
        if tok == "-o":
            return i + 1
    return None


def _retarget(argv: Sequence[str], src_at: int, obj_at: int,
              source: Path, obj: Path) -> str:
    out = list(argv)
    out[src_at] = str(source)
    out[obj_at] = str(obj)
    return shlex.join(out)


def _partial_link(argv: Sequence[str], objects: Sequence[Path], target: Path) -> str:
    """``ld -r`` the part objects into the ONE object torch's link step
    expects. Driven through the same compiler binary torch chose, so the
    toolchain stays exactly the one the cell seal names."""
    return shlex.join([argv[0], "-r", "-nostdlib", "-o", str(target),
                       *(str(o) for o in objects)])


def split_and_compile(cmd_line: str, cwd: str,
                      run: Callable[[str, str], None]) -> Optional[SplitOutcome]:
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
        _emit(SplitOutcome(False, f"unreadable source: {exc}",
                           source=str(source)), lever=_RUNIMPL)
        return None
    split = runimpl.split_run_impl(text)
    if not split.applied:
        _emit(SplitOutcome(False, split.reason, source=str(source)))
        return None

    stem = source.name[: -len(".cpp")]
    units: List[Tuple[Path, Path]] = []
    for name, body in [("main", split.main),
                       *((f"part{i}", p) for i, p in enumerate(split.parts))]:
        cpp = source.with_name(f"{stem}.pgw811.{name}.cpp")
        cpp.write_text(body)
        units.append((cpp, cpp.with_suffix(".o")))

    jobs = host_compile_jobs(len(units))
    errors: List[BaseException] = []

    def build(unit: Tuple[Path, Path]) -> None:
        run(_retarget(argv, src_at, obj_at, unit[0], unit[1]), cwd)

    if jobs == 1:
        for unit in units:
            build(unit)
    else:
        with cf.ThreadPoolExecutor(max_workers=jobs) as pool:
            for future in cf.as_completed([pool.submit(build, u) for u in units]):
                err = future.exception()
                if err is not None:
                    errors.append(err)
        if errors:
            raise errors[0]

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
            f"re-derived), {jobs} concurrent compiles, partial-linked"
        ),
    )


def install() -> bool:
    """Install the transform on torch's single host-compile funnel.

    ``CppBuilder.build()`` reaches ``run_compile_cmd`` through the module
    global, so wrapping that name covers every host compile inductor
    performs and nothing else.

    IDENTITY: this changes neither the compiler, its flags, any inductor
    config, any sealed torch flag, nor any loaded library — so
    ``env_seal.effective_seal()`` and therefore the ``env_seal`` key axis
    are untouched. This module is imported only from the mint path, which
    is outside ``compile_cache.static_code_closure``'s entrypoints, so the
    ``code_closure`` axis is untouched too. No cell is re-keyed. (Both
    facts are asserted by tests/test_aot_wrapper_split_pgw793.py.)

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
        # pgw#754's clamp is asserted at the CONFIG level by
        # host_isa.impose(); this is the argv-level assertion nothing had.
        # `run_compile_cmd` is torch's single host-compile funnel, so every
        # object a mint produces passes through here exactly once. It raises
        # rather than degrading: an unclamped object is not slower, it is
        # unportable, and pgw#754 is a SIGILL-class defect.
        try:
            host_isa.assert_command_is_clamped(shlex.split(cmd_line))
        except ValueError:
            pass                      # unparseable argv is not an ISA claim

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
            original(cmd_line, cwd)          # not a single-source C++ compile
            return
        _emit(outcome)

        # pgw#811 splits run_impl out of whatever source the ctor split left
        # — inductor's own when v1 declined, v1's regrouped one when it did
        # not. It subsumes the single compile: when it fires it drives every
        # compile itself and there is no monolith left to run.
        if not os.environ.get(DISABLE_V2_ENV, "").strip():
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
                    "aot-wrapper-split: pgw#811 split failed (%s); recompiling "
                    "the whole wrapper", type(exc).__name__, exc_info=True)
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


__all__ = [
    "CHUNK",
    "EVENT",
    "MIN_STATEMENTS",
    "DISABLE_V2_ENV",
    "JOBS_ENV",
    "SERVING_HEADROOM_CPUS",
    "SplitOutcome",
    "host_compile_jobs",
    "install",
    "split_and_compile",
    "split_constants_constructor",
    "transform_command",
]
