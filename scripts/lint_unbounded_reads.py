#!/usr/bin/env python3
"""A length read off external bytes may not size a read until something has
bounded it, and a stream may not be copied without a bound checked INSIDE the
loop.

RULE 1. Inside a function, if a name is bound from a binary length prefix —

    (n,) = struct.unpack("<Q", ...)      n = struct.unpack(...)[0]
    n = int.from_bytes(prefix, "little")

— and that same name is later used to size a read or an allocation —

    f.read(n)      os.read(fd, n)      bytearray(n)      b"\\x00" * n

— then the function must also bound it, in one of two ways:

  1. call a sanctioned validator on it, e.g. ``header_len_ok(n)``; or
  2. carry an explicit justification comment, on the read line or in the
     comment block just above it (see JUSTIFICATION_LOOKBACK):
         ``# bound-justified: <why this cannot run away>``

Two ways, deliberately: §4.24 asks for a bound OR a stated reason none is
needed, and a guard that only accepts the bound would push honest exemptions
into silence.

RULE 2 — THE STREAMING-COPY LOOP. A check AFTER the loop is not a bound: the
bytes are already on disk, and the pod can be dead before the comparison runs.
So a loop that pulls a stream and sinks the bytes must, in the loop body,

  * hold a counter accumulated with ``total += len(chunk)``; and
  * compare that counter directly (``total > cap``, ``cap < total``) somewhere
    in the same loop body,

or carry a ``# bound-justified:`` comment, exactly as rule 1 does. Loops that
delegate to ``bounded_stream.copy_bounded`` have no loop here to inspect, which
is the point of that helper existing.

The counter must be a DIRECT operand of the compare: a downloader may compare
``downloaded - last_log >= log_every`` for progress logging, and a rule that
accepted a counter buried in a sub-expression would have taken that for the
bound.

LIMITS — a green run is not proof of absence. Rule 1 matches only a binary
length prefix feeding a read in the SAME function; a size arriving via a return
value, an attribute, a JSON field or a DB column is out of reach. Rule 2 sees
loops, not ``shutil.copyfileobj`` (local file-to-file work in this tree) nor a
sizeless ``.read()`` on a tar member (sites another remainder owns). Widening
either rule means dataflow analysis, and a noisy guard gets suppressed, which
is worse than a narrow one that holds.

Usage: scripts/lint_unbounded_reads.py   (exit 1 on any finding)
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src" / "gen_worker"

# Calls whose result is a length taken from bytes the process did not compute.
LENGTH_SOURCES = {"unpack", "unpack_from", "from_bytes"}

# Names that, called on the length, count as bounding it. A project adds to this
# set when it grows another validator — it is deliberately explicit.
SANCTIONED_VALIDATORS = {"header_len_ok"}

# Call/expression shapes that turn a length into memory or IO.
SIZING_METHODS = {"read", "readinto", "recv", "pread"}

JUSTIFICATION = "bound-justified:"

# How far above the read line a justification comment may sit.
JUSTIFICATION_LOOKBACK = 8

# ---------------------------------------------------------------------------
# Rule 2: the streaming-copy loop
# ---------------------------------------------------------------------------

# Iterators that yield a remote body a block at a time.
STREAM_ITERATORS = {"iter_content", "iter_bytes", "iter_chunks", "iter_raw"}

# Blocking block reads that, inside a `while`, form the same loop by hand.
STREAM_READERS = {"read", "read1", "recv", "recv_into", "readinto"}

# Calls that put the block somewhere it accumulates: a file, a list, a buffer.
SINK_METHODS = {"write", "writelines", "append", "extend", "send", "put"}

# Free functions that write a block through to storage. Named explicitly rather
# than matched by shape — a bare call is not evidence of a sink.
SINK_FUNCTIONS = {"_pwrite_all", "pwrite"}


def _iter_targets(node: ast.AST):
    """Names bound by an assignment target, including tuple unpacking."""
    if isinstance(node, ast.Name):
        yield node.id
    elif isinstance(node, (ast.Tuple, ast.List)):
        for elt in node.elts:
            yield from _iter_targets(elt)


def _call_func_name(call: ast.Call) -> str:
    f = call.func
    if isinstance(f, ast.Attribute):
        return f.attr
    if isinstance(f, ast.Name):
        return f.id
    return ""


def _is_length_source(value: ast.AST) -> bool:
    """struct.unpack(...)  /  int.from_bytes(...)  possibly subscripted."""
    node = value
    if isinstance(node, ast.Subscript):
        node = node.value
    return isinstance(node, ast.Call) and _call_func_name(node) in LENGTH_SOURCES


class FunctionScan(ast.NodeVisitor):
    def __init__(self) -> None:
        self.tainted: set[str] = set()
        self.validated: set[str] = set()
        self.sized: list[tuple[str, int]] = []  # (name, lineno)

    def visit_Assign(self, node: ast.Assign) -> None:
        if _is_length_source(node.value):
            for t in node.targets:
                self.tainted.update(_iter_targets(t))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = _call_func_name(node)
        # A sanctioned validator applied to a tainted name bounds it.
        if name in SANCTIONED_VALIDATORS:
            for a in node.args:
                if isinstance(a, ast.Name):
                    self.validated.add(a.id)
        # A read/alloc sized by a tainted name is the site we care about.
        if name in SIZING_METHODS:
            for a in node.args:
                if isinstance(a, ast.Name):
                    self.sized.append((a.id, node.lineno))
        self.generic_visit(node)


ORDERING_OPS = (ast.Lt, ast.LtE, ast.Gt, ast.GtE)


def _local_file_handles(fn: ast.AST) -> set[str]:
    """Names this function bound to a file IT opened.

    A handle the function opened itself is local IO, not an external stream:
    every upload path in this repo reads its own artifact back in blocks, and
    those loops are the bulk of what a `.read()`-shaped rule would flag. The
    remote sources are the ones the function was HANDED (a socket parameter, a
    guarded-stream context manager).
    """
    handles: set[str] = set()

    def _opens(value: ast.AST) -> bool:
        return isinstance(value, ast.Call) and _call_func_name(value) == "open"

    for sub in _walk_scope(fn):
        if isinstance(sub, ast.withitem):
            if _opens(sub.context_expr) and isinstance(sub.optional_vars, ast.Name):
                handles.add(sub.optional_vars.id)
        elif isinstance(sub, ast.Assign) and _opens(sub.value):
            for t in sub.targets:
                handles.update(_iter_targets(t))
    return handles


def _walk_scope(node: ast.AST):
    """`ast.walk` that stops at a nested function — one scope's own statements."""
    stack = list(ast.iter_child_nodes(node))
    while stack:
        cur = stack.pop()
        yield cur
        if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        stack.extend(ast.iter_child_nodes(cur))


def _stream_block_name(node: ast.AST, local_handles: set[str]) -> "str | None":
    """The name each iteration binds to the block of external bytes, or None.

    None means "not an external stream loop" — either the loop is not one of
    the two shapes, or it reads a handle this function opened itself.
    """
    if isinstance(node, ast.For):
        it = node.iter
        if (
            isinstance(it, ast.Call)
            and _call_func_name(it) in STREAM_ITERATORS
            and isinstance(node.target, ast.Name)
        ):
            return node.target.id
        return None
    if isinstance(node, ast.While):
        for sub in ast.walk(node):
            if not (
                isinstance(sub, ast.Assign)
                and isinstance(sub.value, ast.Call)
                and isinstance(sub.value.func, ast.Attribute)
                and sub.value.func.attr in STREAM_READERS
                and sub.value.args  # a sized block read, not a slurp
            ):
                continue
            src = sub.value.func.value
            if isinstance(src, ast.Name) and src.id in local_handles:
                continue
            if len(sub.targets) == 1 and isinstance(sub.targets[0], ast.Name):
                return sub.targets[0].id
    return None


def _loop_condition_bounds_it(node: ast.AST) -> bool:
    """`while remaining > 0` / `while len(buf) < n` — the test IS the bound.

    A loop whose own condition is an ordering comparison stops on a size, and
    reading its body for a second bound would flag the budget shape that is
    already correct. `while True`, `while b"\\n" not in buf` and
    `while terminal is None` bound nothing and are not covered by this.
    """
    if not isinstance(node, ast.While):
        return False
    test = node.test
    return isinstance(test, ast.Compare) and any(
        isinstance(op, ORDERING_OPS) for op in test.ops
    )


def _sink_names(body: list[ast.stmt], block: str) -> set[str]:
    """The things the loop puts THAT BLOCK into, if any.

    The block itself must be the argument. `chunks.append(ChunkPlan(...))`
    accumulates plans derived from the bytes and drops the bytes; `f.write(
    chunk)`, `buf.extend(chunk)` and `_pwrite_all(fd, b, pos)` accumulate the
    bytes themselves, and only the second kind can run away. An empty result
    means the loop does not accumulate and needs no bound.
    """
    sinks: set[str] = set()
    for stmt in body:
        for sub in ast.walk(stmt):
            if isinstance(sub, ast.Call):
                name = _call_func_name(sub)
                if name in SINK_METHODS or name in SINK_FUNCTIONS:
                    if any(isinstance(a, ast.Name) and a.id == block for a in sub.args):
                        target = sub.func.value if isinstance(sub.func, ast.Attribute) else None
                        sinks.add(target.id if isinstance(target, ast.Name) else name)
            # `buf += chunk` — the quadratic accumulate the frame reader had.
            if (
                isinstance(sub, ast.AugAssign)
                and isinstance(sub.op, ast.Add)
                and isinstance(sub.value, ast.Name)
                and sub.value.id == block
                and isinstance(sub.target, ast.Name)
            ):
                sinks.add(sub.target.id)
    return sinks


def _byte_counters(body: list[ast.stmt]) -> set[str]:
    """Names accumulated with `n += len(...)` — the running byte count."""
    counters: set[str] = set()
    for stmt in body:
        for sub in ast.walk(stmt):
            if (
                isinstance(sub, ast.AugAssign)
                and isinstance(sub.op, ast.Add)
                and isinstance(sub.target, ast.Name)
                and isinstance(sub.value, ast.Call)
                and _call_func_name(sub.value) == "len"
            ):
                counters.add(sub.target.id)
    return counters


def _is_trivial_limit(node: ast.AST) -> bool:
    """`> 0` / `>= 1` is an emptiness test, never a bound."""
    return isinstance(node, ast.Constant) and node.value in (0, 1, None, True, False)


def _measures_growth(node: ast.AST, counters: set[str], sinks: set[str]) -> bool:
    """Does this side of a comparison hold how much has arrived so far?

    Two spellings, because the repo legitimately uses both. An accumulated
    counter (`total += len(chunk)`) must be a DIRECT operand — a counter buried
    in a sub-expression is a progress log (`downloaded - last_log >=
    log_every`), not a bound. A `len(sink)` reading may sit inside an
    expression (`len(buf) + len(chunk) > cap`), because the `len()` call is
    itself the unambiguous measurement, and a buffer that is DRAINED as it goes
    cannot be tracked by an accumulator at all.
    """
    if isinstance(node, ast.Name) and node.id in counters:
        return True
    for sub in ast.walk(node):
        if (
            isinstance(sub, ast.Call)
            and _call_func_name(sub) == "len"
            and len(sub.args) == 1
            and isinstance(sub.args[0], ast.Name)
            and sub.args[0].id in (counters | sinks)
        ):
            return True
    return False


def _counter_is_bounded(body: list[ast.stmt], counters: set[str], sinks: set[str]) -> bool:
    """Is the arrived-so-far quantity compared against a real limit in-loop?"""
    for stmt in body:
        for sub in ast.walk(stmt):
            if not isinstance(sub, ast.Compare):
                continue
            for op, right in zip(sub.ops, sub.comparators):
                if isinstance(op, (ast.Gt, ast.GtE)):
                    grew, limit = sub.left, right
                elif isinstance(op, (ast.Lt, ast.LtE)):
                    grew, limit = right, sub.left
                else:
                    continue
                if _measures_growth(grew, counters, sinks) and not _is_trivial_limit(limit):
                    return True
    return False


def _justified_near(lines: list[str], lineno: int) -> bool:
    start = max(0, lineno - 1 - JUSTIFICATION_LOOKBACK)
    return any(JUSTIFICATION in ln for ln in lines[start:lineno])


def _scan_stream_loops(path: Path, tree: ast.AST, lines: list[str]) -> list[str]:
    # Handles are collected per scope and unioned DOWN the enclosing chain: a
    # `with open(...)` in an enclosing function still names local IO inside a
    # closure it defines, and a name opened in an unrelated function must not
    # silence a socket that happens to share it.
    parents: dict[int, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[id(child)] = node
    handles_of: dict[int, set[str]] = {}

    def _handles_for(node: ast.AST) -> set[str]:
        scopes: list[ast.AST] = []
        cur: "ast.AST | None" = node
        while cur is not None:
            if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)):
                scopes.append(cur)
            cur = parents.get(id(cur))
        acc: set[str] = set()
        for scope in scopes:
            if id(scope) not in handles_of:
                handles_of[id(scope)] = _local_file_handles(scope)
            acc |= handles_of[id(scope)]
        return acc

    findings: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.For, ast.While)):
            continue
        findings.extend(_scan_one_loop(path, node, lines, _handles_for(node)))
    return findings


def _scan_one_loop(
    path: Path, node: ast.AST, lines: list[str], local_handles: set[str]
) -> list[str]:
    block = _stream_block_name(node, local_handles)
    if block is None:
        return []
    body = node.body  # type: ignore[union-attr]
    sinks = _sink_names(body, block)
    if not sinks:
        return []  # counts, hashes or discards; the bytes do not pile up
    if _loop_condition_bounds_it(node):
        return []  # the loop's own test is the bound
    if _justified_near(lines, node.lineno):
        return []
    # A justification may also sit inside the loop, next to the write.
    end = getattr(node, "end_lineno", node.lineno) or node.lineno
    if any(JUSTIFICATION in ln for ln in lines[node.lineno - 1 : end]):
        return []
    rel = path.relative_to(SRC_ROOT.parent.parent)
    counters = _byte_counters(body)
    if _counter_is_bounded(body, counters, sinks):
        return []
    how = (
        f"`{'`, `'.join(sorted(counters))}` counts the bytes but nothing compares it"
        if counters
        else "it keeps no running byte count"
    )
    return [
        f"{rel}:{node.lineno}: this loop streams external bytes into "
        f"`{'`, `'.join(sorted(sinks))}` and {how} inside the loop — any size check "
        f"happens only after every byte is already written.\n"
        f"    Fix: use `bounded_stream.copy_bounded`, or compare the bytes-so-far "
        f"against the declared size IN the loop body, or state why no bound is needed "
        f"with a `# {JUSTIFICATION} ...` comment."
    ]


def scan_file(path: Path, src: str) -> list[str]:
    try:
        tree = ast.parse(src, filename=str(path))
    except SyntaxError as exc:
        return [f"{path}: unparseable ({exc})"]

    lines = src.splitlines()
    findings: list[str] = []

    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        scan = FunctionScan()
        for stmt in fn.body:
            scan.visit(stmt)
        for name, lineno in scan.sized:
            if name not in scan.tainted:
                continue
            if name in scan.validated:
                continue
            # The justification may sit on the read line or in the comment
            # block immediately above it. A one-line-only rule would push a
            # real reason into a cramped trailing comment, which is how
            # justifications become noise nobody reads.
            start = max(0, lineno - 1 - JUSTIFICATION_LOOKBACK)
            window = lines[start:lineno]
            if any(JUSTIFICATION in ln for ln in window):
                continue
            rel = path.relative_to(SRC_ROOT.parent.parent)
            findings.append(
                f"{rel}:{lineno}: `{name}` is a length read off external bytes and sizes a "
                f"read in `{fn.name}()` with nothing bounding it.\n"
                f"    Fix: call a sanctioned validator ({', '.join(sorted(SANCTIONED_VALIDATORS))}) "
                f"on it, or state why none is needed with a `# {JUSTIFICATION} ...` comment."
            )

    findings.extend(_scan_stream_loops(path, tree, lines))
    return findings


def main() -> int:
    findings: list[str] = []
    for py in sorted(SRC_ROOT.rglob("*.py")):
        findings.extend(scan_file(py, py.read_text(encoding="utf-8")))

    if findings:
        print("unbounded-read guard: an external length sizes a read with no bound\n")
        for f in findings:
            print(f)
        print(
            "\npgw#1013 / th#1662: the bounds census could not see these — it enumerated "
            "bounds that exist, not sites that need one."
        )
        return 1

    print("unbounded-read guard: OK — every external length feeding a read, and every\n"
          "streaming copy loop, is bounded or justified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
