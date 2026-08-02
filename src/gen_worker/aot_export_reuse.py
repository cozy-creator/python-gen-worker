"""pgw#847: export ONCE per module, re-specialize per shape row — behind a
fail-closed gate that must PROVE byte-identity or fall back.

**The invariant.** A cell's N entries are one module traced at N shape rows.
An `ExportedProgram`'s ``graph_module.code`` is byte-identical across those
rows — the row lives entirely in node metadata — so re-deriving that metadata
over the same graph reproduces what a fresh `torch.export.export` would have
produced. Measured: wrapper.cpp, kernel.cpp and the **linked `.so`** all
byte-identical, with `torch.export.export` monkeypatched to raise for the
whole reuse arm so the equality could not be accidental.

**Why it is worth doing.** `aot_mint._export_entry` exports once per declared
class row, SERIALLY, in the mint parent — deliberately, since it runs against
the one live pipeline on the one card. sdxl is 36 entries at a banked
``export_s`` of 37.8 s, so that loop is ~22 minutes of mint wall that pgw#809's
K-wide pool divides by ONE.

**Why it is gated.** The invariant is a property of the MODULE, not a law. A
family whose Python control flow branches on a size traces a *different* graph
per row, and reuse would then compile the wrong kernels under the right name —
silently, since the artifact is well-formed. That is the pgw#812 failure shape
and the reason pgw#846 exists. So:

* the flag is **OFF by default** (`GEN_WORKER_AOT_EXPORT_REUSE`);
* the gate is **per family per mint**, never memoised across families — a
  verdict lives on the :class:`ReuseState` the mint creates and dies with it;
* **absence of evidence is a fallback, never a pass** — every failure mode
  (exception, missing artifact, empty digest set, unsupported input spec)
  declines to full per-row export;
* the proof compares **every generated C++ source AND the host command**,
  which on this pinned toolchain determine the object bit for bit — measured
  on the real 6.3 MB sdxl wrapper TU: same source, same command, same build
  path recompiles to a byte-identical object, and a different build path moves
  156 bytes of 15 MB (the embedded path) and nothing else. Both arms stop
  BEFORE their 180 s `g++` and run CONCURRENTLY, which is what keeps the gate
  from costing more than the reuse saves.

**Why the gate is cheap on purpose.** Two full serial entry compiles cost
~780 s against a saving whose measured range BOTTOMS at ~660 s — a change
whose sign depended on a quantity nobody had measured. Stopping before `g++`
and running the arms concurrently takes the gate to roughly one codegen, so
the sign is positive across the whole range instead of at its middle.

Related: [[#847]] (this sweep), [[#846]] (the ruling this serves), [[#793]]
(the per-entry budget), [[#809]] (the pool that never covered the export).
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

#: OFF by default. pgw#846's rule is that the compiled artifact is the fixed
#: point; this changes how a program is DERIVED, so it ships dark until a real
#: family's mint has run the gate on a pod.
ENV_FLAG = "GEN_WORKER_AOT_EXPORT_REUSE"

#: A base needs at least this many rows behind it before reuse can pay: row 0
#: is the base, row 1 is the gate's evidence, so reuse starts at row 2.
MIN_ROWS = 3

#: Files whose trailing `// Compile cmd` trailer records temp paths rather than
#: code. Everything else is compared raw, byte for byte.
_TRAILER_SUFFIXES = (".cpp", ".h", ".hpp")
_TRAILER = b"// Compile cmd"


def enabled() -> bool:
    """True only on an explicit opt-in. Anything else is OFF."""
    return os.environ.get(ENV_FLAG, "").strip().lower() in (
        "1", "true", "yes", "on")


class ReuseUnproven(RuntimeError):
    """The gate could not prove equality. Always a fallback, never a failure."""


class _StopBeforeHostCompile(BaseException):
    """Raised inside the gate's arms at the wrapper's `g++`.

    A `BaseException`, deliberately: `compile_entry_files` and inductor both
    catch `Exception` broadly, and this must reach the gate rather than be
    laundered into a compile failure.
    """


@dataclass(frozen=True)
class GateVerdict:
    """Why reuse was admitted or declined, in words a reader can act on."""

    admitted: bool
    reason: str
    code_equal: Optional[bool] = None
    artifacts_equal: Optional[bool] = None
    own_digests: Mapping[str, str] = field(default_factory=dict)
    reuse_digests: Mapping[str, str] = field(default_factory=dict)
    gate_s: float = 0.0

    def telemetry(self) -> Dict[str, Any]:
        return {
            "admitted": self.admitted,
            "reason": self.reason,
            "code_equal": self.code_equal,
            "artifacts_equal": self.artifacts_equal,
            "files": sorted(self.own_digests),
            "gate_s": round(self.gate_s, 2),
        }


def _digest(path: Path) -> str:
    data = path.read_bytes()
    if path.name.endswith(_TRAILER_SUFFIXES):
        data = data.split(_TRAILER)[0]
    return hashlib.sha256(data).hexdigest()


def respecialize(base: Any, args: Sequence[Any], kwargs: Mapping[str, Any]) -> Any:
    """A REAL ``ExportedProgram`` for a new shape row, from ``base``'s graph.

    Deep-copies the lifted graph module, re-runs fake-tensor propagation with
    this row's inputs (parameters, buffers and constants supplied from the
    base's own state dict, so nothing is invented), and rebuilds an
    ``ExportedProgram`` through torch's own ``_update``. The result survives
    ``torch.export.save``/``load`` — which matters, because that round trip is
    how pgw#809's pool hands an entry to its child.

    Raises :class:`ReuseUnproven` on anything it cannot place exactly; the
    caller falls back to a full export.
    """
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.passes.fake_tensor_prop import FakeTensorProp

    graph_module = copy.deepcopy(base.graph_module)
    signature = base.graph_signature
    state = dict(base.state_dict)
    constants = dict(base.constants)
    user = list(args)
    flat: list[Any] = []
    for spec in signature.input_specs:
        target = getattr(spec, "target", None)
        kind = str(getattr(spec, "kind", ""))
        if target is not None and target in state:
            flat.append(state[target])
        elif target is not None and target in constants:
            flat.append(constants[target])
        elif target is None or "USER_INPUT" in kind:
            if not user:
                raise ReuseUnproven(
                    f"the base graph wants more user inputs than this row "
                    f"supplies ({len(args)} given)")
            flat.append(user.pop(0))
        else:
            raise ReuseUnproven(
                f"unplaceable graph input {target!r} (kind {kind!r}) — the "
                f"base's state dict and constants do not carry it")
    if user:
        raise ReuseUnproven(
            f"this row supplies {len(user)} more input(s) than the base "
            f"graph accepts")

    mode = FakeTensorMode(allow_non_fake_inputs=True)
    with mode:
        fake = tuple(
            mode.from_tensor(t) if isinstance(t, torch.Tensor) else t
            for t in flat)
        FakeTensorProp(graph_module, mode=mode).propagate(*fake)

    program = base._update(graph_module, signature)
    program.example_inputs = (tuple(args), dict(kwargs or {}))
    return program


def _normalise(text: str, cache_dir: Path) -> str:
    """A compile command with its build LOCATION removed.

    Two arms of the gate run in different directories, so the paths differ by
    construction. Everything else — compiler, every flag, every include root —
    must be identical, because "same source implies same object" is only true
    of the same COMMAND.
    """
    return text.replace(str(cache_dir), "<cache>")


def _capture_codegen(
    program: Any, entry: str, cache_dir: Path,
    inductor_configs: Optional[Mapping[str, Any]],
) -> Dict[str, str]:
    """Run the production compile seam and STOP at the wrapper's `g++`.

    Returns a digest of every generated C++ source plus the normalised host
    command, which together determine the object bit for bit — measured on
    this program's real 6.3 MB sdxl wrapper TU: identical source, identical
    command and the same build path compile to a byte-identical object, and
    a different build path moves 156 bytes of 15 MB (the embedded path) and
    nothing else. So comparing THIS is comparing the artifact, at 54 % of the
    cost, and the 180 s `g++` the gate would otherwise pay twice is skipped.

    Deliberately NOT a claim that g++ is reproducible in general: it is a
    measured property of one pinned toolchain, re-checked by
    `test_source_and_command_determine_the_object`.
    """
    from . import aot_mint, aot_wrapper_split
    from torch._inductor import cpp_builder

    shutil.rmtree(cache_dir, ignore_errors=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    captured: Dict[str, str] = {}

    # Installed FIRST so this tap ends up outermost and sees inductor's own
    # command rather than a transformed one.
    aot_wrapper_split.install()
    original = cpp_builder.run_compile_cmd

    def tap(cmd_line: str, cwd: str) -> None:
        argv = shlex.split(cmd_line)
        sources = [t for t in argv if t.endswith(".cpp")]
        if "-c" in argv and any(s.endswith(".wrapper.cpp") for s in sources):
            captured["__cmd__"] = hashlib.sha256(
                _normalise(cmd_line, cache_dir).encode()).hexdigest()
            raise _StopBeforeHostCompile()
        original(cmd_line, cwd)

    previous = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir)
    cpp_builder.run_compile_cmd = tap
    try:
        import torch._inductor.codecache as codecache

        for name in ("cache_dir", "default_cache_dir"):
            resolver = getattr(codecache, name, None)
            clear = getattr(resolver, "cache_clear", None)
            if clear is not None:
                clear()
        aot_mint.compile_entry_files(
            program, entry, inductor_configs=inductor_configs)
    except _StopBeforeHostCompile:
        pass
    finally:
        cpp_builder.run_compile_cmd = original
        if previous is None:
            os.environ.pop("TORCHINDUCTOR_CACHE_DIR", None)
        else:
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = previous

    if "__cmd__" not in captured:
        raise ReuseUnproven(
            "codegen never reached the wrapper's host compile — no evidence "
            "to compare")
    for path in sorted(cache_dir.rglob("*.cpp")):
        # `.wrapper.pgw793.cpp` is aot_wrapper_split's own output, a
        # deterministic function of the wrapper it is skipped in favour of;
        # digesting both would double-count one source.
        if path.name.endswith(".pgw793.cpp"):
            continue
        # The build LOCATION is normalised out of the TEXT, not only out of
        # the command: inductor writes the wrapper's absolute path into
        # `kernel.cpp`'s first line ("Triton kernels are embedded as comments
        # in <path>"), and the arms necessarily run in different directories.
        # That path is not code — measured, it accounts for the entire
        # difference between two objects built from identical source.
        body = _normalise(
            path.read_text().split(_TRAILER.decode())[0], cache_dir)
        captured["".join(path.suffixes[-2:]) or path.suffix] = \
            hashlib.sha256(body.encode()).hexdigest()
    if len(captured) < 2:
        raise ReuseUnproven("codegen emitted no C++ to compare")
    return captured


#: The gate runs each arm in its OWN fresh interpreter, concurrently. Not a
#: thread (inductor's codegen is single-threaded Python, so threads buy
#: nothing) and not a fork (banned after CUDA init — pgw#784). Deliberately
#: NOT pgw#809's `EntryCompilePool`: that pool's contract is "compile one
#: entry", and threading a measurement mode through it would couple the gate
#: to the production child for no gain.
_ARM_ENTRYPOINT = "gen_worker.aot_export_reuse"


def _arm_child_main(job_path: str) -> int:
    """One gate arm, in its own process: codegen, stop, write digests."""
    import torch

    from . import host_isa

    job = json.loads(Path(job_path).read_text())
    out = Path(job["out"])
    try:
        host_isa.impose()
        program = torch.export.load(job["program"])
        digests = _capture_codegen(
            program, job["entry"], Path(job["cache_dir"]),
            job.get("inductor_configs") or None)
        out.write_text(json.dumps({"ok": True, "digests": digests}))
    except BaseException as exc:  # noqa: BLE001
        out.write_text(json.dumps(
            {"ok": False, "error": f"{type(exc).__name__}: {exc}"}))
        return 1
    return 0


def _run_arms(
    arms: Sequence[Tuple[str, Any]], *, workdir: Path,
    inductor_configs: Optional[Mapping[str, Any]],
) -> Dict[str, Dict[str, str]]:
    """Run every arm CONCURRENTLY in its own interpreter; raise on any failure."""
    import torch

    procs: List[Tuple[str, subprocess.Popen, Path]] = []
    for name, program in arms:
        slot = workdir / name
        slot.mkdir(parents=True, exist_ok=True)
        program_path = slot / "program.pt2"
        torch.export.save(program, program_path)
        out = slot / "digests.json"
        job = slot / "job.json"
        job.write_text(json.dumps({
            "program": str(program_path), "entry": name,
            "cache_dir": str(slot / "cache"), "out": str(out),
            "inductor_configs": dict(inductor_configs or {}),
        }))
        procs.append((name, subprocess.Popen(
            [sys.executable, "-m", _ARM_ENTRYPOINT, str(job)],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            start_new_session=True), out))

    results: Dict[str, Dict[str, str]] = {}
    failures: List[str] = []
    for name, proc, out in procs:
        _, stderr = proc.communicate()
        if not out.is_file():
            failures.append(
                f"{name}: arm produced no result "
                f"(rc={proc.returncode}) {stderr.decode()[-400:]}")
            continue
        payload = json.loads(out.read_text())
        if not payload.get("ok"):
            failures.append(f"{name}: {payload.get('error')}")
            continue
        results[name] = payload["digests"]
    if failures:
        raise ReuseUnproven("; ".join(failures))
    return results


def prove(
    base: Any, witness: Any, witness_args: Sequence[Any],
    witness_kwargs: Mapping[str, Any], *, workdir: Path, entry: str,
    inductor_configs: Optional[Mapping[str, Any]] = None,
) -> GateVerdict:
    """Decide whether ``base``'s graph may serve other rows of this family.

    ``witness`` is a FULL export of a DIFFERENT row than the base's. Two
    checks, both required:

    1. **structural** — ``base.graph_module.code`` equals the witness's. This
       is what a family branching on a size fails, and it is nearly free.
    2. **artifact** — a re-specialization of the base at the witness's row
       generates byte-identical C++ under a byte-identical host command,
       which on this pinned toolchain determines the object bit for bit.

    Both arms run CONCURRENTLY, each stopping before its 180 s `g++`. That is
    what keeps the gate cheap enough that reuse cannot be net-negative: two
    full serial entry compiles were ~780 s against a saving whose LOW end is
    ~660 s, i.e. a change whose sign depended on an unmeasured quantity.

    Any exception, any missing evidence and any empty digest set DECLINES.
    """
    started = time.monotonic()
    try:
        code_equal = base.graph_module.code == witness.graph_module.code
    except Exception as exc:  # noqa: BLE001
        return GateVerdict(
            False, f"could not read the graph text: {type(exc).__name__}: "
                   f"{exc}", gate_s=time.monotonic() - started)
    if not code_equal:
        return GateVerdict(
            False,
            "the exported graph TEXT differs between two rows of this family "
            "— the module's structure moves with the shape row, so one export "
            "cannot serve another. Falling back to a full export per row.",
            code_equal=False, gate_s=time.monotonic() - started)

    gate_dir = Path(workdir) / "pgw847-gate"
    own: Dict[str, str] = {}
    reuse: Dict[str, str] = {}
    try:
        candidate = respecialize(base, witness_args, witness_kwargs)
        results = _run_arms(
            (("gate-own", witness), ("gate-reuse", candidate)),
            workdir=gate_dir, inductor_configs=inductor_configs)
        own, reuse = results["gate-own"], results["gate-reuse"]
    except Exception as exc:  # noqa: BLE001
        return GateVerdict(
            False,
            f"the gate could not build its evidence "
            f"({type(exc).__name__}: {exc}) — declining, because an unproven "
            f"reuse is a wrong artifact waiting to happen",
            code_equal=True, gate_s=time.monotonic() - started)
    finally:
        shutil.rmtree(gate_dir, ignore_errors=True)

    if not own or not reuse:
        return GateVerdict(
            False, "one of the gate's arms emitted no files at all",
            code_equal=True, artifacts_equal=False, own_digests=own,
            reuse_digests=reuse, gate_s=time.monotonic() - started)
    if own != reuse:
        differing = sorted(
            k for k in set(own) | set(reuse) if own.get(k) != reuse.get(k))
        return GateVerdict(
            False,
            f"re-specializing the base graph did NOT reproduce a full "
            f"export's artifact: {differing} differ",
            code_equal=True, artifacts_equal=False, own_digests=own,
            reuse_digests=reuse, gate_s=time.monotonic() - started)
    return GateVerdict(
        True,
        f"one export serves this family: graph text equal, and all "
        f"{len(own) - 1} generated source(s) plus the host command "
        f"byte-identical between a full export and a re-specialization",
        code_equal=True, artifacts_equal=True, own_digests=own,
        reuse_digests=reuse, gate_s=time.monotonic() - started)


class ReuseState:
    """One mint's reuse bookkeeping. **Never shared between mints or families.**

    Created per mint by :mod:`aot_mint`; a verdict reached for one family dies
    with the object. There is deliberately no module-level cache anywhere in
    this file — a memoised verdict is a verdict about a module nobody checked.
    """

    def __init__(
        self, workdir: Path, *,
        inductor_configs: Optional[Mapping[str, Any]] = None,
        active: Optional[bool] = None,
    ) -> None:
        self.workdir = Path(workdir)
        self.inductor_configs = dict(inductor_configs or {})
        self.active = enabled() if active is None else bool(active)
        self._bases: Dict[Any, Any] = {}
        self._seen: Dict[Any, int] = {}
        self._verdicts: Dict[Any, GateVerdict] = {}
        #: telemetry the mint publishes; nothing reads it to decide anything
        self.events: list[Dict[str, Any]] = []
        self.reused = 0
        self.exported = 0
        self.respecialize_s = 0.0

    def verdict(self, key: Any) -> Optional[GateVerdict]:
        return self._verdicts.get(key)

    def program(
        self, key: Any, *, entry: str, rows: int,
        args: Sequence[Any], kwargs: Mapping[str, Any],
        full_export: Any,
    ) -> Tuple[Any, str]:
        """Return ``(program, how)`` for one row.

        ``full_export`` is a zero-argument callable performing the real
        ``torch.export.export`` — called for the base row, for the gate's
        witness row, and for every row whenever reuse is not admitted.
        """
        if not self.active or rows < MIN_ROWS:
            self.exported += 1
            return full_export(), "full"

        seen = self._seen.get(key, 0)
        self._seen[key] = seen + 1

        if seen == 0:
            program = full_export()
            self._bases[key] = program
            self.exported += 1
            return program, "full"

        if seen == 1:
            program = full_export()
            self.exported += 1
            base = self._bases.get(key)
            if base is None:
                return program, "full"
            gate = prove(
                base, program, args, kwargs, workdir=self.workdir,
                entry=entry, inductor_configs=self.inductor_configs)
            self._verdicts[key] = gate
            self.events.append({"key": str(key), **gate.telemetry()})
            logger.info(
                "aot-mint: pgw#847 export-reuse gate for %s: %s — %s",
                key, "ADMITTED" if gate.admitted else "DECLINED", gate.reason)
            return program, "full"

        decided = self._verdicts.get(key)
        base = self._bases.get(key)
        if decided is None or not decided.admitted or base is None:
            self.exported += 1
            return full_export(), "full"
        t0 = time.monotonic()
        try:
            program = respecialize(base, args, kwargs)
        except Exception as exc:  # noqa: BLE001
            # A gate that admitted the family does not license a row this
            # code cannot place exactly. Fall back, loudly, per row.
            logger.warning(
                "aot-mint: pgw#847 export-reuse fell back to a full export "
                "for %r: %s: %s", entry, type(exc).__name__, exc)
            self.exported += 1
            return full_export(), "full"
        self.reused += 1
        self.respecialize_s += round(time.monotonic() - t0, 3)
        return program, "reused"

    def telemetry(self) -> Dict[str, Any]:
        return {
            "active": self.active,
            "rows_exported": self.exported,
            "rows_reused": self.reused,
            "respecialize_s": round(self.respecialize_s, 2),
            "gates": list(self.events),
        }


def _main(argv: Sequence[str]) -> int:
    """`python -m gen_worker.aot_export_reuse <job.json>` — one gate arm."""
    if len(argv) != 1:
        print("usage: python -m gen_worker.aot_export_reuse <job.json>",
              file=sys.stderr)
        return 2
    return _arm_child_main(argv[0])


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))


__all__ = [
    "ENV_FLAG",
    "MIN_ROWS",
    "GateVerdict",
    "ReuseState",
    "ReuseUnproven",
    "enabled",
    "prove",
    "respecialize",
]
