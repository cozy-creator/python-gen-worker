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
* the proof compares **every emitted file including the linked `.so`**, with
  both arms built in the SAME cleared cache-dir path, because ``-g1`` embeds
  the source path in the object and two arms in differently-named temp dirs
  can never be byte-equal for a reason that has nothing to do with the code.

Related: [[#847]] (this sweep), [[#846]] (the ruling this serves), [[#793]]
(the per-entry budget), [[#809]] (the pool that never covered the export).
"""
from __future__ import annotations

import copy
import hashlib
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

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


def _compile_digests(
    program: Any, entry: str, cache_dir: Path,
    inductor_configs: Optional[Mapping[str, Any]],
) -> Dict[str, str]:
    """Compile ``program`` through the PRODUCTION seam and digest every file.

    The cache dir is wiped first and is the SAME path for both arms, so the
    `-g1` source path baked into the objects is identical and a difference in
    the `.so` can only be a difference in the code.
    """
    from . import aot_mint

    shutil.rmtree(cache_dir, ignore_errors=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    previous = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir)
    try:
        import torch._inductor.codecache as codecache

        for name in ("cache_dir", "default_cache_dir"):
            resolver = getattr(codecache, name, None)
            clear = getattr(resolver, "cache_clear", None)
            if clear is not None:
                clear()
        files = aot_mint.compile_entry_files(
            program, entry, inductor_configs=inductor_configs)
    finally:
        if previous is None:
            os.environ.pop("TORCHINDUCTOR_CACHE_DIR", None)
        else:
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = previous
    digests: Dict[str, str] = {}
    for handle in files:
        path = Path(str(handle))
        if path.is_file():
            digests["".join(path.suffixes[-2:]) or path.suffix] = _digest(path)
    return digests


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
       compiles to byte-identical files, the witness's own full export
       included, every emitted file compared and the linked `.so` among them.

    Any exception, any missing artifact and any empty digest set DECLINES.
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

    cache = Path(workdir) / "pgw847-gate-cache"
    try:
        own = _compile_digests(
            witness, f"{entry}::gate-own", cache, inductor_configs)
        candidate = respecialize(base, witness_args, witness_kwargs)
        reuse = _compile_digests(
            candidate, f"{entry}::gate-reuse", cache, inductor_configs)
    except Exception as exc:  # noqa: BLE001
        return GateVerdict(
            False,
            f"the gate could not build its evidence "
            f"({type(exc).__name__}: {exc}) — declining, because an unproven "
            f"reuse is a wrong artifact waiting to happen",
            code_equal=True, gate_s=time.monotonic() - started)
    finally:
        shutil.rmtree(cache, ignore_errors=True)

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
        f"one export serves this family: graph text equal and all "
        f"{len(own)} emitted file(s) byte-identical, the linked object "
        f"included",
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
