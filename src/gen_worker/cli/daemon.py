"""The resident endpoint: ONE booted host, one socket, many warm requests.

pgw#1491. This is what ``gen-worker up`` runs and what ``gen-worker run`` is a
client of. There is exactly one execution path for a request in this package and
it is here — ``run`` does not boot, does not load, and does not dispatch. That
is not tidiness: two paths that both "run a request" drift, and the drift is
invisible because both of them keep producing images (measured three times in
one day, pgw#1491).

## Wire

NDJSON over a Unix socket, one request per connection, response on the same
connection:

* request   ``{"function": str, "payload": {...}, "request_id": str?}``
* ok        ``{"ok": true, "result": ..., "warnings": [...], "dispatch": {...}}``
* refusal   ``{"ok": false, "error": {"kind": str, "message": str}}``
* control   ``{"status": {}}``  -> the handle document plus live counters

``dispatch`` is :class:`~gen_worker.serving.dispatch_counter.DispatchCounts` —
every response states whether it was served by a compiled graph or eagerly.

The old ``stream: true`` half of this protocol is NOT restored: entrypoints
return one value today (``EndpointHost.dispatch``/``ServeLoop.invoke`` are not
generators), so a streaming frame shape would be a wire contract with no
producer behind it. It comes back with the producer, not before.

## Requests are serialized

One request at a time, on the accept thread's worker. The card is one lane and
``ModelInstance.admission`` already single-flights per model; a second dispatch
would queue on that lock anyway, so serializing here keeps the counters
attributable to the request that produced them.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import msgspec

from . import endpoint_state, sockaddr
from .workspace import artifacts_root
from .endpoint_state import EndpointHandle
from .protocol import PROTOCOL_VERSION, gen_worker_version

#: How long a client has to finish sending its request line. The long wait is
#: the DISPATCH, which happens after this and is not bounded here.
REQUEST_LINE_TIMEOUT_S = 30.0


class BootError(RuntimeError):
    """The endpoint could not be brought up. Always names what failed."""


@dataclass(slots=True)
class BootSpec:
    """Everything ``up`` decided, frozen before anything is loaded."""

    endpoint_dir: Path
    #: Checkpoint refs this endpoint serves — RUNTIME CONFIG, never a
    #: declaration by the endpoint (DESIGN-RULINGS 2026-08-19). Materialized
    #: through the same code path `gen-worker download` uses.
    checkpoint_refs: Tuple[str, ...] = ()
    #: An already-materialized tree, bypassing the ref resolution entirely.
    checkpoint_dir: Optional[Path] = None
    checkpoint_ref_label: str = "local/checkpoint"
    model: str = ""
    defaults: str = "{}"
    lane: str = ""
    sm: str = ""
    graph_store: Optional[Path] = None
    #: pgw#1526: the BOX cache, stated once in `cli/workspace.py` beside
    #: `graph_cas_root()`. It was `Path(".compiled-graphs")` — relative to
    #: whatever cwd the daemon happened to start in, which put a
    #: machine-scoped artifact store inside an endpoint source tree.
    #: A `default_factory` and not a module-level constant: the address is
    #: config (`COZY_ARTIFACTS`), so it must be read when a BootSpec is
    #: built, not frozen at import — a process that installs Settings after
    #: import would otherwise silently keep the pre-config answer.
    artifacts_dir: Path = field(default_factory=artifacts_root)
    output_dir: Path = Path("outputs")
    #: Background pre-warm posture: "auto" mints the adopt session's holes in
    #: the background, "off" never compiles. `gen-worker compile` is the
    #: explicit front door; this is the serve-coordinated half of the same
    #: work ledger (DESIGN-RULINGS compile addendum 2).
    compile_policy: str = "auto"
    idle_timeout_s: float = 0.0
    env_lockfile: Optional[Path] = None


@dataclass(slots=True)
class Booted:
    """The live endpoint plus the facts the handle publishes."""

    host: Any
    loaded: Any
    counter: Any
    checkpoint_dir: Path
    adopted: Tuple[str, ...] = ()
    holes: Tuple[str, ...] = ()
    #: Per-hole WHY, parallel to nothing — keyed rows, because the graph name
    #: alone answered "which" while the field failure needed "why" (pgw#1564:
    #: fourteen `cannot decompress` reasons sat unread on the session while a
    #: campaign diagnosed the resulting zero as a new defect).
    hole_reasons: Tuple[Tuple[str, str], ...] = ()
    mint: Any = None
    warnings: List[str] = field(default_factory=list)


def boot(spec: BootSpec) -> Booted:
    """Load the endpoint, materialize its checkpoints, adopt, arm the counter.

    The order is the invariant: bytes on disk BEFORE anything is asked to
    serve. ``run`` requires ``up`` for exactly this reason, and a pod's boot is
    the same three steps in one process (th#2204).
    """
    from .. import receipts
    from ..serving.context import DeployBinding
    from ..serving.dispatch_counter import DispatchCounter
    from ..serving.host import EndpointHost
    from ..serving.loader import load_endpoint

    # The store is a directory this operator owns, not a hub delivery: the
    # receipt gate is DECLARED off rather than left unset, because a process
    # that declares neither posture refuses.
    receipts.trust_local_store(
        "gen-worker up: the compiled-graph store is a local directory the "
        "operator owns, not a hub delivery"
    )

    loaded = load_endpoint(spec.endpoint_dir)
    tree, ref_label = _checkpoint_tree(spec, loaded)

    binding = DeployBinding(
        checkpoint_ref=ref_label,
        checkpoint_dir=tree,
        model=spec.model or None,
        defaults=json.loads(spec.defaults or "{}"),
    )
    host = EndpointHost(
        loaded,
        binding,
        lane_contract=spec.lane,
        output_dir=spec.output_dir,
    )
    store, document = _adoption_source(spec, loaded.module_name)
    stack = _stated_stack(spec, document)
    host.setup(
        store=store,
        document=document,
        sm=spec.sm,
        artifacts_dir=spec.artifacts_dir,
        stack=stack,
    )
    counter = DispatchCounter().install(host)

    adopted: Tuple[str, ...] = ()
    holes: Tuple[str, ...] = ()
    hole_reasons: Tuple[Tuple[str, str], ...] = ()
    if host.adoption is not None:
        adopted = tuple(record.graph for record in host.adoption.adopted)
        holes = tuple(hole.record.graph for hole in host.holes)
        hole_reasons = tuple(
            (hole.record.graph, str(hole.reason)[:300]) for hole in host.holes
        )

    booted = Booted(
        host=host,
        loaded=loaded,
        counter=counter,
        checkpoint_dir=tree,
        adopted=adopted,
        holes=holes,
        hole_reasons=hole_reasons,
    )
    if holes and spec.compile_policy == "auto" and store is not None:
        booted.mint = _start_background_mint(spec, host, store)
    return booted


def _checkpoint_tree(spec: BootSpec, loaded: Any) -> Tuple[Path, str]:
    """Put the configured checkpoint on disk and return (tree, ref label).

    A weightless endpoint (zero model slots, pgw#1392) names no checkpoint and
    is not asked for one — absence is legal exactly when nothing can consume
    it, and refused BY NAME the moment something can.
    """
    if spec.checkpoint_dir is not None:
        return Path(spec.checkpoint_dir), spec.checkpoint_ref_label
    if spec.checkpoint_refs:
        from .download import materialize_refs

        trees = materialize_refs(spec.checkpoint_refs)
        first = spec.checkpoint_refs[0]
        return trees[first], first
    if loaded.models:
        raise BootError(
            f"no checkpoint configured: {loaded.module_name} declares model "
            f"slot(s) on "
            f"{', '.join(sorted(m.__name__ for m in loaded.models))}, and a "
            f"model slot is loaded FROM a checkpoint tree.\n"
            f"  Name one with `--checkpoint-ref owner/name@rev` (or "
            f"`--checkpoint <dir>` for a tree you already have), or run "
            f"`gen-worker download` first — it defaults to the endpoint "
            f"author's recommended checkpoint."
        )
    return Path(os.devnull).parent / "__weightless__", spec.checkpoint_ref_label


def _adoption_source(spec: BootSpec, module_name: str) -> Tuple[Any, Any]:
    """(store, document) for this boot, or (None, None) — the eager bridge.

    An UNREADABLE graph-set document is a MISS, never a boot failure (pgw#1525).
    Compiled graphs are derived and disposable: a document this build's torchcg
    cannot decode — a v1 document under a v3 decoder, the ordinary shape after a
    re-vendor — means this box holds nothing usable for this endpoint, which is
    exactly what an empty store means. Both answers are "serve eager and let the
    mint refill", so they must not have different outcomes.

    `LocalGraphStore.get_graphs` RAISES `StoreError` on a decode failure rather
    than answering a clean miss, and this call site used to let it through — so
    a stale store did not degrade, it killed `up` before any author code ran.
    That is the cold-start regression pgw#1525 names, and it is worst precisely
    when recovery matters most: right after the format bump that produced it.

    The refusal is caught HERE and not repaired in the store because the two
    halves belong to different repos. The store-side fix (answer a clean miss
    and GC the unreadable leftover) is torchcg's; this is the serving side
    refusing to die on it either way.
    """
    if spec.graph_store is None:
        return None, None
    if not spec.sm:
        raise BootError(
            "--sm is required to adopt compiled graphs (artifacts are per-sm). "
            "Omit --graph-store to serve eager."
        )
    from .._vendor.torchcg.store import StoreError
    from ..serving.mint_store import graph_store

    # THE ONE STORE (pgw#1573). This built a bare `LocalGraphStore` — no hub
    # tier, no baked tier, and a different object from the one this boot's
    # background mint would publish through — so `up` and a pod disagreed about
    # what "present" means. `graph_store` is what every entry point builds now;
    # `upstream=None` is the honest statement that a box boot has no release to
    # adopt for, not a second class of store.
    store = graph_store(Path(spec.graph_store))
    try:
        document = store.get_graphs(module_name)
    except StoreError as exc:
        # LOUD and typed: silence here would read as "this endpoint compiles to
        # nothing", which is a different fact with the same shape.
        print(
            f"adopt: graph_store_unreadable — the compiled-graph document for "
            f"{module_name} in {spec.graph_store} cannot be decoded by this "
            f"build's torchcg ({exc}).\n"
            f"adopt: SERVING EAGER. Compiled graphs are derived and disposable, "
            f"so this costs speed, never correctness.\n"
            f"adopt: remedy — `gen-worker compile` re-mints this endpoint's "
            f"graphs in the current format; the stale entries are reclaimable.",
            file=sys.stderr,
        )
        # `(store, None)` and NOT `(None, None)`: an undecodable document must
        # land on the SAME value an empty store produces, or the two "I hold
        # nothing usable" states diverge again one layer down. The store stays
        # bound because it is still WRITABLE — `put_graphs` compare-and-swaps a
        # ref and does not care that the old bytes are undecodable, so a re-mint
        # can refill this very store rather than needing it deleted first.
        return store, None
    if document is None:
        # THE REMEDY IS PRINTED FOR THE MISS ITSELF, not only for a raise.
        #
        # tcg#69 made `LocalGraphStore` answer a clean miss and discard the
        # stale bytes, so the branch above no longer fires for the local store
        # — it survives for backends that still raise (the hub-backed one).
        # Printing the remedy only there would have silently retired the
        # operator-facing line on the exact path pgw#1525 exists to fix.
        #
        # Naming ONE outcome for both states is not a loss of information, it
        # is this issue's own conclusion: an undecodable document and an empty
        # store are the same fact — this box holds nothing usable — and the
        # remedy is the same verb. WHICH of the two it was is still said, by
        # torchcg's discard WARNING, which names the graph-set, the bytes and
        # the decode failure. That warning reaches the terminal even with no
        # logging configured (`logging.lastResort` handles WARNING+), which
        # was verified rather than assumed.
        print(
            f"adopt: no compiled-graph document for {module_name} in "
            f"{spec.graph_store}.\n"
            f"adopt: SERVING EAGER. Compiled graphs are derived and "
            f"disposable, so this costs speed, never correctness.\n"
            f"adopt: remedy — `gen-worker compile` mints this endpoint's "
            f"graphs for this card in the current format.",
            file=sys.stderr,
        )
    return store, document


def _stated_stack(spec: BootSpec, document: Any) -> Optional[Any]:
    """This boot's compile stack, off the endpoint's own uv.lock (pgw#1489)."""
    if document is None:
        return None
    from ..env_identity import (
        EnvIdentityError,
        compile_stack_from_lockfile,
        cuda_bucket,
        lockfile_beside,
    )

    lockfile = spec.env_lockfile or lockfile_beside(str(spec.endpoint_dir))
    if lockfile is None:
        return None
    try:
        return dict(compile_stack_from_lockfile(Path(lockfile), bucket=cuda_bucket()))
    except EnvIdentityError as exc:
        raise BootError(f"compile stack from {lockfile}: {exc}") from exc


def _start_background_mint(spec: BootSpec, host: Any, store: Any) -> Any:
    """Fill this boot's holes without blocking a single request.

    Paul's "take over the running compile" lands as the WORK LEDGER, not
    process adoption: this mint and an external ``gen-worker compile`` key work
    by the same CAS entries, so each skips what the other already landed
    (skip-if-present measured at ~10 s/hit) and whichever survives finishes the
    remainder.
    """
    from ..toolchain import toolchain_digest
    from ..serving.self_mint import SelfMint

    box = SelfMint(
        store=store,
        artifacts_dir=Path(spec.artifacts_dir),
        cas_dir=Path(spec.graph_store or (Path(spec.artifacts_dir) / "cas")),
        target_arch=spec.sm,
        toolchain=dict(toolchain_digest()),
    )
    box.arm(host)
    return box


# --------------------------------------------------------------------------
# The resident process
# --------------------------------------------------------------------------


class ResidentEndpoint:
    """One booted endpoint answering NDJSON requests until told to stop."""

    def __init__(self, booted: Booted, spec: BootSpec, handle: EndpointHandle) -> None:
        self.booted = booted
        self.spec = spec
        self.handle = handle
        self._stop = threading.Event()
        self._dispatch_lock = threading.Lock()
        self._served = 0
        self._last_activity = time.time()
        self._booted_at = time.time()

    # -- handle -------------------------------------------------------------

    def _document(self, state: str) -> Dict[str, Any]:
        return {
            "protocol_version": PROTOCOL_VERSION,
            "gen_worker_version": gen_worker_version(),
            "state": state,
            "pid": os.getpid(),
            "endpoint_dir": str(self.spec.endpoint_dir),
            "module": self.booted.loaded.module_name,
            "socket": str(self.handle.socket_path),
            "functions": sorted(self.booted.loaded.entrypoints),
            # The ONE fact a client needs to turn `run "a cat"` into a payload:
            # which field a bare positional fills. Published from the live
            # msgspec struct rather than copied into a schema the client would
            # then have to keep in step — and it is one name, not a type map,
            # because the authoritative decode still happens here.
            "primary_fields": self._primary_fields(),
            "checkpoint_dir": str(self.booted.checkpoint_dir),
            "checkpoint_refs": list(self.spec.checkpoint_refs),
            "output_dir": str(self.spec.output_dir.resolve()),
            "adopted_graphs": list(self.booted.adopted),
            "holes": list(self.booted.holes),
            # WHY each hole is a hole (pgw#1564): the handle is the ONE thing
            # an out-of-process caller can read after boot, and it carried
            # only names — so the 2026-08-20 field zero, whose fourteen
            # reasons all said `cannot decompress`, was reported as
            # reasonless. The verdict block makes the query one field read.
            "hole_reasons": [
                {"graph": graph, "reason": reason}
                for graph, reason in self.booted.hole_reasons
            ],
            "adoption": {
                "engaged": self.booted.host.adoption is not None,
                "armed": len(self.booted.adopted),
                "claimed": len(self.booted.adopted) + len(self.booted.holes),
            },
            "sm": self.spec.sm,
            "lane": self.spec.lane,
            "booted_at": self._booted_at,
            "served": self._served,
        }

    def _primary_fields(self) -> Dict[str, str]:
        from .args import primary_field

        out: Dict[str, str] = {}
        for name, spec in self.booted.loaded.entrypoints.items():
            field_name = primary_field(spec.payload_type)
            if field_name:
                out[name] = field_name
        return out

    def publish(self, state: str) -> None:
        endpoint_state.write_handle(self.handle, self._document(state))

    # -- serving ------------------------------------------------------------

    def dispatch(self, frame: Dict[str, Any]) -> Dict[str, Any]:
        """Run one request. Never raises — every failure is a typed envelope,
        because a transport that dies on a bad payload takes the warm models
        with it."""
        function = frame.get("function")
        payload = frame.get("payload", {})
        request_id = str(frame.get("request_id") or f"run-{self._served}")
        with self._dispatch_lock:
            self._last_activity = time.time()
            self._served += 1
            counter = self.booted.counter
            # A late mint arms artifacts onto the live session; re-wrap before
            # the request so graphs landed since the last one are counted.
            counter.rearm()
            counter.reset()
            ctx = self.booted.host.make_context(request_id)
            try:
                result = self.booted.host.dispatch(
                    str(function), payload, request_id=request_id, ctx=ctx
                )
            except Exception as exc:  # noqa: BLE001 — every failure is a value
                counts = counter.take()
                kind = _error_kind(exc)
                if kind == "user_exception":
                    traceback.print_exc(file=sys.stderr)
                return {
                    "ok": False,
                    "request_id": request_id,
                    "error": {"kind": kind, "message": str(exc)},
                    "dispatch": counts.facts(),
                }
            counts = counter.take()
        sys.stderr.write(f"{counts.summary()}\n")
        sys.stderr.flush()
        return {
            "ok": True,
            "request_id": request_id,
            "result": msgspec.to_builtins(result),
            "warnings": list(getattr(ctx, "warnings", ()) or ()),
            "dispatch": counts.facts(),
        }

    def status(self) -> Dict[str, Any]:
        document = self._document("ready")
        mint = self.booted.mint
        if mint is not None:
            try:
                document["mint"] = dict(mint.facts())
            except Exception:  # noqa: BLE001 — status must never fail
                document["mint"] = {"state": "unreadable"}
        return {"ok": True, "status": document}

    # -- transport ----------------------------------------------------------

    def serve_forever(self) -> int:
        listen = str(self.handle.socket_path)
        server = sockaddr.create_listener(listen, backlog=16)
        server.settimeout(0.5)
        self.publish("ready")
        sys.stderr.write(
            f"gen-worker up: {self.booted.loaded.module_name} ready on {listen}\n"
            f"gen-worker up: functions: "
            f"{', '.join(sorted(self.booted.loaded.entrypoints))}\n"
            f"gen-worker up: run one with "
            f"`gen-worker run \"<prompt>\"` in another shell\n"
        )
        sys.stderr.flush()
        try:
            while not self._stop.is_set():
                try:
                    conn, _ = server.accept()
                except socket.timeout:
                    if self._idle_expired():
                        sys.stderr.write(
                            f"gen-worker up: idle for {self.spec.idle_timeout_s:.0f}s; "
                            f"shutting down and freeing the card\n"
                        )
                        break
                    continue
                except OSError:
                    break
                threading.Thread(
                    target=self._handle_conn, args=(conn,), daemon=True
                ).start()
        finally:
            server.close()
            sockaddr.cleanup_listener(listen)
            self.teardown()
        return 0

    def _idle_expired(self) -> bool:
        timeout = float(self.spec.idle_timeout_s or 0.0)
        if timeout <= 0.0:
            return False
        with self._dispatch_lock:
            idle_for = time.time() - self._last_activity
        return idle_for >= timeout

    def _handle_conn(self, conn: socket.socket) -> None:
        try:
            frame = _read_frame(conn)
            if frame is None:
                return
            if "status" in frame:
                response: Dict[str, Any] = self.status()
            elif "shutdown" in frame:
                response = {"ok": True, "shutting_down": True}
                self._stop.set()
            elif not isinstance(frame.get("function"), str) or not frame["function"]:
                response = {
                    "ok": False,
                    "error": {
                        "kind": "usage",
                        "message": "request.function (string) is required",
                    },
                }
            else:
                response = self.dispatch(frame)
            _send(conn, response)
        except OSError:
            pass
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def stop(self) -> None:
        self._stop.set()

    def teardown(self) -> None:
        self.publish("stopping")
        try:
            self.booted.counter.close()
        except Exception:  # noqa: BLE001
            pass
        mint = self.booted.mint
        if mint is not None:
            try:
                mint.cancel()
            except Exception:  # noqa: BLE001 — the mint is best-effort work
                pass
        try:
            self.booted.host.teardown()
        finally:
            endpoint_state.clear_handle(self.handle)
        sys.stderr.write("gen-worker up: down\n")
        sys.stderr.flush()


def _error_kind(exc: BaseException) -> str:
    """The typed word for a dispatch failure. One vocabulary, closed."""
    from ..serving.host import ServeDispatchError

    if isinstance(exc, ServeDispatchError):
        return "no_such_function"
    if isinstance(exc, msgspec.ValidationError):
        return "payload_invalid"
    if isinstance(exc, msgspec.DecodeError):
        return "payload_invalid"
    return "user_exception"


def _read_frame(conn: socket.socket) -> Optional[Dict[str, Any]]:
    buf = bytearray()
    conn.settimeout(REQUEST_LINE_TIMEOUT_S)
    try:
        while b"\n" not in buf:
            chunk = conn.recv(65536)
            if not chunk:
                break
            if len(buf) + len(chunk) > sockaddr.MAX_NDJSON_LINE_BYTES:
                return None
            buf.extend(chunk)
    except (socket.timeout, OSError):
        return None
    line = bytes(buf).split(b"\n", 1)[0].strip()
    if not line:
        return None
    try:
        frame = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {"function": ""}
    return frame if isinstance(frame, dict) else {"function": ""}


def _send(conn: socket.socket, envelope: Dict[str, Any]) -> None:
    line = json.dumps(envelope, separators=(",", ":"), default=str) + "\n"
    conn.settimeout(None)
    conn.sendall(line.encode("utf-8"))


def serve(spec: BootSpec, handle: EndpointHandle) -> int:
    """Boot and serve until SIGINT/SIGTERM. The body of ``up``, both modes."""
    booted = boot(spec)
    resident = ResidentEndpoint(booted, spec, handle)

    def _on_signal(signum: int, _frame: Any) -> None:
        name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
        sys.stderr.write(f"\ngen-worker up: {name} — tearing down\n")
        sys.stderr.flush()
        resident.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _on_signal)
        except (ValueError, OSError):  # pragma: no cover - non-main thread
            pass
    return resident.serve_forever()


__all__ = [
    "BootError",
    "BootSpec",
    "Booted",
    "ResidentEndpoint",
    "boot",
    "serve",
]
