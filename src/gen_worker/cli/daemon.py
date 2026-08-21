"""The resident endpoint: ONE booted host, one socket, many warm requests; `run` is a client and executes no inference. Wire: NDJSON over a Unix socket, one request per connection — request {"function", "payload", "request_id"?}; ok {"ok":true,"result",...,"warnings","dispatch"}; refusal {"ok":false,"error":{"kind","message"}}; control {"status":{}}. Requests are serialized, one at a time."""

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

REQUEST_LINE_TIMEOUT_S = 30.0


class BootError(RuntimeError):
    """The endpoint could not be brought up."""


@dataclass(slots=True)
class BootSpec:
    """Everything ``up`` decided, frozen before anything is loaded."""

    endpoint_dir: Path
    checkpoint_refs: Tuple[str, ...] = ()
    checkpoint_dir: Optional[Path] = None
    checkpoint_ref_label: str = "local/checkpoint"
    model: str = ""
    defaults: str = "{}"
    lane: str = ""
    sm: str = ""
    graph_store: Optional[Path] = None
    artifacts_dir: Path = field(default_factory=artifacts_root)
    output_dir: Path = Path("outputs")
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
    hole_reasons: Tuple[Tuple[str, str], ...] = ()
    mint: Any = None
    warnings: List[str] = field(default_factory=list)


def boot(spec: BootSpec) -> Booted:
    """Load the endpoint, materialize its checkpoints, adopt, arm the counter."""
    from .. import receipts
    from ..serving.context import DeployBinding
    from ..serving.dispatch_counter import DispatchCounter
    from ..serving.host import EndpointHost
    from ..serving.loader import load_endpoint

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
    if spec.graph_store is None:
        return None, None
    if not spec.sm:
        raise BootError(
            "--sm is required to adopt compiled graphs (artifacts are per-sm). "
            "Omit --graph-store to serve eager."
        )
    from .._vendor.torchcg.store import StoreError
    from ..serving.mint_store import graph_store

    store = graph_store(Path(spec.graph_store))
    try:
        document = store.get_graphs(module_name)
    except StoreError as exc:
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
        return store, None
    if document is None:
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

    def _document(self, state: str) -> Dict[str, Any]:
        from .. import settings_authority

        return {
            "protocol_version": PROTOCOL_VERSION,
            "gen_worker_version": gen_worker_version(),
            "state": state,
            "pid": os.getpid(),
            # The SERVING process's own reading of the declared env — the
            # allocator config a local floor/benchmark row was actually taken
            # under, confessed rather than assumed (pgw#1640).
            "declared_env": settings_authority.process_env_readback(),
            "endpoint_dir": str(self.spec.endpoint_dir),
            "module": self.booted.loaded.module_name,
            "socket": str(self.handle.socket_path),
            "functions": sorted(self.booted.loaded.entrypoints),
            "primary_fields": self._primary_fields(),
            "checkpoint_dir": str(self.booted.checkpoint_dir),
            "checkpoint_refs": list(self.spec.checkpoint_refs),
            "output_dir": str(self.spec.output_dir.resolve()),
            "adopted_graphs": list(self.booted.adopted),
            "holes": list(self.booted.holes),
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

    def dispatch(self, frame: Dict[str, Any]) -> Dict[str, Any]:
        """Run one request. Never raises — every failure is a typed envelope, because a transport that dies on a bad payload takes the warm models with it."""
        function = frame.get("function")
        payload = frame.get("payload", {})
        request_id = str(frame.get("request_id") or f"run-{self._served}")
        with self._dispatch_lock:
            self._last_activity = time.time()
            self._served += 1
            counter = self.booted.counter
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


def seal_declared_env() -> Dict[str, str]:
    """Refuse to serve on an allocator this platform did not declare, and CONFESS the one it got.

    The pod (`python -m gen_worker.entrypoint`) has always imposed
    `settings_authority.DECLARED_ENV`; `gen-worker up` did not, so the two front
    doors ran different allocators and a tight-VRAM row that served on a pod
    refused on the CLI (pgw#1639). The CLI package imposes it at import; this is
    the read-back, taken from `os.environ` in the process that is about to
    serve — the only place the answer is authoritative.
    """
    from .. import settings_authority

    try:
        settings_authority.verify_process_env()
    except settings_authority.SettingsImpositionError as exc:
        raise BootError(str(exc)) from exc
    effective = settings_authority.process_env_readback()
    sys.stderr.write(
        "gen-worker up: declared env in effect — "
        + " ".join(f"{k}={v}" for k, v in sorted(effective.items()))
        + "\n"
    )
    sys.stderr.flush()
    return effective


def serve(spec: BootSpec, handle: EndpointHandle) -> int:
    """Boot and serve until SIGINT/SIGTERM."""
    seal_declared_env()
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
    "seal_declared_env",
    "serve",
]
