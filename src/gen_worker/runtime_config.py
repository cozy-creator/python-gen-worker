"""th#1087 worker-local mutable release config.

The hub's desired state carries (release_id, config_generation); parameter
values for declared knobs propagate over the existing gRPC stream. On each
push the worker updates memory AND atomically rewrites a local snapshot file
at a known path — per-invoke SUBPROCESSES read that file (`read_snapshot`)
and pick up the latest values on their next invoke. Envs are boot-injected
and never mutated here; bindings ride desired-residency (lifecycle.py).
"""

from __future__ import annotations

import logging
import os
import tempfile
import threading
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import msgspec

from .api.errors import RetryableError

logger = logging.getLogger(__name__)

# Known path contract: the worker exports this env var so every subprocess
# it spawns can locate the snapshot without plumbing.
SNAPSHOT_PATH_ENV = "GEN_WORKER_CONFIG_SNAPSHOT_PATH"
DEFAULT_SNAPSHOT_PATH = "/app/.tensorhub/runtime_config.msgpack"


class ConfigSnapshot(msgspec.Struct, frozen=True, kw_only=True):
    """Full current mutable config as seen by this worker."""

    config_generation: int = 0
    release_id: str = ""
    # function name -> parameter name -> value
    parameters: Dict[str, Dict[str, Any]] = msgspec.field(default_factory=dict)


class ConfigSnapshotWriteError(RetryableError):
    """The durable worker-local config snapshot could not be replaced."""


def _decode_snapshot(raw: bytes) -> ConfigSnapshot:
    return msgspec.msgpack.decode(raw, type=ConfigSnapshot)


def read_snapshot(path: Optional[str] = None) -> ConfigSnapshot:
    """Subprocess-side read of the worker's config snapshot file. Returns an
    empty generation-0 snapshot when the file is absent/unreadable."""
    p = path or os.environ.get(SNAPSHOT_PATH_ENV) or DEFAULT_SNAPSHOT_PATH
    try:
        with open(p, "rb") as f:
            return _decode_snapshot(f.read())
    except FileNotFoundError:
        return ConfigSnapshot()
    except Exception:
        logger.warning("unreadable config snapshot at %s", p, exc_info=True)
        return ConfigSnapshot()


class ConfigStore:
    """Current config + the snapshot-file writer. One per worker process."""

    def __init__(self, path: str = "") -> None:
        self._path = path or os.environ.get(SNAPSHOT_PATH_ENV) or DEFAULT_SNAPSHOT_PATH
        # Export the known path so every child process (subproc.run_process
        # or arbitrary Popen in endpoint code) inherits it.
        os.environ[SNAPSHOT_PATH_ENV] = self._path
        self._lock = threading.Lock()
        self._snap = ConfigSnapshot()
        self._parameter_snapshot_generation = 0
        self._on_parameter_snapshot: Callable[[int], None] = lambda _gen: None
        self._on_snapshot_failure: Callable[[str], None] = lambda _detail: None

    @property
    def path(self) -> str:
        return self._path

    @property
    def generation(self) -> int:
        return self._snap.config_generation

    @property
    def parameter_snapshot_generation(self) -> int:
        return self._parameter_snapshot_generation

    def parameters_for(self, function_name: str) -> Dict[str, Any]:
        with self._lock:
            return dict(self._snap.parameters.get(function_name, {}))

    def set_projection_callbacks(
        self,
        *,
        on_parameter_snapshot: Callable[[int], None],
        on_snapshot_failure: Callable[[str], None],
    ) -> None:
        self._on_parameter_snapshot = on_parameter_snapshot
        self._on_snapshot_failure = on_snapshot_failure

    def observe(self, generation: int, *, release_id: str = "") -> bool:
        """A hub desired-state push advertised ``generation`` (th#1085
        interface: DesiredResidency.config_generation). Advances the observed
        gen monotonically, keeping known parameter values; stale/duplicate
        generations are ignored. Returns True when state advanced (memory
        updated + snapshot file atomically rewritten)."""
        gen = int(generation)
        with self._lock:
            if gen <= self._snap.config_generation:
                if gen < self._snap.config_generation:
                    logger.info(
                        "ignoring stale config generation %d (observed %d)",
                        gen,
                        self._snap.config_generation,
                    )
                return False
            next_snapshot = ConfigSnapshot(
                config_generation=gen,
                release_id=release_id or self._snap.release_id,
                parameters=dict(self._snap.parameters),
            )
            self._write_snapshot_locked(next_snapshot)
            self._snap = next_snapshot
            logger.info("config generation %d observed; snapshot at %s", gen, self._path)
            return True

    def apply_parameter_snapshot(
        self,
        raw: bytes,
        generation: int,
        *,
        release_id: str,
    ) -> bool:
        """Atomically apply the hub's full declared-parameter snapshot."""
        gen = int(generation)
        release = str(release_id or "").strip()
        try:
            decoded = msgspec.msgpack.decode(raw)
            if not isinstance(decoded, dict):
                raise TypeError("parameter snapshot must be a map")
            parameters: Dict[str, Dict[str, Any]] = {}
            for function_name, values in decoded.items():
                if not isinstance(function_name, str) or not isinstance(values, dict):
                    raise TypeError("parameter snapshot entries must be maps")
                parameters[function_name] = {str(name): value for name, value in values.items()}
        except Exception as exc:
            raise ConfigSnapshotWriteError("invalid config parameter snapshot") from exc

        with self._lock:
            current_release = self._snap.release_id
            if current_release and release != current_release:
                raise ConfigSnapshotWriteError("config parameter snapshot release_id mismatch")
            if gen < self._snap.config_generation:
                return False
            next_snapshot = ConfigSnapshot(
                config_generation=gen,
                release_id=release,
                parameters=parameters,
            )
            changed = next_snapshot != self._snap
            if changed:
                self._write_snapshot_locked(next_snapshot)
                self._snap = next_snapshot
            self._parameter_snapshot_generation = max(
                self._parameter_snapshot_generation,
                gen,
            )
            self._on_parameter_snapshot(gen)
            return changed

    def stamp_function(
        self,
        function_name: str,
        values: Mapping[str, Any],
        generation: int,
    ) -> bool:
        """RunJob-stamped effective parameter values for one function (the
        class-1 read-at-dispatch carrier). Values at a generation older than
        the observed one are ignored; otherwise the function's values are
        full-replaced and the snapshot rewritten, so per-invoke subprocesses
        read the latest values as of this invoke."""
        gen = int(generation)
        with self._lock:
            if gen < self._snap.config_generation:
                return False
            vals = dict(values)
            if (
                gen == self._snap.config_generation
                and self._snap.parameters.get(function_name) == vals
            ):
                self._parameter_snapshot_generation = max(
                    self._parameter_snapshot_generation,
                    gen,
                )
                self._on_parameter_snapshot(gen)
                return False
            parameters = dict(self._snap.parameters)
            parameters[function_name] = vals
            next_snapshot = ConfigSnapshot(
                config_generation=max(gen, self._snap.config_generation),
                release_id=self._snap.release_id,
                parameters=parameters,
            )
            try:
                self._write_snapshot_locked(next_snapshot)
            except ConfigSnapshotWriteError as exc:
                self._on_snapshot_failure(str(exc))
                raise
            self._snap = next_snapshot
            self._parameter_snapshot_generation = max(
                self._parameter_snapshot_generation,
                gen,
            )
            self._on_parameter_snapshot(gen)
            logger.info(
                "config generation %d applied for %s; snapshot at %s",
                self._snap.config_generation,
                function_name,
                self._path,
            )
            return True

    def invocation_snapshot(
        self,
        function_name: str,
        values: Mapping[str, Any],
        generation: int,
    ) -> bytes:
        """Encode the config stamped on one invocation.

        ``run_process(ctx=...)`` gives this immutable snapshot to its child,
        so an older in-flight job cannot accidentally read a newer global
        generation that arrived before the child started.
        """
        with self._lock:
            snap = ConfigSnapshot(
                config_generation=int(generation),
                release_id=self._snap.release_id,
                parameters={str(function_name): dict(values)},
            )
        return msgspec.msgpack.encode(snap)

    def _write_snapshot_locked(self, snapshot: ConfigSnapshot) -> None:
        """Atomic write: tmp file in the same dir + os.replace, so a
        subprocess mid-read never sees a torn file."""
        try:
            d = os.path.dirname(self._path) or "."
            os.makedirs(d, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=d, prefix=".runtime_config-")
            try:
                with os.fdopen(fd, "wb") as f:
                    f.write(msgspec.msgpack.encode(snapshot))
                os.replace(tmp, self._path)
            except BaseException:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
        except Exception as exc:
            logger.error("config snapshot write failed at %s", self._path, exc_info=True)
            raise ConfigSnapshotWriteError(f"config snapshot write failed at {self._path}") from exc


def extract_config_push(ack: Any) -> Optional[Tuple[int, str]]:
    """(config_generation, release_id) from a HelloAck's DesiredResidency,
    using the A+C wire contract (``release_id = 5;
    config_generation = 6``)."""
    desired = ack.desired_residency
    gen = int(desired.config_generation or 0)
    if gen <= 0:
        return None
    return gen, str(desired.release_id or "")


def extract_job_config(run: Any) -> Tuple[int, Optional[Dict[str, Any]]]:
    """(config_generation, values) stamped on the RunJob (class-1
    read-at-dispatch; ``config_generation = 16; bytes config_params = 17``,
    a msgpack param-name -> value map for the invoked function). A
    dispatched job runs with the values it was stamped with — a gen bump
    never rewrites it. (0, None) when unstamped."""
    gen = int(run.config_generation or 0)
    raw = run.config_params or b""
    if not raw:
        return gen, None
    try:
        out = msgspec.msgpack.decode(raw)
    except Exception:
        logger.warning("undecodable RunJob.config_params: %r", raw[:200])
        return gen, None
    if not isinstance(out, dict):
        return gen, None
    return gen, {str(k): v for k, v in out.items()}


__all__ = [
    "ConfigSnapshot",
    "ConfigSnapshotWriteError",
    "ConfigStore",
    "DEFAULT_SNAPSHOT_PATH",
    "SNAPSHOT_PATH_ENV",
    "extract_config_push",
    "extract_job_config",
    "read_snapshot",
]
