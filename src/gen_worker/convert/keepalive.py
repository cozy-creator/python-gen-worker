"""Touch the hub while the job is busy elsewhere."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Optional
from ..http_origin import is_definite_hub_answer
from ..hubio.client import _http_session

logger = logging.getLogger(__name__)

__all__ = ["HubKeepalive"]

_INTERVAL_S = 120.0


class HubKeepalive:
    """Background prober: one cheap authenticated GET on a fixed cadence."""

    def __init__(
        self,
        client: Any,
        repo_path: str,
        *,
        interval_s: float = _INTERVAL_S,
        log: Optional[Callable[[str], None]] = None,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self._client = client
        self._repo_path = str(repo_path or "")
        self._interval_s = float(interval_s)
        self._log = log
        self._now = now
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.probes = 0
        self.reachable: Optional[bool] = None
        self.unreachable_since: Optional[float] = None
        self.longest_outage_s = 0.0

    def __enter__(self) -> "HubKeepalive":
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    def start(self) -> None:
        if self._thread is not None or not self._repo_path:
            return
        self._thread = threading.Thread(
            target=self._loop, name="hub-keepalive", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=5.0)

    def _say(self, msg: str) -> None:
        logger.info("%s", msg)
        if callable(self._log):
            try:
                self._log(msg)
            except Exception:  # noqa: BLE001 - observability never breaks a job
                pass

    def _loop(self) -> None:
        while not self._stop.wait(self._interval_s):
            self.probe()

    def probe(self) -> bool:
        """One touch."""

        self.probes += 1
        answered = False
        try:
            resp = _http_session().get(
                f"{self._client.base_url}{self._repo_path}",
                headers=self._client._headers(),
                timeout=(15.0, 30.0),
            )
            answered = is_definite_hub_answer(resp)
            detail = f"status={resp.status_code}"
        except Exception as exc:  # noqa: BLE001 - a probe may never raise
            detail = f"{type(exc).__name__}: {exc}"
        self._record(answered, detail)
        return answered

    def _record(self, answered: bool, detail: str) -> None:
        was = self.reachable
        self.reachable = answered
        if answered:
            if self.unreachable_since is not None:
                out = self._now() - self.unreachable_since
                self.longest_outage_s = max(self.longest_outage_s, out)
                self.unreachable_since = None
                self._say(
                    f"hub reachable again after {out:.0f}s unreachable "
                    f"({detail})")
            return
        if self.unreachable_since is None:
            self.unreachable_since = self._now()
        if was is not False:
            self._say(
                f"hub NOT answering — a proxy or nothing is in front of it "
                f"({detail}); uploads later in this job will have to ride "
                f"this out")
