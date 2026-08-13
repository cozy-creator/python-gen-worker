"""Touch the hub while the job is busy elsewhere.

A clone spends ~58 minutes downloading from HuggingFace and makes NOT ONE hub
request in that time. Losses of a paid 53 GiB download land on the very first
upload after such a gap, with an HTML 503 from the proxy in front of the hub,
not the hub itself. That timing is the whole reason this module exists:

  * The recorded hypothesis is an idle tunnel. A worker->hub path that carries
    no traffic for an hour has NAT entries, tunnel circuits and pooled sockets
    expiring underneath it, and the first request afterwards pays for it. A
    periodic touch means there is never an hour-old idle path to be surprised
    by, and the touch — not a multi-GB upload — is what pays if there is.

  * Whether or not that hypothesis holds, this makes the outage OBSERVABLE and
    dates it: the log says when reachability was lost and regained, rather than
    leaving the corpse of the job as the first evidence the hub had been gone.

The alternative considered and rejected was starting uploads earlier by
sub-chunking the source tier. It does not fit clone's architecture: the output
tree does not exist until the whole source snapshot has been downloaded and
run through repackage/cast (`build_flavor_tree` reshards across the complete
file set), so "upload sooner" means restructuring the pipeline into a streaming
one, not adding a call. This is ~60 lines and orthogonal to it.

Deliberately toothless: the probe NEVER fails the job, never retries, never
raises. It only observes and keeps the path warm. Deciding what an outage means
belongs to the publisher's retry loop, which is the code that knows what work
is at stake.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Optional
from ..http_origin import is_definite_hub_answer
from ..hubio.client import _http_session

logger = logging.getLogger(__name__)

__all__ = ["HubKeepalive"]

# Comfortably inside the idle-eviction timers this is defending against (NAT
# UDP/TCP mappings and tunnel circuits are commonly 60s-15min) while adding a
# rounding error of traffic: one small GET every two minutes against an hour of
# multi-GB downloading.
_INTERVAL_S = 120.0


class HubKeepalive:
    """Background prober: one cheap authenticated GET on a fixed cadence.

    Used as a context manager around a long hub-silent phase::

        with HubKeepalive(client, repo_path, log=ctx.log):
            download_58_minutes()

    The probe is a plain repo GET on the destination the job is already
    entitled to. It is idempotent, tiny, and — importantly — a 404 answers it
    just as well as a 200: the question is "did the HUB answer", not "does the
    repo exist".
    """

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
        # Public, for tests and for the caller's own record.
        self.probes = 0
        self.reachable: Optional[bool] = None
        self.unreachable_since: Optional[float] = None
        self.longest_outage_s = 0.0

    # ---- lifecycle ----

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

    # ---- internals ----

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
        """One touch. Returns whether the hub itself answered."""

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
