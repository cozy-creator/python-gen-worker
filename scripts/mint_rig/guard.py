"""podguard, behind a Protocol — and the rule that it is never optional.

th#1327's argument in one line: *renting a pod and arming its teardown are ONE
operation, because agent sessions die routinely and any cleanup that depends on
the renter surviving leaks by construction.* The rig does not re-implement that;
it uses podguard, which already carries a pod-side watchdog armed inside the
CREATE call and a box-side reaper for the pod whose container never started.

The Protocol exists so the driver is testable at $0, not so the guard is
swappable for nothing. :func:`real_guard` REFUSES when podguard is unreachable
rather than degrading to an unguarded rental.

The rig therefore has THREE layers of stoppability, and they fail independently:

  podguard Layer A   the pod kills itself when nobody renews (needs no box)
  podguard Layer B   the cron reaper kills a pod whose container never started
  mint_rig killset   a durable box-side record written BEFORE the create call,
                     so even the pod nobody ever learned the id of is nameable
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Protocol, Sequence


class Guard(Protocol):
    """Arming, attending and closing out a rental."""

    def rent(
        self,
        api_key: str,
        body: Mapping[str, Any],
        *,
        lane: str,
        lease_seconds: float,
        orig_cmd: Sequence[str],
        post: Callable[[Mapping[str, Any]], dict[str, Any]],
    ) -> dict[str, Any]:
        """Arm `body`, create through `post`, and record the lease.

        Returns the create response. `post` is the rig's own REST client, so the
        guard never owns the wire — which is what makes the whole driver
        testable without one.
        """

    def release(self, pod_id: str, *, confirmed_dead: bool, reason: str) -> None: ...


class PodguardGuard:
    """The real one. Thin by design — podguard is the implementation."""

    def __init__(self, module: Any) -> None:
        self._pg = module

    def rent(
        self,
        api_key: str,
        body: Mapping[str, Any],
        *,
        lane: str,
        lease_seconds: float,
        orig_cmd: Sequence[str],
        post: Callable[[Mapping[str, Any]], dict[str, Any]],
    ) -> dict[str, Any]:
        lease = self._pg.rent(
            api_key,
            dict(body),
            lane=lane,
            lease_seconds=lease_seconds,
            orig_cmd=list(orig_cmd),
            post=post,
        )
        raw = getattr(lease, "raw", None)
        if isinstance(raw, dict) and raw:
            return dict(raw)
        return {"id": lease.pod_id, "name": lease.name, "costPerHr": lease.rate_per_hr}

    def release(self, pod_id: str, *, confirmed_dead: bool, reason: str) -> None:
        self._pg.record_release(pod_id, confirmed_dead, reason)


def real_guard() -> PodguardGuard:
    from .runpod import load_podguard

    return PodguardGuard(load_podguard())
