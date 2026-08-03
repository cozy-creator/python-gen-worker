"""pgw#763 split-harness endpoints: real uncatchable deaths and hangs.

Separate from toy_endpoints so the process-split suite owns its fixture file
outright (shared-worktree etiquette).
"""

from __future__ import annotations

import asyncio
import os
import signal
import time

import msgspec

from gen_worker import RequestContext, activity, endpoint


class ProbeIn(msgspec.Struct):
    text: str = ""


class ProbeOut(msgspec.Struct):
    response: str


@endpoint
class SplitProbe:
    def echo(self, ctx: RequestContext, data: ProbeIn) -> ProbeOut:
        return ProbeOut(response=f"echo:{data.text}")

    def whoami(self, ctx: RequestContext, data: ProbeIn) -> ProbeOut:
        """Report which execution GROUP this child is (pgw#783 G>1 routing
        proof): the ordinal the parent stamped and the cards it was scoped to.
        A single-group worker reports g0 with no CUDA_VISIBLE_DEVICES."""
        return ProbeOut(response=(
            f"g={os.environ.get('GEN_WORKER_GROUP_ORDINAL', '0')}"
            f" cvd={os.environ.get('CUDA_VISIBLE_DEVICES', '-')}"
            f" sib={os.environ.get('GEN_WORKER_HOST_SIBLINGS', '1')}"
        ))

    # ---- driver-3 probes: TENANT CODE going after platform credentials -----
    # These handlers do what the threat model says untrusted endpoint code can
    # do — it runs in this process, so every one of these is reachable. The
    # security suite asserts each comes back EMPTY.

    def steal_credentials(self, ctx: RequestContext, data: ProbeIn) -> ProbeOut:
        """Sweep this process for the pod's signing identity.

        Three routes, because closing one is not closing the class: the
        environment (`WORKER_JWT` at pod-launch), the loaded Settings, and the
        transport object the framework hands every handler's process.
        """
        found = []
        if str(os.environ.get("WORKER_JWT", "") or "").strip():
            found.append("env:WORKER_JWT")
        try:
            from gen_worker.config import get_settings

            if str(getattr(get_settings(), "worker_jwt", "") or "").strip():
                found.append("settings.worker_jwt")
        except Exception:
            pass
        try:
            import gc

            from gen_worker.procsplit.child import ChildTransport

            for obj in gc.get_objects():
                if isinstance(obj, ChildTransport) and obj.current_worker_jwt:
                    found.append("transport.current_worker_jwt")
                    break
        except Exception:
            pass
        return ProbeOut(response=",".join(found))

    def forge_hub_call(self, ctx: RequestContext, data: ProbeIn) -> ProbeOut:
        """Ask the control parent to make a call the allowlist does not name.

        The IPC surface is the child's only route to worker authority, so it is
        an authorization surface: an un-named path must be REFUSED, not proxied.
        """
        from gen_worker.procsplit import broker

        if not broker.active():
            return ProbeOut(response="no-broker")
        try:
            broker.request("GET", str(data.text or "/v1/worker/secrets"))
        except Exception as exc:
            return ProbeOut(response=f"refused:{exc}")
        return ProbeOut(response="PERFORMED")

    def c2pa_sign(self, ctx: RequestContext, data: ProbeIn) -> ProbeOut:
        """Sign a claim through the REAL content_credentials path (delta 5).

        The child has no worker JWT, so the ask must reach the hub through the
        parent. `data.text` is the hub base the handler would use — it is
        ignored under the split, which is itself the point.
        """
        from gen_worker import content_credentials as cc

        remote = cc._RemoteSigner(base_url=str(data.text or ""), worker_jwt=lambda: "")
        try:
            sig = cc._hub_sign_claim(remote, "es256", b"claim-to-be-signed")
        except Exception as exc:
            return ProbeOut(response=f"refused:{exc}")
        return ProbeOut(response=f"signed:{sig.decode(errors='replace')}")

    def forge_capability_renew(self, ctx: RequestContext, data: ProbeIn) -> ProbeOut:
        """Renew a capability for a request this worker was never given.

        The path IS allowlisted, so only the parent's own in-flight table can
        refuse it — the check that needs parent state, not a path pattern.
        """
        from gen_worker.procsplit import broker

        if not broker.active():
            return ProbeOut(response="no-broker")
        try:
            broker.request(
                "POST",
                "/v1/worker/capability/renew",
                json={
                    "request_id": str(data.text or "someone-elses-request"),
                    "attempt": 1,
                    "capability_token": "stolen",
                },
            )
        except Exception as exc:
            return ProbeOut(response=f"refused:{exc}")
        return ProbeOut(response="PERFORMED")

    def die_hard(self, ctx: RequestContext, data: ProbeIn) -> ProbeOut:
        """SIGKILL self: the cgroup-OOM shape — no exception, no finally."""
        os.kill(os.getpid(), signal.SIGKILL)
        return ProbeOut(response="unreachable")

    def segfault(self, ctx: RequestContext, data: ProbeIn) -> ProbeOut:
        """Real native fault: exercises per-group faulthandler attribution."""
        import ctypes

        ctypes.string_at(0)
        return ProbeOut(response="unreachable")

    def sleepy(self, ctx: RequestContext, data: ProbeIn) -> ProbeOut:
        """Long cancellable job: measures cancel latency across the seam."""
        for _ in range(1200):
            ctx.raise_if_cancelled()
            time.sleep(0.05)
        return ProbeOut(response="done")

    def freeze(self, ctx: RequestContext, data: ProbeIn) -> ProbeOut:
        """SIGSTOP self: a wedge the WatchdogSec analog must detect."""
        os.kill(os.getpid(), signal.SIGSTOP)
        return ProbeOut(response="unfrozen")

    async def starve_loop(self, ctx: RequestContext, data: ProbeIn) -> ProbeOut:
        """pgw#771: an inductor compile's shape — an ASYNC handler that burns
        CPU without yielding, so the event loop (and every ping riding it) goes
        silent while the process is demonstrably alive and working.

        Wrapped in the real activity + evidence watchdog bracket, because that
        is what a self-mint compile does and it is what the parent's hang
        verdict is required to defer to.
        """
        seconds = float(data.text or "8")
        with activity.running("self_mint_compile") as act:
            with activity.watchdog(act):
                deadline = time.monotonic() + seconds
                while time.monotonic() < deadline:
                    pow(7, 4001, 10**9 + 7)   # real CPU, no await, no yield
        return ProbeOut(response=f"compiled:{seconds:.0f}")

    async def async_wait(self, ctx: RequestContext, data: ProbeIn) -> ProbeOut:
        """A job that legitimately WAITS: real asyncio sleeps, so the process
        burns no CPU and moves no disk while its loop keeps turning.

        The shape that falsified the parent's first stall report — a healthy
        15s marco-polo-slow was called stalled on the first real-stack run
        because /proc evidence alone cannot tell waiting from wedged.
        """
        for _ in range(int(float(data.text or "8") / 0.1)):
            await asyncio.sleep(0.1)
        return ProbeOut(response="waited")
