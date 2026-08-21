from __future__ import annotations

import asyncio
import os
import signal
import time

import msgspec

from gen_worker import RequestContext, activity, entrypoint


class ProbeIn(msgspec.Struct):
    text: str = ""


class ProbeOut(msgspec.Struct):
    response: str

@entrypoint
def echo(ctx: RequestContext, data: ProbeIn) -> ProbeOut:
    return ProbeOut(response=f"echo:{data.text}")


@entrypoint
def whoami(ctx: RequestContext, data: ProbeIn) -> ProbeOut:
    return ProbeOut(response=(
        f"g={os.environ.get('GEN_WORKER_GROUP_ORDINAL', '0')}"
        f" cvd={os.environ.get('CUDA_VISIBLE_DEVICES', '-')}"
        f" sib={os.environ.get('GEN_WORKER_HOST_SIBLINGS', '1')}"
    ))

@entrypoint
def steal_credentials(ctx: RequestContext, data: ProbeIn) -> ProbeOut:
    """Sweep this process for the pod's signing identity."""
    found = []
    if str(os.environ.get("WORKER_JWT", "") or "").strip():
        found.append("env:WORKER_JWT")
    try:
        from gen_worker import config as gw_config

        if str(getattr(gw_config.current(), "worker_jwt", "") or "").strip():
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


@entrypoint
def forge_hub_call(ctx: RequestContext, data: ProbeIn) -> ProbeOut:
    """Ask the control parent to make a call the allowlist does not name."""
    from gen_worker.procsplit import broker

    if not broker.active():
        return ProbeOut(response="no-broker")
    try:
        broker.request("GET", str(data.text or "/v1/worker/secrets"))
    except Exception as exc:
        return ProbeOut(response=f"refused:{exc}")
    return ProbeOut(response="PERFORMED")


@entrypoint
def c2pa_sign(ctx: RequestContext, data: ProbeIn) -> ProbeOut:
    """Sign a claim through the REAL content_credentials path (delta 5)."""
    from gen_worker import content_credentials as cc

    remote = cc._RemoteSigner(base_url=str(data.text or ""), worker_jwt=lambda: "")
    try:
        sig = cc._hub_sign_claim(remote, "es256", b"claim-to-be-signed")
    except Exception as exc:
        return ProbeOut(response=f"refused:{exc}")
    return ProbeOut(response=f"signed:{sig.decode(errors='replace')}")


@entrypoint
def who_am_i(ctx: RequestContext, data: ProbeIn) -> ProbeOut:
    from gen_worker import worker_identity

    try:
        me = worker_identity.viewer()
    except Exception as exc:
        return ProbeOut(response=f"refused:{type(exc).__name__}:{exc}")
    return ProbeOut(
        response=f"endpoint={me.endpoint_id} org={me.org_id}")


@entrypoint
def forge_capability_renew(ctx: RequestContext, data: ProbeIn) -> ProbeOut:
    """Renew a capability for a request this worker was never given."""
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


@entrypoint
def die_hard(ctx: RequestContext, data: ProbeIn) -> ProbeOut:
    """SIGKILL self: the cgroup-OOM shape — no exception, no finally."""
    os.kill(os.getpid(), signal.SIGKILL)
    return ProbeOut(response="unreachable")


@entrypoint
def segfault(ctx: RequestContext, data: ProbeIn) -> ProbeOut:
    """Real native fault: exercises per-group faulthandler attribution."""
    import ctypes

    ctypes.string_at(0)
    return ProbeOut(response="unreachable")


@entrypoint
async def activity_die(ctx: RequestContext, data: ProbeIn) -> ProbeOut:
    act = activity.begin(str(data.text or "g_hold"), phase="holding")
    act.heartbeat()
    await asyncio.sleep(0.2)
    os.kill(os.getpid(), signal.SIGKILL)
    return ProbeOut(response="unreachable")


@entrypoint
async def activity_hold(ctx: RequestContext, data: ProbeIn) -> ProbeOut:
    """Open the same activity kind and keep it open, beating, until cancelled — the LIVE group whose fact must survive a sibling's death."""
    act = activity.begin(str(data.text or "g_hold"), phase="holding")
    for _ in range(600):
        ctx.raise_if_cancelled()
        act.heartbeat()
        await asyncio.sleep(0.1)
    return ProbeOut(response="held")


@entrypoint
def sleepy(ctx: RequestContext, data: ProbeIn) -> ProbeOut:
    """Long cancellable job: measures cancel latency across the seam."""
    for _ in range(1200):
        ctx.raise_if_cancelled()
        time.sleep(0.05)
    return ProbeOut(response="done")


@entrypoint
def freeze(ctx: RequestContext, data: ProbeIn) -> ProbeOut:
    """SIGSTOP self: a wedge the WatchdogSec analog must detect."""
    os.kill(os.getpid(), signal.SIGSTOP)
    return ProbeOut(response="unfrozen")


@entrypoint
async def starve_loop(ctx: RequestContext, data: ProbeIn) -> ProbeOut:
    seconds = float(data.text or "8")
    with activity.running("self_mint_compile") as act:
        with activity.watchdog(act):
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                pow(7, 4001, 10**9 + 7)
    return ProbeOut(response=f"compiled:{seconds:.0f}")


@entrypoint
async def async_wait(ctx: RequestContext, data: ProbeIn) -> ProbeOut:
    """A job that legitimately WAITS: real asyncio sleeps, so the process burns no CPU and moves no disk while its loop keeps turning."""
    for _ in range(int(float(data.text or "8") / 0.1)):
        await asyncio.sleep(0.1)
    return ProbeOut(response="waited")
