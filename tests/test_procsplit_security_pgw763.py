"""pgw#763 driver 3: the parent/child seam as an AUTHORIZATION boundary.

Stages 1-4 built the seam for resilience. These rows prove it is also the
security boundary the issue's driver 3 says it must be: that the things tenant
endpoint code could forge or steal while it shared the worker's process are now
either impossible to reach or refused when asked for.

Every row is the same shape as the fix it guards — an attack that no longer
works, plus the legitimate use of the same path still working. The attacks are
run by REAL endpoint handlers in a REAL compute child (that is the threat model:
tenant code is imported into this process), or by a hostile frame peer that
speaks the seam protocol directly (for the forgeries a well-behaved child would
never emit).

Run: uv run pytest tests/test_procsplit_security_pgw763.py -q
"""

from __future__ import annotations

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import msgspec
import pytest

from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.procsplit import actions

from harness.hub_double import is_ready, is_result_for
from test_procsplit_pgw763 import (  # noqa: F401 — fixtures come with it
    BOOT_TIMEOUT_S,
    CHILD_MAIN,
    SplitHarness,
    _payload,
    captured_dials,
    isolated_postmortem,
)

WORKER_JWT = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ3LXBhcmVudCIsInJlbGVhc2VfaWQiOiJyZWwtNzYzIn0.sig"


def _text(msg: pb.WorkerMessage) -> str:
    """The handler's ProbeOut.response out of an OK JobResult."""
    return msgspec.msgpack.decode(msg.job_result.inline)["response"]


# --------------------------------------------------------------------------
# A hub double for the parent's MEDIATED calls: a real HTTP server, so the
# "legitimate path still works" half is an end-to-end round trip (child asks ->
# parent authorizes -> parent attaches the JWT -> hub answers -> child reads).
# --------------------------------------------------------------------------


class _HubHTTP(BaseHTTPRequestHandler):
    def _answer(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}") \
            if self.command == "POST" else {}
        self.server.calls.append({  # type: ignore[attr-defined]
            "method": self.command,
            "path": self.path,
            "authorization": self.headers.get("Authorization", ""),
            "body": body,
        })
        payload = json.dumps(
            self.server.reply  # type: ignore[attr-defined]
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    do_GET = _answer
    do_POST = _answer

    def log_message(self, *a: Any) -> None:
        pass


@pytest.fixture()
def hub_http():
    srv = HTTPServer(("127.0.0.1", 0), _HubHTTP)
    srv.calls: List[Dict[str, Any]] = []          # type: ignore[attr-defined]
    srv.reply = {"capability_token": "fresh-token", "expires_at_unix": 4102444800}  # type: ignore[attr-defined]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield srv
    finally:
        srv.shutdown()
        srv.server_close()


@pytest.fixture()
def credentialed_split(tmp_path, captured_dials, monkeypatch, hub_http):
    """A split whose PARENT holds a real-shaped worker JWT, delivered the way a
    pod gets one: in the process environment."""
    monkeypatch.setenv("WORKER_JWT", WORKER_JWT)
    h = SplitHarness(
        tmp_path,
        extra_child_env={"PGW763_CHILD_MODULES": "harness.procsplit_endpoints"},
    )
    h.scheduler.file_base_url = f"http://127.0.0.1:{hub_http.server_address[1]}"
    # The harness builds Settings with worker_jwt="" (its own default); give the
    # parent the credential a real pod is launched with.
    h.pc._settings = msgspec.structs.replace(h.pc._settings, worker_jwt=WORKER_JWT)
    h.pc.transport._settings = h.pc._settings
    try:
        yield h
    finally:
        h.close()


# ==========================================================================
# DELTA 1 — the worker JWT is not in the child, and cannot be borrowed freely
# ==========================================================================


def test_delta1_tenant_code_finds_no_worker_jwt_in_its_process(credentialed_split):
    """THE ATTACK: an endpoint handler reads the pod's signing identity.

    It is one `os.environ["WORKER_JWT"]` away in a single-process worker, and
    the deleted T_TOKEN frame used to re-deliver it on every rotation. The
    handler sweeps all three routes — environment, loaded Settings, the
    transport object — and must come back with nothing.
    """
    conn = credentialed_split.scheduler.wait_connection(0, timeout=BOOT_TIMEOUT_S)
    conn.wait_for(is_ready, timeout=BOOT_TIMEOUT_S)

    conn.send(run_job=pb.RunJob(
        request_id="r-steal", attempt=1, function_name="steal-credentials",
        input_payload=_payload()))
    got = conn.wait_for(is_result_for("r-steal"), timeout=60.0)
    assert got.job_result.status == pb.JOB_STATUS_OK
    leaked = _text(got)
    assert leaked == "", (
        f"tenant code reached the worker JWT via {leaked} — the compute child "
        "must hold no signing identity (pgw#763 delta 1 / th#1311)"
    )

    # ...and the PARENT still has it: the credential moved, it did not vanish.
    assert credentialed_split.pc.transport.current_worker_jwt == WORKER_JWT


def test_delta1_parent_refuses_a_hub_call_the_allowlist_does_not_name(
    credentialed_split, hub_http, captured_dials,
):
    """THE ATTACK: with no credential of its own, the child asks the parent to
    make the call for it — an un-named path, i.e. the parent used as an open
    proxy for its own JWT.

    Mediation is only a boundary if the parent DECIDES. An un-named path is
    refused, the credential never goes on the wire, and the refusal is banked
    as a security event rather than logged and forgotten.
    """
    conn = credentialed_split.scheduler.wait_connection(0, timeout=BOOT_TIMEOUT_S)
    conn.wait_for(is_ready, timeout=BOOT_TIMEOUT_S)

    conn.send(run_job=pb.RunJob(
        request_id="r-forge", attempt=1, function_name="forge-hub-call",
        input_payload=_payload("/v1/admin/orgs")))
    got = conn.wait_for(is_result_for("r-forge"), timeout=60.0)
    assert got.job_result.status == pb.JOB_STATUS_OK
    answer = _text(got)
    assert answer.startswith("refused:"), (
        f"the parent performed an un-allowlisted hub call for the child ({answer})"
    )
    assert "not an allowlisted parent-mediated action" in answer
    assert credentialed_split.pc.actions_refused >= 1
    # Nothing reached the hub, so the JWT was never presented.
    assert not [c for c in hub_http.calls if "/v1/admin/" in c["path"]]
    assert any("compute_action_refused" in d for d in captured_dials)


def test_delta1_parent_refuses_capability_renewal_for_a_foreign_request(
    credentialed_split, hub_http,
):
    """THE ATTACK: the path IS allowlisted, so only parent STATE can refuse it —
    a renewal for a request this worker was never dispatched. A path allowlist
    alone would have forwarded it and let the hub decide with less context than
    the parent has."""
    conn = credentialed_split.scheduler.wait_connection(0, timeout=BOOT_TIMEOUT_S)
    conn.wait_for(is_ready, timeout=BOOT_TIMEOUT_S)

    conn.send(run_job=pb.RunJob(
        request_id="r-renew-forge", attempt=1,
        function_name="forge-capability-renew",
        input_payload=_payload("victim-request-id")))
    got = conn.wait_for(is_result_for("r-renew-forge"), timeout=60.0)
    answer = _text(got)
    assert answer.startswith("refused:"), (
        f"the parent renewed a capability for a job it never dispatched ({answer})"
    )
    assert "not an in-flight job on this worker" in answer
    assert not [c for c in hub_http.calls if "capability/renew" in c["path"]]


def test_delta1_the_legitimate_mediated_call_still_works(credentialed_split, hub_http):
    """The other half: an ALLOWLISTED action for a job that IS in flight goes
    through — the parent attaches the credential, the hub sees a normal
    authenticated call, and the child gets the answer without ever holding the
    bearer."""
    pc = credentialed_split.pc
    conn = credentialed_split.scheduler.wait_connection(0, timeout=BOOT_TIMEOUT_S)
    conn.wait_for(is_ready, timeout=BOOT_TIMEOUT_S)

    # Stand in for a dispatched job (the relay records exactly this).
    pc._in_flight[("r-live", 1)] = "echo"
    status, body = _ask(pc, {
        "method": "POST",
        "path": "/v1/worker/capability/renew",
        "json": {"request_id": "r-live", "attempt": 1, "capability_token": "old"},
    })
    assert status == 200, body
    assert json.loads(body)["capability_token"] == "fresh-token"

    call = [c for c in hub_http.calls if "capability/renew" in c["path"]][-1]
    assert call["authorization"] == f"Bearer {WORKER_JWT}", (
        "the parent must present the worker JWT on the child's behalf"
    )
    assert call["body"]["request_id"] == "r-live"


def _ask(pc, req: Dict[str, Any]) -> Tuple[int, str]:
    """Drive one action through the parent's real authorization path."""
    import asyncio

    fut = asyncio.run_coroutine_threadsafe(pc._perform_action(req), pc._loop)
    out = fut.result(60.0)
    return int(out["status"]), str(out["body"])


# ==========================================================================
# DELTA 2 — hardware and the canary are measured by the PARENT, pre-import
# ==========================================================================


FAKE_CHILD = Path(__file__).resolve().parent / "harness" / "procsplit_fake_child.py"


@pytest.fixture()
def forging_split(tmp_path, captured_dials, monkeypatch):
    """A hostile compute child: it answers the Hello request with fabricated
    identity, silicon and canary numbers. The real child's Hello builder is
    reachable by tenant code, so this is what that code could send."""
    monkeypatch.setenv("WORKER_JWT", WORKER_JWT)
    h = SplitHarness(
        tmp_path,
        child_cmd=[sys.executable, str(FAKE_CHILD)],
        extra_child_env={"PGW763_FAKE_MODE": "forge_hello"},
    )
    h.pc._settings = msgspec.structs.replace(
        h.pc._settings, worker_jwt=WORKER_JWT, worker_image_digest="sha256:real")
    h.pc.transport._settings = h.pc._settings
    try:
        yield h
    finally:
        h.close()


def test_delta2_parent_measurement_replaces_a_forged_hello(forging_split):
    """THE ATTACK: the child reports the hardware and the boot canary.

    Each of these is a fleet-wide verdict key — HardwareUnsuitable fences a
    machine, HostCanary condemns a SKU on the SPFabricLedger, and gpu_name
    chooses which key gets written — and they were measured in
    `Lifecycle.build_resources`, i.e. AFTER `Worker.__init__` imported the
    tenant's modules. The parent measures them in a subprocess that imports no
    endpoint code and stamps its own numbers onto the Hello.
    """
    from harness.procsplit_fake_child import (  # type: ignore
        FORGED_GPU_NAME,
        FORGED_MEMCPY_GBPS,
        FORGED_RELEASE_ID,
        FORGED_VRAM_BYTES,
        FORGED_WORKER_ID,
    )

    conn = forging_split.scheduler.wait_connection(0, timeout=BOOT_TIMEOUT_S)
    hello = conn.hello
    assert hello is not None

    # Identity (delta 1): asserted by the credential holder, not the child.
    assert hello.worker_id != FORGED_WORKER_ID, (
        "the child named another worker and the hub believed it"
    )
    assert hello.release_id != FORGED_RELEASE_ID

    # Silicon + canary (delta 2): the parent's measurement, or nothing at all.
    res = hello.resources
    assert res.gpu_name != FORGED_GPU_NAME, (
        f"gpu_name={res.gpu_name!r} came from the child — a forged SKU picks "
        "the fleet-wide verdict key (th#1310)"
    )
    assert res.vram_total_bytes != FORGED_VRAM_BYTES
    assert res.gpu_sm != "90"
    assert res.torch_version != "9.9.9"
    assert res.gen_worker_version != "0.0.0-forged"
    assert res.host_canary.memcpy_gbps != FORGED_MEMCPY_GBPS, (
        "a fabricated HostCanary reached the hub; it condemns SKUs"
    )
    assert res.host_canary.d2h_gbps != FORGED_MEMCPY_GBPS
    assert res.host_canary.interconnect != "nvlink"
    # instance_id/image_digest identify the POD, so they come from the process
    # that holds the pod's credential.
    assert res.instance_id != "pod-belonging-to-someone-else"
    assert res.image_digest in ("", "sha256:real")


def test_delta2_the_parent_measures_the_real_host(forging_split):
    """The legitimate half: the numbers on the wire are the ones this box
    actually has, produced by the parent's pre-import measurement — not zeros
    from having simply deleted the child's report."""
    forging_split.scheduler.wait_connection(0, timeout=BOOT_TIMEOUT_S)
    pc = forging_split.pc
    assert pc._measurement is not None, "the parent never measured the host"

    from gen_worker.procsplit.measure import measure

    truth = measure()
    hw = pc._measurement.get("hardware") or {}
    assert hw.get("gpu_name", "") == (truth.get("hardware") or {}).get("gpu_name", "")
    assert hw.get("gpu_count", 0) == (truth.get("hardware") or {}).get("gpu_count", 0)
    assert pc._measurement.get("gen_worker_version") == truth.get("gen_worker_version")
    # The canary is measured on a box with a GPU; on a CPU-only box it is
    # absent, and absent is the honest answer — never a fabricated one.
    assert ("canary" in pc._measurement) == ("canary" in truth)


def test_delta2_measurement_process_imports_no_endpoint_module():
    """The property that makes the measurement trustworthy at all: the process
    that produces it never names a tenant module. A regression here (someone
    importing `worker` or `registry` for convenience) silently restores the
    forgery, so it is asserted on the source, not on behaviour."""
    import ast

    src = (
        Path(__file__).resolve().parent.parent
        / "src" / "gen_worker" / "procsplit" / "measure.py"
    ).read_text()
    tree = ast.parse(src)

    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add("." * node.level + (node.module or ""))
            imported.update(a.name for a in node.names)
    # Endpoint discovery lives behind exactly these two names; either one in
    # this process means tenant code ran before the numbers were taken.
    for banned in ("collect_endpoints", "registry", "worker", "..worker",
                   "..registry", "Worker"):
        assert banned not in imported, (
            f"the pre-import measurement imports {banned!r} — it must reach no "
            "endpoint-discovery code (pgw#763 delta 2)"
        )
    # And nothing dynamic can smuggle one in.
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in ("__import__", "eval", "exec"), (
                "dynamic import in the pre-import measurement"
            )


# ==========================================================================
# DELTA 3 — billables the parent can observe are attested by the parent
# ==========================================================================


@pytest.fixture()
def billing_split(tmp_path, captured_dials, monkeypatch):
    monkeypatch.setenv("WORKER_JWT", WORKER_JWT)
    h = SplitHarness(
        tmp_path,
        child_cmd=[sys.executable, str(FAKE_CHILD)],
        extra_child_env={"PGW763_FAKE_MODE": "forge_metrics"},
    )
    try:
        yield h
    finally:
        h.close()


def test_delta3_forged_billables_are_replaced_by_the_parents_observation(
    billing_split, captured_dials,
):
    """THE ATTACK: the child reports its own billing quantities (th#1309).

    Here it claims three HOURS of runtime, queue, slot-held and finalize wall
    for a job the parent watched take milliseconds, plus a fabricated
    concurrency and a 999 GiB RSS. Everything the parent could observe from
    outside the process comes back to what the parent observed.
    """
    from harness.procsplit_fake_child import (  # type: ignore
        FORGED_CONCURRENCY,
        FORGED_RSS_BYTES,
        FORGED_RUNTIME_MS,
    )

    conn = billing_split.scheduler.wait_connection(0, timeout=BOOT_TIMEOUT_S)
    conn.send(run_job=pb.RunJob(
        request_id="r-bill", attempt=1, function_name="echo",
        input_payload=_payload()))
    got = conn.wait_for(is_result_for("r-bill"), timeout=60.0)
    m = got.job_result.metrics

    assert m.runtime_ms < FORGED_RUNTIME_MS, (
        f"runtime_ms={m.runtime_ms} survived: the code being billed set its own "
        "billable wall clock (th#1309)"
    )
    assert m.runtime_ms < 60_000, "the clamp must be the OBSERVED wall, not a cap"
    for name in ("queue_ms", "slot_held_ms", "finalize_wall_ms"):
        assert getattr(m, name) < FORGED_RUNTIME_MS, f"{name} survived unattested"
    assert m.concurrency_at_start != FORGED_CONCURRENCY
    assert m.concurrency_at_start == 0, (
        "concurrency_at_start must be the parent's own dispatch-time count"
    )
    assert m.rss_at_end_bytes != FORGED_RSS_BYTES, (
        "rss_at_end_bytes is a /proc reading the parent takes; a process is not "
        "the witness for its own resource use"
    )
    # The divergence is BANKED, not merely clamped away in silence.
    assert billing_split.pc.metric_divergences >= 1
    assert any("compute_billing_attestation" in d for d in captured_dials)
    # ...and the quantity the parent CANNOT corroborate is named, not faked:
    # output_media_duration_s=0 with outputs present is th#1309's $0 bill.
    assert any("output_media_duration_s=0" in d for d in captured_dials)


def test_delta3_an_honest_report_passes_through_unchanged():
    """The other half: attestation is not a rewrite. A child whose numbers
    agree with what the parent watched keeps them — including the quantities
    the parent deliberately does not measure."""
    from gen_worker.procsplit import attest

    metrics = pb.JobMetrics(
        runtime_ms=1200, queue_ms=30, slot_held_ms=1100, finalize_wall_ms=90,
        concurrency_at_start=2, rss_at_end_bytes=4 << 30,
        output_media_duration_s=8.5, output_count=1,
        input_tokens=120, output_tokens=64, lane="fp8-w8a8-dynamic+compiled",
    )
    obs = attest.JobObservation(
        function="generate",
        relayed_at=0.0,
        concurrency_at_relay=2,
    )
    divergences = attest.attest(
        metrics, obs, now=1.6, child_rss_bytes=(4 << 30) + 1000, status_ok=True)

    assert divergences == [], divergences
    assert metrics.runtime_ms == 1200 and metrics.queue_ms == 30
    assert metrics.concurrency_at_start == 2
    # Untouched by design — measuring these parent-side would mean routing the
    # data plane through the parent's interpreter (th#1309 owns the hub bound).
    assert metrics.output_media_duration_s == 8.5
    assert metrics.input_tokens == 120 and metrics.output_tokens == 64
    assert metrics.lane == "fp8-w8a8-dynamic+compiled"


# ==========================================================================
# DELTA 5 — C2PA signing is a parent-side action; the child sends a hash
# ==========================================================================


def test_delta5_the_child_signs_through_the_parent_holding_no_credential(
    credentialed_split, hub_http,
):
    """th#1307, finished at the seam.

    The platform private key already left the pod (the credbound lane made
    signing a hub oracle). What remained was WHO makes the call: the oracle is
    authenticated with the pod's worker JWT, and that call was made from the
    process running tenant code. Now the child sends the claim's to-be-signed
    octets and receives a signature; the credential is the parent's.
    """
    import base64

    hub_http.reply = {"signature_b64": base64.b64encode(b"SIGNATURE").decode()}
    conn = credentialed_split.scheduler.wait_connection(0, timeout=BOOT_TIMEOUT_S)
    conn.wait_for(is_ready, timeout=BOOT_TIMEOUT_S)

    # The handler passes a base URL of its own choosing. It is IGNORED: the
    # parent aims its own credential, at the host the hub named (th#1312).
    conn.send(run_job=pb.RunJob(
        request_id="r-sign", attempt=1, function_name="c2pa-sign",
        input_payload=_payload("http://attacker.invalid")))
    got = conn.wait_for(is_result_for("r-sign"), timeout=60.0)
    assert got.job_result.status == pb.JOB_STATUS_OK
    assert _text(got) == "signed:SIGNATURE", _text(got)

    call = [c for c in hub_http.calls if c["path"] == "/v1/worker/c2pa/sign"][-1]
    assert call["authorization"] == f"Bearer {WORKER_JWT}", (
        "the signing oracle must be authenticated by the parent's credential"
    )
    # A hash and the algorithm. Nothing else crosses — not the media, not a
    # destination, not a header.
    assert set(call["body"]) == {"alg", "claim_b64"}, call["body"]
    assert base64.b64decode(call["body"]["claim_b64"]) == b"claim-to-be-signed"


def test_delta5_the_sign_action_cannot_be_widened(credentialed_split, hub_http):
    """The narrowness IS the fix: the child asks the parent to sign a hash it
    was given, and nothing more. A body key the action does not name is
    refused, so the oracle cannot be turned into a general signer."""
    with pytest.raises(actions.ActionRefused):
        actions.authorize({
            "method": "POST", "path": "/v1/worker/c2pa/sign",
            "json": {"alg": "es256", "claim_b64": "AA==", "key_id": "platform"},
        })


# ==========================================================================
# DELTA 4 — the parent DECIDES on the per-job capability token
# ==========================================================================


def _cap_token(**claims: Any) -> str:
    """An unsigned-shaped JWT carrying the hub's real capability claims. The
    parent reads claims unverified by design — it is deciding whether the grant
    matches the job, which no signature can answer."""
    import base64

    def seg(obj: Dict[str, Any]) -> str:
        raw = json.dumps(obj).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    body: Dict[str, Any] = {
        "cap_kind": "worker_capability",
        "iat": int(__import__("time").time()),
        "exp": int(__import__("time").time()) + 900,
        "grants": [{"resource": "media", "actions": ["write"]}],
    }
    body.update(claims)
    return f"{seg({'alg': 'RS256'})}.{seg(body)}.sig"


def test_delta4_a_grant_for_another_request_is_withheld(split_for_capability):
    """THE ATTACK (or the hub bug that looks like one): a job arrives carrying a
    capability token minted for a DIFFERENT caller's request.

    The token was relayed verbatim, so handler code would have run under
    authority derived from someone else's request. The parent now refuses,
    strips the value, and answers the job typed instead of letting the child
    discover it mid-upload.
    """
    pc, conn = split_for_capability
    conn.send(run_job=pb.RunJob(
        request_id="r-mine", attempt=1, function_name="echo",
        capability_token=_cap_token(
            request_id="r-someone-else", attempt=1, worker_id="split-parent"),
        input_payload=_payload()))
    got = conn.wait_for(is_result_for("r-mine"), timeout=30.0)
    assert got.job_result.status == pb.JOB_STATUS_FATAL
    assert "CapabilityWithheld" in got.job_result.safe_message
    assert "scoped to request r-someone-else" in got.job_result.safe_message
    assert pc.capability_withheld >= 1
    # Never dispatched: no accounting for a job the parent refused.
    assert ("r-mine", 1) not in pc._in_flight


def test_delta4_an_expired_grant_is_withheld_retryable(split_for_capability):
    """A grant that is already dead cannot upload the job's output. Refusing at
    dispatch turns a mid-job auth failure into a legible retry."""
    import time as _t

    pc, conn = split_for_capability
    conn.send(run_job=pb.RunJob(
        request_id="r-stale", attempt=1, function_name="echo",
        capability_token=_cap_token(
            request_id="r-stale", attempt=1, worker_id="split-parent",
            iat=int(_t.time()) - 7200, exp=int(_t.time()) - 600),
        input_payload=_payload()))
    got = conn.wait_for(is_result_for("r-stale"), timeout=30.0)
    assert got.job_result.status == pb.JOB_STATUS_RETRYABLE
    assert "expired" in got.job_result.safe_message


def test_delta4_a_correctly_scoped_grant_is_forwarded(split_for_capability):
    """The other half: a grant that names this job on this worker goes through
    untouched — least authority in the child is explicitly allowed, and the
    child genuinely needs it (inputs and outputs go child -> object store)."""
    pc, conn = split_for_capability
    token = _cap_token(request_id="r-ok", attempt=1, function_name="echo",
                       worker_id="split-parent")
    conn.send(run_job=pb.RunJob(
        request_id="r-ok", attempt=1, function_name="echo",
        capability_token=token, input_payload=_payload()))
    got = conn.wait_for(is_result_for("r-ok"), timeout=30.0)
    assert got.job_result.status == pb.JOB_STATUS_OK
    assert pc.capability_withheld == 0


@pytest.fixture()
def split_for_capability(tmp_path, captured_dials, monkeypatch):
    monkeypatch.setenv("WORKER_JWT", WORKER_JWT)
    h = SplitHarness(
        tmp_path,
        child_cmd=[sys.executable, str(FAKE_CHILD)],
        extra_child_env={"PGW763_FAKE_MODE": "result_then_exit"},
    )
    try:
        conn = h.scheduler.wait_connection(0, timeout=BOOT_TIMEOUT_S)
        yield h.pc, conn
    finally:
        h.close()


@pytest.mark.parametrize(
    "claims,forward",
    [
        ({"request_id": "r-1", "attempt": 1, "worker_id": "w-1"}, True),
        ({"request_id": "r-2", "attempt": 1, "worker_id": "w-1"}, False),
        ({"request_id": "r-1", "attempt": 2, "worker_id": "w-1"}, False),
        ({"request_id": "r-1", "attempt": 1, "worker_id": "w-other"}, False),
        ({"request_id": "r-1", "attempt": 1, "worker_id": "w-1",
          "function_name": "other-fn"}, False),
        ({"request_id": "r-1", "attempt": 1, "worker_id": "w-1",
          "cap_kind": "org_access_token"}, False),
    ],
)
def test_capability_policy_matrix(claims, forward):
    from gen_worker.procsplit import capability

    d = capability.decide(
        _cap_token(**claims),
        request_id="r-1", attempt=1, function_name="generate", worker_id="w-1")
    assert d.forward is forward, d.reason


def test_capability_policy_reports_an_over_long_ttl_without_refusing():
    """Only the hub can shorten a TTL, so refusing legitimate work over one
    would trade a real outage for a theoretical exposure. Report it."""
    import time as _t

    from gen_worker.procsplit import capability

    d = capability.decide(
        _cap_token(request_id="r-1", attempt=1, worker_id="w-1",
                   iat=int(_t.time()),
                   exp=int(_t.time()) + capability.MAX_EXPECTED_TTL_S + 3600),
        request_id="r-1", attempt=1, worker_id="w-1")
    assert d.forward is True
    assert "TTL" in d.note


def test_capability_policy_passes_a_job_with_no_grant():
    """Jobs with no file authority legitimately exist; a missing token is not a
    withheld one."""
    from gen_worker.procsplit import capability

    assert capability.decide("", request_id="r", attempt=1).forward is True


# ==========================================================================
# The allowlist itself — unit rows, because the table IS the policy
# ==========================================================================


@pytest.mark.parametrize(
    "req,why",
    [
        ({"method": "GET", "path": "/v1/worker/secrets"}, "unlisted path"),
        ({"method": "POST", "path": "/v1/worker/cells/receipt"}, "wrong method"),
        ({"method": "GET", "path": "/api/v1/repos/a/b/../../admin/resolve"},
         "traversal out of the allowlisted prefix"),
        ({"method": "GET", "path": "/v1/worker/cells/receipt",
          "query": {"blake3": "x", "owner": "root"}}, "query key not in the action"),
        ({"method": "POST", "path": "/v1/worker/c2pa/sign",
          "json": {"alg": "es256", "claim_b64": "AA", "callback_url": "http://evil"}},
         "body key not in the action"),
        ({"method": "POST", "path": "/v1/worker/capability/renew",
          "json": {"request_id": "r", "attempt": 1,
                   "capability_token": "x" * (300 * 1024)}},
         "oversized body: the seam carries control, not data"),
    ],
)
def test_action_table_refuses(req, why):
    with pytest.raises(actions.ActionRefused):
        actions.authorize(req)


def test_action_table_admits_exactly_the_named_actions():
    for req in (
        {"method": "POST", "path": "/v1/worker/capability/renew",
         "json": {"request_id": "r", "attempt": 1, "capability_token": "t"}},
        {"method": "POST", "path": "/v1/worker/c2pa/sign",
         "json": {"alg": "es256", "claim_b64": "AA=="}},
        {"method": "GET", "path": "/v1/worker/cells/receipt",
         "query": {"blake3": "b3", "cell_key": "k"}},
        {"method": "GET", "path": "/v1/worker/cells/revocations"},
        {"method": "GET", "path": "/api/v1/repos/root/system-sdxl/checkpoints",
         "query": {"limit": "50"}},
        {"method": "GET", "path": "/api/v1/repos/root/system-sdxl/resolve",
         "query": {"digest": "ck5-abc"}},
    ):
        actions.authorize(req)


def test_no_frame_carries_the_worker_jwt():
    """A ratchet over the frame vocabulary itself: T_TOKEN is gone and nothing
    replaced it. Catches a future frame that re-opens the same hole under a
    different name."""
    from gen_worker.procsplit import frames

    assert not hasattr(frames, "T_TOKEN")
    names = [n for n in dir(frames) if n.startswith("T_")]
    for name in names:
        assert "TOKEN" not in name and "JWT" not in name, (
            f"frame {name} looks like it carries a credential to the compute child"
        )
