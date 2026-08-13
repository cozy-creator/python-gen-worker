"""pgw#1122: the cell RECEIPT TRUST GATE runs in the process that holds no
credential — so it must not answer "who am I?" by decoding one.

MEASURED, three real pods, 2026-08-11 (RTX 4090, sm_89, gen-worker 0.104.0).
All three derived ``ck1-f023c374…``, issued ``POST /v1/worker/cells/resolve``
(hub GIN log: 3 calls, all 200), were answered with the cell + receipt,
materialized it — and then failed identically::

    OrderedArmError: artifact_receipt_refused: publisher_untrusted:
    this pod cannot name its own endpoint or org (no cell_read_* claim on the
    worker credential), so it may adopt platform-tier cells only

``receipts._self_endpoint_id``/``_self_org_id`` decoded ``cell_read_*`` out of
*this process's* worker JWT, and the gate is armed at ``lifecycle.py`` with
``executor.worker_jwt_provider`` — in the COMPUTE CHILD, whose
``current_worker_jwt`` is ``""`` by construction (``procsplit/child.py``,
pgw#763 delta 1). Both viewer ids were therefore ``""`` on every split serving
pod, so every ORG-tier cell was unadoptable, always. The blast radius was not a
wasted download: the hub logged ``worker_function_unavailable
reason=compile_cell_failed``, the pod never served, the autoscaler reaped it
``state_blocked_idle`` and bought a replacement — twice.

This is byte-for-byte the pgw#1108 defect class one gate later. The rows below
are therefore about the CLASS, not the call site:

1. the real compute child, in a real split, names itself without holding a JWT;
2. the receipt gate arms an org-tier cell from that child;
3. "nobody can name us" is a DIFFERENT typed refusal from "we asked and the
   answer says no" — the conflation is what made this read like an attack;
4. a refused arm on a cell this pod adopted BY ITS OWN KEY degrades to the
   ordinary boot instead of taking the function down;
5. that degrade is a typed ``boot_adopt`` event, so the next occurrence is one
   query rather than three pods of archaeology;
6. the fence: a NEW credential read has to be classified, and there is no
   classification that means "identity".

Every row fails on ``origin/master``. Rows 1-3 fail as ASSERTIONS against the
shipped behaviour; rows 4-6 fail on master because master cannot express the
distinction at all (no ``adopt`` on ``_ArmOrder``, no ``arm_refused`` terminus,
no fence) — which is the fix, not a gap in the test.

Run: uv run pytest tests/test_receipt_identity_seam_pgw1122.py -q
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import msgspec
import pytest

from gen_worker import (
    activity as activity_mod,
    boot_adopt,
    cell_adopt,
    executor as executor_mod,
    fleet_cells,
    receipts,
    worker_credential,
    worker_identity,
)
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.procsplit import actions, broker

from harness.hub_double import is_ready, is_result_for
from test_procsplit_pgw763 import (  # noqa: F401 — fixtures come with it
    CHILD_MAIN,
    SplitHarness,
    _payload,
    captured_dials,
    isolated_postmortem,
)
from test_receipts_pgw709 import (  # noqa: F401 — fixtures come with it
    FAMILY,
    OTHER_ENDPOINT,
    SELF_ENDPOINT,
    SELF_ORG,
    HubStub,
    hub,
    make_artifact,
    pub_map,
    rsa_key,
    worker_jwt_for,
)

#: The parent's credential, shaped exactly as `cellgrant.Stamp` writes it:
#: the scheduler subject plus the two VIEWER claims the hub stamps from its OWN
#: record of the release (th#1657/th#1680). Nothing here is invented — the
#: fixture manufactures no field production does not set.
POD_ENDPOINT = SELF_ENDPOINT
POD_ORG = SELF_ORG


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def stamped_worker_jwt(
    *, endpoint_id: str = POD_ENDPOINT, org_id: str = POD_ORG,
) -> str:
    claims: Dict[str, Any] = {
        "sub": "w-parent",
        "release_id": "rel-1122",
        "cap_kind": "worker_capability",
        "cell_read_endpoint_id": endpoint_id,
    }
    if org_id:
        claims["cell_read_org_id"] = org_id
    return (
        _b64(json.dumps({"alg": "HS256"}).encode())
        + "." + _b64(json.dumps(claims).encode())
        + ".sig"
    )


@pytest.fixture(autouse=True)
def _clean_identity() -> Iterator[None]:
    """Identity is a PROCESS fact; unwind it around every row."""
    worker_identity.reset()
    worker_credential.reset()
    yield
    worker_identity.reset()
    worker_credential.reset()
    receipts.reset()
    broker.install(None)


# ===========================================================================
# 1. THE REAL CHILD. A real split, a real compute child running real endpoint
#    code, holding no credential — and it can still name itself.
# ===========================================================================


@pytest.fixture()
def stamped_split(tmp_path, captured_dials, monkeypatch):  # noqa: F811
    """A split whose PARENT holds a credential carrying the viewer claims.

    Delivered the way a pod gets one — in the process environment at launch —
    and then stripped from the child by the parent, which is the whole premise.
    """
    token = stamped_worker_jwt()
    monkeypatch.setenv("WORKER_JWT", token)
    h = SplitHarness(
        tmp_path,
        extra_child_env={"PGW763_CHILD_MODULES": "harness.procsplit_endpoints"},
    )
    h.pc._settings = msgspec.structs.replace(
        h.pc._settings, bootstrap_worker_jwt=token)
    h.pc.transport._settings = h.pc._settings
    try:
        yield h
    finally:
        h.close()


def test_the_compute_child_names_itself_without_holding_a_credential(
    stamped_split,
) -> None:
    """THE regression, on the production path: tenant-adjacent code inside the
    real compute child asks who this pod is.

    On master the child had no way to find out at all — the only answer
    available to it was the empty string its own ``current_worker_jwt``
    returns, which the arm gate reads as "adopt platform-tier only".
    """
    conn = stamped_split.scheduler.wait_connection(0)
    conn.wait_for(is_ready)

    conn.send(run_job=pb.RunJob(
        request_id="r-whoami", attempt=1, function_name="who-am-i",
        input_payload=_payload()))
    got = conn.wait_for(is_result_for("r-whoami"), timeout=60.0)
    assert got.job_result.status == pb.JOB_STATUS_OK
    answer = msgspec.msgpack.decode(got.job_result.inline)["response"]

    assert answer == f"endpoint={POD_ENDPOINT} org={POD_ORG}", (
        "a compute child could not name the endpoint/org it serves "
        f"({answer!r}) — the pgw#1122 gate reads exactly this and refuses "
        "every org-tier cell when it comes back empty")


def test_the_relay_carries_the_claims_and_never_the_credential(
    stamped_split,
) -> None:
    """A claim is not a credential. The parent answers ``identity.viewer`` with
    two ids and nothing else — a relay that shipped the token would hand tenant
    code the pod's signing identity, which is the whole of pgw#763 delta 1."""
    pc = stamped_split.pc
    answer = pc._viewer_identity()

    assert set(answer) == {"endpoint_id", "org_id"}
    assert answer["endpoint_id"] == POD_ENDPOINT
    assert answer["org_id"] == POD_ORG
    token = pc.transport.current_worker_jwt
    assert token, "the fixture's premise is that the PARENT holds a credential"
    assert token not in json.dumps(answer)


def test_the_parent_refuses_to_invent_an_identity_it_does_not_have(
    tmp_path, captured_dials,  # noqa: F811
) -> None:
    """A parent with no credential REFUSES rather than answering ``("", "")``.

    An empty answer would be indistinguishable from a hub that stamped no
    claims — a legal state that narrows the pod — and the whole cost of
    pgw#1122 was two states sharing one value.
    """
    h = SplitHarness(tmp_path)
    try:
        h.pc._settings = msgspec.structs.replace(
            h.pc._settings, bootstrap_worker_jwt="")
        h.pc.transport._settings = h.pc._settings
        # pgw#893 §2 deleted the transport's stream-local credential cache;
        # `worker_credential` is now the ONE home, so emptying it is what
        # "this parent holds no credential" means.
        worker_credential.reset()
        with pytest.raises(actions.ActionRefused) as exc:
            h.pc._viewer_identity()
        assert "holds no worker credential" in str(exc.value)
    finally:
        h.close()


# ===========================================================================
# 2. THE GATE. Same seam, one gate later: the receipt trust check.
# ===========================================================================


class _FakeParent:
    """A control seam that answers exactly what the parent answers.

    Not a mock of the code under test: the child-side call is the real
    ``broker.viewer_identity`` over the real action name, and this stands in
    for the process on the other end of the socket. Row 1 already proves the
    real one.
    """

    def __init__(self, endpoint_id: str, org_id: str, base_url: str = "") -> None:
        self.endpoint_id = endpoint_id
        self.org_id = org_id
        self.base_url = base_url.rstrip("/")
        self.asks: List[str] = []

    def call_action(
        self, action: str, args: Dict[str, Any], *, timeout: float = 30.0,
    ) -> Dict[str, Any]:
        self.asks.append(action)
        assert action == actions.ACTION_VIEWER_IDENTITY
        assert args == {}, "the child names no field in an identity ask"
        return {"endpoint_id": self.endpoint_id, "org_id": self.org_id}

    def call(
        self, method: str, path: str, *, params: Any = None, json: Any = None,
        timeout: float = 30.0,
    ) -> broker.HubResponse:
        """The parent's half of a mediated HTTP call: it names the host, it
        attaches the credential, the child sees only the response."""
        import requests

        self.asks.append(f"{method} {path}")
        resp = requests.request(
            method, self.base_url + path, params=params, json=json,
            headers={"Authorization": "Bearer parent-holds-this"},
            timeout=timeout)
        return broker.HubResponse(status_code=resp.status_code, text=resp.text)


def _child_gate(stub: HubStub, parent: Optional[_FakeParent]) -> None:
    """Arm the receipt gate exactly as ``lifecycle.on_hello_ack`` does IN THE
    COMPUTE CHILD: the provider is the child's, so it returns ``""``."""
    worker_credential.reset()
    worker_identity.reset()
    receipts.configure(base_url=stub.base_url, worker_jwt=lambda: "")
    if parent is not None:
        parent.base_url = stub.base_url
    broker.install(parent)


def test_the_child_arms_an_org_tier_cell_it_is_entitled_to(
    tmp_path: Path, hub: HubStub,  # noqa: F811
) -> None:
    """THE POD FAILURE, reproduced: a resolved, materialized, correctly-owned
    org-tier cell reaching the arm gate in a process with no credential.

    On master this raises ``publisher_untrusted`` — on 3 of 3 real pods.
    """
    artifact = make_artifact(tmp_path)
    hub.serve_receipt_for(
        artifact, owning_endpoint_id=POD_ENDPOINT,
        publisher_tier="org", publisher_org_id=POD_ORG)
    parent = _FakeParent(POD_ENDPOINT, POD_ORG)
    _child_gate(hub, parent)

    receipt = receipts.verify_delivered_artifact(artifact, FAMILY)

    assert receipt.publisher_org_id == POD_ORG
    assert receipts.gate_delivered_artifact(artifact, FAMILY) is True
    assert parent.asks.count(actions.ACTION_VIEWER_IDENTITY) == 1, (
        "identity does not change for the life of a pod; asking per arm puts a "
        "seam round trip on every cell")


def test_a_sibling_endpoint_in_the_same_org_arms_from_the_child(
    tmp_path: Path, hub: HubStub,  # noqa: F811
) -> None:
    """th#1680's rule, now actually reachable under the split: the org matches
    even when the endpoint does not. Master could not apply it at all — with
    both ids empty the ``not mine and not my_org`` branch fired first."""
    artifact = make_artifact(tmp_path)
    hub.serve_receipt_for(
        artifact, owning_endpoint_id=OTHER_ENDPOINT,
        publisher_tier="org", publisher_org_id=POD_ORG)
    _child_gate(hub, _FakeParent(POD_ENDPOINT, POD_ORG))

    assert receipts.verify_delivered_artifact(
        artifact, FAMILY).publisher_org_id == POD_ORG


def test_another_orgs_cell_is_still_refused_from_the_child(
    tmp_path: Path, hub: HubStub,  # noqa: F811
) -> None:
    """The threat is unchanged and this fix must not widen it: the artifact is
    a ``.so`` this process is about to ``dlopen``. A relayed identity that
    matches everybody would be worse than one that matches nobody."""
    artifact = make_artifact(tmp_path)
    hub.serve_receipt_for(
        artifact, owning_endpoint_id=OTHER_ENDPOINT, publisher_tier="org",
        publisher_org_id="99999999-0000-0000-0000-000000000000")
    _child_gate(hub, _FakeParent(POD_ENDPOINT, POD_ORG))

    with pytest.raises(receipts.ReceiptError) as exc:
        receipts.verify_delivered_artifact(artifact, FAMILY)
    assert exc.value.reason == "publisher_untrusted"


def test_no_identity_at_all_refuses_LOUDLY_and_by_its_own_name(
    tmp_path: Path, hub: HubStub,  # noqa: F811
) -> None:
    """The structurally-impossible case: no credential here, no seam to ask
    over. It still fails CLOSED — but under its own name.

    ``identity_unavailable`` is a wiring defect on our side;
    ``publisher_untrusted`` is a trust decision about somebody else's cell.
    Master emitted the second for the first, which is why three pods' worth of
    evidence read like an attack on the platform by the platform.
    """
    artifact = make_artifact(tmp_path)
    hub.serve_receipt_for(
        artifact, owning_endpoint_id=POD_ENDPOINT,
        publisher_tier="org", publisher_org_id=POD_ORG)
    _child_gate(hub, None)  # no parent, no credential

    with pytest.raises(receipts.ReceiptError) as exc:
        receipts.verify_delivered_artifact(artifact, FAMILY)
    assert exc.value.reason == "identity_unavailable", (
        "a pod that could not be ASKED about its identity reported the same "
        "reason as a pod whose identity does not match the publisher")

    with pytest.raises(worker_identity.IdentityUnavailable) as ident:
        worker_identity.viewer()
    assert ident.value.reason == "no_credential"


def test_a_platform_tier_cell_still_arms_with_no_identity(
    tmp_path: Path, hub: HubStub,  # noqa: F811
) -> None:
    """The refusal must stay scoped to the org-tier decision it is about: a
    platform-tier cell needs no viewer identity and never asks for one."""
    artifact = make_artifact(tmp_path)
    hub.serve_receipt_for(
        artifact, owning_endpoint_id="", publisher_tier="platform",
        publisher_org_id="")
    _child_gate(hub, None)

    assert receipts.verify_delivered_artifact(artifact, FAMILY).publisher_tier \
        == "platform"


def test_a_hub_that_stamped_no_claims_is_an_ANSWER_not_a_failure(
    tmp_path: Path, hub: HubStub,  # noqa: F811
) -> None:
    """``cellgrant.Stamp`` omits both claims when the hub cannot resolve them,
    which legally narrows the pod to platform-tier. That is a resolved identity
    that names nothing — and it must not be reported as an unreachable one."""
    _child_gate(hub, _FakeParent("", ""))
    me = worker_identity.viewer()
    assert not me.named

    artifact = make_artifact(tmp_path)
    hub.serve_receipt_for(
        artifact, owning_endpoint_id=POD_ENDPOINT,
        publisher_tier="org", publisher_org_id=POD_ORG)
    with pytest.raises(receipts.ReceiptError) as exc:
        receipts.verify_delivered_artifact(artifact, FAMILY)
    assert exc.value.reason == "publisher_untrusted"


# ===========================================================================
# 3. BLAST RADIUS. A refused arm on a self-adopted cell costs a download, not
#    a pod. Master: `worker_function_unavailable reason=compile_cell_failed`.
# ===========================================================================


class _Events:
    """Collect the typed activity events this boot emitted."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.rows: List[Tuple[str, str, str]] = []
        monkeypatch.setattr(
            activity_mod, "emit_event",
            lambda kind, detail, phase="", duration_ms=0, **_kw: self.rows.append(
                (kind, phase, detail)))

    def phases(self, kind: str) -> List[str]:
        return [p for k, p, _ in self.rows if k == kind]

    def detail(self, kind: str, phase: str) -> str:
        return next(d for k, p, d in self.rows if k == kind and p == phase)


def _hit(family: str = FAMILY, function: str = "generate") -> Any:
    """The ``BootAdoptOutcome`` §4.27 produces on a HIT — the exact object the
    executor now carries onto the arm order."""
    return boot_adopt.BootAdoptOutcome(
        adoption=None, reason=boot_adopt.HIT,
        derived_key="ek1-" + "f0" * 28, derive_ms=10_895,
        family=family, function=function)


def _executor(tmp_path: Path) -> Any:
    from gen_worker.executor import Executor, ModelStore

    async def _send(msg: Any) -> None:
        pass

    return Executor([], _send, store=ModelStore(_send, cache_dir=tmp_path / "cas"))


class _Cfg:
    family = FAMILY
    lora_bucket = 0


def _refusing_arm(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(*a: Any, **k: Any) -> Any:
        raise fleet_cells.OrderedArmError(
            "artifact_receipt_refused",
            "publisher_untrusted: this pod cannot name its own endpoint or org")

    monkeypatch.setattr(fleet_cells, "arm_ordered", _raise)


def test_an_adopted_cell_that_will_not_arm_does_not_kill_the_function(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE COST. On master this ``OrderedArmError`` escaped setup, the hub
    logged ``worker_function_unavailable reason=compile_cell_failed``, the pod
    never served, was reaped ``state_blocked_idle``, and a replacement was
    bought — three pods, twice.

    Nothing ORDERED this arm: the pod derived the key, asked, and was answered.
    So the refusal means what every other boot-adopt refusal means.
    """
    events = _Events(monkeypatch)
    _refusing_arm(monkeypatch)
    monkeypatch.setattr(
        fleet_cells, "enable_compiled",
        lambda *a, **k: fleet_cells.ArmOutcome(armed=False))
    monkeypatch.setattr(
        executor_mod.compile_cache, "mandatory_serving", lambda pipe: False)
    ex = _executor(tmp_path)
    order = executor_mod._ArmOrder(
        backend="aot_cell", publisher_org=POD_ORG, adopt=_hit())

    outcome = ex._enable_compiled(object(), _Cfg(), tmp_path / "cell.tar.gz",
                                  None, order)

    assert outcome.armed is False
    assert outcome.eager_reason == cell_adopt.EagerPhase.ADOPTED_CELL_REFUSED

    # ...and it says so ON THE WIRE, under the kind that already carries the
    # rest of this journey, with the refusing gate named (pgw#1116's shape).
    assert "arm_refused" in events.phases(activity_mod.KIND_BOOT_ADOPT)
    detail = events.detail(activity_mod.KIND_BOOT_ADOPT, "arm_refused")
    assert "cause=artifact_receipt_refused" in detail
    assert "publisher_untrusted" in detail
    assert f"family={FAMILY}" in detail and "key=ek1-" in detail


def test_a_HUB_ordered_arm_stays_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other half, and the one this fix must not erode: when the HUB named
    an exact artifact (pgw#904), a substitute would not be it. That refusal is
    still typed and terminal — the degrade is scoped to arms nobody ordered."""
    _Events(monkeypatch)
    _refusing_arm(monkeypatch)
    ex = _executor(tmp_path)
    order = executor_mod._ArmOrder(backend="aot_cell", publisher_org=POD_ORG)

    with pytest.raises(fleet_cells.OrderedArmError) as exc:
        ex._enable_compiled(object(), _Cfg(), tmp_path / "cell.tar.gz",
                            None, order)
    assert exc.value.reason == "artifact_receipt_refused"


def test_a_mandatory_lane_still_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A w8a8/w4a4 lane serves ONLY from a cell (pgw#1010), so "boot as
    yesterday" is not available: degrading there would serve numerics the
    release does not describe. Fail closed, named."""
    _Events(monkeypatch)
    _refusing_arm(monkeypatch)
    monkeypatch.setattr(
        executor_mod.compile_cache, "mandatory_serving", lambda pipe: True)
    ex = _executor(tmp_path)
    order = executor_mod._ArmOrder(
        backend="aot_cell", publisher_org=POD_ORG, adopt=_hit())

    with pytest.raises(fleet_cells.OrderedArmError):
        ex._enable_compiled(object(), _Cfg(), tmp_path / "cell.tar.gz",
                            None, order)


def test_the_degrade_reruns_the_ordinary_policy_with_no_delivered_cell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """"Boot as this pod booted yesterday" is a claim with a mechanism: the
    order is dropped and the FLEET policy runs with no artifact — bit for bit
    the call this method makes on a boot-adopt MISS. The refused cell is not
    handed back in, or the policy would retry the artifact that just refused.
    """
    _Events(monkeypatch)
    _refusing_arm(monkeypatch)
    monkeypatch.setattr(
        executor_mod.compile_cache, "mandatory_serving", lambda pipe: False)
    seen: List[Tuple[Any, ...]] = []

    def _policy(pipe: Any, cfg: Any, cache_dir: Any, artifact: Any,
                **kw: Any) -> Any:
        seen.append((artifact, kw.get("delivered_ref"),
                     kw.get("delivered_digest")))
        return fleet_cells.ArmOutcome(armed=True)

    monkeypatch.setattr(fleet_cells, "enable_compiled", _policy)
    ex = _executor(tmp_path)
    order = executor_mod._ArmOrder(
        backend="aot_cell", publisher_org=POD_ORG, adopt=_hit())

    outcome = ex._enable_compiled(object(), _Cfg(), tmp_path / "cell.tar.gz",
                                  None, order)

    assert outcome.armed is True
    assert seen == [(None, "", "")], (
        "the degrade re-offered the refused artifact to the fleet policy")


def test_arm_refused_is_in_the_boot_adopt_vocabulary() -> None:
    """pgw#1116's fence, extended: the journey's LAST terminus has to be
    enumerable too, or the event that promised to name the gate stops one gate
    short — which is exactly what the three pods measured."""
    assert "arm_refused" in boot_adopt.REASONS


# ===========================================================================
# 4. THE FENCE. The class, not the call site: a THIRD instance has no label.
# ===========================================================================


def _lint() -> Any:
    import sys

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    try:
        import lint_credential_identity as lint
    finally:
        sys.path.remove(str(root / "scripts"))
    return lint


def test_the_fence_catches_a_new_unclassified_credential_read(
    tmp_path: Path,
) -> None:
    """A check that cannot go red proves nothing (pgw#1113). Write the pgw#1122
    bug into a fresh module and prove the fence names it."""
    lint = _lint()
    (tmp_path / "new_gate.py").write_text(
        "class Gate:\n"
        "    def decide(self, cfg):\n"
        "        return cfg.worker_jwt()\n",
        encoding="utf-8")

    sites = lint.scan(tmp_path)
    assert any(site.endswith("Gate.decide::worker_jwt") for _, site in sites)
    problems = lint.check(sites, {})
    assert problems and "UNCLASSIFIED worker-credential read" in problems[0]
    assert "worker_identity.viewer()" in problems[0]


def test_there_is_no_classification_that_means_identity() -> None:
    """The class-closer. Every other credential use has a label; wanting to
    know WHO THIS POD IS does not, because that answer comes from one resolver
    that can ask the parent. A third instance of pgw#1108/pgw#1122 has nothing
    to write in the allowlist."""
    lint = _lint()
    assert "IDENTITY" not in lint.CLASSIFICATIONS
    assert lint.RESOLVER_FILES == {"worker_identity.py"}


def test_the_live_tree_is_fully_classified() -> None:
    """And the allowlist is exact in both directions: an unclassified read is
    red, a row matching nothing is red."""
    lint = _lint()
    allowed, errors = lint.load_allowlist()
    assert not errors
    assert not lint.check(lint.scan(), allowed)
