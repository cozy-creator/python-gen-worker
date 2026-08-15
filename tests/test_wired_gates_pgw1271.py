"""pgw#1271 — the gates that could not go red, and the caller each now has.

Every row here is a WIRING test, not a logic test. The logic of each gate was
already correct, already documented and already unit-tested; what did not exist
was a production call site, so the gate's failure mode was invisible and its
own prose read like enforcement. A unit test is structurally blind to that
class — the unit test IS the caller production is missing — so each row below
drives the PRODUCTION function and asserts the gate fires THROUGH it.

Every row was verified RED against the pre-fix tree before the fix landed.

Nothing here compiles, mints or touches a GPU.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast

import pytest

from gen_worker import (
    aot_serve,
    author_ci,
    boot_key,
    compiled_graph_key,
    compile_cache,
    lifecycle,
    mint_supervisor,
    numerics_ladder,
    numerics_probe,
    presigned_upload,
    receipts,
    rigcheck,
)
from gen_worker.api.errors import ArtifactTransferError
from gen_worker.hubio.transport import TransportError
from gen_worker.parallel.cp import (
    ContextParallelUnavailable,
    CpComms,
    install_context_parallel,
)
from gen_worker.topology import TopologyError

from harness.receipt_hub import (  # noqa: F401 — fixtures ride along
    FAMILY, SELF_ENDPOINT,
    HubStub, _configure, hub, make_artifact, rsa_key,
)


# ---------------------------------------------------------------------------
# G1 — boot_key.assert_memo_honest now has a production caller
# ---------------------------------------------------------------------------


def _class_hash(dim: int) -> str:
    """One TCG class hash, in the only shape the memo accepts (16 lower hex).

    pgw#1270: TCG derives `class_hash` inside `GraphClassDeclaration` and the
    memo stores those hashes directly, so there is no worker-side keying block
    left to stamp. `graph_witness` is one of the facts TCG folds INTO
    `class_hash`, so two dims disagreeing here is exactly what a drifted
    witness or a drifted graph body looks like at this seam.
    """
    return hashlib.sha256(f"class-dim-{int(dim)}".encode()).hexdigest()[:16]


def _entry_block(*, dim: int = 64) -> Dict[str, Any]:
    """The `entry` block a minted artifact carries, as the seam reads it."""
    return {
        "name": "a",
        "target": "unet",
        "class_hash": _class_hash(dim),
    }


class _Cfg:
    """The parent's `CompileCell`, as `cfg_spec` reads it."""

    family = "tiny"
    targets = ("unet",)
    shapes = ((1024, 1024),)
    text_lens = (77,)
    guidance_scales = (7.5,)
    lora_bucket = 0


class _Pending:
    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.family = "tiny"


class _Row:
    def __init__(self, entry: str, block: Dict[str, Any]) -> None:
        self.entry = entry
        self.key = f"cg-{entry}"
        self.metadata = {compiled_graph_key.ENTRY_BLOCK_KEY: block}


class _Result:
    def __init__(self, rows: List[_Row]) -> None:
        self.entries = tuple(rows)


def _mint_task(tmp_path: Path) -> Any:
    return mint_supervisor.MintTask(
        pending=_Pending(tmp_path),
        pipe=object(),
        function="generate",
        modules=("endpoint",),
        slots={},
    )


@pytest.fixture()
def _runtime_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the runtime key-complete on a GPU-less box.

    Only the PROBE is faked; every value is a real fact of some runtime and
    nothing about how they fold into a key is touched.
    """
    monkeypatch.setattr(compile_cache, "runtime_key", lambda: {
        "sku": "l4", "sm": "sm_89", "torch": "2.13.0+cu130",
        "triton": "3.6.0", "cuda": "13.0",
        "image_digest": "sha256:" + "ab" * 32,
    })


def _write_memo(tmp_path: Path, hashes: Dict[str, str]) -> str:
    cfg = _Cfg()
    digest = boot_key.closure_digest(
        "tiny", mint_supervisor.cfg_spec(cfg), function="generate", slots={})
    assert boot_key.write_memo(tmp_path, digest, hashes)
    return digest


def test_the_mint_publish_seam_rules_on_a_DISHONEST_boot_memo(
    tmp_path: Path, _runtime_key: None,
) -> None:
    """THE headline. `assert_memo_honest` had zero src/ callers, so a memo that
    answered the boot with the WRONG graph half was never contradicted — and
    the memo path skips the traces, so nothing else in the pod ever held a
    traced truth to contradict it with.

    Here the memo holds one closure's class hash and the mint traces a
    DIFFERENT one. The production seam must name the disagreement and
    invalidate the entry, so the next boot re-traces rather than re-reading a
    proven-wrong hash.
    """
    digest = _write_memo(tmp_path, {"a": _class_hash(64)})

    reason = mint_supervisor.rule_on_boot_memo(
        _mint_task(tmp_path), _Cfg(),
        _Result([_Row("a", _entry_block(dim=128))]), declared=1)

    assert "DISHONEST" in reason and "a: memo" in reason
    # The entry is GONE: the next boot re-traces instead of answering from it.
    assert boot_key.read_memo(tmp_path, digest) == {}


def test_an_HONEST_boot_memo_is_silence_at_the_publish_seam(
    tmp_path: Path, _runtime_key: None,
) -> None:
    hashes = {"a": _class_hash(64)}
    digest = _write_memo(tmp_path, hashes)

    assert mint_supervisor.rule_on_boot_memo(
        _mint_task(tmp_path), _Cfg(),
        _Result([_Row("a", _entry_block(dim=64))]), declared=1) == ""
    # An honest memo SURVIVES — the whole economic point of having one.
    assert boot_key.read_memo(tmp_path, digest) == hashes


def test_a_PARTIAL_class_set_rules_on_nothing(
    tmp_path: Path, _runtime_key: None,
) -> None:
    """Coverage accretes (pgw#1176), so a mint that packed 1 of 2 declared
    classes cannot tell "the memo holds a class we did not trace" from "we have
    not traced it yet". Ruling anyway would fire `class set differs` on every
    partial mint — a gate that cries wolf is uninstalled within a week."""
    hashes = {"a": _class_hash(64), "b": _class_hash(128)}
    digest = _write_memo(tmp_path, hashes)

    assert mint_supervisor.rule_on_boot_memo(
        _mint_task(tmp_path), _Cfg(),
        _Result([_Row("a", _entry_block(dim=64))]), declared=2) == ""
    assert boot_key.read_memo(tmp_path, digest) == hashes


def test_the_dishonest_verdict_reaches_the_wire_as_a_TYPED_EVENT(
    tmp_path: Path, _runtime_key: None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Not a log line: a hub-spawned pod's stdout goes nowhere (pgw#760), and a
    dishonest memo is a KEY-SPACE fault the fleet has to be able to count."""
    from gen_worker import activity as activity_mod

    seen: List[Tuple[str, str, str]] = []
    monkeypatch.setattr(
        activity_mod, "emit_event",
        lambda kind, detail, **kw: seen.append(
            (kind, detail, str(kw.get("phase") or ""))))

    _write_memo(tmp_path, {"a": _class_hash(64)})
    mint_supervisor._rule_on_boot_memo(
        _mint_task(tmp_path), _Cfg(), _Result([_Row("a", _entry_block(dim=128))]),
        declared=1, family="tiny")

    assert [(k, p) for k, _d, p in seen] == [
        (activity_mod.KIND_BOOT_MEMO, "memo_dishonest")]
    assert "DISHONEST" in seen[0][1]


# ---------------------------------------------------------------------------
# G2 — the rig's fleet line, and the CUDA half
# ---------------------------------------------------------------------------


def test_the_SDK_is_not_its_own_fleet_line_authority(tmp_path: Path) -> None:
    """`_collect_authorities` appended "gen-worker" to the chain, so the SDK
    certified the very torch floor the rig was being asked to prove:
    `FleetLineUnknown` was unreachable on any machine with gen-worker
    installed — i.e. every machine that can run a rig."""
    with pytest.raises(rigcheck.FleetLineUnknown):
        rigcheck.resolve_fleet_line(start=tmp_path, endpoint_dists=())


def test_a_host_whose_DIAGNOSTIC_is_broken_still_refuses_to_measure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The refusal used to be gated on `env["driver"]` being readable, i.e. on
    `nvidia-smi` WORKING. A host broken enough that it does not is exactly the
    host this refusal exists for, so the check skipped the machines it was
    written for."""
    monkeypatch.setattr(rigcheck, "resolve_fleet_line", lambda **_: rigcheck.FleetLine(
        torch=(2, 13, 0), cuda=(13, 0), authorities=()))
    monkeypatch.setattr(rigcheck, "resolve_environment", lambda: {
        "python": "3.12.0",
        "torch": "2.13.0+cu130",
        "cuda": "13.0",
        # nvidia-smi did not answer. The allocation still failed.
        "driver": "",
        "cuda_usable": False,
        "cuda_unusable_reason": "CUDA driver version is insufficient",
        "cuda_unusable_class": "driver_too_old",
    })

    with pytest.raises(rigcheck.CudaUnusable):
        rigcheck.assert_fleet_line("rig", start=tmp_path, stream=None)


# ---------------------------------------------------------------------------
# G3 — the direct-final PUT speaks the caller's exception vocabulary
# ---------------------------------------------------------------------------


def test_an_expired_presign_on_the_DIRECT_FINAL_leg_is_typed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The re-plan at `presigned_upload_file` catches `ArtifactTransferError`
    with `phase == "put"` and `status_code == 403`. The multipart leg produced
    exactly that; the direct-final leg — the one production ALWAYS takes, since
    `_stream` always sends `sha256` and the hub therefore always mints
    `put_url` — raised `TransportError` raw. It matched no `except`, so an
    expired presign lost a whole completed render as an untyped RuntimeError.
    """
    def _boom(**_kw: Any) -> str:
        raise TransportError("presigned PUT 403", retryable=False, status_code=403)

    monkeypatch.setattr(presigned_upload, "upload_part_to_presigned_url", _boom)
    body = tmp_path / "render.png"
    body.write_bytes(b"x" * 32)

    with pytest.raises(ArtifactTransferError) as caught:
        presigned_upload._put_whole_object(
            url="https://r2.example/final",
            file_path=str(body),
            size_bytes=32,
            extra_headers={},
            on_progress=None,
            cancel_check=None,
            put_pool=None,
        )

    exc = caught.value
    # The exact predicate the re-plan loop tests. Asserted as the predicate,
    # not as two fields, because the fields only matter through it.
    assert exc.phase == "put" and getattr(exc, "status_code", None) == 403


def test_a_cancel_on_the_direct_final_leg_is_a_CANCEL(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from gen_worker.api.errors import CanceledError

    def _boom(**_kw: Any) -> str:
        raise InterruptedError("canceled")

    monkeypatch.setattr(presigned_upload, "upload_part_to_presigned_url", _boom)
    body = tmp_path / "render.png"
    body.write_bytes(b"x")

    with pytest.raises(CanceledError):
        presigned_upload._put_whole_object(
            url="https://r2.example/final", file_path=str(body), size_bytes=1,
            extra_headers={}, on_progress=None, cancel_check=None, put_pool=None)


# ---------------------------------------------------------------------------
# G5 — a wedged NVLink fabric is not "card 0, fine"
# ---------------------------------------------------------------------------


def test_a_wedged_fabric_is_not_swallowed_as_a_free_vram_number(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`topology.delivered_topology` raises `TopologyError` on peer access with
    0.0 GB/s measured — every collective on that host blocks forever, and the
    refusal exists so the hub RE-PACKS. Two `except Exception: pass` in this
    module ate it without a log, and the pod reported a perfectly ordinary
    free-VRAM number and went on to serve."""
    def _wedged(*_a: Any, **_kw: Any) -> Any:
        raise TopologyError(
            "topology_fabric_wedged_peer_access_zero_bandwidth",
            "peer access reported with 0.0 GB/s measured")

    monkeypatch.setattr(lifecycle, "delivered_topology", _wedged)
    with pytest.raises(TopologyError):
        lifecycle.free_vram_bytes()


def test_an_ordinary_topology_failure_is_still_swallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The catch is NARROWED, not removed: a box with no CUDA at all must still
    report 0 rather than failing a heartbeat."""
    def _no_torch(*_a: Any, **_kw: Any) -> Any:
        raise RuntimeError("no CUDA driver")

    monkeypatch.setattr(lifecycle, "delivered_topology", _no_torch)
    assert lifecycle.free_vram_bytes() == 0


# ---------------------------------------------------------------------------
# author_ci — a report's EXISTENCE is not its verdict
# ---------------------------------------------------------------------------


def _report(cosine: float, *, measured: bool = True) -> numerics_probe.CompiledGraphNumerics:
    thresholds = numerics_ladder.Thresholds(
        floor=0.90, warn=0.99, label="test",
        source=numerics_ladder.SOURCE_SDK_DEFAULT)
    comparison = numerics_ladder.Comparison(
        reference="eager", subject="unet", thresholds=thresholds,
        rows=(numerics_ladder.RowStat(
            name="out", elements=16, cosine=cosine, retention=1.0),),
        cosine=cosine, retention=1.0)
    axis = numerics_probe.ProbeAxis(entry="unet", target="unet")
    verdict = numerics_probe.AxisVerdict(
        axis=axis, comparison=comparison if measured else None,
        reason="" if measured else "compiled_graph_forward_failed")
    return numerics_probe.CompiledGraphNumerics(
        family="tiny", compiled_graph_key="cg-1", thresholds=thresholds,
        threshold_source=thresholds.source, verdicts=(verdict,), axes_total=1)


class _Subject:
    def __init__(self, pipe: Any) -> None:
        self._pipe = pipe

    def armed_pipeline(self) -> Any:
        return self._pipe


def test_a_DEGRADED_mint_report_does_not_report_PASS(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`read_parity` set `passed = True` in the mint branch from the mere
    EXISTENCE of a report — the report's presence scored as its verdict. So a
    cell whose worst axis sits in the DEGRADED band reported PASS with the
    failing cosine printed on the same line."""
    monkeypatch.setattr(aot_serve, "entry_states", lambda _p: {
        "unet": {"state": "armed", "target": "unet", "calls": 1}})

    parity = author_ci.read_parity(
        cast(Any, _Subject(object())), declaration=None, minted=_report(0.95))

    assert parity.passed is False
    assert parity.cosine == pytest.approx(0.95)


def test_a_HEALTHY_mint_report_still_reports_PASS(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(aot_serve, "entry_states", lambda _p: {
        "unet": {"state": "armed", "target": "unet", "calls": 1}})
    assert author_ci.read_parity(
        cast(Any, _Subject(object())), declaration=None, minted=_report(0.999)).passed is True


def test_a_DE_ARMED_entry_fails_the_parity_whatever_the_cosine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cross-check. A class the arm de-armed is a class this cell does not
    deliver; a verdict that describes the cell without consulting what the
    pipeline SERVES is a verdict about something else."""
    monkeypatch.setattr(aot_serve, "entry_states", lambda _p: {
        "unet": {"state": "armed", "target": "unet", "calls": 1},
        "vae.decode": {"state": "de_armed", "target": "vae",
                       "reason": "ingress_contract"},
    })
    parity = author_ci.read_parity(
        cast(Any, _Subject(object())), declaration=None, minted=_report(0.999))
    assert parity.passed is False
    assert "NOT armed" in parity.detail


# ---------------------------------------------------------------------------
# parallel.cp.refuse_unless_divisible now has a caller
# ---------------------------------------------------------------------------


class _Expert:
    """A sharding candidate: declares a `_cp_plan` and `enable_parallelism`."""

    _cp_plan = {"hidden_states": object()}

    def __init__(self, heads: int) -> None:
        self.config = type("Cfg", (), {"num_attention_heads": heads})()

    def enable_parallelism(self, *_a: Any, **_k: Any) -> None:  # pragma: no cover
        raise AssertionError("the group-aware install replaces this")


class _Pipeline:
    def __init__(self, heads: int) -> None:
        self.transformer = _Expert(heads)


def test_an_indivisible_HEAD_COUNT_is_refused_before_the_pod_commits() -> None:
    """diffusers #12536: the all-to-all shards the head dimension. An expert
    whose declared head count does not divide the degree fails as an
    inscrutable shape mismatch mid-denoise, on a paying request, on a pod that
    has already rented. `refuse_unless_divisible` said so, was exported, and
    was called by nothing."""
    comms = CpComms(pg=object(), rank=0, device="cuda:0")
    with pytest.raises(ContextParallelUnavailable, match="head count"):
        install_context_parallel(_Pipeline(heads=6), degree=4, comms=comms)


def test_a_divisible_head_count_gets_past_the_divisibility_gate() -> None:
    """Proves the gate is not a blanket refusal: heads=8 at degree 4 clears it
    and the call proceeds into the diffusers install, which then fails on this
    toy component for an unrelated reason. Whatever comes back, it is not the
    divisibility refusal."""
    comms = CpComms(pg=object(), rank=0, device="cuda:0")
    with pytest.raises(Exception) as caught:
        install_context_parallel(_Pipeline(heads=8), degree=4, comms=comms)
    assert "head count" not in str(caught.value)


# ---------------------------------------------------------------------------
# aot_serve.runtime_key: pgw#1270 deleted the second probe outright
# ---------------------------------------------------------------------------
#
# pgw#1271 made `aot_serve.runtime_key` delegate to `compile_cache.runtime_key`
# and asserted the projection here. The cut then removed its last caller — TCG
# owns the artifact key — so the duplicate implementation is GONE rather than
# delegating. That is the same guarantee in its strongest form: there is one
# runtime probe because there is one function.

# ---------------------------------------------------------------------------
# receipts — no call that reads as the last gate and checks nothing
# ---------------------------------------------------------------------------


class TestReceiptTrustGate:
    """Driven through the real gate against the real signing hub."""

    def test_the_platform_tier_path_makes_no_VACUOUS_trust_call(
        self, tmp_path: Path, hub: HubStub, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """`verify_delivered_artifact` called `refuse_untrusted_publisher(receipt,
        "", "")` INSIDE that callee's own early-return condition, so it executed
        exactly zero checks — immediately before the caller dlopens the artifact.

        Driven through the real gate against the real signing hub: the platform
        branch must reach its return having made no trust call at all, rather than
        one that is structurally incapable of refusing.
        """
        calls: List[Tuple[str, str]] = []
        real = receipts.refuse_untrusted_publisher

        def _record(receipt: Any, endpoint_id: str, org_id: str = "") -> None:
            calls.append((endpoint_id, org_id))
            real(receipt, endpoint_id, org_id)

        monkeypatch.setattr(receipts, "refuse_untrusted_publisher", _record)

        artifact = make_artifact(tmp_path)
        hub.serve_receipt_for(artifact, publisher_tier="platform")
        _configure(hub)

        receipt = receipts.verify_delivered_artifact(artifact, FAMILY)
        assert receipt.publisher_tier == "platform"
        assert calls == [], (
            "the platform branch made a trust call that can only ever return")


    def test_an_ORG_tier_cell_still_reaches_the_real_trust_gate(
        self, tmp_path: Path, hub: HubStub, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The other half, so the deletion above cannot have removed enforcement:
        an org-tier receipt from a foreign endpoint IS refused, and the call that
        refuses it carries THIS pod's identity — never two empty strings."""
        calls: List[Tuple[str, str]] = []
        real = receipts.refuse_untrusted_publisher

        def _record(receipt: Any, endpoint_id: str, org_id: str = "") -> None:
            calls.append((endpoint_id, org_id))
            real(receipt, endpoint_id, org_id)

        monkeypatch.setattr(receipts, "refuse_untrusted_publisher", _record)

        artifact = make_artifact(tmp_path)
        hub.serve_receipt_for(
            artifact, owning_endpoint_id="ep_someone_else",
            publisher_org_id="org_someone_else")
        _configure(hub, endpoint_id=SELF_ENDPOINT)

        with pytest.raises(receipts.ReceiptError):
            receipts.verify_delivered_artifact(artifact, FAMILY)
        assert calls and calls[0][0] == SELF_ENDPOINT
