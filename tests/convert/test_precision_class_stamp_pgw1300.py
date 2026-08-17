"""pgw#1300 / th#2055: the placement stamp is DELETED, and `precision_class`
is the one key that survives it.

Replaces `tests/convert/test_placement_stamp_pgw1286.py`, which is deleted
rather than skipped. That fence transcribed tensorhub's `defaultPlacement`
mirror and asserted, at the wire, that pgw stamped the same SM allow-lists,
SM floors and engine lists. **The thing it fenced no longer exists**: th#2055
(`65f0882f2`) deleted `PlacementFromMetadata`, `Placement`, `defaultPlacement`,
`admitted()` and `AdmitLiteral`'s sm/engine arms, so pod purchase depends only
on the endpoint owner's (GPU, lane) ladder and the compute floor is the
registered contract's own `Requires.MinComputeCapability`. Restoring the old
fence would re-derive a stamp the hub cannot read and would re-create the
defect that opened pgw#1300: `sm_allowed=(120, 121) + engines=("nunchaku",)`
told the hub that svdq-fp4 needs a wheel the fleet does not install, on
silicon a B200 is not — vetoing a flavor the native engine serves on sm_100.
A CORRECTED allow-list was rejected too: it still vetoes an owner's own rung.

What IS fenced here is the surviving half. `precision.StoredPrecisionOf` reads
`placement.precision_class` as its strongest evidence for a stored class where
no tensor-layout contract is proven, so pgw must keep writing exactly that key
and no other.

Revert-turns-red: re-add any admission key to `_precision_class_block`
(`convert/publish.py`), or drop the block, and a row fails at the wire.

    pytest tests/convert/test_precision_class_stamp_pgw1300.py -q
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fake_hub import _FakeHub

from gen_worker.convert.produced import ProducedFlavor
from gen_worker.convert.publish import publish_flavors
from gen_worker.models import ladder

#: Cut by `tests/convert/conftest.py`'s fake repo.
RELEASE = "2026.08"

#: The block the hub must receive, per flavor token. Written as literals: a
#: fence that derived them from `classify_flavor_token` would agree with itself.
#: `None` = no block at all, which is exact — the hub's fallback for an
#: unstamped row is `ClassBase`.
EXPECTED: dict[str, dict[str, Any] | None] = {
    "bf16": None,
    "fp16": None,
    "fp8": {"precision_class": "fp8"},
    "fp8-w8a8": {"precision_class": "fp8"},
    "svdq-int4": {"precision_class": "svdq-int4"},
    "svdq-fp4": {"precision_class": "svdq-fp4"},
    "nvfp4-w4a4": {"precision_class": "nvfp4-w4a4"},
    # Unrecognized tokens stay opaque and are never ladder rungs.
    "q4_k_m": None,
}

#: The keys th#2055 stopped reading. Any of them at the wire is the regression.
DELETED_ADMISSION_KEYS = ("sm_allowed", "sm_min", "engines")


class _Ctx:
    def __init__(self, base_url: str) -> None:
        self._file_api_base_url = base_url
        self._worker_capability_token = "cap-token"
        self.owner = "acme"

    def log(self, message: str, **fields: Any) -> None:
        pass


def _tree(tmp_path: Path, name: str) -> Path:
    out = tmp_path / name
    out.mkdir()
    (out / "diffusion.safetensors").write_bytes(b"\x11" * 2048)
    return out


def _publish(fake_hub: Any, tmp_path: Path, flavor: str, **attrs: str) -> dict:
    ctx = _Ctx(f"http://127.0.0.1:{fake_hub.server_port}")
    publish_flavors(
        ctx,
        [ProducedFlavor(
            path=_tree(tmp_path, "out"), flavor=flavor, attributes=dict(attrs),
        )],
        destination_repo="acme/qwen-image",
        release=RELEASE,
    )
    return dict(_FakeHub.state["publish_request"].get("metadata") or {})


@pytest.mark.parametrize("token", sorted(EXPECTED))
def test_the_declared_block_is_precision_class_and_nothing_else(
    fake_hub: Any, tmp_path: Path, token: str
) -> None:
    """Read off the declare request the fake hub actually receives: every
    laddered class carries its `precision_class`, and no class carries an
    admission key the hub deleted."""
    meta = _publish(fake_hub, tmp_path, token)

    expected = EXPECTED[token]
    if expected is None:
        assert "placement" not in meta, (
            f"{token!r} is not a ladder rung; a block here restates the hub's "
            "own ClassBase fallback")
        return
    assert meta["placement"] == expected
    assert set(meta["placement"]) == {"precision_class"}, (
        "th#2055 deleted every reader but `precision_class`; a second key is "
        "unread JSON at best and a resurrected purchase veto at worst")


@pytest.mark.parametrize("token", ["svdq-fp4", "svdq-int4", "nvfp4-w4a4", "fp8"])
def test_no_admission_key_survives_at_the_wire(
    fake_hub: Any, tmp_path: Path, token: str
) -> None:
    """The pgw#1300 defect, stated as the thing that must not come back: the
    svdq-fp4 row used to declare `sm_allowed=[120, 121]` + `engines=
    ["nunchaku"]`, which refused a B200 (sm_100) the native engine serves."""
    block = _publish(fake_hub, tmp_path, token)["placement"]
    present = [k for k in DELETED_ADMISSION_KEYS if k in block]
    assert present == [], f"{token!r} re-stamped deleted admission keys: {present}"


def test_a_producer_precision_class_attr_still_wins_over_the_token(
    fake_hub: Any, tmp_path: Path
) -> None:
    """A producer that knows its class states it; the token is the fallback.
    The DELETED `placement_*` override attrs are dropped rather than published
    as prose — `training-endpoints`' modelopt quantizer still writes
    `placement_sm_allowed`, and an attr pgw no longer reads is not metadata."""
    meta = _publish(
        fake_hub, tmp_path, "q4_k_m",
        precision_class="nvfp4-w4a4",
        placement_sm_allowed="89", placement_engines="nunchaku,triton",
    )

    assert meta["placement"] == {"precision_class": "nvfp4-w4a4"}
    assert not [k for k in meta if k.startswith("placement_")]


def test_the_ladder_module_no_longer_exports_a_placement() -> None:
    """The deletion asserted rather than remembered. `Placement`,
    `default_placement` and `placement_to_metadata` are gone, and with them the
    reason `models/svdq.py` kept nunchaku's kernel windows alive — pgw#1298's
    two deliberate survivors, whose only consumers were this stamp and a
    tensorhub peer pin th#2055 deleted."""
    from gen_worker.models import svdq

    assert set(ladder.__all__) == {
        "CLASS_BASE", "CLASS_FP8", "CLASS_NVFP4", "CLASS_NVFP4_W4A4",
        "CLASS_SVDQ_FP4", "CLASS_SVDQ_INT4", "classify_flavor_token",
    }
    leftovers = [n for n in vars(ladder) if "PLACEMENT" in n.upper()]
    assert leftovers == [], f"placement survived the cut: {leftovers}"
    assert not hasattr(svdq, "SVDQ_FP4_SMS")
    assert not hasattr(svdq, "SVDQ_INT4_SMS")


def test_the_worker_no_longer_reports_a_nunchaku_admission_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pgw#1298's other deliberate survivor. The probe list is hardcoded and
    nunchaku is installed nowhere here, so asking the real environment would
    pass with the token still in the list — a vacuous fence. Every probe is
    forced importable instead, which makes the reported set the LIST itself:
    `nunchaku` must be absent while its neighbours are present."""
    from gen_worker.models import hub_policy

    monkeypatch.setattr(hub_policy, "_is_importable", lambda _name: True)
    libs = set(hub_policy.detect_worker_capabilities().installed_libs)

    assert {"torchao", "modelopt", "deepcompressor"} <= libs, (
        "the probe list itself did not answer — this fence is measuring the "
        "wrong thing")
    assert "nunchaku" not in libs
