from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from gen_worker.convert.produced import ProducedFlavor
from gen_worker.convert.publish import publish_flavors

from fake_hub import _FakeHub


class _Ctx:
    def __init__(self, base_url: str, source_ref: str = "") -> None:
        self._file_api_base_url = base_url
        self._worker_capability_token = "cap-token"
        self.owner = "acme"
        self.lines: list[str] = []
        if source_ref:
            self.source = {"ref": source_ref, "attributes": {}}

    def log(self, message: str, **fields: Any) -> None:
        self.lines.append(message)


def _tree(tmp_path: Path) -> Path:
    out = tmp_path / "fp8"
    out.mkdir()
    (out / "diffusion.safetensors").write_bytes(b"\x11" * 2048)
    return out


def _resolve_body(
    *, objective: str, distilled: bool, distilled_status: str = "classified",
) -> dict:
    payload = b"\x33" * 64
    return {
        "snapshot_digest": "sha256:" + "cd" * 32,
        "files": [{
            "path": "transformer/diffusion.safetensors",
            "size_bytes": len(payload),
            "digest": "sha256:" + hashlib.sha256(payload).hexdigest(),
            "url": "http://127.0.0.1:9/blob",
        }],
        "objective": objective,
        "distilled": distilled,
        "distilled_status": distilled_status,
    }


RELEASE = "2026.08"


def _publish(ctx: _Ctx, tree: Path, **kw: Any) -> Any:
    kw.setdefault("release", RELEASE)
    return publish_flavors(
        ctx,
        [ProducedFlavor(path=str(tree), attributes={"precision_class": "fp8"})],
        destination_repo="acme/qwen-image",
        **kw,
    )


def test_source_stamps_are_restated_by_default(
    fake_hub: Any, tmp_path: Path,
) -> None:
    _FakeHub.state["resolve_body"] = _resolve_body(
        objective="flow", distilled=True)
    ctx = _Ctx(f"http://127.0.0.1:{fake_hub.server_port}",
               source_ref="acme/qwen-image")

    _publish(ctx, _tree(tmp_path))

    req = _FakeHub.state["publish_request"]
    assert req["objective"] == "flow"
    assert req["distilled"] is True
    assert _FakeHub.state["resolve_gets"], "stamps must come from the hub"


def test_classified_false_source_restates_distilled_false_and_no_objective(
    fake_hub: Any, tmp_path: Path,
) -> None:
    """Distillation evidence is independent of objective evidence: a classified false is restated even when objective is not classified."""
    _FakeHub.state["resolve_body"] = _resolve_body(
        objective="", distilled=False)
    ctx = _Ctx(f"http://127.0.0.1:{fake_hub.server_port}",
               source_ref="acme/base")

    _publish(ctx, _tree(tmp_path))

    req = _FakeHub.state["publish_request"]
    assert "objective" not in req
    assert req["distilled"] is False


@pytest.mark.parametrize("status", ["unclassified", "inconclusive"])
def test_unknown_source_distillation_is_not_authored_as_false(
    fake_hub: Any, tmp_path: Path, status: str,
) -> None:
    _FakeHub.state["resolve_body"] = _resolve_body(
        objective="", distilled=False, distilled_status=status
    )
    ctx = _Ctx(f"http://127.0.0.1:{fake_hub.server_port}",
               source_ref="acme/base")

    _publish(ctx, _tree(tmp_path))

    req = _FakeHub.state["publish_request"]
    assert "objective" not in req
    assert "distilled" not in req


def test_absent_status_is_not_authored_as_false_either(
    fake_hub: Any, tmp_path: Path,
) -> None:
    body = _resolve_body(objective="", distilled=False)
    body.pop("distilled_status")
    _FakeHub.state["resolve_body"] = body
    ctx = _Ctx(f"http://127.0.0.1:{fake_hub.server_port}",
               source_ref="acme/base")

    _publish(ctx, _tree(tmp_path))

    assert "distilled" not in _FakeHub.state["publish_request"]


def test_explicit_caller_override_wins(fake_hub: Any, tmp_path: Path) -> None:
    _FakeHub.state["resolve_body"] = _resolve_body(
        objective="flow", distilled=True)
    ctx = _Ctx(f"http://127.0.0.1:{fake_hub.server_port}",
               source_ref="acme/qwen-image")

    _publish(ctx, _tree(tmp_path), objective="v_prediction", distilled=False)

    req = _FakeHub.state["publish_request"]
    assert req["objective"] == "v_prediction"
    assert req["distilled"] is False
    assert not _FakeHub.state.get("resolve_gets"), \
        "fully stated stamps must not cost a resolve round-trip"


def test_no_source_and_failed_resolve_stay_unstamped(
    fake_hub: Any, tmp_path: Path,
) -> None:
    """No ctx.source (e.g."""
    base = f"http://127.0.0.1:{fake_hub.server_port}"
    _publish(_Ctx(base), _tree(tmp_path))
    req = _FakeHub.state["publish_request"]
    assert "objective" not in req and "distilled" not in req

    ctx = _Ctx(base, source_ref="acme/missing")
    out = tmp_path / "second"
    out.mkdir()
    (out / "model.safetensors").write_bytes(b"\x44" * 256)
    _publish(ctx, out)
    req = _FakeHub.state["publish_request"]
    assert "objective" not in req and "distilled" not in req
    assert any("source-stamp read failed" in line for line in ctx.lines)


def test_the_tag_axis_is_gone_from_the_declare(
    fake_hub: Any, tmp_path: Path,
) -> None:
    tree = _tree(tmp_path)
    ctx = _Ctx(f"http://127.0.0.1:{fake_hub.server_port}")
    with pytest.raises(TypeError):
        _publish(ctx, tree, tags=[])  # type: ignore[call-arg]
    _publish(ctx, tree)
    req = _FakeHub.state["publish_request"]
    assert "tags" not in req
    assert req["release"] == RELEASE
