"""pgw#1159: the publish body stops carrying fields th#1803 deleted.

`flavor` / `flavors` / `default_flavor` were decoded hub-side and read by
NOTHING after th#1803 (§1.32(d) deleted the flavor as an address). A producer
that sent them believed it had named a catalog row it had not — a silent drop.

What replaces them: `tags[].head` (this publish owns the bare row) and
`artifact_contract` (what the bytes ARE, PROVEN against the header).

Revert-turns-red: put `flavor=`/`default_flavor=` back into
`HubClient.publish_v2`'s body and every assertion here fails on the field it
names.

    pytest tests/convert/test_publish_flavor_fields_pgw1159.py -q
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import msgspec
import pytest

from gen_worker.convert.produced import ProducedFlavor
from gen_worker.convert.publish import publish_flavors

from fake_hub import _FakeHub

_DEAD_FIELDS = ("flavor", "flavors", "default_flavor")


class _Ctx:
    def __init__(self, base_url: str) -> None:
        self._file_api_base_url = base_url
        self._worker_capability_token = "cap-token"
        self.owner = "acme"

    def log(self, message: str, **fields: Any) -> None:
        pass


def _tree(tmp_path: Path) -> Path:
    out = tmp_path / "w8a8"
    out.mkdir()
    (out / "diffusion.safetensors").write_bytes(b"\x11" * 4096)
    (out / "config.json").write_text('{"architectures": ["Fake"]}')
    return out


def _publish(ctx: _Ctx, tree: Path, **attrs: str) -> Any:
    return publish_flavors(
        ctx,
        [ProducedFlavor(
            path=str(tree), flavor="fp8-w8a8",
            attributes={"dtype": "fp8", "quantization_method": "w8a8", **attrs},
        )],
        destination_repo="acme/quant",
        tags=["prod"],
    )


def test_no_dead_flavor_field_reaches_the_publish_body(
    fake_hub: Any, tmp_path: Path,
) -> None:
    """The whole point: not one of the three names is on the wire, at the top
    level OR inside a tag binding."""
    ctx = _Ctx(f"http://127.0.0.1:{fake_hub.server_port}")
    _publish(ctx, _tree(tmp_path))

    req = _FakeHub.state["publish_request"]
    for field in _DEAD_FIELDS:
        assert field not in req, f"{field} is dead hub-side (th#1803)"
    for binding in req["tags"]:
        assert "default_flavor" not in binding
    # Not smuggled through the metadata bag either — the producer's local
    # label is not a catalog statement under any key.
    assert "flavor" not in (req.get("metadata") or {})


def test_the_producer_label_still_classifies_placement(
    fake_hub: Any, tmp_path: Path,
) -> None:
    """`ProducedFlavor.flavor` survives as a PRODUCER-LOCAL label: it stamps
    placement (th#697). Deleting it silently would drop the stamp."""
    ctx = _Ctx(f"http://127.0.0.1:{fake_hub.server_port}")
    _publish(ctx, _tree(tmp_path))

    meta = _FakeHub.state["publish_request"]["metadata"]
    assert meta["placement"]["precision_class"] == "fp8"


def test_artifact_contract_rides_its_own_proven_field(
    fake_hub: Any, tmp_path: Path,
) -> None:
    """th#1580/§1.33: the statement of what the bytes ARE. It goes to the
    typed field the hub PROVES, and is not duplicated into metadata."""
    ctx = _Ctx(f"http://127.0.0.1:{fake_hub.server_port}")
    _publish(ctx, _tree(tmp_path), artifact_contract="cozy.w8a8@2")

    req = _FakeHub.state["publish_request"]
    assert req["artifact_contract"] == "cozy.w8a8@2"
    assert "artifact_contract" not in req["metadata"]


def test_head_replaces_default_flavor_on_the_tag_binding(
    fake_hub: Any, tmp_path: Path,
) -> None:
    """`default_flavor`'s only real job was "this publish owns the bare row".
    That is `tags[].head` now — stated, not inferred from a token."""
    from gen_worker.convert.hub import CommitFile, HubClient

    tree = _tree(tmp_path)
    client = HubClient(
        base_url=f"http://127.0.0.1:{fake_hub.server_port}", token="cap-token")
    files = [CommitFile(path="config.json", local_path=tree / "config.json")]

    client.publish_v2(destination_repo="acme/quant", files=files,
                      tags=["latest"], head=True)
    assert _FakeHub.state["publish_request"]["tags"] == [
        {"tag": "latest", "head": True}]

    client.publish_v2(destination_repo="acme/quant", files=files,
                      tags=["latest"])
    assert _FakeHub.state["publish_request"]["tags"] == [{"tag": "latest"}]


def test_publish_v2_refuses_the_deleted_keyword_arguments() -> None:
    """A caller still writing `flavor=`/`default_flavor=` gets a TypeError,
    not a body field the hub throws away."""
    from gen_worker.convert.hub import HubClient

    client = HubClient(base_url="http://127.0.0.1:1", token="t")
    for kw in _DEAD_FIELDS:
        with pytest.raises(TypeError):
            client.publish_v2(destination_repo="a/b", files=[], **{kw: "fp8"})


def test_produced_flavor_has_no_flavor_label_set() -> None:
    """N artifacts are N publishes joining one tag group; a single publish
    never carried a label SET the hub could record."""
    with pytest.raises(TypeError):
        ProducedFlavor(path="/tmp/x", flavors=["fp8", "aio"])  # type: ignore[call-arg]
    names = {f.name for f in msgspec.structs.fields(ProducedFlavor)}
    assert "flavors" not in names and "flavor" in names
