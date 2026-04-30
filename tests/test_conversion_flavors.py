from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from gen_worker.conversion import ProducedFlavor
from gen_worker.conversion.dispatch import _finalize_produced_variants


def test_produced_flavor_fields() -> None:
    produced = ProducedFlavor(path=Path("/tmp/out"), flavor="int4", attributes={"quant_recipe": "awq"})

    assert produced.path == Path("/tmp/out")
    assert produced.flavor == "int4"
    assert produced.attributes["quant_recipe"] == "awq"


def test_produced_flavor_positional_attributes_still_work() -> None:
    produced = ProducedFlavor(Path("/tmp/out"), {"flavor": "bf16"})

    assert produced.attributes == {"flavor": "bf16"}
    assert produced.flavor == ""


def test_produced_flavor_accepts_multi_label_flavors() -> None:
    produced = ProducedFlavor(
        path=Path("/tmp/out"),
        flavor="fp8",
        flavors=["fp8", "flashpack", "aio"],
    )

    assert produced.flavor == "fp8"
    assert produced.flavors == ["fp8", "flashpack", "aio"]


def test_finalize_produced_flavors_publishes_one_tag_group(tmp_path: Path) -> None:
    out_a = tmp_path / "bf16.safetensors"
    out_b = tmp_path / "fp8.flashpack"
    out_a.write_bytes(b"a")
    out_b.write_bytes(b"b")

    calls: list[dict] = []

    class Ctx:
        request_id = "req1"
        job_id = "job1"
        destination = {"ref": "org/model", "tags": ["prod"], "release_visibility": "public"}
        source = {"ref": "org/base", "checkpoint_id": "sha256:base"}

        def save_checkpoint(self, ref: str, path: str, **_: object) -> SimpleNamespace:
            p = Path(path)
            return SimpleNamespace(blob_digest=f"blake3:{p.stem}", size_bytes=p.stat().st_size)

        def publish_repo_revision(self, **kwargs: object) -> None:
            calls.append(dict(kwargs))

    _finalize_produced_variants(
        Ctx(),
        [
            ProducedFlavor(path=out_a, flavor="bf16", flavors=["bf16", "diffusers", "safetensors"]),
            ProducedFlavor(path=out_b, flavor="fp8", flavors=["fp8", "flashpack", "aio"]),
        ],
        kind="quantization",
    )

    assert len(calls) == 1
    metadata = calls[0]["metadata"]
    flavors = metadata["checkpoint_flavors"]
    assert [item["flavor"] for item in flavors] == ["bf16", "fp8"]
    assert flavors[0]["flavors"] == ["bf16", "diffusers", "safetensors"]
    assert flavors[1]["flavors"] == ["fp8", "flashpack", "aio"]
    assert calls[0]["destination_repo_tags"] == ["prod"]
    assert calls[0]["relationship_kind"] == "quantization"
