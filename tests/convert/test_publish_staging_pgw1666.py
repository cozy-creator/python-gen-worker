"""pgw#1666 — the publish's bytes are the pipeline's bytes, and it says so.

The measured defect: a 50.19 GB `clone-huggingface` + bf16 cast passed a
preflight that demanded ~87 GiB, paid its whole download and its whole cast,
and then died `OSError: [Errno 28]` inside `publish_v2` — which was copying
the produced tree into the local CAS, a term the budget had no name for.

Two halves, both here:
  * the publish no longer writes that copy (it uploads byte RANGES of the
    producer's own files), so the real need is what the budget already said;
  * the budget ASKS the publish what it costs, so an omission like the one
    that cost 250 GPU-s cannot come back silently.
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fake_hub import _client, _FakeHub

from gen_worker.convert.clone import (
    CAS,
    CloneDiskSpaceError,
    OutputSpec,
    _preflight_disk,
    plan_disk_demand,
    run_clone,
)
from gen_worker.convert.ingest import IngestedSource
from gen_worker.hubio.client import CommitFile, files_from_tree
from gen_worker.models.cache_paths import tensorhub_cas_dir

GIB = 1024**3
BF16 = OutputSpec(dtype="bf16", file_layout="multi-file", file_type="safetensors")


def _bytes_under(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


def _safetensors(tensors: dict[str, tuple[str, int]]) -> bytes:
    """One valid safetensors file: `name -> (dtype, element count)`."""

    width = {"F32": 4, "BF16": 2}
    header: dict[str, Any] = {}
    offset = 0
    for name, (dtype, count) in tensors.items():
        end = offset + count * width[dtype]
        header[name] = {"dtype": dtype, "shape": [count], "data_offsets": [offset, end]}
        offset = end
    blob = json.dumps(header).encode()
    body = bytes(bytearray((i * 37 + 11) % 251 for i in range(offset)))
    return struct.pack("<Q", len(blob)) + blob + body


class _Ctx:
    def __init__(self, server: Any) -> None:
        self._file_api_base_url = f"http://127.0.0.1:{server.server_port}"
        self._worker_capability_token = "cap-token"
        self.owner = "tensorhub"
        self.request_id = "req-1666"
        self.destination = {"repo": "tensorhub/fallback"}


# ---------------------------------------------------------------- the copy


def test_a_publish_writes_no_second_copy_of_the_tree_it_publishes(
    fake_hub: Any, tmp_path: Path,
) -> None:
    """THE DEFECT, at fixture scale: publishing must not consume disk.

    Pre-fix this stored every object under the CAS root — the whole tree
    again, on the same filesystem, before a byte was uploaded.
    """

    tree = tmp_path / "flavor"
    tree.mkdir()
    (tree / "config.json").write_bytes(b'{"architectures":["Fake"]}')
    (tree / "model.safetensors").write_bytes(
        _safetensors({f"blocks.{i}.weight": ("F32", 4096) for i in range(8)}))
    published = _bytes_under(tree)
    assert published > 100_000

    cas_root = tensorhub_cas_dir()
    before = _bytes_under(cas_root)

    result = _client(fake_hub).publish_v2(
        destination_repo="acme/model", release="r1",
        files=files_from_tree(tree),
    )

    assert result.checkpoint_id
    assert result.uploaded == 10  # 8 tensors + header + config
    assert _bytes_under(cas_root) - before == 0, (
        "publish staged a local copy of the tree it was publishing")


def test_the_bytes_the_hub_receives_are_the_bytes_on_disk(
    fake_hub: Any, tmp_path: Path,
) -> None:
    """A range upload is only a saving if it sends the right range.

    The fake hub refuses any PUT whose body does not hash to its key, so a
    mis-seeked region cannot pass — and the declared per-tensor digests must
    match the file's real bytes at those offsets.
    """

    raw = _safetensors({"a.weight": ("F32", 512), "b.weight": ("BF16", 1024)})
    path = tmp_path / "model.safetensors"
    path.write_bytes(raw)

    _client(fake_hub).publish_v2(
        destination_repo="acme/model", release="r1",
        files=[CommitFile(path="model.safetensors", local_path=path,
                          size_bytes=len(raw))],
    )

    declared = next(iter(_FakeHub.state["publishes"].values()))["files"][0]
    offset = 0
    for chunk in declared["chunks"]:
        window = raw[offset:offset + int(chunk["len"])]
        assert hashlib.sha256(window).hexdigest() == chunk["digest"]
        assert chunk["digest"] in _FakeHub.state["v2_cas"]
        offset += int(chunk["len"])
    assert offset == len(raw)


def test_a_grant_for_an_undeclared_object_is_refused_not_guessed(
    fake_hub: Any, tmp_path: Path,
) -> None:
    """Staging by reference must not become "upload whatever is lying around"."""

    from gen_worker._vendor.tensorfs.refs import CASRef
    from gen_worker.transfer.grants import StagedRegions

    source = StagedRegions({})
    with pytest.raises(Exception) as excinfo:
        source.region(CASRef("00" * 32), 7)
    assert "never declared" in str(excinfo.value)


# --------------------------------------------------------- the whole clone


def _fake_plan(source_dir: Path) -> Any:
    """The network-derived plan for a tree that is already on this disk."""

    files = [
        (p.relative_to(source_dir).as_posix(), p.stat().st_size,
         hashlib.sha256(p.read_bytes()).hexdigest())
        for p in sorted(source_dir.rglob("*")) if p.is_file()
    ]
    return SimpleNamespace(
        provider="huggingface",
        paths=[name for name, _, _ in files],
        source_storage_bits=32,
        classification=SimpleNamespace(
            strategy="transformers",
            attrs={"dtype": "fp32", "file_layout": "single-file",
                   "file_type": "safetensors"},
        ),
        bank_files=lambda: list(files),
    )


def test_a_real_clone_stays_inside_the_disk_it_demanded(
    fake_hub: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole pipeline — plan, cast, publish — measured against its budget.

    Sampled at the peak (the instant the uploader is handed its grants: the
    source tree, the cast tree and everything staged coexist). Three bars,
    because a 2 GiB margin will absorb a fixture-scale tree and prove nothing:
    the peak is inside the demand, the publish's own contribution is ZERO, and
    what the stage list does not name is journal-scale, not tree-scale.
    """

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "config.json").write_bytes(b'{"architectures":["Fake"]}')
    (source_dir / "model.safetensors").write_bytes(
        _safetensors({f"blocks.{i}.weight": ("F32", 65536) for i in range(8)}))

    plan = _fake_plan(source_dir)
    demand = plan_disk_demand(plan, [BF16])
    assert demand is not None
    accounted = sum(stage.bytes for stage in demand.stages)

    cas_root = tensorhub_cas_dir()
    work_root = tmp_path / "work"
    peak = {"pipeline": 0, "cas": 0}
    real_upload = __import__(
        "gen_worker.hubio.client", fromlist=["upload"]).upload

    def _sampling_upload(*args: Any, **kwargs: Any) -> Any:
        peak["pipeline"] = max(
            peak["pipeline"],
            _bytes_under(work_root) + _bytes_under(source_dir))
        peak["cas"] = max(peak["cas"], _bytes_under(cas_root))
        return real_upload(*args, **kwargs)

    monkeypatch.setattr("gen_worker.hubio.client.upload", _sampling_upload)
    monkeypatch.setenv("COZY_CONVERT_WORKDIR", str(work_root))
    monkeypatch.setattr("gen_worker.convert.clone.plan_huggingface",
                        lambda *a, **k: plan)
    monkeypatch.setattr(
        "gen_worker.convert.clone.ingest_huggingface",
        lambda source_ref, dest_dir, **kwargs: IngestedSource(
            provider="huggingface", source_ref=source_ref,
            source_revision="5be7df96", dir=source_dir, layout="single-file",
            model_family="fake", model_family_variant="fake1",
            classification=plan.classification,
            attrs=dict(plan.classification.attrs),
            metadata={"source_provider": "huggingface"},
            repo_spec={"kind": "model", "library_name": "transformers"},
        ))

    result = run_clone(
        _Ctx(fake_hub), provider="huggingface", source_ref="fake/tree",
        destination_repo="tensorhub/fake-tree", destination_release="r1",
        outputs=[{"dtype": "bf16", "file_layout": "multi-file",
                  "file_type": "safetensors"}],
    )

    assert not result.failed_flavors, result.failed_flavors
    assert result.published and result.published[0]["dtype"] == "bf16"
    assert peak["pipeline"] > 0, "the sampler never ran; the peak is unmeasured"
    total = peak["pipeline"] + peak["cas"]
    assert total <= demand.required, (
        f"peak {total} bytes on disk against {demand.describe()}")
    assert peak["cas"] == 0, (
        f"the publish staged {peak['cas']} bytes of its own — the term that "
        "made a 50 GB mirror die at ENOSPC after paying for everything")
    assert total - accounted < 64 * 1024, (
        f"peak {total} exceeds the {accounted} bytes the stage list names by "
        f"more than a journal's worth: {demand.describe()}")


# ------------------------------------------------------------ the accounting


def _sensenova_plan() -> Any:
    """se#840's tree as the plan measured it: 50.19 GB over 13 shards."""

    shard = 50_190_000_000 // 13
    files = [(f"model-{i:05d}-of-00013.safetensors", shard, f"{i:064x}")
             for i in range(1, 14)]
    files.append(("config.json", 4096, "c" * 64))
    return SimpleNamespace(
        provider="huggingface",
        paths=[name for name, _, _ in files],
        source_storage_bits=32,
        classification=SimpleNamespace(
            strategy="transformers",
            attrs={"dtype": "fp32", "file_layout": "single-file",
                   "file_type": "safetensors"},
        ),
        bank_files=lambda: list(files),
    )


def _with_free(monkeypatch: pytest.MonkeyPatch, free: int) -> None:
    import shutil as _shutil

    monkeypatch.setattr(
        "gen_worker.convert.clone.shutil.disk_usage",
        lambda _p: _shutil._ntuple_diskusage(  # type: ignore[attr-defined]
            total=free * 2, used=free, free=free))


def test_the_demand_names_the_publish_even_when_it_costs_nothing(
    tmp_path: Path,
) -> None:
    """The stage the old budget had no name for is now always in the list."""

    demand = plan_disk_demand(_sensenova_plan(), [BF16])
    assert demand is not None
    names = [stage.name for stage in demand.stages]
    assert names == ["source tree", "1 materialized output tree(s)",
                     "publish staging"]
    assert demand.required_on(CAS) == 0
    assert "publish staging 0.0 GiB" in demand.describe()


def test_a_publish_that_stages_bytes_is_refused_at_zero_dollars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wiring, falsified: put the copy back and the guard fires FIRST.

    `publish_staging_bytes` is the publish's own statement of what it writes.
    Restoring the pre-fix answer (a whole local copy of the published tree)
    must show up in the demand as a whole extra tree and refuse se#840's
    mirror before it rents anything — instead of the 250 GPU-s / 41,950 uUSD
    the real job spent to learn the same fact at `cas.ingest_file`.
    """

    plan = _sensenova_plan()
    source_bytes = sum(size for _, size, _ in plan.bank_files())
    cast_bytes = source_bytes // 2  # fp32 source, bf16 output
    honest = plan_disk_demand(plan, [BF16])
    assert honest is not None
    assert honest.required == source_bytes + cast_bytes + 2 * GIB

    monkeypatch.setattr("gen_worker.convert.clone.publish_staging_bytes",
                        lambda published: published)
    copies = plan_disk_demand(plan, [BF16])
    assert copies is not None
    assert copies.required_on(CAS) == cast_bytes
    assert copies.required == honest.required + cast_bytes

    # The pod that ran the real job had room for the honest demand and not
    # for the true one. Pre-fix that pod was ADMITTED and billed.
    _with_free(monkeypatch, honest.required + GIB)
    with pytest.raises(CloneDiskSpaceError) as excinfo:
        _preflight_disk(tmp_path, plan, [BF16])
    assert "publish staging" in str(excinfo.value)
    assert "not enough disk for clone" in str(excinfo.value)
