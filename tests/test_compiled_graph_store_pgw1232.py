"""TCG/HashRepo sidecars are strict, durable, and restart-safe."""

from __future__ import annotations

import inspect
import json
import multiprocessing
import os
import stat
import struct
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from hashrepo import LocalCAS
from torch_compiled_graphs import (
    Engine,
    GraphClassDeclaration,
)
from torch_compiled_graphs.artifact import build_metadata, pack_artifact
from torch_compiled_graphs.host_isa import _host_requirement

from gen_worker import compiled_graph_store, fleet_cells


KEY = "cg-key-v1-" + "1" * 56
OTHER_KEY = "cg-key-v1-" + "2" * 56


def _record(key: str) -> dict[str, object]:
    return {
        "format": 1,
        "compiled_graph_key": key,
        "family": "micro",
        "arm_token": "arm-v1-abc",
        "bytes": 7,
        "content_digest": "sha256:" + "4" * 64,
        "stored_at": 1.0,
        "manifest": "sha256:" + "3" * 64,
        "verdict": compiled_graph_store.VERDICT_ADMITTED,
        "sink": compiled_graph_store.SINK_NONE,
    }


def _record_path(root: Path, key: str = KEY) -> Path:
    return compiled_graph_store.sidecar_dir(key, root) / compiled_graph_store.RECORD_NAME


def _write_record(root: Path, record: dict[str, object]) -> Path:
    path = _record_path(root)
    compiled_graph_store._write_json_atomic(path, record)
    return path


def _elf() -> bytes:
    names = b"\0.shstrtab\0.lrodata\0"
    section_offset = 64
    section_size = 64
    string_offset = section_offset + section_size * 3
    image = bytearray(string_offset + len(names))
    image[:4] = b"\x7fELF"
    image[4:7] = bytes((2, 1, 1))
    struct.pack_into("<Q", image, 0x28, section_offset)
    struct.pack_into("<HHH", image, 0x3A, section_size, 3, 1)
    struct.pack_into("<II", image, section_offset + section_size, 1, 3)
    struct.pack_into(
        "<QQ",
        image,
        section_offset + section_size + 0x18,
        string_offset,
        len(names),
    )
    struct.pack_into("<II", image, section_offset + 2 * section_size, 11, 1)
    struct.pack_into(
        "<QQ",
        image,
        section_offset + 2 * section_size + 0x18,
        len(image),
        0,
    )
    image[string_offset:] = names
    return bytes(image)


def _real_artifact(tmp_path: Path) -> tuple[Path, str]:
    name = "denoiser/h=64,w=64"
    package = tmp_path / "model.pt2"
    wrapper = "AOTInductorModelBase(1, 1, 0, device_str, std::move(cubin_dir), false)"
    with zipfile.ZipFile(package, "w") as archive:
        root = f"data/aotinductor/{name}"
        archive.writestr(f"{root}/model.wrapper.cpp", wrapper)
        archive.writestr(f"{root}/model.so", _elf())
    graph = {
        "v": 3,
        "constant_fqns": [],
        "lifted_inputs": [],
        "pytree": {"in": "leaf", "out": "leaf"},
        "specialization": {},
    }
    declaration = GraphClassDeclaration(
        graph_class=name,
        target="unet",
        graph=graph,
        graph_witness="fedcba9876543210",
        range_digest="0123456789abcdef" * 2,
    )
    toolchain = {"torch": "torch-content", "triton": "triton-content"}
    metadata = build_metadata(
        graph_class={
            "name": declaration.graph_class,
            "target": declaration.target,
            "class_hash": declaration.class_hash,
            "graph": dict(declaration.graph),
            "graph_witness": declaration.graph_witness,
            "range_digest": declaration.range_digest,
            "fork": [],
            "class_dims": [],
            "strict": True,
            "lora_bucket": 0,
            "literal_values": "",
            "literal_payload_values": "",
            "placement": list(declaration.placement),
            "constants": [],
        },
        sm="sm_89",
        toolchain=toolchain,
        host_isa=_host_requirement().facts(),
    )
    artifact = pack_artifact(package, tmp_path / "compiled_graph.tar.gz", metadata)
    return artifact, str(metadata["compiled_graph_key"])


def _mark_process(
    root: str,
    key: str,
    field: str,
    value: str,
    ready: Any,
    go: Any,
    results: Any,
) -> None:
    ready.set()
    go.wait(5)
    kwargs = {field: value}
    results.put(compiled_graph_store.mark(key, root=Path(root), **kwargs))


def test_real_tcg_hashrepo_sidecar_resolves_after_engine_and_process_restart(
    tmp_path: Path,
) -> None:
    artifact, key = _real_artifact(tmp_path)
    cas_root = tmp_path / "cas"

    stored = compiled_graph_store.store(
        artifact,
        key=key,
        family="micro",
        arm_token="arm-v1-real",
        verdict=compiled_graph_store.VERDICT_ADMITTED,
        sink=compiled_graph_store.SINK_OWED,
        root=cas_root,
    )

    assert stored is not None
    record = compiled_graph_store._read_record(_record_path(cas_root, key))
    assert record is not None
    manifest = LocalCAS(cas_root).load_manifest(str(record["manifest"]))
    assert stored.bytes == manifest.files[0].size_bytes
    assert stored.content_digest == str(manifest.files[0].digest)
    assert stored.artifact.read_bytes() == artifact.read_bytes()
    fresh = Engine(LocalCAS(cas_root)).resolve(key, tmp_path / "fresh-engine")
    assert fresh is not None and fresh.key == key

    destination = tmp_path / "fresh-process"
    code = """
import sys
from pathlib import Path
from hashrepo import LocalCAS
from torch_compiled_graphs import Engine
from gen_worker import compiled_graph_store
root, key, destination = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
sidecar = compiled_graph_store.lookup(key, root)
assert sidecar is not None and sidecar.compiled_graph_key == key
resolved = Engine(LocalCAS(root)).resolve(key, destination)
assert resolved is not None and resolved.key == key
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, str(cas_root), key, str(destination)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_production_root_uses_only_the_canonical_worker_cas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = tmp_path / "canonical-cas"
    calls: list[Path | None] = []

    def opened(root: Path | None = None) -> Any:
        calls.append(root)
        return SimpleNamespace(root=canonical if root is None else Path(root))

    monkeypatch.setenv("GEN_WORKER_LOCAL_COMPILED_GRAPHS_DIR", str(tmp_path / "split"))
    monkeypatch.setattr(compiled_graph_store, "open_worker_cas", opened)

    assert compiled_graph_store.store_root() == canonical
    assert compiled_graph_store.sidecars_root() == canonical / compiled_graph_store.SIDECARS_DIRNAME
    scoped = tmp_path / "scoped-test-cas"
    assert compiled_graph_store._cas(scoped).root == scoped
    assert calls == [None, None, scoped]
    source = inspect.getsource(compiled_graph_store)
    assert "GEN_WORKER_LOCAL_COMPILED_GRAPHS_DIR" not in source


def test_same_process_concurrent_atomic_writes_have_unique_temporaries(
    tmp_path: Path,
) -> None:
    path = tmp_path / "rows" / "record.json"

    def write(index: int) -> None:
        compiled_graph_store._write_json_atomic(path, {"index": index})

    with ThreadPoolExecutor(max_workers=32) as executor:
        list(executor.map(write, range(512)))

    assert json.loads(path.read_text())["index"] in range(512)
    assert not list(path.parent.glob(".record.json.tmp-*"))


def test_atomic_write_fsyncs_file_and_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kinds: list[int] = []
    directories: set[Path] = set()
    real_fsync = os.fsync

    def recording_fsync(descriptor: int) -> None:
        mode = os.fstat(descriptor).st_mode
        kinds.append(mode)
        if stat.S_ISDIR(mode):
            directories.add(Path(os.readlink(f"/proc/self/fd/{descriptor}")))
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", recording_fsync)
    path = tmp_path / "new-parent" / "nested" / "record.json"
    compiled_graph_store._write_json_atomic(path, {"v": 1})

    assert any(stat.S_ISREG(mode) for mode in kinds)
    assert directories == {tmp_path, path.parent.parent, path.parent}


def test_root_parent_output_is_readable_after_privilege_drop(tmp_path: Path) -> None:
    previous_umask = os.umask(0o077)
    try:
        path = _write_record(tmp_path, _record(KEY))
    finally:
        os.umask(previous_umask)

    assert stat.S_IMODE(path.stat().st_mode) == 0o644
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o755
    assert stat.S_IMODE(path.parent.parent.stat().st_mode) == 0o755
    assert not list(path.parent.glob("*.lock"))


def test_sidecar_is_readable_by_a_real_unprivileged_process(tmp_path: Path) -> None:
    image = "nginx:alpine"
    available = subprocess.run(
        ["docker", "image", "inspect", image],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if available.returncode != 0:
        pytest.skip(f"cached {image} image is unavailable")
    path = _write_record(tmp_path, _record(KEY))

    completed = subprocess.run(
        [
            "docker", "run", "--rm", "--user", "65534:65534",
            "--volume", f"{path.parent}:/sidecar:ro",
            image, "cat", "/sidecar/record.json",
        ],
        check=False,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr.decode()
    assert json.loads(completed.stdout) == _record(KEY)


def test_existing_root_parent_tree_requires_no_child_chmod(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write_record(tmp_path, _record(KEY))

    def refused_chmod(_path: Path, _mode: int) -> None:
        raise PermissionError("dropped child cannot chmod root-owned directories")

    monkeypatch.setattr(os, "chmod", refused_chmod)
    assert compiled_graph_store.mark(
        KEY,
        sink=compiled_graph_store.SINK_NONE,
        root=tmp_path,
    )
    assert compiled_graph_store._read_record(path) is not None


def test_cross_process_mark_is_locked_fresh_read_modify_write(tmp_path: Path) -> None:
    record = _record(KEY)
    record["verdict"] = compiled_graph_store.VERDICT_UNVERIFIED
    record["sink"] = compiled_graph_store.SINK_OWED
    path = _write_record(tmp_path, record)
    context = multiprocessing.get_context("spawn")
    ready = (context.Event(), context.Event())
    go = context.Event()
    results = context.Queue()
    processes = (
        context.Process(
            target=_mark_process,
            args=(
                str(tmp_path),
                KEY,
                "verdict",
                compiled_graph_store.VERDICT_QUARANTINED,
                ready[0],
                go,
                results,
            ),
        ),
        context.Process(
            target=_mark_process,
            args=(
                str(tmp_path),
                KEY,
                "sink",
                compiled_graph_store.SINK_DELIVERED,
                ready[1],
                go,
                results,
            ),
        ),
    )
    with compiled_graph_store._record_lock(path):
        for process in processes:
            process.start()
        assert all(event.wait(5) for event in ready)
        go.set()
        time.sleep(0.2)
        assert all(process.is_alive() for process in processes), (
            "mark ignored the cross-process record lock"
        )
    for process in processes:
        process.join(10)
        assert process.exitcode == 0
    assert [results.get(timeout=2), results.get(timeout=2)] == [True, True]
    final = compiled_graph_store._read_record(path)
    assert final is not None
    assert final["verdict"] == compiled_graph_store.VERDICT_QUARANTINED
    assert final["sink"] == compiled_graph_store.SINK_DELIVERED
    assert not list(path.parent.glob("*.lock"))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("verdict", "banana"),
        ("sink", "teleported"),
        ("family", "../escape"),
        ("arm_token", ".hidden"),
        ("bytes", True),
        ("stored_at", float("nan")),
        ("manifest", "sha256:not-a-digest"),
        ("content_digest", "4" * 64),
    ],
)
def test_sidecar_schema_rejects_invalid_recursive_fields(
    field: str,
    value: object,
    tmp_path: Path,
) -> None:
    record = _record(KEY)
    record[field] = value
    if isinstance(value, float) and not value == value:
        path = _record_path(tmp_path)
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(record))
    else:
        path = _write_record(tmp_path, record)

    with pytest.raises(compiled_graph_store._PersistedStateError):
        compiled_graph_store._read_record(path)
    assert compiled_graph_store.lookup(KEY, tmp_path) is None


def test_persisted_json_rejects_duplicate_keys_recursively(tmp_path: Path) -> None:
    path = _record_path(tmp_path)
    path.parent.mkdir(parents=True)
    encoded = json.dumps(_record(KEY))
    path.write_text(encoded[:-1] + ',"sink":"owed"}')

    with pytest.raises(compiled_graph_store._PersistedStateError):
        compiled_graph_store._read_record(path)
    nested = tmp_path / "nested.json"
    nested.write_text('{"outer":{"key":1,"key":2}}')
    with pytest.raises(ValueError, match="duplicate JSON key 'key'"):
        compiled_graph_store._read_json(nested)


def test_mark_rejects_unknown_and_backward_states(tmp_path: Path) -> None:
    path = _write_record(tmp_path, _record(KEY))

    assert not compiled_graph_store.mark(KEY, verdict="banana", root=tmp_path)
    assert not compiled_graph_store.mark(KEY, sink="teleported", root=tmp_path)
    assert compiled_graph_store.mark(
        KEY,
        verdict=compiled_graph_store.VERDICT_QUARANTINED,
        root=tmp_path,
    )
    assert not compiled_graph_store.mark(
        KEY,
        verdict=compiled_graph_store.VERDICT_ADMITTED,
        root=tmp_path,
    )
    record = compiled_graph_store._read_record(path)
    assert record is not None
    assert record["verdict"] == compiled_graph_store.VERDICT_QUARANTINED


def test_owed_scan_filters_sidecars_before_exporting_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owed = _record(KEY)
    owed["sink"] = compiled_graph_store.SINK_OWED
    _write_record(tmp_path, owed)
    other_path = _record_path(tmp_path, OTHER_KEY)
    other = _record(OTHER_KEY)
    compiled_graph_store._write_json_atomic(other_path, other)
    monkeypatch.setattr(
        compiled_graph_store,
        "_engine",
        lambda _root=None: pytest.fail("owed scan resolved TCG"),
    )
    monkeypatch.setattr(
        compiled_graph_store,
        "_export",
        lambda *_args: pytest.fail("owed scan exported artifact bytes"),
    )

    rows = compiled_graph_store.graphs_owed_to_sink(tmp_path)
    assert [row.compiled_graph_key for row in rows] == [KEY]
    assert rows[0].content_digest == owed["content_digest"]
    assert rows[0].bytes == owed["bytes"]


def test_sidecar_key_must_equal_its_directory_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _record_path(tmp_path)
    compiled_graph_store._write_json_atomic(path, _record(OTHER_KEY))
    monkeypatch.setattr(
        compiled_graph_store,
        "_engine",
        lambda _root=None: pytest.fail("wrong sidecar reached TCG resolve"),
    )

    assert compiled_graph_store.lookup(KEY, tmp_path) is None
    assert not compiled_graph_store.has_graphs(tmp_path)


def test_load_runner_refuses_a_runner_for_another_exact_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _record(KEY)
    _write_record(tmp_path, record)
    graph = SimpleNamespace(
        key=KEY,
        manifest=record["manifest"],
        metadata={},
    )
    wrong_runner = SimpleNamespace(key=OTHER_KEY)

    class FakeCAS:
        def load_manifest(self, manifest: object) -> Any:
            assert str(manifest) == record["manifest"]
            return SimpleNamespace(files=(SimpleNamespace(
                digest=record["content_digest"], size_bytes=record["bytes"]
            ),))

    class FakeEngine:
        def __init__(self, _cas: object) -> None:
            pass

        def resolve(self, key: str, destination: Path) -> Any:
            assert key == KEY
            assert destination.name == KEY
            return graph

        def runner(self, key: str, destination: Path) -> Any:
            assert key == KEY
            assert destination.name == KEY
            return wrong_runner

    monkeypatch.setattr(compiled_graph_store, "_cas", lambda _root=None: FakeCAS())
    monkeypatch.setattr(compiled_graph_store, "Engine", FakeEngine)

    assert compiled_graph_store.load_runner(KEY, tmp_path) is None


def test_malformed_key_never_reaches_tcg(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        compiled_graph_store,
        "_engine",
        lambda _root=None: pytest.fail("malformed key reached TCG"),
    )
    assert compiled_graph_store.load_runner("cell-not-a-key", tmp_path) is None


@pytest.mark.parametrize("duplicate", [False, True])
def test_store_never_overwrites_present_invalid_sidecar(
    duplicate: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact, key = _real_artifact(tmp_path)
    root = tmp_path / "cas"
    path = _record_path(root, key)
    path.parent.mkdir(parents=True)
    if duplicate:
        encoded = json.dumps(_record(key))
        original = (encoded[:-1] + ',"sink":"owed"}').encode()
    else:
        original = b'{"format":1,"verdict":"banana"}'
    path.write_bytes(original)

    class NoImport:
        def __init__(self, _cas: object) -> None:
            pass

        def import_artifact(self, _key: str, _artifact: Path) -> object:
            pytest.fail("present-invalid worker state mutated TCG")

    monkeypatch.setattr(compiled_graph_store, "Engine", NoImport)

    assert compiled_graph_store.store(
        artifact, key=key, family="micro", root=root
    ) is None
    assert path.read_bytes() == original


def test_store_refuses_contextually_wrong_key_before_tcg_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact, key = _real_artifact(tmp_path)
    root = tmp_path / "cas"
    path = _record_path(root, key)
    compiled_graph_store._write_json_atomic(path, _record(OTHER_KEY))
    original = path.read_bytes()

    class NoImport:
        def __init__(self, _cas: object) -> None:
            pass

        def import_artifact(self, _key: str, _artifact: Path) -> object:
            pytest.fail("wrong-key worker state mutated TCG")

    monkeypatch.setattr(compiled_graph_store, "Engine", NoImport)
    assert compiled_graph_store.store(
        artifact, key=key, family="micro", root=root
    ) is None
    assert path.read_bytes() == original


def test_persisted_json_is_bounded_and_schema_predicate_is_total(
    tmp_path: Path,
) -> None:
    path = _record_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_bytes(b'{"padding":"' + b"x" * compiled_graph_store._MAX_JSON_BYTES)

    with pytest.raises(compiled_graph_store._PersistedStateError, match="exceeds"):
        compiled_graph_store._read_record(path)
    assert not compiled_graph_store._valid_record([])
    for field, value in (("verdict", []), ("sink", {})):
        record = _record(KEY)
        record[field] = value
        assert not compiled_graph_store._valid_record(record)


@pytest.mark.parametrize("state", ["missing", "quarantined", "wrong-key"])
def test_load_runner_requires_matching_nonquarantined_sidecar_before_tcg(
    state: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if state != "missing":
        record = _record(OTHER_KEY if state == "wrong-key" else KEY)
        if state == "quarantined":
            record["verdict"] = compiled_graph_store.VERDICT_QUARANTINED
        compiled_graph_store._write_json_atomic(_record_path(tmp_path), record)

    class NoEngine:
        def __init__(self, _cas: object) -> None:
            pytest.fail("refused sidecar reached TCG")

    monkeypatch.setattr(compiled_graph_store, "Engine", NoEngine)
    assert compiled_graph_store.load_runner(KEY, tmp_path) is None


@pytest.mark.parametrize("field", ["manifest", "content_digest", "bytes"])
def test_load_and_lookup_bind_sidecar_to_resolved_manifest_content(
    field: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actual = _record(KEY)
    record = dict(actual)
    record[field] = {
        "manifest": "sha256:" + "5" * 64,
        "content_digest": "sha256:" + "6" * 64,
        "bytes": 8,
    }[field]
    _write_record(tmp_path, record)

    class FakeCAS:
        def load_manifest(self, manifest: object) -> Any:
            assert str(manifest) == actual["manifest"]
            return SimpleNamespace(files=(SimpleNamespace(
                digest=actual["content_digest"], size_bytes=actual["bytes"]
            ),))

    class FakeEngine:
        def __init__(self, _cas: object) -> None:
            pass

        def resolve(self, key: str, _destination: Path) -> Any:
            assert key == KEY
            return SimpleNamespace(
                key=KEY, manifest=actual["manifest"], metadata={}
            )

        def runner(self, _key: str, _destination: Path) -> object:
            pytest.fail("mismatched sidecar reached package load")

        def export_artifact(self, _key: str, _destination: Path) -> Path:
            pytest.fail("mismatched sidecar reached artifact export")

    monkeypatch.setattr(compiled_graph_store, "_cas", lambda _root=None: FakeCAS())
    monkeypatch.setattr(compiled_graph_store, "Engine", FakeEngine)

    assert compiled_graph_store.lookup(KEY, tmp_path) is None
    assert compiled_graph_store.load_runner(KEY, tmp_path) is None


def test_tcg_corruption_can_be_repaired_without_worker_quarantine_resurrection(
    tmp_path: Path,
) -> None:
    artifact, key = _real_artifact(tmp_path)
    root = tmp_path / "cas"
    assert compiled_graph_store.store(
        artifact, key=key, family="micro", root=root
    ) is not None
    path = _record_path(root, key)
    record = compiled_graph_store._read_record(path)
    assert record is not None
    LocalCAS(root).object_path(str(record["content_digest"])).write_bytes(b"corrupt")

    assert compiled_graph_store.lookup(key, root) is None
    after_failure = compiled_graph_store._read_record(path)
    assert after_failure is not None
    assert after_failure["verdict"] == compiled_graph_store.VERDICT_ADMITTED
    assert compiled_graph_store.store(
        artifact, key=key, family="micro", root=root
    ) is not None
    assert compiled_graph_store.lookup(key, root) is not None

    compiled_graph_store.drop(key, root)
    assert compiled_graph_store.store(
        artifact, key=key, family="micro", root=root
    ) is None
    quarantined = compiled_graph_store._read_record(path)
    assert quarantined is not None
    assert quarantined["verdict"] == compiled_graph_store.VERDICT_QUARANTINED


def test_store_requires_arm_memo_and_preserves_all_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact, key = _real_artifact(tmp_path)
    refused_root = tmp_path / "refused"
    monkeypatch.setattr(compiled_graph_store, "note_memo", lambda *_args: False)
    assert compiled_graph_store.store(
        artifact,
        key=key,
        family="micro",
        arm_token="arm-v1-refused",
        root=refused_root,
    ) is None

    monkeypatch.undo()
    root = tmp_path / "aliases"
    for token in ("arm-v1-first", "arm-v1-second"):
        assert compiled_graph_store.store(
            artifact, key=key, family="micro", arm_token=token, root=root
        ) is not None
        assert compiled_graph_store.lookup_for_arm(token, root) is not None
    record = compiled_graph_store._read_record(_record_path(root, key))
    assert record is not None and record["arm_token"] == "arm-v1-first"


@pytest.mark.parametrize("invalid", [False, True])
def test_arm_memo_collision_or_invalid_state_is_never_overwritten(
    invalid: bool,
    tmp_path: Path,
) -> None:
    path = compiled_graph_store.memo_path("arm-v1-alias", tmp_path)
    if invalid:
        path.parent.mkdir(parents=True)
        original = b'{"format":1,"compiled_graph_key":"broken"}'
        path.write_bytes(original)
    else:
        assert compiled_graph_store.note_memo("arm-v1-alias", KEY, tmp_path)
        original = path.read_bytes()

    assert not compiled_graph_store.note_memo(
        "arm-v1-alias", OTHER_KEY, tmp_path
    )
    assert path.read_bytes() == original


def test_duplicate_trust_state_is_never_overwritten(tmp_path: Path) -> None:
    path = compiled_graph_store.sidecars_root(tmp_path) / compiled_graph_store.TRUST_CLASS_NAME
    path.parent.mkdir(parents=True)
    original = (
        b'{"format":1,"class":"untrusted","code":'
        b'"compiled_graph_publish_untrusted_tier","detail":"a",'
        b'"detail":"b","learned_at":1}'
    )
    path.write_bytes(original)

    assert not compiled_graph_store.note_refusal(
        compiled_graph_store.UNTRUSTED_REFUSAL_CODE, root=tmp_path
    )
    assert path.read_bytes() == original


def test_resume_filters_inflight_before_resolve_and_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owed = [
        compiled_graph_store.OwedCompiledGraph(
            key, "sha256:" + "4" * 64, "micro", "arm-v1-a", 7
        )
        for key in (KEY, OTHER_KEY)
    ]
    looked_up: list[str] = []
    published: list[tuple[Path, dict[str, object]]] = []
    thread = SimpleNamespace()

    monkeypatch.setattr(compiled_graph_store, "graphs_owed_to_sink", lambda: owed)
    monkeypatch.setattr(fleet_cells, "publishes_in_flight", lambda: [KEY])

    def lookup(key: str) -> Any:
        looked_up.append(key)
        assert key == OTHER_KEY
        return compiled_graph_store.LocalCompiledGraph(
            key,
            tmp_path / "resolved.tar.gz",
            "sha256:" + "4" * 64,
            "micro",
            "arm-v1-b",
            7,
            compiled_graph_store.SINK_OWED,
            {"compiled_graph_key": key},
        )

    def publish(_publisher: object, _family: str, artifact: Path, meta: Any,
                **_kwargs: object) -> object:
        published.append((artifact, dict(meta)))
        return thread

    monkeypatch.setattr(compiled_graph_store, "lookup", lookup)
    monkeypatch.setattr(fleet_cells, "_publish_async", publish)
    publisher = SimpleNamespace(enabled=lambda: True)

    assert fleet_cells.resume_owed_publishes(cast(Any, publisher)) == [thread]
    assert looked_up == [OTHER_KEY]
    assert published == [(tmp_path / "resolved.tar.gz", {
        "compiled_graph_key": OTHER_KEY,
    })]
