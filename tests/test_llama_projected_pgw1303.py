"""pgw#1303 tier 3: the llama.cpp runtime over a PROJECTED tree.

Paul's ruling puts `llama-server -m` and `gguf.GGUFReader` at tier 3 on the
serving fleet — a pod has no FUSE, so a third-party binary that insists on a
real file gets a real file, permanently. The seam existed after pgw#1308 step
⑥ but reached only ONE of the runtime's three consumers, and the two it
missed both fail the way a projection fails worst: quietly.

* `llama-server -m` was handed the projected path itself (`server.py`'s
  `resolve_gguf(model_source)`), i.e. a ~128 byte `TFSSTUB1` file. llama.cpp
  refuses it at its own parse site — correct, and useless.
* `read_gguf_info().size_bytes` sums `st_size` over the shard group, and a
  stub's `st_size` is ~128 bytes REGARDLESS of the model behind it. That
  number is the whole VRAM fit: a 30 GiB model reports as ~128 bytes, every
  layer "fits", `-ngl` goes to full offload, and the degraded-boot ladder
  this runtime is built around never engages. Nothing raises.

Every arm runs the real consumer over a real `LocalCAS` and the real
`project_snapshot`, with the VENDORED `stories260K.gguf` as the model, so the
header is one the `gguf` package actually parses.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from gen_worker._vendor.tensorfs import LocalCAS, project_snapshot
from gen_worker.models import projection
from gen_worker.models.projection import REF_PREFIX, SNAPSHOTS_DIR
from gen_worker.runtimes.llama import plan_fit, read_gguf_info, resolve_gguf
from gen_worker.runtimes.server import DegradingBoot, llama_server

_TINY_GGUF = Path(__file__).resolve().parent / "testdata" / "stories260K.gguf"
_TINY_GGUF_SHA256 = "270cba1bd5109f42d03350f60406024560464db173c0e387d91f0426d3bd256d"


def _tiny_gguf_bytes() -> bytes:
    """The vendored model, digest-checked so a bad checkout says so."""

    payload = _TINY_GGUF.read_bytes()
    got = hashlib.sha256(payload).hexdigest()
    assert got == _TINY_GGUF_SHA256, (
        f"{_TINY_GGUF} is not the vendored model (sha256 {got}, expected "
        f"{_TINY_GGUF_SHA256})"
    )
    return payload


def _projected(base: Path, files: dict[str, bytes], key: str = "c" * 64) -> Path:
    """A real projected snapshot holding ``files``. No materialization."""

    source = base / "source"
    source.mkdir(parents=True, exist_ok=True)
    for name, payload in files.items():
        (source / name).write_bytes(payload)
    cas = LocalCAS(base)
    manifest = cas.ingest_repository(source)
    # Exactly `cozy_snapshot._pin_manifest`, so the production resolver runs.
    cas.compare_and_swap_ref(
        REF_PREFIX + key, cas.store_manifest(manifest), expected=None
    )
    tree = base / SNAPSHOTS_DIR / key
    project_snapshot(cas, manifest, tree)
    return tree


def test_resolve_gguf_hands_back_a_real_file_not_a_stub(tmp_path: Path) -> None:
    payload = _tiny_gguf_bytes()
    tree = _projected(tmp_path, {"model.gguf": payload})
    # The fixture is the article: without this the rest proves nothing.
    assert projection.stub_at(tree / "model.gguf") is not None

    resolved = resolve_gguf(tree)

    assert projection.stub_at(resolved) is None
    assert resolved.read_bytes() == payload
    assert resolved.stat().st_size == len(payload)


def test_every_shard_is_real_not_only_the_one_named(tmp_path: Path) -> None:
    """llama.cpp opens the siblings ITSELF, told only about the first.

    A per-file copy would satisfy `-m` and leave `-00002-of-00002` a stub —
    a failure that appears only for split models, i.e. only for the large
    ones, i.e. only in production.
    """

    payload = _tiny_gguf_bytes()
    tree = _projected(
        tmp_path,
        {
            "m-00001-of-00002.gguf": payload,
            "m-00002-of-00002.gguf": payload + b"\0" * 64,
        },
    )

    resolved = resolve_gguf(tree)

    assert resolved.name == "m-00001-of-00002.gguf"
    sibling = resolved.parent / "m-00002-of-00002.gguf"
    assert projection.stub_at(sibling) is None
    assert sibling.stat().st_size == len(payload) + 64


def test_the_vram_fit_is_sized_from_real_bytes(tmp_path: Path) -> None:
    """`size_bytes` decides `-ngl`. A stub's `st_size` decides nothing true."""

    payload = _tiny_gguf_bytes()
    tree = _projected(tmp_path, {"model.gguf": payload})

    info = read_gguf_info(resolve_gguf(tree))

    assert info.architecture == "llama"
    assert info.n_layers > 0
    assert info.size_bytes == len(payload)
    # The number that would have been read off the stub is ~128 bytes, and it
    # is not merely wrong — it is small enough that any model "fits".
    assert info.size_bytes > 100_000
    plan = plan_fit(info, free_vram_gb=80.0)
    assert plan.n_gpu_layers == info.n_layers + 1 and not plan.degraded


def test_read_gguf_info_sizes_the_model_even_when_handed_a_stub(
    tmp_path: Path,
) -> None:
    """`read_gguf_info` is public and `plan_for` passes on what it was given.

    Routing the seam through `resolve_gguf` alone would leave this entry
    reading its sizes off stubs — and the arm above could not see it, because
    it hands over an already-real path. This one hands over the stub.
    """

    payload = _tiny_gguf_bytes()
    tree = _projected(tmp_path, {"model.gguf": payload})
    stub = tree / "model.gguf"
    assert projection.stub_at(stub) is not None
    # The number the fit used to read, on a model of any size whatsoever.
    assert stub.stat().st_size < 1024

    info = read_gguf_info(stub)

    assert info.size_bytes == len(payload)


def test_the_server_argv_names_a_real_file(tmp_path: Path) -> None:
    """The consumer that is a SUBPROCESS: what `-m` actually points at."""

    payload = _tiny_gguf_bytes()
    tree = _projected(tmp_path, {"model.gguf": payload})

    booter = llama_server(str(tree), port=4321)
    proc = booter.candidates[0] if isinstance(booter, DegradingBoot) else booter

    named = Path(proc.command[proc.command.index("-m") + 1])
    assert projection.stub_at(named) is None
    assert named.read_bytes() == payload


def test_a_tree_that_is_not_projected_is_untouched(tmp_path: Path) -> None:
    """The seam is unconditional at the call site and must stay a no-op here.

    An ordinary directory — a bare download, a test fixture, a conversion
    workdir — is already real files, and resolution must not route it through
    a snapshot view that does not exist.
    """

    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "model.gguf").write_bytes(b"GGUF-not-really")

    assert resolve_gguf(plain) == plain / "model.gguf"
    assert resolve_gguf(plain / "model.gguf") == plain / "model.gguf"


def test_a_missing_binding_still_refuses_by_name(tmp_path: Path) -> None:
    """The refusals resolution owns must survive the seam being added."""

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(Exception, match="no .gguf file"):
        resolve_gguf(empty)
