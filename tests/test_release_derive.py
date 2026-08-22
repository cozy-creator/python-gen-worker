from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("transformers")
import gen_worker._vendor.torchcg  # noqa: E402,F401

FIXTURES = Path(__file__).resolve().parent / "release_fixtures"


@pytest.fixture(scope="module")
def config_only_tree(tmp_path_factory: pytest.TempPathFactory) -> Path:
    sys.path.insert(0, str(FIXTURES))
    try:
        import tiny_tree
    finally:
        sys.path.remove(str(FIXTURES))
    return tiny_tree.save_config_only(tmp_path_factory.mktemp("tiny-config-only"))


LOCK = (
    'version = 1\n'
    '\n[[package]]\nname = "torch"\nversion = "2.13.0"\n'
    '\n[[package]]\nname = "triton"\nversion = "3.7.1"\n'
    '\n[[package]]\nname = "nvidia-cublas"\nversion = "13.1.1.3"\n'
    '\n[[package]]\nname = "diffusers"\nversion = "0.39.0"\n'
)


@pytest.fixture(scope="module")
def lockfile(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("tiny-lock") / "uv.lock"
    path.write_text(LOCK)
    return path


def _derive(module: str, tree: Path, out: Path, *extra: str) -> int:
    from gen_worker.cli import main

    if "--lockfile" not in extra:
        default = out.parent / "default-uv.lock"
        default.write_text(LOCK)
        extra = ("--lockfile", str(default), *extra)
    return main(
        [
            "release",
            "derive",
            "--dir",
            str(FIXTURES),
            "--module",
            module,
            "--checkpoint",
            str(tree),
            "--out",
            str(out),
            *extra,
        ]
    )


def test_derive_discovers_the_auto_enumerated_graph_set(
    config_only_tree: Path, tmp_path: Path
) -> None:
    out = tmp_path / "release.json"
    assert _derive("tiny_endpoint", config_only_tree, out) == 0
    document = json.loads(out.read_bytes())
    assert document["kind"] == "gen-worker.release-metadata@1"
    assert document["endpoint"].endswith(":TinyModel")

    (lane,) = document["graphs"]["lanes"]
    assert lane["contract"] == "sd15.diffusers@1+plain.f32@1"
    assert lane["unobserved_targets"] == []
    assert len(lane["graphs"]) == 4
    assert {record["target"] for record in lane["graphs"]} == {"unet"}
    batches = sorted(
        record["ingress"]["inputs"][0]["shape"][0] for record in lane["graphs"]
    )
    assert batches == [1, 1, 2, 2]

    assert lane["graphs"][0]["ingress"]["inputs"][0]["shape"] == [2, 4, 64, 64]

    schema = document["checkpoint_defaults_schema"]
    assert schema is not None
    rendered = json.dumps(schema)
    assert "cfg" in rendered and "steps" in rendered and "guidance" in rendered


def test_document_is_byte_stable_across_a_subprocess_fence(
    config_only_tree: Path, tmp_path: Path, lockfile: Path
) -> None:
    """Two FRESH interpreters derive identical bytes."""

    first = tmp_path / "first.json"
    again = tmp_path / "again.json"
    assert _derive("tiny_endpoint", config_only_tree, first) == 0
    assert _derive("tiny_endpoint", config_only_tree, again) == 0
    assert first.read_bytes() == again.read_bytes()

    fenced: list[bytes] = []
    for run_name in ("fence-a.json", "fence-b.json"):
        out = tmp_path / run_name
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "gen_worker.cli",
                "release",
                "derive",
                "--dir",
                str(FIXTURES),
                "--module",
                "tiny_endpoint",
                "--checkpoint",
                str(config_only_tree),
                "--lockfile",
                str(lockfile),
                "--out",
                str(out),
            ],
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr[-2000:]
        fenced.append(out.read_bytes())
    assert fenced[0] == fenced[1]


def test_a_lane_naming_a_missing_path_fails_red_with_the_path(
    config_only_tree: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _derive("bad_lane_endpoint", config_only_tree, tmp_path / "o.json") == 1
    stderr = capsys.readouterr().err
    assert "does_not_exist" in stderr


def test_a_never_called_target_fails_red_not_silent(
    config_only_tree: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`vae` exists but only `vae.decode()` runs -- the hook sees nothing."""

    assert _derive("unobserved_endpoint", config_only_tree, tmp_path / "o.json") == 1
    stderr = capsys.readouterr().err
    assert "never CALLED" in stderr and "'vae'" in stderr


def test_an_endpoint_that_marks_nothing_derives_zero_graphs(
    config_only_tree: Path, tmp_path: Path
) -> None:

    out = tmp_path / "eager.json"
    assert _derive("eager_endpoint", config_only_tree, out) == 0
    document = json.loads(out.read_bytes())
    assert document["graphs"]["lanes"] == []
    assert document["checkpoint_defaults_schema"] == {}
    assert document["model_type"] is not None


def test_graph_identity_does_not_come_from_the_declaring_MODULE(
    config_only_tree: Path, tmp_path: Path
) -> None:

    declared = tmp_path / "declared.json"
    twin = tmp_path / "twin.json"
    assert _derive("tiny_endpoint", config_only_tree, declared) == 0
    assert _derive("derived_twin_endpoint", config_only_tree, twin) == 0

    (contract_lane,) = json.loads(declared.read_bytes())["graphs"]["lanes"]
    document = json.loads(twin.read_bytes())
    (twin_lane,) = document["graphs"]["lanes"]

    assert contract_lane["contract"] == "sd15.diffusers@1+plain.f32@1"
    assert twin_lane["contract"] == "sd15.diffusers@1+plain.f32@1"
    assert [record["graph"] for record in twin_lane["graphs"]] == [
        record["graph"] for record in contract_lane["graphs"]
    ]
    assert len(twin_lane["graphs"]) == 4

    # Every lane row names a REAL stamp pair. The `derived: true` marker is
    # gone with the identity it flagged — there is no second kind of lane left
    # for a reader to have to tell apart.
    #
    # ⚠️ THE INLINE `document` IS DELETED (pgw#1621), and this assertion says
    # so rather than being dropped. v1 inlined the lane's whole tensorfs
    # contract document plus its digest into every release, because a v1 lane
    # WAS a stored document. A v2 layout is `quant(topology)` and is COMPUTED
    # by the Go engine, never stored, so there is nothing to inline: the stamp
    # pair IS the identity, and both halves are ratified documents the hub
    # already holds. Re-adding a `document` key here would be a second copy of
    # a layout that has one producer.
    row = document["lane_contracts"]["sd15.diffusers@1+plain.f32@1"]
    assert row["stamp"] == "sd15.diffusers@1+plain.f32@1"
    assert "document" not in row and "digest" not in row
    assert "derived" not in row
    # The map KEY and the row's own `stamp` are ONE expression in the derive
    # (`lane_contract_handle`), because the hub REFUSES the release
    # (`release_compiled_graphs_invalid_lane`) when they disagree.
    assert set(document["lane_contracts"]) == {twin_lane["contract"]}


def test_an_unmarked_endpoint_traces_and_reports_nothing_to_compile(
    config_only_tree: Path, tmp_path: Path
) -> None:
    """The middle state: traced, and the author marked no compile target."""

    out = tmp_path / "unmarked.json"
    assert _derive("unmarked_endpoint", config_only_tree, out) == 0
    document = json.loads(out.read_bytes())
    assert document["graphs"]["lanes"] == []
    assert document["lane_contracts"] == {}
    assert set(document["entrypoints"]) == {"analyze"}
    assert document["entrypoints"]["analyze"]["model_slots"] == {
        "model": "UnmarkedModel"
    }


def test_binding_enumeration_reaches_adapter_and_cfg_arms(
    config_only_tree: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """is_trace is DELETED: arm coverage is the derive's binding enumeration."""

    from gen_worker import Adapter, DistillationAdapter
    from gen_worker.models import SDXL
    from gen_worker.release.derive import _defaults_variants, _fake_adapter

    variants = _defaults_variants(SDXL)
    assert [variant.cfg for variant in variants] == [True, False]

    fake = _fake_adapter(SDXL)
    assert isinstance(fake, Adapter)
    assert fake.defaults is not None and fake.defaults.cfg is False
    assert fake.defaults.steps.default == 4
    marked = _fake_adapter(SDXL, DistillationAdapter)
    assert isinstance(marked, DistillationAdapter)

    out = tmp_path / "release.json"
    assert _derive("tiny_endpoint", config_only_tree, out) == 0
    stderr = capsys.readouterr().err
    assert "refused by the author's own validation" in stderr


def test_the_compile_stack_is_the_env_identity_and_only_the_compiler_is_in_it(
    config_only_tree: Path, tmp_path: Path
) -> None:

    stack_only = tmp_path / "a.lock"
    stack_only.write_text(LOCK)
    drifted = tmp_path / "b.lock"
    drifted.write_text(LOCK.replace('"0.39.0"', '"0.40.0"')
                       + '\n[[package]]\nname = "pillow"\nversion = "12.0.0"\n')
    bumped = tmp_path / "c.lock"
    bumped.write_text(LOCK.replace('"2.13.0"', '"2.14.0"'))

    documents = {}
    for tag, lock in (("stack", stack_only), ("drift", drifted), ("bump", bumped)):
        out = tmp_path / f"{tag}.json"
        assert _derive("tiny_endpoint", config_only_tree, out,
                       "--lockfile", str(lock)) == 0
        documents[tag] = json.loads(out.read_bytes())

    assert documents["stack"]["graphs"]["stack"] == documents["drift"]["graphs"]["stack"]
    assert documents["stack"]["graphs"]["stack"] != documents["bump"]["graphs"]["stack"]
    assert ["diffusers", "0.39.0"] not in documents["stack"]["graphs"]["stack"]
    assert ["torch", "2.13.0"] in documents["stack"]["graphs"]["stack"]
    graphs = [
        [record["graph"] for lane in document["graphs"]["lanes"]
         for record in lane["graphs"]]
        for document in documents.values()
    ]
    assert graphs[0] == graphs[1] == graphs[2]


def test_a_derive_with_no_lockfile_refuses_instead_of_restating_the_env(
    config_only_tree: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from gen_worker.cli import main

    code = main([
        "release", "derive", "--dir", str(FIXTURES), "--module", "tiny_endpoint",
        "--checkpoint", str(config_only_tree), "--out", str(tmp_path / "x.json"),
    ])
    assert code == 1
    assert "pass `lockfile=`" in capsys.readouterr().err


def test_the_derive_cli_can_resolve_its_own_imports() -> None:
    from gen_worker.cli import release as release_cli

    source = Path(release_cli.__file__).read_text()
    assert "cli.run" not in source and "from .run import" not in source

    proc = subprocess.run(
        [sys.executable, "-c",
         "from gen_worker.discovery.discover import prime_sys_path;"
         "from gen_worker.release.derive import DeriveError, derive_release;"
         "print('ok')"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]


def test_priming_puts_src_ahead_of_root_and_repeats_cleanly(tmp_path: Path) -> None:
    """The ORDER is load-bearing and was nearly lost in the extraction."""
    from gen_worker.discovery.discover import prime_sys_path

    root = tmp_path / "endpoint"
    (root / "src").mkdir(parents=True)
    saved = list(sys.path)
    try:
        prime_sys_path(root)
        assert sys.path.index(str(root / "src")) < sys.path.index(str(root))
        before = list(sys.path)
        prime_sys_path(root)
        assert sys.path == before
    finally:
        sys.path[:] = saved


def test_priming_omits_a_src_dir_that_does_not_exist(tmp_path: Path) -> None:
    """A flat endpoint has no `src/`, and a path entry to a missing directory is a silent import-order hazard rather than a harmless no-op."""
    from gen_worker.discovery.discover import prime_sys_path

    root = tmp_path / "flat"
    root.mkdir()
    saved = list(sys.path)
    try:
        prime_sys_path(root)
        assert str(root) in sys.path
        assert str(root / "src") not in sys.path
    finally:
        sys.path[:] = saved


def test_the_blob_keeps_ONE_device_story_and_carries_no_weights(
    config_only_tree: Path, tmp_path: Path
) -> None:

    import json
    import zipfile

    out = tmp_path / "doc.json"
    cas = tmp_path / "graph-cas"
    assert _derive(
        "tiny_endpoint", config_only_tree, out, "--graph-cas", str(cas)
    ) == 0

    written = [
        path for path in cas.rglob("*")
        if path.is_file() and zipfile.is_zipfile(path)
    ]
    assert written, "the derive stored no exported-program blob to inspect"

    for blob in written:
        devices: set[str] = set()
        payload = 0
        with zipfile.ZipFile(blob) as archive:
            for name in archive.namelist():
                if name.endswith(
                    ("model_weights_config.json", "model_constants_config.json")
                ):
                    for entry in json.loads(archive.read(name))["config"].values():
                        devices.add(str(entry["tensor_meta"]["device"]["type"]))
                elif "/weights/" in name and not name.endswith(".json"):
                    payload += archive.getinfo(name).file_size
        assert "meta" not in devices, (
            f"{blob.name} records META-device tensors: the demotion is back, and "
            f"a meta state dict against a device-stamped graph is the pgw#1465 "
            f"two-device-stories failure AOTI refuses"
        )
        assert len(devices) <= 1, (
            f"{blob.name} tells more than one device story {sorted(devices)!r}; "
            f"AOTI reads the graph AND the state dict and refuses a mismatch"
        )
        assert payload == 0, (
            f"{blob.name} carries {payload} bytes of weight payload; a graph "
            f"artifact carries structure and the miner binds real weights"
        )


_BLOCKER = """
import sys
class _NoTopLevel:
    BLOCKED = {"torchcg", "tensorfs"}
    def find_module(self, name, path=None):
        return None
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in self.BLOCKED:
            raise ImportError(
                "top-level %s is not importable in a real release env" % name
            )
        return None
sys.meta_path.insert(0, _NoTopLevel())
"""


def test_derive_runs_in_a_release_env_with_no_top_level_torchcg_or_tensorfs(
    config_only_tree: Path, tmp_path: Path, lockfile: Path
) -> None:
    out = tmp_path / "bare.json"
    cas = tmp_path / "graph-cas"
    sitecustomize = tmp_path / "sitecustomize.py"
    sitecustomize.write_text(_BLOCKER)

    import os

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(tmp_path), *([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])]
    )
    completed = subprocess.run(
        [
            sys.executable, "-m", "gen_worker.cli", "release", "derive",
            "--dir", str(FIXTURES), "--module", "tiny_endpoint",
            "--checkpoint", str(config_only_tree),
            "--graph-cas", str(cas),
            "--lockfile", str(lockfile),
            "--out", str(out),
        ],
        capture_output=True, text=True, timeout=900, check=False, env=env,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]

    document = json.loads(out.read_bytes())
    graphs = [
        record["graph"]
        for lane in document["graphs"]["lanes"]
        for record in lane["graphs"]
    ]
    assert graphs, document["graphs"]
    assert all(g.startswith("cg-graph-v1-") for g in graphs), graphs

    assert not any(
        "program" in record
        for lane in document["graphs"]["lanes"]
        for record in lane["graphs"]
    ), document["graphs"]

    from gen_worker._vendor.tensorfs import LocalCAS
    from gen_worker.graphs.store import LocalGraphStore

    assert cas.is_dir() and any(cas.rglob("*")), "the graph CAS holds no blob"
    store = LocalGraphStore(LocalCAS(cas))
    missing = [g for g in graphs if not store.has_program(g)]
    assert not missing, f"the CAS holds no program for {missing}"


def test_the_blocker_itself_can_go_red(config_only_tree: Path, tmp_path: Path) -> None:
    """The negative control: with the blocker armed, the OLD import fails."""
    import os

    sitecustomize = tmp_path / "sitecustomize.py"
    sitecustomize.write_text(_BLOCKER)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(tmp_path), *([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])]
    )
    proc = subprocess.run(
        [sys.executable, "-c", "import torchcg"],
        capture_output=True, text=True, env=env, check=False,
    )
    assert proc.returncode != 0 and "not importable in a real release env" in proc.stderr


def test_the_default_lockfile_is_the_one_BESIDE_the_endpoint(
    config_only_tree: Path, tmp_path: Path
) -> None:
    """The production shape: no `--lockfile`, and the lock is found anyway."""

    endpoint = tmp_path / "app"
    endpoint.mkdir()
    for module in FIXTURES.glob("*.py"):
        (endpoint / module.name).symlink_to(module)
    (endpoint / "uv.lock").write_text(LOCK)

    from gen_worker.cli import main

    out = tmp_path / "beside.json"
    assert main([
        "release", "derive", "--dir", str(endpoint), "--module", "tiny_endpoint",
        "--checkpoint", str(config_only_tree), "--out", str(out),
    ]) == 0
    stack = json.loads(out.read_bytes())["graphs"]["stack"]
    assert ["torch", "2.13.0"] in stack
