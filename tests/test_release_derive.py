"""pgw#1370: `gen-worker release derive` over real author-shaped code.

Integration through the actual CLI codepath: a main_v2-shaped endpoint
(stock diffusers pipeline, lanes-only surface) against a CONFIG-ONLY
checkpoint tree, on CPU, no weights, no GPU. Coverage is auto-enumerated
from the payload schemas; the document is byte-reproducible across a
subprocess fence; every refusal names its cause.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("transformers")
# se#786/pgw#1462: the derive imports the VENDORED torchcg (the miner's copy),
# which ships in this wheel — so this is a hard import, never a skip.
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


#: A derive STATES the compile stack it traced under, and reads it from the
#: endpoint's uv.lock (pgw#1489) — so every derive here states one. Restating
#: the installed set instead is exactly what that issue deleted.
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
    assert lane["contract"] == "tiny.diffusers-fp32@1"
    assert lane["unobserved_targets"] == []
    # 2 Size values x {CFG batch-2 generate, batch-1 turbo} = 4 graph specializations.
    assert len(lane["graphs"]) == 4
    assert {record["target"] for record in lane["graphs"]} == {"unet"}
    batches = sorted(
        record["ingress"]["inputs"][0]["shape"][0] for record in lane["graphs"]
    )
    assert batches == [1, 1, 2, 2]

    # pgw#1384: the DEFAULT-parameter class LEADS the document -- the serving
    # hole list inherits document order and the miner mints in it, so the
    # class an all-defaults request needs is the first one published.
    # GenerateInput defaults size=LARGE (64px; the tiny VAE has ONE block, so
    # the latent equals the pixel size) under the platform (cfg-on) row: the
    # first row is the batch-2 latent-64 class even though SMALL precedes
    # LARGE in enum declaration order.
    assert lane["graphs"][0]["ingress"]["inputs"][0]["shape"] == [2, 4, 64, 64]

    # The exported defaults schema: the successor of the hub-embedded
    # per-family defaults registry (one schema per release, endpoint-owned).
    schema = document["checkpoint_defaults_schema"]
    assert schema is not None
    rendered = json.dumps(schema)
    assert "cfg" in rendered and "steps" in rendered and "guidance" in rendered


def test_document_is_byte_stable_across_a_subprocess_fence(
    config_only_tree: Path, tmp_path: Path, lockfile: Path
) -> None:
    """Two FRESH interpreters derive identical bytes.

    The production artifact is what the CLI process emits, so the fence
    compares subprocess against subprocess. (In-process determinism is
    covered separately below; comparing in-process against subprocess would
    make this test assert that every OTHER test sharing this interpreter
    left torch's global state untouched, which is not this test's claim.)
    """

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


def test_a_no_lane_endpoint_stamps_the_explicit_eager_marker(
    config_only_tree: Path, tmp_path: Path
) -> None:
    out = tmp_path / "eager.json"
    assert _derive("eager_endpoint", config_only_tree, out) == 0
    document = json.loads(out.read_bytes())
    assert document["graphs"]["lanes"] == []
    assert document["checkpoint_defaults_schema"] is None


def test_a_contractless_endpoint_traces_under_a_derived_lane(
    config_only_tree: Path, tmp_path: Path
) -> None:
    """pgw#1488 fix (1) + (3): no contract, no `lanes=`, and it still traces.

    The twin fixture is `tiny_endpoint` with its `lanes=` emptied and nothing
    else changed, so this is a controlled measurement rather than an
    assertion about a second endpoint: SAME graphs, DIFFERENT lane name.
    A contract handle is a name, not a key — which is the whole argument for
    letting a contract be optional without rekeying anything that exists.
    """

    declared = tmp_path / "declared.json"
    derived = tmp_path / "derived.json"
    assert _derive("tiny_endpoint", config_only_tree, declared) == 0
    assert _derive("derived_twin_endpoint", config_only_tree, derived) == 0

    (contract_lane,) = json.loads(declared.read_bytes())["graphs"]["lanes"]
    document = json.loads(derived.read_bytes())
    (derived_lane,) = document["graphs"]["lanes"]

    assert contract_lane["contract"] == "tiny.diffusers-fp32@1"
    assert derived_lane["contract"] == "derived.sdxl@1"
    assert [record["graph"] for record in derived_lane["graphs"]] == [
        record["graph"] for record in contract_lane["graphs"]
    ]
    assert len(derived_lane["graphs"]) == 4

    # The lane row says WHICH KIND of identity it carries. `document: null` is
    # a bug on a declared contract (pgw#1391) and the honest state on a derived
    # one, and a reader cannot tell those apart from a null alone.
    row = document["lane_contracts"]["derived.sdxl@1"]
    assert row == {"stamp": "derived.sdxl@1", "document": None, "digest": "",
                   "derived": True}


def test_an_unmarked_endpoint_traces_and_reports_nothing_to_compile(
    config_only_tree: Path, tmp_path: Path
) -> None:
    """The middle state: traced, and the author marked no compile target.

    Before pgw#1488 this endpoint could not be derived at all — its model type
    has no canonical contract, so the derive refused the class by name, and the
    remedy that refusal named (`lanes=()`) silently disabled compilation. Now
    `load` RUNS and zero graphs is a measurement.
    """

    out = tmp_path / "unmarked.json"
    assert _derive("unmarked_endpoint", config_only_tree, out) == 0
    document = json.loads(out.read_bytes())
    assert document["graphs"]["lanes"] == []
    assert document["lane_contracts"] == {}
    # It is NOT weightless and NOT eager-by-declaration: the entrypoint's
    # envelope is published exactly as a compiled endpoint's is.
    assert set(document["entrypoints"]) == {"analyze"}
    assert document["entrypoints"]["analyze"]["model_slots"] == {
        "model": "UnmarkedModel"
    }


def test_binding_enumeration_reaches_adapter_and_cfg_arms(
    config_only_tree: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """is_trace is DELETED: arm coverage is the derive's binding enumeration.

    The fixture's guidance-free batch-1 arm is reachable two ways -- a fake
    riding adapter, and the cfg-flipped defaults variant -- and the
    impossible combination (adapter stacked on a distilled checkpoint) is
    REFUSED by the author's own ValidationError and skipped, counted.
    """

    from gen_worker import Adapter, DistillationAdapter
    from gen_worker.models import SDXL
    from gen_worker.release.derive import _defaults_variants, _fake_adapter

    variants = _defaults_variants(SDXL)
    assert [variant.cfg for variant in variants] == [True, False]

    fake = _fake_adapter(SDXL)
    assert isinstance(fake, Adapter)
    assert fake.defaults is not None and fake.defaults.cfg is False
    assert fake.defaults.steps.default == 4
    # A distillation SLOT enumerates its own KIND (the typed-takeover guard).
    marked = _fake_adapter(SDXL, DistillationAdapter)
    assert isinstance(marked, DistillationAdapter)

    out = tmp_path / "release.json"
    assert _derive("tiny_endpoint", config_only_tree, out) == 0
    stderr = capsys.readouterr().err
    assert "refused by the author's own validation" in stderr


def test_the_compile_stack_is_the_env_identity_and_only_the_compiler_is_in_it(
    config_only_tree: Path, tmp_path: Path
) -> None:
    """pgw#1489: the derive stamps the lock's COMPILE STACK, not its closure.

    Two lockfiles differing only in a package no compiler can see stamp the
    SAME env; a torch bump stamps a different one. Env identity never leaks
    into graph identity either way.
    """

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
    # Env identity never leaks into GRAPH identity: same graphs every time.
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


# --- the sys.path priming the derive shares with discovery -------------------


def test_the_derive_cli_can_resolve_its_own_imports() -> None:
    """pgw#1440 REGRESSION. `_run_derive` imported `_ensure_sys_path` from
    `cli/run.py`, which pgw#1373 (`cd46c957`) deleted with the v1 SDK, so
    `gen-worker release derive` raised `ModuleNotFoundError` on the second
    line of its handler — every derive, on every tree.

    Asserted by RESOLVING the handler's imports rather than by grepping for
    the module: the failure was an import that could not be satisfied, so the
    check has to be one that a second dead import would also fail.
    """
    from gen_worker.cli import release as release_cli

    source = Path(release_cli.__file__).read_text()
    assert "cli.run" not in source and "from .run import" not in source

    # The real gate: run the handler's own import block in a clean process.
    proc = subprocess.run(
        [sys.executable, "-c",
         "from gen_worker.discovery.discover import prime_sys_path;"
         "from gen_worker.release.derive import DeriveError, derive_release;"
         "print('ok')"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]


def test_priming_puts_src_ahead_of_root_and_repeats_cleanly(tmp_path: Path) -> None:
    """The ORDER is load-bearing and was nearly lost in the extraction.

    Both inserts go to position 0, so writing them root-then-src yields
    `src` ahead of `root` — which is what a `src/`-layout endpoint needs.
    Swapping the two statements reverses the precedence silently, and the
    only thing that catches it is an assertion on the resulting order.
    """
    from gen_worker.discovery.discover import prime_sys_path

    root = tmp_path / "endpoint"
    (root / "src").mkdir(parents=True)
    saved = list(sys.path)
    try:
        prime_sys_path(root)
        assert sys.path.index(str(root / "src")) < sys.path.index(str(root))
        before = list(sys.path)
        prime_sys_path(root)  # idempotent: no duplicate entries
        assert sys.path == before
    finally:
        sys.path[:] = saved


def test_priming_omits_a_src_dir_that_does_not_exist(tmp_path: Path) -> None:
    """A flat endpoint has no `src/`, and a path entry to a missing directory
    is a silent import-order hazard rather than a harmless no-op."""
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
    """pgw#1465: the graph's device and the state dict's device must AGREE.

    `_demote_fakes_to_meta` rewrote every fake tensor to META. Its rationale
    was real -- a phantom storage must not make the archive claim bytes it
    does not have -- but the cure destroyed the device, and pgw#1458 made the
    device load-bearing. One blob then told TWO device stories (1,922 graph
    node metas on cuda:0 against 686 state-dict entries on meta), AOTI reads
    both, and every sd1.5 class died on `FakeTensorDeviceMismatchError`.

    So this reads the blob's OWN recorded devices -- the same archive metadata
    a CPU-side check can read without a GPU -- and asserts they are the single
    device the trace ran on, plus that no real weight rode along.
    """

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


# se#786 / pgw#1462 / th#2192 — THE BARE RELEASE ENV, AND THE PROGRAM BLOB.
#
# Measured 2026-08-19 in `serverless-endpoints/sdxl`'s own venv (the release
# env, built from its uv.lock): `release derive` refused with
# "torchcg is a release dependency there ... gen-worker deliberately does not
# bundle it", and NO endpoint in the fleet pins torchcg. Every other production
# site — `serving/mint_child.py`, `serving/host.py`, `serving/hub_store.py` —
# imports `gen_worker._vendor.torchcg`, so an endpoint-pinned torchcg would let
# the publish-time TRACE and the serving-time MINT run different compilers.
#
# The blocker below is the whole test: this env HAS both distributions
# installed as dev dependencies, so without it the assertion would pass on a
# derive that never touched the vendored copy.

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

    # th#2192's consequence, asserted on the OUTPUT: an empty `program` is the
    # miner's `MissingProgramDigest`, so a derive that stores no blob is not a
    # mintable derive. `--graph-cas` is the only thing that fills it.
    document = json.loads(out.read_bytes())
    programs = [
        record["program"]
        for lane in document["graphs"]["lanes"]
        for record in lane["graphs"]
    ]
    assert programs and all(p.strip() for p in programs), document["graphs"]
    assert cas.is_dir() and any(cas.rglob("*")), "the graph CAS holds no blob"


def test_the_blocker_itself_can_go_red(config_only_tree: Path, tmp_path: Path) -> None:
    """The negative control: with the blocker armed, the OLD import fails.

    Without this, a blocker that silently stopped blocking would make the test
    above pass for the wrong reason — the exact fixture-instead-of-emitter hole
    that kept th#2192 green.
    """
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
    """The production shape: no `--lockfile`, and the lock is found anyway.

    tensorhub's builder runs `gen-worker release derive --module M --checkpoint
    /derive/checkpoint` with no lockfile flag, in an image whose `WORKDIR /app`
    holds `pyproject.toml` and `uv.lock` (docs/dockerfile.md). `--dir` defaults
    to that cwd, so the endpoint's own lock is beside it — which is the ONLY
    reason a required lockfile does not break the fleet's derive path. This
    reproduces that layout with real files rather than trusting the Dockerfile.
    """

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
