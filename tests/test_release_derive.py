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
pytest.importorskip("torchcg")

FIXTURES = Path(__file__).resolve().parent / "release_fixtures"


@pytest.fixture(scope="module")
def config_only_tree(tmp_path_factory: pytest.TempPathFactory) -> Path:
    sys.path.insert(0, str(FIXTURES))
    try:
        import tiny_tree
    finally:
        sys.path.remove(str(FIXTURES))
    return tiny_tree.save_config_only(tmp_path_factory.mktemp("tiny-config-only"))


def _derive(module: str, tree: Path, out: Path, *extra: str) -> int:
    from gen_worker.cli import main

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
    config_only_tree: Path, tmp_path: Path
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


def test_lockfile_closure_is_the_env_identity(
    config_only_tree: Path, tmp_path: Path
) -> None:
    lockfile = tmp_path / "uv.lock"
    lockfile.write_text(
        'version = 1\n\n[[package]]\nname = "torch"\nversion = "2.13.0"\n'
        '\n[[package]]\nname = "diffusers"\nversion = "0.39.0"\n'
    )
    out_locked = tmp_path / "locked.json"
    out_installed = tmp_path / "installed.json"
    assert _derive(
        "tiny_endpoint", config_only_tree, out_locked, "--lockfile", str(lockfile)
    ) == 0
    assert _derive("tiny_endpoint", config_only_tree, out_installed) == 0
    locked = json.loads(out_locked.read_bytes())
    installed = json.loads(out_installed.read_bytes())
    assert locked["graphs"]["closure"] != installed["graphs"]["closure"]
    # Env identity never leaks into GRAPH identity: same graphs either way.
    assert [record["graph"] for lane in locked["graphs"]["lanes"] for record in lane["graphs"]] == [
        record["graph"] for lane in installed["graphs"]["lanes"] for record in lane["graphs"]
    ]


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
