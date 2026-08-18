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
    # 2 Size values x {CFG batch-2 generate, batch-1 turbo} = 4 graph classes.
    assert len(lane["graphs"]) == 4
    assert {record["target"] for record in lane["graphs"]} == {"unet"}
    batches = sorted(
        record["ingress"]["inputs"][0]["shape"][0] for record in lane["graphs"]
    )
    assert batches == [1, 1, 2, 2]

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

    from gen_worker.api.model_base import Adapter
    from gen_worker.models import SDXL
    from gen_worker.release.derive import _defaults_variants, _fake_adapter

    variants = _defaults_variants(SDXL)
    assert [variant.cfg for variant in variants] == [True, False]

    fake = _fake_adapter(SDXL)
    assert isinstance(fake, Adapter)
    assert fake.defaults is not None and fake.defaults.cfg is False
    assert fake.defaults.steps.default == 4

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
