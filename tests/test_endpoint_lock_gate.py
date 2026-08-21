"""`gen-worker lock` as a COMMITTED SOURCE artifact: deterministic, checkable."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

MAIN = '''
from __future__ import annotations

import msgspec

from gen_worker import LoadContext, Model, RequestContext, entrypoint, lane
from gen_worker.demand import GiB, const

#: pgw#1621: a lane is the `(topology, quant)` STAMP PAIR, both halves
#: ratified in the vendored `spec/v2` corpus. `contracts.SDXL_DIFFUSERS_BF16`
#: is deleted with the v1 vocabulary.
SDXL_BF16 = ("sdxl.diffusers@1", "plain.bf16@1")


class Unlaned(msgspec.Struct):
    """The fixture's model type. It borrows nothing — the class header below
    names the contract, which is the only place a lane can come from."""

    steps: int = 4


class In(msgspec.Struct):
    text: str
{extra_field}

class Out(msgspec.Struct):
    echoed: str


class UnlanedModel(
    Model[Unlaned],
    lanes={SDXL_BF16: lane(request=const(GiB(1)))},
    self_loading="a lock/--check gate fixture: there is no pipeline at all",
):
    """A real lane, no ctx.compile mark — and it locks."""

    def load(self, ctx: LoadContext[Unlaned]) -> None:
        self.ready = True


@entrypoint
def analyze(ctx: RequestContext, payload: In, model: UnlanedModel) -> Out:
    return Out(echoed=payload.text)
'''


def _write_endpoint(root: Path, *, extra_field: str = "") -> None:
    package = root / "src" / "lockcheck_fixture"
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    # `.replace`, not `.format`: the source now contains real dict braces
    # (`lanes={SDXL_BF16: lane(…)}`) and `str.format` would read them as
    # fields.
    (package / "main.py").write_text(
        MAIN.replace("{extra_field}", extra_field), encoding="utf-8"
    )
    (root / "endpoint.toml").write_text(
        'schema_version = 1\nmain = "lockcheck_fixture.main"\n', encoding="utf-8"
    )
    (root / "pyproject.toml").write_text(
        '[project]\nname = "lockcheck-fixture"\nversion = "0"\n\n'
        '[tool.gen_worker]\nmain = "lockcheck_fixture.main"\n',
        encoding="utf-8",
    )
    (root / "uv.lock").write_text(
        'version = 1\n\n[[package]]\nname = "torch"\nversion = "2.13.0"\n'
        '\n[[package]]\nname = "triton"\nversion = "3.7.1"\n',
        encoding="utf-8",
    )


@pytest.fixture()
def endpoint(tmp_path: Path) -> Path:
    root = tmp_path / "endpoint"
    _write_endpoint(root)
    return root


@pytest.fixture()
def checkpoint(tmp_path: Path) -> Path:
    tree = tmp_path / "checkpoint"
    tree.mkdir()
    (tree / "config.json").write_text(
        json.dumps({"torch_dtype": "bfloat16"}), encoding="utf-8"
    )
    return tree


def _lock(
    endpoint: Path, checkpoint: Path, graph_cas: Path, *extra: str
) -> tuple[int, dict]:

    completed = subprocess.run(
        [sys.executable, "-m", "gen_worker.cli", "lock", str(endpoint),
         "--checkpoint", str(checkpoint), "--graph-cas", str(graph_cas), *extra],
        capture_output=True, text=True, check=False,
    )
    summary = json.loads(completed.stdout) if completed.stdout.strip() else {}
    return completed.returncode, summary


def test_lock_is_deterministic_and_check_gates_on_the_document(
    endpoint: Path,
    checkpoint: Path,
    tmp_path: Path,
) -> None:
    graph_cas = tmp_path / "graph-cas"
    lock = endpoint / "endpoint.lock"

    code, summary = _lock(endpoint, checkpoint, graph_cas)
    assert code == 0
    first = lock.read_bytes()
    assert summary["derive"] == "traced"
    assert summary["posture"] == "traced-no-compile-targets"
    assert summary["specializations"] == 0

    code, summary = _lock(endpoint, checkpoint, graph_cas)
    assert code == 0
    assert lock.read_bytes() == first
    assert summary["derive"] == "reused"

    code, summary = _lock(endpoint, checkpoint, graph_cas, "--check")
    assert code == 0
    assert summary["check"] == "current"
    assert lock.read_bytes() == first

    source = endpoint / "src" / "lockcheck_fixture" / "main.py"
    source.write_text(
        source.read_text(encoding="utf-8") + "\n# a comment moves the inputs\n",
        encoding="utf-8",
    )
    code, result = _lock(endpoint, checkpoint, graph_cas, "--check")
    assert code == 0
    assert result["check"] == "current"
    assert result["committed_document_digest"] == result["derived_document_digest"]
    assert lock.read_bytes() == first


def test_check_goes_red_on_a_stale_committed_lock(
    endpoint: Path,
    checkpoint: Path,
    tmp_path: Path,
) -> None:

    graph_cas = tmp_path / "graph-cas"
    lock = endpoint / "endpoint.lock"
    assert _lock(endpoint, checkpoint, graph_cas)[0] == 0
    before = lock.read_bytes()

    _write_endpoint(endpoint, extra_field="    drifted: bool = False\n")

    code, result = _lock(endpoint, checkpoint, graph_cas, "--check")
    assert code == 1
    assert result["check"] == "drift"
    assert result["committed_document_digest"] != result["derived_document_digest"]
    assert lock.read_bytes() == before


def test_check_refuses_when_there_is_no_lock_to_check(
    endpoint: Path, checkpoint: Path, tmp_path: Path
) -> None:
    assert _lock(endpoint, checkpoint, tmp_path / "graph-cas", "--check")[0] == 2


def _safetensors(path: Path, dtypes: dict[str, int]) -> None:

    import struct

    header: dict[str, object] = {}
    offset = 0
    for spelling, count in dtypes.items():
        for index in range(count):
            header[f"w{spelling}{index}"] = {
                "dtype": spelling, "shape": [1], "data_offsets": [offset, offset + 2],
            }
            offset += 2
    blob = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\0" * offset)


def test_a_derived_lane_takes_its_dtype_from_the_checkpoint(tmp_path: Path) -> None:
    import torch

    from gen_worker.serving.checkpoint_dtype import checkpoint_dtype

    assert checkpoint_dtype(None) is None
    assert checkpoint_dtype(tmp_path / "nowhere") is None

    tree = tmp_path / "raw"
    tree.mkdir()
    _safetensors(tree / "a-encoder.safetensors", {"F32": 2})
    _safetensors(tree / "b-denoiser.safetensors", {"BF16": 9})
    assert checkpoint_dtype(tree) is torch.bfloat16

    (tree / "model_index.json").write_text(
        json.dumps({"_class_name": "X", "torch_dtype": "float16"}), encoding="utf-8"
    )
    assert checkpoint_dtype(tree) is torch.float16

    empty = tmp_path / "empty"
    empty.mkdir()
    assert checkpoint_dtype(empty) is None
