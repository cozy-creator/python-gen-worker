from __future__ import annotations

import json
import socket
import threading
import time
from pathlib import Path

import msgspec
import requests


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    _, port = s.getsockname()
    s.close()
    return int(port)


def _wait_ready(base: str) -> None:
    for _ in range(50):
        try:
            r = requests.get(base + "/v1/status", timeout=0.25)
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(0.05)
    raise AssertionError("dev http runner did not become ready")


def test_dev_http_runner_writes_outputs(tmp_path: Path, monkeypatch) -> None:
    # Create a minimal tenant module.
    pkg_dir = tmp_path / "tenant"
    pkg_dir.mkdir(parents=True, exist_ok=True)

    (pkg_dir / "tenant_mod_auto.py").write_text(
        """
import msgspec
from gen_worker.api.decorators import worker_function
from gen_worker.worker import RequestContext

class In(msgspec.Struct):
    prompt: str

class Out(msgspec.Struct):
    ref: str

@worker_function()
def generate(ctx: RequestContext, payload: In) -> Out:
    ref = f"jobs/{ctx.request_id}/outputs/out.txt"
    ctx.save_bytes(ref, (payload.prompt + "\\n").encode("utf-8"))
    return Out(ref=ref)
""".lstrip(),
        encoding="utf-8",
    )

    # Fake baked manifest pointing at that module.
    man = {
        "endpoint_name": "t",
        "functions": [
            {
                "name": "generate",
                "python_name": "generate",
                "module": "tenant_mod_auto",
                "required_models": [],
                "payload_repo_selectors": [],
            }
        ],
        "models_by_function": {"generate": {}},
    }
    manifest_path = tmp_path / "endpoint.lock"
    manifest_path.write_text(msgspec.toml.encode(man).decode("utf-8"), encoding="utf-8")

    outputs = tmp_path / "out"
    outputs.mkdir(parents=True, exist_ok=True)

    port = _free_port()
    monkeypatch.setenv("GEN_WORKER_HTTP_LISTEN", f"127.0.0.1:{port}")
    monkeypatch.setenv("GEN_WORKER_MANIFEST_PATH", str(manifest_path))
    monkeypatch.setenv("GEN_WORKER_ENDPOINT_ROOT", str(pkg_dir))
    monkeypatch.setenv("GEN_WORKER_OUTPUT_DIR", str(outputs))

    # Start server in background thread.
    def run_server() -> None:
        from gen_worker.testing.http_runner import main

        main([])

    t = threading.Thread(target=run_server, daemon=True)
    t.start()

    base = f"http://127.0.0.1:{port}"
    _wait_ready(base)

    r = requests.post(
        base + "/v1/request/generate",
        json={"payload": {"prompt": "hello"}},
        timeout=10,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["success"] is True
    request_id = body["request_id"]
    # The output file should exist on disk.
    p = outputs / "jobs" / request_id / "outputs" / "out.txt"
    assert p.exists()
    assert p.read_text(encoding="utf-8").strip() == "hello"


def test_dev_http_runner_auto_uploads_returned_local_asset(tmp_path: Path, monkeypatch) -> None:
    pkg_dir = tmp_path / "tenant"
    pkg_dir.mkdir(parents=True, exist_ok=True)

    (pkg_dir / "tenant_mod.py").write_text(
        """
from pathlib import Path

import msgspec
from gen_worker.api.decorators import worker_function
from gen_worker.api.types import Asset
from gen_worker.worker import RequestContext

class In(msgspec.Struct):
    payload: str

class Out(msgspec.Struct):
    artifact: Asset

@worker_function()
def convert_local(ctx: RequestContext, payload: In) -> Out:
    local_file = Path(f"/tmp/{ctx.request_id}-converted.bin")
    local_file.write_bytes((payload.payload + "\\n").encode("utf-8"))
    return Out(artifact=Asset(ref="", local_path=str(local_file)))
""".lstrip(),
        encoding="utf-8",
    )

    man = {
        "endpoint_name": "t",
        "functions": [
            {
                "name": "convert-local",
                "python_name": "convert_local",
                "module": "tenant_mod",
                "required_models": [],
                "payload_repo_selectors": [],
            }
        ],
        "models_by_function": {"convert-local": {}},
    }
    manifest_path = tmp_path / "endpoint.lock"
    manifest_path.write_text(msgspec.toml.encode(man).decode("utf-8"), encoding="utf-8")

    outputs = tmp_path / "out"
    outputs.mkdir(parents=True, exist_ok=True)

    port = _free_port()
    monkeypatch.setenv("GEN_WORKER_HTTP_LISTEN", f"127.0.0.1:{port}")
    monkeypatch.setenv("GEN_WORKER_MANIFEST_PATH", str(manifest_path))
    monkeypatch.setenv("GEN_WORKER_ENDPOINT_ROOT", str(pkg_dir))
    monkeypatch.setenv("GEN_WORKER_OUTPUT_DIR", str(outputs))

    def run_server() -> None:
        from gen_worker.testing.http_runner import main

        main([])

    t = threading.Thread(target=run_server, daemon=True)
    t.start()

    base = f"http://127.0.0.1:{port}"
    _wait_ready(base)

    r = requests.post(
        base + "/v1/request/convert-local",
        json={"payload": {"payload": "weights"}},
        timeout=10,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["success"] is True
    request_id = body["request_id"]

    out_asset = body["output"]["artifact"]
    out_ref = str(out_asset["ref"])
    assert out_ref.startswith(f"jobs/{request_id}/outputs/auto/")
    assert out_ref.endswith(".bin")

    persisted = outputs / out_ref
    assert persisted.exists()
    assert persisted.read_text(encoding="utf-8").strip() == "weights"


def test_dev_http_runner_auto_uploads_returned_local_tensors(tmp_path: Path, monkeypatch) -> None:
    pkg_dir = tmp_path / "tenant"
    pkg_dir.mkdir(parents=True, exist_ok=True)

    (pkg_dir / "tenant_mod_tensors.py").write_text(
        """
from pathlib import Path

import msgspec
from gen_worker.api.decorators import worker_function
from gen_worker.api.types import Tensors
from gen_worker.worker import RequestContext

class In(msgspec.Struct):
    payload: str

class Out(msgspec.Struct):
    weights: Tensors

@worker_function()
def convert_local(ctx: RequestContext, payload: In) -> Out:
    local_file = Path(f"/tmp/{ctx.request_id}-converted.safetensors")
    local_file.write_bytes((payload.payload + "\\n").encode("utf-8"))
    return Out(weights=Tensors(ref="", local_path=str(local_file), format="safetensors"))
""".lstrip(),
        encoding="utf-8",
    )

    man = {
        "endpoint_name": "t",
        "functions": [
            {
                "name": "convert-local",
                "python_name": "convert_local",
                "module": "tenant_mod_tensors",
                "required_models": [],
                "payload_repo_selectors": [],
            }
        ],
        "models_by_function": {"convert-local": {}},
    }
    manifest_path = tmp_path / "endpoint.lock"
    manifest_path.write_text(msgspec.toml.encode(man).decode("utf-8"), encoding="utf-8")

    outputs = tmp_path / "out"
    outputs.mkdir(parents=True, exist_ok=True)

    port = _free_port()
    monkeypatch.setenv("GEN_WORKER_HTTP_LISTEN", f"127.0.0.1:{port}")
    monkeypatch.setenv("GEN_WORKER_MANIFEST_PATH", str(manifest_path))
    monkeypatch.setenv("GEN_WORKER_ENDPOINT_ROOT", str(pkg_dir))
    monkeypatch.setenv("GEN_WORKER_OUTPUT_DIR", str(outputs))

    def run_server() -> None:
        from gen_worker.testing.http_runner import main

        main([])

    t = threading.Thread(target=run_server, daemon=True)
    t.start()

    base = f"http://127.0.0.1:{port}"
    _wait_ready(base)

    r = requests.post(
        base + "/v1/request/convert-local",
        json={"payload": {"payload": "weights"}},
        timeout=10,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["success"] is True
    request_id = body["request_id"]

    out_tensors = body["output"]["weights"]
    out_ref = str(out_tensors["ref"])
    assert out_ref.startswith(f"jobs/{request_id}/outputs/auto/")
    assert out_ref.endswith(".safetensors")
    assert out_tensors["format"] == "safetensors"

    persisted = outputs / out_ref
    assert persisted.exists()
    assert persisted.read_text(encoding="utf-8").strip() == "weights"
