from __future__ import annotations

import json
import socket
import threading
import time
from pathlib import Path

import requests


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    _, port = s.getsockname()
    s.close()
    return int(port)


def test_dev_http_runner_writes_outputs(tmp_path: Path, monkeypatch) -> None:
    # Create a minimal tenant module.
    pkg_dir = tmp_path / "tenant"
    pkg_dir.mkdir(parents=True, exist_ok=True)

    (pkg_dir / "tenant_mod.py").write_text(
        """
import msgspec
from gen_worker.decorators import worker_function
from gen_worker.worker import ActionContext

class In(msgspec.Struct):
    prompt: str

class Out(msgspec.Struct):
    ref: str

@worker_function()
def generate(ctx: ActionContext, payload: In) -> Out:
    ref = f"runs/{ctx.run_id}/outputs/out.txt"
    ctx.save_bytes(ref, (payload.prompt + "\\n").encode("utf-8"))
    return Out(ref=ref)
""".lstrip(),
        encoding="utf-8",
    )

    # Fake baked manifest pointing at that module.
    man = {
        "project_name": "t",
        "functions": [
            {
                "name": "generate",
                "endpoint_name": "generate",
                "module": "tenant_mod",
                "required_models": [],
                "payload_repo_selectors": [],
            }
        ],
        "models_by_function": {"generate": {}},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(man), encoding="utf-8")

    outputs = tmp_path / "out"
    outputs.mkdir(parents=True, exist_ok=True)

    port = _free_port()
    monkeypatch.setenv("GEN_WORKER_HTTP_LISTEN", f"127.0.0.1:{port}")
    monkeypatch.setenv("GEN_WORKER_MANIFEST_PATH", str(manifest_path))
    monkeypatch.setenv("GEN_WORKER_PROJECT_ROOT", str(pkg_dir))
    monkeypatch.setenv("GEN_WORKER_OUTPUT_DIR", str(outputs))

    # Start server in background thread.
    def run_server() -> None:
        from gen_worker.testing.http_runner import main

        main([])

    t = threading.Thread(target=run_server, daemon=True)
    t.start()

    # Wait for it to be reachable.
    base = f"http://127.0.0.1:{port}"
    for _ in range(50):
        try:
            r = requests.get(base + "/v1/status", timeout=0.25)
            if r.status_code == 200:
                break
        except Exception:
            time.sleep(0.05)

    r = requests.post(
        base + "/v1/run/generate",
        json={"payload": {"prompt": "hello"}},
        timeout=10,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["success"] is True
    run_id = body["run_id"]
    # The output file should exist on disk.
    p = outputs / "runs" / run_id / "outputs" / "out.txt"
    assert p.exists()
    assert p.read_text(encoding="utf-8").strip() == "hello"
