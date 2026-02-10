from __future__ import annotations

from pathlib import Path

from gen_worker.worker import ActionContext


def test_action_context_local_output_backend(tmp_path: Path) -> None:
    ctx = ActionContext(
        "rid1",
        local_output_dir=str(tmp_path),
        owner="o1",
        user_id="u1",
    )
    ref = "runs/rid1/outputs/hello.bin"
    asset = ctx.save_bytes(ref, b"abc")
    assert asset.ref == ref
    assert asset.local_path is not None
    p = Path(asset.local_path)
    assert p.exists()
    assert p.read_bytes() == b"abc"

