from __future__ import annotations

import threading
import time
from pathlib import Path

import gen_worker.presigned_upload as presigned_upload_module
from gen_worker.presigned_upload import _upload_parts_to_s3
from gen_worker.request_context._concurrent_upload import optimal_file_concurrency


def test_optimal_file_concurrency_is_fixed() -> None:
    assert optimal_file_concurrency(0) == 1
    assert optimal_file_concurrency(1) == 1
    assert optimal_file_concurrency(2) == 2
    assert optimal_file_concurrency(4) == 4
    assert optimal_file_concurrency(32) == 4


def test_presigned_part_uploads_obey_global_put_budget(monkeypatch, tmp_path: Path) -> None:
    src = tmp_path / "parts.bin"
    src.write_bytes(b"abcdef")

    monkeypatch.setattr(
        presigned_upload_module,
        "_presigned_put_slots",
        threading.BoundedSemaphore(2),
    )

    active = 0
    max_active = 0
    lock = threading.Lock()
    release = threading.Event()

    def fake_upload_part_to_presigned_url(**kwargs) -> str:
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
            if max_active == 2:
                release.set()
        release.wait(timeout=1.0)
        time.sleep(0.01)
        with lock:
            active -= 1
        return f"etag-{kwargs['offset']}"

    monkeypatch.setattr(
        presigned_upload_module,
        "upload_part_to_presigned_url",
        fake_upload_part_to_presigned_url,
    )

    progress: list[tuple[int, int, int]] = []
    etags = _upload_parts_to_s3(
        file_path=str(src),
        part_urls=[f"https://example.test/part-{i}" for i in range(6)],
        part_size=1,
        total_parts=6,
        on_progress=lambda done, total, bytes_up: progress.append((done, total, bytes_up)),
        cancel_check=None,
    )

    assert max_active == 2
    assert len(etags) == 6
    assert [part_number for part_number, _etag in etags] == [1, 2, 3, 4, 5, 6]
    assert progress[-1] == (6, 6, 6)
