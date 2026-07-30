"""pgw#738 harness: a non-reentrant class whose handler saves a blob
mid-handler — the exact shape that deadlocked 62922680/d0cbf910 (save_bytes
yields the GPU permit while holding the instance run_lock; a second job
parked on the permit grabbed it and blocked on the run_lock forever)."""

from __future__ import annotations

import threading

import msgspec

from gen_worker import RequestContext, endpoint

UPLOADER_STARTED = threading.Event()
UPLOAD_GATE = threading.Event()

CALLS: list[str] = []


def reset() -> None:
    CALLS.clear()
    UPLOADER_STARTED.clear()
    UPLOAD_GATE.clear()


class UploadIn(msgspec.Struct):
    text: str = ""


class UploadOut(msgspec.Struct):
    response: str


@endpoint
class UploadJobs:
    def uploader(self, ctx: RequestContext, data: UploadIn) -> UploadOut:
        CALLS.append("uploader")
        UPLOADER_STARTED.set()
        # Held open until the test has parked a second job behind this one.
        UPLOAD_GATE.wait(timeout=30.0)
        # The real eval-evidence codepath: save_bytes -> output stream ->
        # _gpu_slot_yielded (yield permit, upload, REACQUIRE).
        asset = ctx.save_bytes("evidence/blob-1.bin", b"x" * 1024)
        return UploadOut(response=f"saved:{asset.ref}")

    def bystander(self, ctx: RequestContext, data: UploadIn) -> UploadOut:
        CALLS.append("bystander")
        return UploadOut(response="ran")
