"""th#2082 harness: a handler that saves one small MEDIA blob.

Small enough to sit under ``_SAVE_BYTES_INLINE_THRESHOLD``, so the client's
``Prefer: bytes=`` preference is the ONLY thing deciding whether the bytes are
uploaded or handed back raw. The handler reports what it observed, so the fence
can read the worker's own view of the decision as well as the upload sink's.
"""

from __future__ import annotations

import msgspec

from gen_worker import RequestContext, endpoint


class MediaIn(msgspec.Struct):
    text: str = ""


class MediaOut(msgspec.Struct):
    ref: str
    inline: bool
    size_bytes: int


@endpoint
class MediaBytesJobs:
    def render(self, ctx: RequestContext, data: MediaIn) -> MediaOut:
        asset = ctx.save_bytes("media/th2082.bin", b"m" * 2048)
        return MediaOut(
            ref=asset.ref,
            inline=bool(asset.inline_bytes),
            size_bytes=int(asset.size_bytes or 0),
        )
