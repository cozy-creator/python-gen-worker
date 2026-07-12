"""hub-artifact-echo — CPU endpoint bound to a TAG-QUALIFIED Hub artifact.

The binding carries an EXPLICIT non-default tag, so its wire ref is the
gw#492 normal form ``owner/repo:prod`` — the spelling the hub must match
byte-for-byte in keep/snapshot/ModelOp refs (th#763: a cold worker's FIRST
request per unseen tag-qualified ref must complete, never fatal
MissingSnapshotError). No GPU, no torch: ``setup`` annotates the slot as
``str`` and receives the snapshot's local path.

Driven by the cozy e2e th#763 cold-ref journey; ``ARTIFACT_ECHO_REF`` /
``ARTIFACT_ECHO_TAG`` override the binding for other stacks.
"""

from __future__ import annotations

import os
from pathlib import Path

import msgspec

from gen_worker import Hub, RequestContext, endpoint

HUB_REF = os.environ.get("ARTIFACT_ECHO_REF", "tensorhub/th763-cold-tiny")
HUB_TAG = os.environ.get("ARTIFACT_ECHO_TAG", "prod")


class StatInput(msgspec.Struct):
    text: str = ""


class StatOutput(msgspec.Struct):
    files: list[str]
    total_bytes: int


@endpoint(model=Hub(HUB_REF, tag=HUB_TAG))
class HubArtifactEcho:
    def setup(self, model: str) -> None:
        # str annotation = path injection: the snapshot's local directory.
        self._root = Path(model)

    def artifact_stat(self, ctx: RequestContext, data: StatInput) -> StatOutput:
        ctx.raise_if_cancelled()
        files: list[str] = []
        total = 0
        for p in sorted(self._root.rglob("*")):
            if p.is_file():
                files.append(str(p.relative_to(self._root)))
                total += p.stat().st_size
        return StatOutput(files=files, total_bytes=total)
