from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


# In-memory shape of an orchestrator-resolved cozy: ref. These used to be
# populated by an HTTP resolve call against tensorhub from inside the worker;
# now the orchestrator pre-resolves every cozy ref a job needs and ships the
# manifest + presigned URLs via JobExecutionRequest.resolved_repos_by_id over
# gRPC. The worker never talks to tensorhub's resolve endpoint directly.


@dataclass(frozen=True)
class WorkerResolvedRepoFile:
    path: str
    size_bytes: int
    blake3: str
    url: Optional[str]


@dataclass(frozen=True)
class WorkerResolvedRepo:
    snapshot_digest: str
    files: List[WorkerResolvedRepoFile]
