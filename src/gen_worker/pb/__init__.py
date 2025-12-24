"""Protocol buffer module for worker communication."""

from __future__ import annotations

import sys
from typing import Any

from . import frontend_pb2, worker_scheduler_pb2

__all__ = [
    "worker_scheduler_pb2",
    "worker_scheduler_pb2_grpc",
    "frontend_pb2",
    "frontend_pb2_grpc",
]

# Compatibility: generated grpc stubs use absolute imports (e.g., worker_scheduler_pb2).
# Register module aliases before importing *_grpc stubs.
sys.modules.setdefault("worker_scheduler_pb2", worker_scheduler_pb2)  # type: ignore[arg-type]
sys.modules.setdefault("frontend_pb2", frontend_pb2)  # type: ignore[arg-type]

# Import grpc stubs after aliases are registered.
from . import frontend_pb2_grpc, worker_scheduler_pb2_grpc

# Also expose grpc stubs for convenience.
sys.modules.setdefault("worker_scheduler_pb2_grpc", worker_scheduler_pb2_grpc)  # type: ignore[arg-type]
sys.modules.setdefault("frontend_pb2_grpc", frontend_pb2_grpc)  # type: ignore[arg-type]
