from __future__ import annotations

# Worker<->scheduler gRPC wire protocol version.
# MAJOR: breaking changes, MINOR: additive/backward-compatible changes.
WIRE_PROTOCOL_MAJOR = 1
WIRE_PROTOCOL_MINOR = 0


def wire_protocol_version_string() -> str:
    return f"{WIRE_PROTOCOL_MAJOR}.{WIRE_PROTOCOL_MINOR}"
