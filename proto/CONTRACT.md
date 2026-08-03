# The wire contract lives in tensorhub — not here

The canonical worker↔orchestrator gRPC contract is:

    tensorhub: internal/orchestrator/grpc/proto/CONTRACT.md

**Do not edit a second copy here.** This file used to be a full 988-line fork of
that document. The two drifted into 38 conflicting hunks — the document written
to prevent drift became the cause of it. It is a pointer now, permanently.

`worker_scheduler.proto` in this directory is a vendored copy of tensorhub's
proto for codegen. Wire semantics are documented on the tensorhub side only.
