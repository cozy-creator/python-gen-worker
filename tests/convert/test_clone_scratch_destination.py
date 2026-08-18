"""pgw#1385: the SDK's clone destination validator admits the hub's OWN scratch
repo, and nothing else that starts with an underscore.

Measured on the production path, not inferred: a `publishes=true` job's
destination is not the submitter's — the hub rewrites `payload.destination_repo`
on the wire to `<org>/_job-<job-id>` (th#2068), and the hub carved that reserved
prefix out of its own normalizer (`normalizeSchedulerRepoName`) precisely so the
rewrite survives validation. The SDK never got the twin, so `run_clone` raised
`ValueError: destination_repo must be '<owner>/<repo>'` in the job body on a real
L4 — every BYOM/model-ingest job dead at the one address the platform hands out.

The carve-out is a STRIP-VALIDATE-REATTACH of ONE reserved prefix, never a
widened `^[a-z0-9_]` charset: `_job-` is platform vocabulary, not a legal public
name, so a user-authored `_anything` must still refuse — otherwise a submitted
payload could name a scratch-shaped repo and the write authority that the prefix
anchors becomes squattable.
"""

from __future__ import annotations

import pytest

from gen_worker.convert.clone import normalize_destination_ref
from gen_worker.scratchrepo import PREFIX


def test_hub_authored_scratch_destination_round_trips():
    # The literal shape the hub minted for job 01a01390-… in the run that found
    # this (transcribed, not derived from the code under test).
    ref = "tensorhub/_job-01a01390-ea8a-70fc-8796-205f5dd2a012"
    assert normalize_destination_ref(ref) == ref
    # Case/whitespace normalization still applies through the prefix.
    assert normalize_destination_ref("  TensorHub/_Job-01A01390-EA8A  ") == "tensorhub/_job-01a01390-ea8a"
    assert PREFIX == "_job-"


@pytest.mark.parametrize("ref", [
    "tensorhub/_notjob-x",        # underscore, but not the reserved prefix
    "tensorhub/_",                # bare underscore
    "tensorhub/_job-",            # the prefix with nothing behind it
    "tensorhub/_job-bad_name",    # prefix stripped, REST still fails the charset
    "_job-01a01390/mymodel",      # scratch shape in the OWNER half: the hub
                                  # carves out the repo half only
])
def test_user_authored_underscore_names_still_refuse(ref):
    with pytest.raises(ValueError, match="destination_repo must be"):
        normalize_destination_ref(ref)
