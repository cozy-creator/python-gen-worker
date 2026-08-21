from __future__ import annotations

import pytest

from gen_worker.convert.clone import normalize_destination_ref
from gen_worker.scratchrepo import PREFIX


def test_hub_authored_scratch_destination_round_trips():
    ref = "tensorhub/_job-01a01390-ea8a-70fc-8796-205f5dd2a012"
    assert normalize_destination_ref(ref) == ref
    assert normalize_destination_ref("  TensorHub/_Job-01A01390-EA8A  ") == "tensorhub/_job-01a01390-ea8a"
    assert PREFIX == "_job-"


@pytest.mark.parametrize("ref", [
    "tensorhub/_notjob-x",
    "tensorhub/_",
    "tensorhub/_job-",
    "tensorhub/_job-bad_name",
    "_job-01a01390/mymodel",
])
def test_user_authored_underscore_names_still_refuse(ref):
    with pytest.raises(ValueError, match="destination_repo must be"):
        normalize_destination_ref(ref)
