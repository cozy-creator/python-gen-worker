from __future__ import annotations

from gen_worker.mount_backend import mount_backend_for_path, volume_key_for_path


def test_mount_backend_for_path_parses_mountinfo_and_classifies_nfs() -> None:
    mountinfo = "\n".join(
        [
            # id parent major:minor root mountpoint opts ... - fstype source superopts
            "36 25 0:32 / / rw,relatime - overlay overlay rw,lowerdir=/lower,upperdir=/upper,workdir=/work",
            "50 36 0:51 / /workspace rw,relatime - nfs4 10.0.0.1:/vol rw,vers=4.2",
            "60 36 8:1 / /tmp rw,relatime - ext4 /dev/nvme0n1p1 rw",
        ]
    )

    mb = mount_backend_for_path("/workspace/models/some-model", mountinfo_text=mountinfo)
    assert mb is not None
    assert mb.fstype == "nfs4"
    assert mb.is_nfs is True

    mb2 = mount_backend_for_path("/tmp/cozy/local-model-cache", mountinfo_text=mountinfo)
    assert mb2 is not None
    assert mb2.fstype == "ext4"
    assert mb2.is_nfs is False

    # Volume key is a stable hash of mount identity (no raw source exposure).
    k1 = volume_key_for_path("/workspace/models/some-model", mountinfo_text=mountinfo)
    k2 = volume_key_for_path("/workspace/other", mountinfo_text=mountinfo)
    assert k1 is not None and k2 is not None
    assert k1 == k2
