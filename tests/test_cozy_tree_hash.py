from gen_worker.cozy_cas import blake3_tree_hash


def test_blake3_tree_hash_stable_sorting() -> None:
    a = blake3_tree_hash([("b.txt", 2, "aa"), ("a.txt", 1, "bb")])
    b = blake3_tree_hash([("a.txt", 1, "bb"), ("b.txt", 2, "aa")])
    assert a == b

