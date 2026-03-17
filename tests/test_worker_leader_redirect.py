import unittest

from gen_worker.worker import Worker


class TestWorkerLeaderRedirect(unittest.TestCase):
    def test_extract_leader_addr(self):
        self.assertEqual(Worker._extract_leader_addr("not_leader:127.0.0.1:8080"), "127.0.0.1:8080")
        self.assertIsNone(Worker._extract_leader_addr("not_leader:"))
        self.assertIsNone(Worker._extract_leader_addr("other_error"))
        self.assertIsNone(Worker._extract_leader_addr(None))

    def test_normalize_scheduler_addrs(self):
        addrs = Worker._normalize_scheduler_addrs("a:1", ["b:2", "a:1", " ", "c:3"])
        self.assertEqual(addrs, ["a:1", "b:2", "c:3"])


if __name__ == "__main__":
    unittest.main()
