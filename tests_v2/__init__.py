# tests_v2 is a package so the declarative endpoint catalog is importable as
# "tests_v2.catalog" by the in-process Worker, by entrypoint subprocesses
# (via PYTHONPATH=<repo root>), and by every suite.
