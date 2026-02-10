import tempfile
import unittest
from pathlib import Path

from gen_worker.project_validation import validate_project


class TestProjectValidation(unittest.TestCase):
    def test_requires_dockerfile_cozy_toml_and_pyproject(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            res = validate_project(root)
            self.assertFalse(res.ok)
            self.assertIn("missing Dockerfile", res.errors)
            self.assertIn("missing cozy.toml", res.errors)
            self.assertIn("missing pyproject.toml", res.errors)

            (root / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
            (root / "cozy.toml").write_text(
                "schema_version = 1\nname = 'x'\nmain = 'x.main'\ngen_worker = '>=0'\n",
                encoding="utf-8",
            )
            (root / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
            res2 = validate_project(root)
            self.assertTrue(res2.ok)

    def test_rejects_requirements_txt(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
            (root / "cozy.toml").write_text(
                "schema_version = 1\nname = 'x'\nmain = 'x.main'\ngen_worker = '>=0'\n",
                encoding="utf-8",
            )
            (root / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
            (root / "requirements.txt").write_text("msgspec\n", encoding="utf-8")

            res = validate_project(root)
            self.assertFalse(res.ok)
            self.assertTrue(any("requirements.txt" in e for e in res.errors))

    def test_requires_project_name(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
            (root / "cozy.toml").write_text(
                "schema_version = 1\nname = 'x'\nmain = 'x.main'\ngen_worker = '>=0'\n",
                encoding="utf-8",
            )
            (root / "pyproject.toml").write_text("[project]\nversion='0.1.0'\n", encoding="utf-8")
            res = validate_project(root)
            self.assertFalse(res.ok)
            self.assertTrue(any("missing [project].name" in e for e in res.errors))

    def test_accepts_non_slug_project_name(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
            (root / "cozy.toml").write_text(
                "schema_version = 1\nname = 'My Cool Project'\nmain = 'x.main'\ngen_worker = '>=0'\n",
                encoding="utf-8",
            )
            (root / "pyproject.toml").write_text(
                "[project]\nname='My Cool Project'\n",
                encoding="utf-8",
            )
            res = validate_project(root)
            self.assertTrue(res.ok)
