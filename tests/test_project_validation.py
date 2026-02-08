import tempfile
import unittest
from pathlib import Path

from gen_worker.project_validation import validate_project


class TestProjectValidation(unittest.TestCase):
    def test_requires_pyproject_and_tool_cozy(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            res = validate_project(root)
            self.assertFalse(res.ok)
            self.assertIn("missing pyproject.toml", res.errors)

            (root / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
            res2 = validate_project(root)
            self.assertFalse(res2.ok)
            self.assertTrue(any("missing [tool.cozy]" in e for e in res2.errors))

    def test_rejects_cozy_toml_and_requirements_txt(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "pyproject.toml").write_text("[tool.cozy]\nfunctions.modules=['x']\n", encoding="utf-8")
            (root / "requirements.txt").write_text("msgspec\n", encoding="utf-8")
            (root / "cozy.toml").write_text("[tool.cozy]\n", encoding="utf-8")

            res = validate_project(root)
            self.assertFalse(res.ok)
            self.assertTrue(any("requirements.txt" in e for e in res.errors))
            self.assertTrue(any("cozy.toml" in e for e in res.errors))

    def test_requires_project_name(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "pyproject.toml").write_text("[tool.cozy]\nfunctions.modules=['x']\n", encoding="utf-8")
            res = validate_project(root)
            self.assertFalse(res.ok)
            self.assertTrue(any("missing [project].name" in e for e in res.errors))

    def test_accepts_non_slug_project_name(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "pyproject.toml").write_text(
                "[project]\nname='My Cool Project'\n[tool.cozy]\nfunctions.modules=['x']\n",
                encoding="utf-8",
            )
            res = validate_project(root)
            self.assertTrue(res.ok)
