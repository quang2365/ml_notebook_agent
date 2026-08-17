"""Tests for the notebook execution node without starting a real kernel."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nbformat
from nbclient.exceptions import CellExecutionError

from nodes.execute_notebook_note import execute_notebook_node


def write_test_notebook(path: Path) -> None:
    """Write a minimal valid notebook used by execution-node tests."""
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell(
                "result = 1 + 1\nprint(result)"
            )
        ]
    )
    nbformat.write(notebook, path)


class ExecuteNotebookNodeTests(unittest.TestCase):
    def test_missing_notebook_path(self) -> None:
        result = execute_notebook_node(
            {"notebook_path": None}
        )

        self.assertEqual(result["execution_status"], "failed")
        self.assertEqual(
            result["execution_error"]["error_type"],
            "missing_notebook_path",
        )

    def test_notebook_file_does_not_exist(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_path = Path(temp_dir) / "missing.ipynb"
            result = execute_notebook_node(
                {"notebook_path": str(missing_path)}
            )

        self.assertEqual(result["execution_status"], "failed")
        self.assertEqual(
            result["execution_error"]["error_type"],
            "notebook_not_found",
        )

    @patch("nodes.execute_notebook_note.NotebookClient")
    def test_successful_execution(self, mock_client_class) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            notebook_path = Path(temp_dir) / "success.ipynb"
            write_test_notebook(notebook_path)

            result = execute_notebook_node(
                {"notebook_path": str(notebook_path)}
            )

        mock_client_class.return_value.execute.assert_called_once()
        mock_client_class.return_value.execute.assert_called_once_with(
            cwd=str(Path.cwd().resolve())
        )
        self.assertEqual(result["execution_status"], "success")
        self.assertIsNone(result["execution_error"])
        self.assertIsNone(result["error"])

    @patch("nodes.execute_notebook_note.NotebookClient")
    def test_cell_execution_error(self, mock_client_class) -> None:
        mock_client_class.return_value.execute.side_effect = (
            CellExecutionError(
                "Traceback: NameError",
                "NameError",
                "name 'missing_name' is not defined",
            )
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            notebook_path = Path(temp_dir) / "runtime_error.ipynb"
            write_test_notebook(notebook_path)

            result = execute_notebook_node(
                {"notebook_path": str(notebook_path)}
            )

        self.assertEqual(result["execution_status"], "failed")
        self.assertEqual(
            result["execution_error"]["error_type"],
            "cell_execution_error",
        )
        self.assertIn("NameError", result["execution_error"]["message"])


if __name__ == "__main__":
    unittest.main()
