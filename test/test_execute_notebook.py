"""Tests for the notebook execution node without starting a real kernel."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nbformat
from nbclient.exceptions import CellExecutionError

from nodes.execute_notebook_note import (
    execute_notebook_node,
    extract_failed_cell,
)
from route.route_after_execution import route_after_execution


def write_test_notebook(path: Path) -> None:
    """Write a minimal valid notebook used by execution-node tests."""
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell(
                "result = 1 + 1\nprint(result)",
                id="normalized-cell-id",
                metadata={
                    "agent": {
                        "cell_id": "section_1_code_1",
                        "section_id": "section_1",
                        "title": "Runtime test",
                    }
                },
            ),
        ]
    )
    nbformat.write(notebook, path)


class ExecuteNotebookNodeTests(unittest.TestCase):
    def test_extract_failed_cell_prefers_agent_cell_id(self) -> None:
        notebook = nbformat.v4.new_notebook(
            cells=[
                nbformat.v4.new_code_cell(
                    "print(missing_name)",
                    id="normalized-cell-id",
                    metadata={
                        "agent": {
                            "cell_id": "section_3_code_2",
                        }
                    },
                    outputs=[
                        nbformat.v4.new_output(
                            output_type="error",
                            ename="NameError",
                            evalue="missing_name is not defined",
                            traceback=["Traceback", "NameError"],
                        )
                    ],
                )
            ]
        )

        result = extract_failed_cell(notebook)

        self.assertIsNotNone(result)
        self.assertEqual(result["cell_id"], "section_3_code_2")
        self.assertEqual(result["exception_name"], "NameError")
        self.assertIn("print(missing_name)", result["source"])

    def test_extract_failed_cell_falls_back_to_notebook_id(self) -> None:
        notebook = nbformat.v4.new_notebook(
            cells=[
                nbformat.v4.new_code_cell(
                    "1 / 0",
                    id="fallback-cell-id",
                    outputs=[
                        nbformat.v4.new_output(
                            output_type="error",
                            ename="ZeroDivisionError",
                            evalue="division by zero",
                            traceback=["ZeroDivisionError"],
                        )
                    ],
                )
            ]
        )

        result = extract_failed_cell(notebook)

        self.assertEqual(result["cell_id"], "fallback-cell-id")

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
        def build_failing_client(notebook, **kwargs):
            class FailingClient:
                def execute(self, **execute_kwargs):
                    notebook.cells[0]["outputs"] = [
                        nbformat.v4.new_output(
                            output_type="error",
                            ename="NameError",
                            evalue="name 'missing_name' is not defined",
                            traceback=[
                                "Traceback: NameError",
                                "NameError: missing_name",
                            ],
                        )
                    ]
                    raise CellExecutionError(
                        "Traceback: NameError",
                        "NameError",
                        "name 'missing_name' is not defined",
                    )

            return FailingClient()

        mock_client_class.side_effect = build_failing_client

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
        self.assertEqual(
            result["execution_error"]["cell_id"],
            "section_1_code_1",
        )
        self.assertEqual(
            result["execution_error"]["exception_name"],
            "NameError",
        )
        self.assertIn("NameError", result["execution_error"]["message"])


class ExecutionRouteTests(unittest.TestCase):
    def test_success_ends_workflow(self) -> None:
        self.assertEqual(
            route_after_execution(
                {
                    "execution_status": "success",
                    "execution_error": None,
                    "execution_fix_attempts": 0,
                }
            ),
            "success",
        )

    def test_failed_cell_is_fixable(self) -> None:
        self.assertEqual(
            route_after_execution(
                {
                    "execution_status": "failed",
                    "execution_error": {
                        "cell_id": "section_1_code_1",
                    },
                    "execution_fix_attempts": 0,
                }
            ),
            "fix",
        )

    def test_missing_cell_id_or_retry_limit_stops(self) -> None:
        self.assertEqual(
            route_after_execution(
                {
                    "execution_status": "failed",
                    "execution_error": None,
                    "execution_fix_attempts": 0,
                }
            ),
            "failed",
        )
        self.assertEqual(
            route_after_execution(
                {
                    "execution_status": "failed",
                    "execution_error": {
                        "cell_id": "section_1_code_1",
                    },
                    "execution_fix_attempts": 2,
                }
            ),
            "failed",
        )


if __name__ == "__main__":
    unittest.main()
