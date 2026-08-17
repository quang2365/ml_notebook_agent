"""Offline tests for runtime cell repair."""

import unittest
from unittest.mock import patch

from nodes.fix_execution_cell_node import fix_execution_cell_node
from schemas.fixed_cell_schema import FixedCell
from test.fakes import FakeRunnable, make_agent_cell


class FixExecutionCellTests(unittest.TestCase):
    def test_missing_runtime_cell_id_fails_safely(self) -> None:
        result = fix_execution_cell_node(
            {
                "notebook_cells": [make_agent_cell()],
                "execution_error": {
                    "error_type": "cell_execution_error",
                },
                "execution_fix_attempts": 0,
            }
        )

        self.assertEqual(result["execution_status"], "failed")
        self.assertEqual(result["execution_fix_attempts"], 1)

    def test_successful_runtime_fix_returns_to_validation(self) -> None:
        fake = FakeRunnable(
            [
                FixedCell(
                    cell_id="section_1_code_1",
                    source="x = 2\nprint(x)",
                    changes="Corrected runtime value.",
                )
            ]
        )
        state = {
            "notebook_cells": [make_agent_cell()],
            "execution_error": {
                "error_type": "cell_execution_error",
                "cell_id": "section_1_code_1",
                "exception_name": "ValueError",
                "message": "runtime failed",
                "traceback": "ValueError: runtime failed",
            },
            "execution_fix_attempts": 0,
            "target_column": "median_house_value",
            "problem_type": "regression",
        }

        with patch(
            "nodes.fix_execution_cell_node.runtime_fix_llm",
            fake,
        ):
            result = fix_execution_cell_node(state)

        self.assertEqual(
            result["notebook_cells"][0]["source"],
            "x = 2\nprint(x)",
        )
        self.assertEqual(result["validation_cell_status"], "pending")
        self.assertEqual(result["pipeline_review_status"], "pending")
        self.assertEqual(result["build_status"], "pending")
        self.assertEqual(result["execution_status"], "pending")
        self.assertEqual(result["execution_fix_attempts"], 1)


if __name__ == "__main__":
    unittest.main()
