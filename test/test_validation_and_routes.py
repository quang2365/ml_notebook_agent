"""Tests for deterministic validators and graph routes."""

from __future__ import annotations

import unittest

from nodes.validate_cells_node import validate_cells_node
from nodes.validate_plan_node import validate_plan_node
from route.route_after_plan_validation import route_after_plan_validation
from route.route_after_validation_cell import route_after_validation_cell
from test.fakes import make_agent_cell, make_plan


class PlanValidationTests(unittest.TestCase):
    def test_valid_plan(self) -> None:
        plan = make_plan().model_dump()
        state = {
            "notebook_plan": plan,
            "target_column": "median_house_value",
            "problem_type": "regression",
        }

        result = validate_plan_node(state)

        self.assertEqual(result["plan_validation_status"], "valid")
        self.assertEqual(result["plan_validation_errors"], [])

    def test_collects_multiple_plan_errors(self) -> None:
        plan = make_plan().model_dump()
        plan["target_column"] = "wrong_target"
        plan["sections"][0]["tasks"] = []
        plan["sections"][1]["section_id"] = "section_1"
        state = {
            "notebook_plan": plan,
            "target_column": "median_house_value",
            "problem_type": "regression",
        }

        result = validate_plan_node(state)

        self.assertEqual(result["plan_validation_status"], "invalid")
        error_types = {
            error["error_type"]
            for error in result["plan_validation_errors"]
        }
        self.assertIn("target_changed", error_types)
        self.assertIn("invalid_task_count", error_types)
        self.assertIn("duplicate_section_id", error_types)


class CellValidationTests(unittest.TestCase):
    def test_valid_cells(self) -> None:
        state = {
            "notebook_cells": [make_agent_cell()],
        }

        result = validate_cells_node(state)

        self.assertEqual(result["validation_status"], "valid")
        self.assertEqual(result["validation_errors"], [])

    def test_invalid_syntax(self) -> None:
        state = {
            "notebook_cells": [make_agent_cell(source="print('broken'")],
        }

        result = validate_cells_node(state)

        self.assertEqual(result["validation_status"], "invalid")
        error_types = {
            error["error_type"]
            for error in result["validation_errors"]
        }
        self.assertIn("syntax_error", error_types)


class RouteTests(unittest.TestCase):
    def test_plan_route(self) -> None:
        self.assertEqual(
            route_after_plan_validation(
                {
                    "plan_validation_status": "valid",
                    "plan_fix_attempts": 0,
                }
            ),
            "valid",
        )
        self.assertEqual(
            route_after_plan_validation(
                {
                    "plan_validation_status": "invalid",
                    "plan_fix_attempts": 0,
                }
            ),
            "fix",
        )
        self.assertEqual(
            route_after_plan_validation(
                {
                    "plan_validation_status": "invalid",
                    "plan_fix_attempts": 3,
                }
            ),
            "failed",
        )

    def test_cell_route(self) -> None:
        cells = [make_agent_cell()]
        self.assertEqual(
            route_after_validation_cell(
                {
                    "validation_status": "valid",
                    "fix_attempts": 0,
                    "notebook_cells": cells,
                }
            ),
            "valid",
        )
        self.assertEqual(
            route_after_validation_cell(
                {
                    "validation_status": "invalid",
                    "fix_attempts": 0,
                    "notebook_cells": cells,
                }
            ),
            "fix",
        )
        self.assertEqual(
            route_after_validation_cell(
                {
                    "validation_status": "invalid",
                    "fix_attempts": 3,
                    "notebook_cells": cells,
                }
            ),
            "failed",
        )


if __name__ == "__main__":
    unittest.main()
