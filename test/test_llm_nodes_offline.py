"""Offline tests for nodes that normally call an LLM."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage
from nodes.analyze_dataset_node import analyze_dataset_note
from nodes.fix_cells_node import fix_cells_node
from nodes.fix_plan_node import fix_plan_node
from nodes.generate_section_node import generate_section_node
from nodes.plan_notebook_node import plan_notebook_node
from schemas.fixed_cell_schema import FixedCell
from test.fakes import FakeRunnable, make_generated_section, make_plan
from tools.section_generation import build_dataset_context


class AnalyzeDatasetOfflineTests(unittest.TestCase):
    def test_analyze_dataset_uses_fake_llm(self) -> None:
        fake = FakeRunnable(
            [AIMessage(content="Offline dataset analysis")]
        )
        state = {
            "summary": {
                "analysis_context": {
                    "rows": 100,
                    "columns": ["feature", "target"],
                }
            }
        }

        with patch("nodes.analyze_dataset_node.llm", fake):
            result = analyze_dataset_note(state)

        self.assertEqual(result["summary_llm"], "Offline dataset analysis")
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(len(fake.calls[0]["input"]), 2)


class PlanNodeOfflineTests(unittest.TestCase):
    def test_plan_node_accepts_structured_fake_response(self) -> None:
        fake_plan = make_plan()
        fake = FakeRunnable([fake_plan])
        state = {
            "summary": {
                "analysis_context": {
                    "column_names": ["feature", "median_house_value"],
                    "numeric_columns": ["feature", "median_house_value"],
                }
            },
            "target_analysis": {
                "unique_count": 50,
                "missing_count": 0,
            },
            "target_column": "median_house_value",
            "problem_type": "regression",
        }

        with patch("nodes.plan_notebook_node.structured_llm", fake):
            result = plan_notebook_node(state)

        self.assertIsNone(result["error"])
        self.assertEqual(len(result["notebook_plan"]["sections"]), 8)
        self.assertEqual(len(fake.calls), 1)


class FixPlanOfflineTests(unittest.TestCase):
    def test_fix_plan_receives_old_plan_and_errors(self) -> None:
        fixed_plan = make_plan()
        fake = FakeRunnable([fixed_plan])
        old_plan = fixed_plan.model_dump()
        old_plan["sections"][5]["tasks"] = []
        state = {
            "notebook_plan": old_plan,
            "plan_validation_errors": [
                {
                    "error_type": "invalid_task_count",
                    "location": "section_6",
                    "message": "Section 6 has no tasks.",
                }
            ],
            "fix_plan_attempts": 0,
        }

        with patch("nodes.fix_plan_node.fix_plan_llm", fake):
            result = fix_plan_node(state)

        self.assertEqual(result["plan_validation_status"], "pending")
        self.assertEqual(result["fix_plan_attempts"], 1)
        self.assertEqual(len(fake.calls), 1)

        prompt = fake.calls[0]["input"][1].content
        self.assertIn("section_6", prompt)
        self.assertIn("invalid_task_count", prompt)


class GenerateSectionsOfflineTests(unittest.TestCase):
    def test_generate_all_sections_without_network(self) -> None:
        plan = make_plan()
        responses = []

        for index in range(1, 9):
            section_id = f"section_{index}"
            if index == 1:
                source = (
                    'dataset_path = "./data/housing.csv"\n'
                    'target_column = "median_house_value"\n'
                    "print(dataset_path, target_column)"
                )
            else:
                source = f"section_value_{index} = {index}"

            responses.append(
                make_generated_section(section_id, source)
            )

        fake = FakeRunnable(responses)
        state = {
            "notebook_plan": plan.model_dump(),
            "dataset_path": "./data/housing.csv",
            "target_column": "median_house_value",
            "problem_type": "regression",
            "summary": {
                "column_names": ["feature", "median_house_value"],
                "numeric_columns": ["feature", "median_house_value"],
                "categorical_columns": [],
                "possible_id_columns": [],
            },
            "target_analysis": {
                "unique_count": 50,
                "missing_count": 0,
            },
            "notebook_cells": [],
            "current_section_index": 0,
            "generated_section_ids": [],
            "generation_cell_errors": [],
            "section_retry_attempts": 0,
        }

        with (
            patch("tools.section_generation.section_llm", fake),
            patch("tools.section_generation.time.sleep", return_value=None),
            patch("builtins.print"),
        ):
            result = dict(state)
            for _ in plan.sections:
                result.update(generate_section_node(result))

        self.assertEqual(result["generation_cell_status"], "success")
        self.assertEqual(len(result["notebook_cells"]), 8)
        self.assertEqual(len(fake.calls), 8)


class FixCellsOfflineTests(unittest.TestCase):
    def test_fix_without_validation_errors_uses_cell_status(self) -> None:
        result = fix_cells_node(
            {
                "notebook_cells": [
                    {
                        "cell_id": "section_1_1",
                        "cell_type": "code",
                        "source": "value = 1",
                    }
                ],
                "validation_cell_errors": [],
                "fix_cell_attempts": 0,
            }
        )

        self.assertEqual(result["validation_cell_status"], "invalid")
        self.assertEqual(result["fix_cell_attempts"], 1)

    def test_fix_syntax_error_with_fake_structured_response(self) -> None:
        fixed = FixedCell(
            cell_id="section_1_code_1",
            source="print('fixed')",
            changes="Closed the string and function call.",
        )
        fake = FakeRunnable([fixed])
        state = {
            "notebook_cells": [
                {
                    "cell_id": "section_1_code_1",
                    "section_id": "section_1",
                    "cell_type": "code",
                    "title": "Broken syntax",
                    "source": "print('broken'",
                    "purpose": "Exercise offline repair.",
                    "expected_output": None,
                }
            ],
            "validation_cell_errors": [
                {
                    "cell_id": "section_1_code_1",
                    "error_type": "syntax_error",
                    "line": 1,
                    "message": "'(' was never closed",
                }
            ],
            "fix_cell_attempts": 0,
        }

        with patch("nodes.fix_cells_node.fix_llm", fake):
            result = fix_cells_node(state)

        self.assertEqual(result["fix_cell_attempts"], 1)
        self.assertEqual(result["fixed_cell_ids"], ["section_1_code_1"])
        self.assertEqual(
            result["notebook_cells"][0]["source"],
            "print('fixed')",
        )
        self.assertEqual(len(fake.calls), 1)

class DatasetContextTests(unittest.TestCase):
    def test_build_dataset_context(self) -> None:
        state = {
            "summary": {
                "column_names": [
                    "income",
                    "ocean_proximity",
                    "median_house_value",
                ],
                "numeric_columns": [
                    "income",
                    "median_house_value",
                ],
                "categorical_columns": [
                    "ocean_proximity",
                ],
                "possible_id_columns": [],
            },
            "target_analysis": {
                "target_column": (
                    "median_house_value"
                ),
                "problem_type": "regression",
            },
        }

        context = build_dataset_context(
            state
        )

        self.assertEqual(
            context["column_names"],
            [
                "income",
                "ocean_proximity",
                "median_house_value",
            ],
        )

        self.assertEqual(
            context["numeric_columns"],
            [
                "income",
                "median_house_value",
            ],
        )

        self.assertEqual(
            context["categorical_columns"],
            [
                "ocean_proximity",
            ],
        )

        self.assertEqual(
            context["possible_id_columns"],
            [],
        )

        self.assertEqual(
            context["target_analysis"][
                "problem_type"
            ],
            "regression",
        )
    def test_build_dataset_context_with_empty_state(self,) -> None:
        context = build_dataset_context({})

        self.assertEqual(
            context,
            {
                "column_names": None,
                "numeric_columns": None,
                "categorical_columns": None,
                "possible_id_columns": None,
                "target_analysis": {},
            },
        )
if __name__ == "__main__":
    unittest.main()
