"""Broad offline simulation of a complete LLM-rendered ML notebook."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nodes.generate_cells_node import generate_cells_node
from nodes.notebook_builder_node import notebook_builder
from nodes.validate_cells_node import validate_cells_node
from test.fakes import FakeRunnable, make_full_rendered_sections, make_plan


class FullRenderedNotebookTests(unittest.TestCase):
    def setUp(self) -> None:
        self.plan = make_plan(section_count=10)
        self.rendered_sections = make_full_rendered_sections()
        self.state = {
            "notebook_plan": self.plan.model_dump(),
            "dataset_path": "./data/housing.csv",
            "target_column": "median_house_value",
            "problem_type": "regression",
            "summary": {
                "column_names": [
                    "longitude",
                    "latitude",
                    "housing_median_age",
                    "total_rooms",
                    "total_bedrooms",
                    "population",
                    "households",
                    "median_income",
                    "median_house_value",
                    "ocean_proximity",
                ],
                "numeric_columns": [
                    "longitude",
                    "latitude",
                    "housing_median_age",
                    "total_rooms",
                    "total_bedrooms",
                    "population",
                    "households",
                    "median_income",
                    "median_house_value",
                ],
                "categorical_columns": ["ocean_proximity"],
                "possible_id_columns": [],
            },
            "target_analysis": {
                "unique_count": 3842,
                "missing_count": 0,
                "distribution": {
                    "mean": 206855.82,
                    "median": 179700.0,
                    "std": 115395.62,
                    "skewness": 0.98,
                },
            },
        }

    def _generate_offline(self) -> tuple[dict, FakeRunnable]:
        fake = FakeRunnable(self.rendered_sections.copy())

        with (
            patch("nodes.generate_cells_node.section_llm", fake),
            patch("nodes.generate_cells_node.time.sleep", return_value=None),
            patch("builtins.print"),
        ):
            result = generate_cells_node(self.state)

        return result, fake

    def test_complete_render_contains_30_cells(self) -> None:
        result, fake = self._generate_offline()

        self.assertEqual(result["generation_cell_status"], "success")
        self.assertEqual(len(result["notebook_cells"]), 30)
        self.assertEqual(len(fake.calls), 10)

        code_count = sum(
            cell["cell_type"] == "code"
            for cell in result["notebook_cells"]
        )
        markdown_count = sum(
            cell["cell_type"] == "markdown"
            for cell in result["notebook_cells"]
        )
        self.assertEqual(code_count, 20)
        self.assertEqual(markdown_count, 10)

    def test_complete_render_passes_static_and_dependency_validation(self) -> None:
        generated, _ = self._generate_offline()

        with patch("builtins.print"):
            validation = validate_cells_node(
                {
                    "notebook_cells": generated["notebook_cells"],
                }
            )

        self.assertEqual(
            validation["validation_cell_status"],
            "valid",
            msg=json.dumps(
                validation.get("validation_cell_errors"),
                ensure_ascii=False,
                indent=2,
            ),
        )

    def test_complete_render_builds_30_cell_ipynb(self) -> None:
        generated, _ = self._generate_offline()

        with tempfile.TemporaryDirectory() as directory:
            notebook_path = Path(directory) / "full_render.ipynb"
            build_result = notebook_builder(
                {
                    "notebook_cells": generated["notebook_cells"],
                    "notebook_path": str(notebook_path),
                }
            )

            self.assertEqual(build_result["build_status"], "success")
            notebook = json.loads(
                notebook_path.read_text(encoding="utf-8")
            )
            self.assertEqual(len(notebook["cells"]), 30)

    def test_validator_detects_error_in_large_render(self) -> None:
        generated, _ = self._generate_offline()
        generated["notebook_cells"][16]["source"] = "print('broken'"

        with patch("builtins.print"):
            validation = validate_cells_node(
                {
                    "notebook_cells": generated["notebook_cells"],
                }
            )

        self.assertEqual(validation["validation_cell_status"], "invalid")
        self.assertTrue(
            any(
                error.get("error_type") == "syntax_error"
                for error in validation["validation_cell_errors"]
            )
        )


if __name__ == "__main__":
    unittest.main()
