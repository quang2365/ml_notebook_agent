import unittest
from unittest.mock import patch

from nodes.generate_section_node import (
    generate_section_node,
)
from route.route_after_section_generation import (
    route_after_section_generation,
)


class GenerateSectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.state = {
            "notebook_plan": {
                "sections": [
                    {
                        "section_id": "section_1",
                        "tasks": ["Load data"],
                    },
                    {
                        "section_id": "section_2",
                        "tasks": ["Explore data"],
                    },
                ]
            },
            "dataset_path": "./data/housing.csv",
            "target_column": "median_house_value",
            "problem_type": "regression",
            "summary": {},
            "target_analysis": {},
            "notebook_cells": [],
            "current_section_index": 0,
            "generated_section_ids": [],
            "generation_cell_errors": [],
            "section_retry_attempts": 0,
        }
    @patch(
        "nodes.generate_section_node."
        "generate_one_section"
    )
    def test_generates_one_section(
        self,
        mock_generate,
    ) -> None:
        mock_generate.return_value = {
            "status": "success",
            "section_id": "section_1",
            "cells": [
                {
                    "cell_id": "section_1_1",
                    "section_id": "section_1",
                    "cell_type": "code",
                    "title": "Load data",
                    "source": "value = 1",
                    "purpose": "Test",
                }
            ],
            "error": None,
        }

        result = generate_section_node(
            self.state
        )

        mock_generate.assert_called_once()

        self.assertEqual(
            result["current_section_index"],
            1,
        )

        self.assertEqual(
            result["generated_section_ids"],
            ["section_1"],
        )

        self.assertEqual(
            len(result["notebook_cells"]),
            1,
        )

        # Vẫn còn section 2.
        self.assertEqual(
            result["generation_cell_status"],
            "pending",
        )
    @patch(
        "nodes.generate_section_node."
        "generate_one_section"
    )
    def test_appends_to_existing_cells(
        self,
        mock_generate,
    ) -> None:
        old_cell = {
            "cell_id": "section_1_1",
            "section_id": "section_1",
            "cell_type": "code",
            "title": "Old cell",
            "source": "value = 1",
            "purpose": "Previous section",
        }

        state = {
            **self.state,
            "current_section_index": 1,
            "notebook_cells": [old_cell],
            "generated_section_ids": [
                "section_1",
            ],
        }

        mock_generate.return_value = {
            "status": "success",
            "section_id": "section_2",
            "cells": [
                {
                    "cell_id": "section_2_1",
                    "section_id": "section_2",
                    "cell_type": "code",
                    "title": "New cell",
                    "source": "value += 1",
                    "purpose": "Current section",
                }
            ],
            "error": None,
        }

        result = generate_section_node(state)

        self.assertEqual(
            [
                cell["cell_id"]
                for cell in result[
                    "notebook_cells"
                ]
            ],
            [
                "section_1_1",
                "section_2_1",
            ],
        )

        self.assertEqual(
            result["current_section_index"],
            2,
        )

        self.assertEqual(
            result["generation_cell_status"],
            "success",
        )
    @patch(
        "nodes.generate_section_node."
        "generate_one_section"
    )
    def test_failure_preserves_progress(
        self,
        mock_generate,
    ) -> None:
        old_cell = {
            "cell_id": "section_1_1",
            "section_id": "section_1",
            "cell_type": "code",
            "title": "Old cell",
            "source": "value = 1",
            "purpose": "Previous section",
        }

        state = {
            **self.state,
            "current_section_index": 1,
            "notebook_cells": [old_cell],
            "generated_section_ids": [
                "section_1",
            ],
        }

        mock_generate.return_value = {
            "status": "failed",
            "section_id": "section_2",
            "cells": [],
            "error": "Too Many Requests",
        }

        result = generate_section_node(state)

        self.assertEqual(
            result["notebook_cells"],
            [old_cell],
        )

        # Không tăng vì section 2 chưa thành công.
        self.assertEqual(
            result["current_section_index"],
            1,
        )

        self.assertEqual(
            result["generated_section_ids"],
            ["section_1"],
        )

        self.assertEqual(
            result["generation_cell_status"],
            "failed",
        )

        self.assertEqual(
            result["section_retry_attempts"],
            1,
        )
class SectionGenerationRouteTests(
    unittest.TestCase
):
    def test_continue_when_sections_remain(
        self,
    ) -> None:
        result = route_after_section_generation(
            {
                "notebook_plan": {
                    "sections": [{}, {}],
                },
                "current_section_index": 1,
                "generation_cell_status": (
                    "pending"
                ),
            }
        )

        self.assertEqual(result, "continue")

    def test_complete_after_last_section(
        self,
    ) -> None:
        result = route_after_section_generation(
            {
                "notebook_plan": {
                    "sections": [{}, {}],
                },
                "current_section_index": 2,
                "generation_cell_status": (
                    "success"
                ),
            }
        )

        self.assertEqual(result, "complete")

    def test_failed_generation_stops(
        self,
    ) -> None:
        result = route_after_section_generation(
            {
                "notebook_plan": {
                    "sections": [{}, {}],
                },
                "current_section_index": 1,
                "generation_cell_status": (
                    "failed"
                ),
            }
        )

        self.assertEqual(result, "failed")