import unittest

from nodes.prepare_generation_node import (
    prepare_generation_node,
)


class PrepareGenerationTests(
    unittest.TestCase
):
    def test_resets_old_generation_data(
        self,
    ) -> None:
        """
        Verify that data from the previous generation
        is reset when a new generation begins.
        """

        state = {
            "notebook_plan": {
                "sections": [
                    {
                        "section_id": "section_1",
                    },
                    {
                        "section_id": "section_2",
                    },
                ]
            },
            "notebook_cells": [
                {
                    "cell_id": "old_cell",
                }
            ],
            "section_generation_status": "success",
            "section_generation_errors": [
                {
                    "message": "Old error",
                }
            ],
            "current_section_index": 7,
            "generated_section_ids": [
                "old_section",
            ],
            "section_retry_attempts": 2,
            "error": "Old workflow error",
        }

        result = prepare_generation_node(
            state
        )

        self.assertEqual(
            result["section_generation_status"],
            "pending",
        )

        self.assertEqual(
            result["notebook_cells"],
            [],
        )

        self.assertEqual(
            result["section_generation_errors"],
            [],
        )

        self.assertEqual(
            result["current_section_index"],
            0,
        )

        self.assertEqual(
            result["generated_section_ids"],
            [],
        )

        self.assertEqual(
            result["section_retry_attempts"],
            0,
        )

        self.assertIsNone(
            result["error"]
        )

    def test_fails_without_notebook_plan(
        self,
    ) -> None:
        """
        Without a plan, generation must stop
        before calling the LLM.
        """

        result = prepare_generation_node({})

        self.assertEqual(
            result["section_generation_status"],
            "failed",
        )

        self.assertEqual(
            result["section_generation_errors"][0][
                "error_type"
            ],
            "missing_sections",
        )

    def test_fails_with_empty_sections(
        self,
    ) -> None:
        """
        A plan with empty sections also cannot
        start generation.
        """

        result = prepare_generation_node(
            {
                "notebook_plan": {
                    "sections": [],
                }
            }
        )

        self.assertEqual(
            result["section_generation_status"],
            "failed",
        )

        self.assertIn(
            "has no section",
            result["error"],
        )


if __name__ == "__main__":
    unittest.main()
