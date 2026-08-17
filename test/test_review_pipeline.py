"""Offline tests for semantic pipeline review."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from nodes.review_pipeline_node import review_pipeline_node
from schemas.pipeline_review_schema import (
    PipelineReviewError,
    PipelineReviewResult,
)
from test.fakes import FakeRunnable, make_agent_cell, make_plan


def make_state() -> dict:
    return {
        "dataset_path": "./data/housing.csv",
        "summary": {"rows": 100},
        "summary_llm": "Dataset summary",
        "target_column": "median_house_value",
        "problem_type": "regression",
        "target_analysis": {"unique_count": 50},
        "notebook_plan": make_plan().model_dump(),
        "notebook_cells": [make_agent_cell()],
    }


class ReviewPipelineNodeTests(unittest.TestCase):
    def test_valid_review(self) -> None:
        fake = FakeRunnable(
            [
                PipelineReviewResult(
                    status="valid",
                    summary="Pipeline hợp lệ.",
                    errors=[],
                )
            ]
        )

        with patch(
            "nodes.review_pipeline_node.review_pipeline_llm",
            fake,
        ):
            result = review_pipeline_node(make_state())

        self.assertEqual(result["pipeline_review_status"], "valid")
        self.assertEqual(result["pipeline_review_errors"], [])

    def test_invalid_review_forwards_errors_to_fixer(self) -> None:
        fake = FakeRunnable(
            [
                PipelineReviewResult(
                    status="invalid",
                    summary="Pipeline có lỗi.",
                    errors=[
                        PipelineReviewError(
                            cell_id="section_1_code_1",
                            error_type="pipeline_incompatibility",
                            message="DataFrame bị chuyển thành ndarray.",
                            suggestion="Đổi thứ tự transformer.",
                        )
                    ],
                )
            ]
        )

        with patch(
            "nodes.review_pipeline_node.review_pipeline_llm",
            fake,
        ):
            result = review_pipeline_node(make_state())

        self.assertEqual(result["pipeline_review_status"], "invalid")
        self.assertEqual(result["validation_cell_status"], "invalid")
        self.assertEqual(len(result["validation_cell_errors"]), 1)

    def test_unknown_cell_id_fails_review(self) -> None:
        fake = FakeRunnable(
            [
                PipelineReviewResult(
                    status="invalid",
                    summary="Pipeline có lỗi.",
                    errors=[
                        PipelineReviewError(
                            cell_id="missing_cell",
                            error_type="other",
                            message="Không ánh xạ được cell.",
                            suggestion="Kiểm tra cell ID.",
                        )
                    ],
                )
            ]
        )

        with patch(
            "nodes.review_pipeline_node.review_pipeline_llm",
            fake,
        ):
            result = review_pipeline_node(make_state())

        self.assertEqual(result["pipeline_review_status"], "failed")
        self.assertIn("không tồn tại", result["error"])

    def test_llm_exception_fails_review(self) -> None:
        fake = FakeRunnable([RuntimeError("API unavailable")])

        with patch(
            "nodes.review_pipeline_node.review_pipeline_llm",
            fake,
        ):
            result = review_pipeline_node(make_state())

        self.assertEqual(result["pipeline_review_status"], "failed")
        self.assertIn("API unavailable", result["error"])


if __name__ == "__main__":
    unittest.main()
