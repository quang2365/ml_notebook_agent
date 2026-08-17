"""Offline integration tests for the complete LangGraph workflow."""

from __future__ import annotations

import tempfile
import unittest
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
import nbformat

from nbclient.exceptions import (
    CellExecutionError,
)

from schemas.fixed_cell_schema import (
    FixedCell,
)
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
)
from langgraph.types import Command

from graph import build_graph
from schemas.pipeline_review_schema import (
    PipelineReviewResult,
)
from test.fakes import (
    FakeRunnable,
    make_generated_section,
    make_plan,
)


os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"


def make_dataset_summary() -> dict:
    """Giả lập kết quả inspect_dataset lần đầu."""

    return {
        "file_name": "housing.csv",
        "rows": 100,
        "columns": 3,
        "column_names": [
            "feature_1",
            "feature_2",
            "median_house_value",
        ],
        "numeric_columns": [
            "feature_1",
            "feature_2",
            "median_house_value",
        ],
        "categorical_columns": [],
        "total_missing_values": 0,
        "total_missing_percentage": 0,
        "duplicate_rows": 0,
        "possible_id_columns": [],
        "target_candidates": [
            {
                "column": "median_house_value",
                "suggested_problem_type": "regression",
                "score": 3,
                "unique_count": 90,
                "reasons": [
                    "Tên cột chứa từ khóa value",
                ],
            }
        ],
        "analysis_context": {
            "shape": {
                "rows": 100,
                "columns": 3,
            },
            "data_quality": {
                "quality_warnings": [],
            },
            "columns": [
                "feature_1",
                "feature_2",
                "median_house_value",
            ],
        },
    }


def make_target_summary() -> dict:
    """Giả lập kết quả inspect target lần thứ hai."""

    summary = make_dataset_summary()

    summary["target_analysis"] = {
        "column": "median_house_value",
        "problem_type": "regression",
        "unique_count": 90,
        "missing_count": 0,
        "distribution": {
            "mean": 200000.0,
            "median": 180000.0,
            "std": 50000.0,
            "skewness": 0.5,
        },
    }

    return summary


def make_initial_state(
    notebook_path: str,
) -> dict:
    """State tối thiểu để chạy toàn graph."""

    return {
        "messages": [
            HumanMessage(
                content=(
                    "Hãy tạo Machine Learning notebook."
                )
            )
        ],
        "dataset_path": "./data/housing.csv",
        "summary": None,
        "summary_llm": None,
        "problem_proposal": None,
        "target_column": None,
        "problem_type": None,
        "approval_status": None,
        "user_feedback": None,
        "target_analysis": None,
        "notebook_plan": None,
        "plan_validation_status": None,
        "plan_validation_errors": None,
        "fix_plan_attempts": 0,
        "notebook_cells": None,
        "section_generation_status": None,
        "section_generation_errors": [],
        "current_section_index": 0,
        "generated_section_ids": [],
        "section_retry_attempts": 0,
        "validation_cell_status": None,
        "validation_cell_errors": None,
        "fix_cell_attempts": 0,
        "fixed_cell_ids": [],
        "fix_cell_failures": [],
        "pipeline_review_status": "pending",
        "pipeline_review_errors": [],
        "notebook_path": notebook_path,
        "build_status": "pending",
        "build_error": None,
        "execution_status": "pending",
        "execution_error": None,
        "execution_attempts": 0,
        "execution_fix_attempts": 0,
        "error": None,
    }


class CompleteGraphIntegrationTests(
    unittest.TestCase
):
    def test_complete_graph_without_network(
        self,
    ) -> None:
        dataset_tool = MagicMock()


        dataset_tool.invoke.return_value = (
            make_dataset_summary()
        )

        target_tool = MagicMock()


        target_tool.invoke.return_value = (
            make_target_summary()
        )

        analyze_llm = FakeRunnable(
            [
                AIMessage(
                    content=(
                        "Dataset phù hợp với regression."
                    )
                )
            ]
        )

        plan = make_plan(
            section_count=8
        )

        plan_llm = FakeRunnable([plan])

        review_llm = FakeRunnable(
            [
                PipelineReviewResult(
                    status="valid",
                    summary=(
                        "Pipeline Machine Learning hợp lệ."
                    ),
                    errors=[],
                )
            ]
        )

        def fake_generate_one_section(
            section_id: str,
            **kwargs,
        ) -> dict:
            generated = make_generated_section(
                section_id=section_id,
                source=(
                    f"value_{section_id} = "
                    f"'{section_id}'"
                ),
            )

            return {
                "status": "success",
                "cells": [
                    cell.model_dump()
                    for cell in generated.cells
                ],
                "error": None,
            }

        with tempfile.TemporaryDirectory() as directory:
            notebook_path = (
                Path(directory)
                / "integration_test.ipynb"
            )

            with (
                patch(
                    "nodes.inspect_dataset_node."
                    "inspect_dataset",
                    dataset_tool,
                ),
                patch(
                    "nodes.analyze_target_node."
                    "inspect_dataset",
                    target_tool,
                ),
                patch(
                    "nodes.analyze_dataset_node.llm",
                    analyze_llm,
                ),
                patch(
                    "nodes.plan_notebook_node."
                    "structured_llm",
                    plan_llm,
                ),
                patch(
                    "nodes.generate_section_node."
                    "generate_one_section",
                    side_effect=(
                        fake_generate_one_section
                    ),
                ),
                patch(
                    "nodes.review_pipeline_node."
                    "review_pipeline_llm",
                    review_llm,
                ),
                patch(
                    "nodes.execute_notebook_note."
                    "NotebookClient",
                ) as notebook_client,
            ):
                graph = build_graph()

                config = {
                    "configurable": {
                        "thread_id": (
                            "offline-integration-test"
                        )
                    },
                    "recursion_limit": 50,
                }

                # ==============================
                # PHASE 1: CHẠY ĐẾN INTERRUPT
                # ==============================

                interrupted_state = graph.invoke(
                    make_initial_state(
                        str(notebook_path)
                    ),
                    config=config,
                )

                self.assertIn(
                    "__interrupt__",
                    interrupted_state,
                )

                self.assertEqual(
                    interrupted_state[
                        "approval_status"
                    ],
                    "pending",
                )

                # ==============================
                # PHASE 2: HUMAN APPROVE
                # ==============================

                final_state = graph.invoke(
                    Command(
                        resume={
                            "action": "approve",
                        }
                    ),
                    config=config,
                )

            # ==============================
            # ASSERT TOÀN PIPELINE
            # ==============================

            self.assertEqual(
                final_state["approval_status"],
                "approved",
            )

            self.assertEqual(
                final_state[
                    "plan_validation_status"
                ],
                "valid",
            )

            self.assertEqual(
                final_state[
                    "section_generation_status"
                ],
                "success",
            )

            self.assertEqual(
                len(
                    final_state[
                        "generated_section_ids"
                    ]
                ),
                8,
            )

            self.assertEqual(
                final_state[
                    "validation_cell_status"
                ],
                "valid",
            )

            self.assertEqual(
                final_state[
                    "pipeline_review_status"
                ],
                "valid",
            )

            self.assertEqual(
                final_state["build_status"],
                "success",
            )

            self.assertEqual(
                final_state["execution_status"],
                "success",
            )

            self.assertTrue(
                notebook_path.exists()
            )

            notebook_client.assert_called_once()

    def test_graph_repairs_runtime_error(
        self,
    ) -> None:
        dataset_tool = MagicMock()
        dataset_tool.invoke.return_value = (
            make_dataset_summary()
        )

        target_tool = MagicMock()
        target_tool.invoke.return_value = (
            make_target_summary()
        )

        analyze_llm = FakeRunnable(
            [
                AIMessage(
                    content="Offline dataset analysis."
                )
            ]
        )

        plan_llm = FakeRunnable(
            [
                make_plan(section_count=8)
            ]
        )


        # trước execution và sau runtime fix.
        review_llm = FakeRunnable(
            [
                PipelineReviewResult(
                    status="valid",
                    summary="Pipeline hợp lệ.",
                    errors=[],
                ),
                PipelineReviewResult(
                    status="valid",
                    summary=(
                        "Pipeline hợp lệ sau runtime fix."
                    ),
                    errors=[],
                ),
            ]
        )

        runtime_fix_llm = FakeRunnable(
            [
                FixedCell(
                    cell_id="section_1_code_1",
                    source=(
                        "value_section_1 = "
                        "'section_1_fixed'"
                    ),
                    changes=(
                        "Đã sửa lỗi runtime giả lập."
                    ),
                )
            ]
        )

        def fake_generate_one_section(
            section_id: str,
            **kwargs,
        ) -> dict:
            generated = make_generated_section(
                section_id=section_id,
                source=(
                    f"value_{section_id} = "
                    f"'{section_id}'"
                ),
            )

            return {
                "status": "success",
                "cells": [
                    cell.model_dump()
                    for cell in generated.cells
                ],
                "error": None,
            }

        execution_count = {
            "value": 0,
        }

        created_clients = []

        def fake_notebook_client(
            notebook,
            **kwargs,
        ):
            """
            Lần execute đầu tiên gây lỗi.
            Lần execute thứ hai thành công.
            """

            class FakeNotebookClient:
                def execute(
                    self,
                    **execute_kwargs,
                ):
                    execution_count["value"] += 1

                    if execution_count["value"] == 1:
                        failed_cell = (
                            notebook.cells[0]
                        )


                        # trước khi raise CellExecutionError.
                        failed_cell["outputs"] = [
                            nbformat.v4.new_output(
                                output_type="error",
                                ename="NameError",
                                evalue=(
                                    "name 'broken_value' "
                                    "is not defined"
                                ),
                                traceback=[
                                    "Traceback: NameError",
                                    (
                                        "NameError: name "
                                        "'broken_value' "
                                        "is not defined"
                                    ),
                                ],
                            )
                        ]

                        raise CellExecutionError(
                            "Traceback: NameError",
                            "NameError",
                            (
                                "name 'broken_value' "
                                "is not defined"
                            ),
                        )

                    # Lần thứ hai không raise:
                    # notebook được xem là chạy thành công.
                    return notebook

            client = FakeNotebookClient()
            created_clients.append(client)

            return client

        with tempfile.TemporaryDirectory() as directory:
            notebook_path = (
                Path(directory)
                / "runtime_repair_test.ipynb"
            )

            with (
                patch(
                    "nodes.inspect_dataset_node."
                    "inspect_dataset",
                    dataset_tool,
                ),
                patch(
                    "nodes.analyze_target_node."
                    "inspect_dataset",
                    target_tool,
                ),
                patch(
                    "nodes.analyze_dataset_node.llm",
                    analyze_llm,
                ),
                patch(
                    "nodes.plan_notebook_node."
                    "structured_llm",
                    plan_llm,
                ),
                patch(
                    "nodes.generate_section_node."
                    "generate_one_section",
                    side_effect=(
                        fake_generate_one_section
                    ),
                ),
                patch(
                    "nodes.review_pipeline_node."
                    "review_pipeline_llm",
                    review_llm,
                ),
                patch(
                    "nodes.fix_execution_cell_node."
                    "runtime_fix_llm",
                    runtime_fix_llm,
                ),
                patch(
                    "nodes.execute_notebook_note."
                    "NotebookClient",
                    side_effect=(
                        fake_notebook_client
                    ),
                ),
            ):
                graph = build_graph()

                config = {
                    "configurable": {
                        "thread_id": (
                            "runtime-repair-integration"
                        )
                    },
                    "recursion_limit": 50,
                }

                interrupted_state = graph.invoke(
                    make_initial_state(
                        str(notebook_path)
                    ),
                    config=config,
                )

                self.assertIn(
                    "__interrupt__",
                    interrupted_state,
                )

                final_state = graph.invoke(
                    Command(
                        resume={
                            "action": "approve",
                        }
                    ),
                    config=config,
                )

            # ==============================
            # KIỂM TRA RUNTIME REPAIR
            # ==============================


            self.assertEqual(
                execution_count["value"],
                2,
            )


            self.assertEqual(
                len(runtime_fix_llm.calls),
                1,
            )

            self.assertEqual(
                final_state[
                    "execution_fix_attempts"
                ],
                1,
            )

            self.assertEqual(
                final_state[
                    "execution_status"
                ],
                "success",
            )

            self.assertIsNone(
                final_state[
                    "execution_error"
                ]
            )

            self.assertEqual(
                final_state[
                    "validation_cell_status"
                ],
                "valid",
            )

            self.assertEqual(
                final_state[
                    "pipeline_review_status"
                ],
                "valid",
            )

            self.assertEqual(
                final_state[
                    "build_status"
                ],
                "success",
            )

            repaired_cell = next(
                cell
                for cell in final_state[
                    "notebook_cells"
                ]
                if cell.get("cell_id")
                == "section_1_code_1"
            )

            self.assertIn(
                "section_1_fixed",
                repaired_cell["source"],
            )


            notebook_was_created = notebook_path.exists()

        self.assertTrue(
            notebook_was_created
        )

if __name__ == "__main__":
    unittest.main()
