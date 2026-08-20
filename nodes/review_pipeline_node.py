import json

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)

from model.model import llm
from model.structured_output import build_structured_llm, invoke_structured
from schemas.pipeline_review_schema import (
    PipelineReviewResult,
)
from state import State

review_pipeline_llm = build_structured_llm(llm,
    PipelineReviewResult,
)


PIPELINE_REVIEW_SYSTEM_PROMPT = """
You are an expert in reviewing Python Machine Learning notebooks.

Your task is to evaluate the correctness of the ENTIRE
Machine Learning pipeline based on:

1. Dataset context.
2. Target column.
3. Problem type.
4. Notebook plan.
5. All code cells in the correct execution order.

The previous Python validator checked the structure, syntax, and a
part of the dependency. You must focus on SEMANTIC errors and
interaction errors between cells.

ERROR GROUPS TO CHECK:

1. DATA LEAKAGE

- Do not fit preprocessing on test data.
- Do not use the target as a feature.
- Do not use test data to select the model.
- The train/test split must occur before the steps that learn
  parameters from data.

2. PREPROCESSING

- Train and test must use the same fitted preprocessor.
- Do not call fit_transform separately on test data.
- Feature engineering must be consistent between train and test.
- The pipeline must be compatible with the data type passed between
  steps.

3. PANDAS AND NUMPY COMPATIBILITY

- If FunctionTransformer accesses a column with the syntax
  X["column"], its input must be a pandas DataFrame.
- SimpleImputer, StandardScaler, or ColumnTransformer
  can convert a DataFrame to a NumPy array.
- Do not place a FunctionTransformer that uses column names after a
  transformer that converted the data to a NumPy array, unless
  the notebook guarantees pandas output.
- The preferred structure is:

  Pipeline([
      ("feature_engineering", FunctionTransformer(...)),
      ("columns", ColumnTransformer(...)),
  ])

4. VARIABLE CONSISTENCY

- A variable must be created before it is used.
- Model names must be consistent among trained_models,
  predictions and model_results.
- Do not accidentally use the model's metric variable before or
  model sau.
- Do not silently overwrite important data with an
  incompatible data type.

5. MODEL TRAINING

- The model must be appropriate for regression or classification.
- The model must be fitted on the correct training features and target.
- Prediction must use the correct fitted model.
- Do not create fake metrics or predictions.

6. METRICS AND MODEL COMPARISON

- The metric must be appropriate for the problem type.
- Each model must store its own metric immediately after evaluation.
- Do not use the final model's metric for all models.
- The model_results table must be created from actual results.
- The selection of best_model must be based on the appropriate metric.

7. TARGET AND DATASET

- The target must match the provided TARGET COLUMN.
- Do not arbitrarily change the dataset path.
- Do not change the problem type.
- Do not include the target in feature preprocessing.

RULES FOR RETURNING RESULTS:

1. Only return status="valid" when the entire pipeline is consistent.
2. When status="valid", errors must be an empty list.
3. When status="invalid", errors must contain at least one error.
4. Each error must indicate the exact cell_id that needs to be fixed.
5. cell_id must exist in the list of code cells.
6. Do not require fixing markdown cells.
7. Do not report errors based only on speculation without evidence.
8. suggestion must describe the smallest possible fix.
9. Do not rewrite the code of the notebook.
10. Do not return a Markdown code fence.
"""


def review_pipeline_node(state: State) -> dict:
    """
    Perform semantic evaluation of the entire notebook pipeline after
    the cells have passed static validation.
    """

    cells = state.get("notebook_cells") or []


    # irrelevant; it interferes with the evaluation process.
    code_cells = [
        {
            "cell_id": cell.get("cell_id"),
            "section_id": cell.get("section_id"),
            "title": cell.get("title"),
            "purpose": cell.get("purpose"),
            "source": normalize_source(
                cell.get("source")
            ),
        }
        for cell in cells
        if cell.get("cell_type") == "code"
    ]

    if not code_cells:
        message = (
            "No code cell to evaluate the pipeline."
        )

        return {
            "pipeline_review_status": "failed",
            "pipeline_review_errors": [
                {
                    "cell_id": "unknown",
                    "error_type": "other",
                    "message": message,
                    "suggestion": (
                        "Check the generate_cells step again."
                    ),
                    "related_cell_ids": [],
                }
            ],
            "error": message,
            "messages": [
                AIMessage(content=message)
            ],
        }

    review_context = {
        "dataset": {
            "dataset_path": state.get(
                "dataset_path"
            ),
            "summary": state.get("summary"),
            "summary_llm": state.get(
                "summary_llm"
            ),
        },
        "problem": {
            "target_column": state.get(
                "target_column"
            ),
            "problem_type": state.get(
                "problem_type"
            ),
            "target_analysis": state.get(
                "target_analysis"
            ),
        },
        "notebook_plan": state.get(
            "notebook_plan"
        ),
        "code_cells": code_cells,
    }

    try:
        result = invoke_structured(review_pipeline_llm, llm, PipelineReviewResult,
            [
                SystemMessage(
                    content=(
                        PIPELINE_REVIEW_SYSTEM_PROMPT
                    )
                ),
                HumanMessage(
                    content=(
                        "Please evaluate the following pipeline:\n\n"
                        + json.dumps(
                            review_context,
                            ensure_ascii=False,
                            default=str,
                        )
                    )
                ),
            ]
        )

        result_dict = result.model_dump()
        status = result_dict["status"]
        errors = result_dict.get("errors") or []


        # Normalize status based on the actual error list.
        if errors:
            status = "invalid"
        else:
            status = "valid"


        valid_cell_ids = {
            cell["cell_id"]
            for cell in code_cells
            if cell.get("cell_id")
        }

        normalized_errors = []
        invalid_cell_references = []

        for review_error in errors:
            cell_id = review_error.get("cell_id")

            if cell_id not in valid_cell_ids:
                invalid_cell_references.append(cell_id)
                continue

            normalized_errors.append(
                {
                    "cell_id": cell_id,
                    "error_type": review_error.get(
                        "error_type",
                        "other",
                    ),
                    "message": review_error.get(
                        "message",
                        "",
                    ),
                    "suggestion": review_error.get(
                        "suggestion",
                        "",
                    ),
                    "related_cell_ids": [
                        related_id
                        for related_id in review_error.get(
                            "related_cell_ids",
                            [],
                        )
                        if related_id in valid_cell_ids
                    ],

                    # semantic errors detected by the pipeline reviewer.
                    "source": "pipeline_review",
                }
            )


        # reviewer returned a nonexistent cell_id.
        if invalid_cell_references:
            invalid_ids = ", ".join(
                str(cell_id)
                for cell_id in invalid_cell_references
            )
            message = (
                "Pipeline reviewer returned a cell_id that does not exist: "
                f"{invalid_ids}."
            )
            return {
                "pipeline_review_status": "failed",
                "pipeline_review_errors": [],
                "error": message,
                "messages": [AIMessage(content=message)],
            }


        final_status = (
            "invalid"
            if normalized_errors
            else "valid"
        )

        summary = result_dict.get(
            "summary",
            "",
        )

        return {
            "pipeline_review_status": final_status,
            "pipeline_review_errors": (
                normalized_errors
            ),

            # fix_cells_node is currently reading.
            "validation_cell_errors": (
                normalized_errors
                if final_status == "invalid"
                else []
            ),
            "validation_cell_status": (
                "invalid"
                if final_status == "invalid"
                else "valid"
            ),
            "error": (
                None
                if final_status == "valid"
                else (
                    "LLM detected "
                    f"{len(normalized_errors)} errors "
                    "trong Machine Learning pipeline."
                )
            ),
            "messages": [
                AIMessage(
                    content=build_review_message(
                        status=final_status,
                        summary=summary,
                        errors=normalized_errors,
                    )
                )
            ],
        }

    except Exception as exc:
        message = (
            "Unable to evaluate notebook pipeline: "
            f"{exc}"
        )

        return {
            "pipeline_review_status": "failed",
            "pipeline_review_errors": [],
            "error": message,
            "messages": [
                AIMessage(content=message)
            ],
        }


def normalize_source(source: str | list | None) -> str:
    """Normalize the source of a notebook cell into a string."""

    if source is None:
        return ""

    if isinstance(source, list):
        return "".join(source)

    return str(source)


def build_review_message(
    status: str,
    summary: str,
    errors: list[dict],
) -> str:
    """Create a short message to display the review result."""

    if status == "valid":
        return (
            "# Pipeline Review\n\n"
            "Pipeline passed semantic evaluation "
            "by the LLM.\n\n"
            f"{summary}"
        )

    lines = [
        "# Pipeline Review",
        "",
        (
            f"Detected **{len(errors)} semantic "
            "errors**."
        ),
        "",
    ]

    for index, review_error in enumerate(
        errors,
        start=1,
    ):
        lines.extend(
            [
                (
                    f"## {index}. "
                    f"{review_error['cell_id']}"
                ),
                (
                    "- Type: "
                    f"`{review_error['error_type']}`"
                ),
                (
                    "- Message: "
                    f"{review_error['message']}"
                ),
                (
                    "- Suggestion: "
                    f"{review_error['suggestion']}"
                ),
                "",
            ]
        )

    return "\n".join(lines)
