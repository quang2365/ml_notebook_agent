from typing import Literal

from pydantic import BaseModel, Field


class PipelineReviewError(BaseModel):
    cell_id: str = Field(
        description=(
            "ID of the cell to edit. "
            "Must exist in the notebook."
        )
    )

    error_type: Literal[
        "data_leakage",
        "pipeline_incompatibility",
        "inconsistent_variable",
        "invalid_model_flow",
        "invalid_metric",
        "invalid_preprocessing",
        "invalid_feature_engineering",
        "invalid_evaluation",
        "wrong_problem_type",
        "wrong_target",
        "other",
    ] = Field(description="the error groups that may appear")

    message: str = Field(description="description of specific errors")

    suggestion: str = Field(description="suggested fix without directly generating code")

    related_cell_ids: list[str] = Field(
        default_factory=list
    )


class PipelineReviewResult(BaseModel):

    status: Literal["valid", "invalid"]

    summary: str

    errors: list[PipelineReviewError] = Field(
        default_factory=list
    )
