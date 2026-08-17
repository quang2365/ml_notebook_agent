from typing import Literal

from pydantic import BaseModel, Field


class PipelineReviewError(BaseModel):
    cell_id: str = Field(
        description=(
            "ID của cell cần sửa. "
            "Phải tồn tại trong notebook."
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
    ] = Field(description="các nhóm lỗi có thể xuất hiện")

    message: str = Field(description="mô tả về lỗi củ thể")

    suggestion: str = Field(description="gợi ý cách sửa lỗi, không trực tiếp sinh code")

    related_cell_ids: list[str] = Field(
        default_factory=list
    )


class PipelineReviewResult(BaseModel):

    status: Literal["valid", "invalid"]

    summary: str

    errors: list[PipelineReviewError] = Field(
        default_factory=list
    )