from typing import Literal

from pydantic import BaseModel, Field


class PipelineReviewError(BaseModel):
    # Cell chính gây ra lỗi.
    cell_id: str = Field(
        description=(
            "ID của cell cần sửa. "
            "Phải tồn tại trong notebook."
        )
    )

    # Nhóm lỗi để route/fix dễ xử lý.
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

    # Mô tả lỗi cụ thể.
    message: str = Field(description="mô tả về lỗi củ thể")

    # Gợi ý cách sửa nhưng không trực tiếp sinh code.
    suggestion: str = Field(description="gợi ý cách sửa lỗi, không trực tiếp sinh code")

    # Các cell liên quan đến lỗi, nếu có.
    related_cell_ids: list[str] = Field(
        default_factory=list
    )


class PipelineReviewResult(BaseModel):
    # valid: pipeline hợp lệ.
    # invalid: phát hiện lỗi cần sửa.
    status: Literal["valid", "invalid"]

    # Tóm tắt đánh giá tổng thể.
    summary: str

    # Danh sách lỗi. Phải rỗng khi status là valid.
    errors: list[PipelineReviewError] = Field(
        default_factory=list
    )