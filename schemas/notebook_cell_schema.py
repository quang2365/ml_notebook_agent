from typing import Literal

from pydantic import BaseModel, Field


class NotebookCell(BaseModel):
    cell_id: str = Field(
        description="Mã duy nhất của cell."
    )

    section_id: str = Field(
        description="Section mà cell thuộc về."
    )
    cell_type: Literal["markdown", "code"] = Field(
        description="Loại của cell: markdown hoặc code."
    )
    title: str = Field(
        description="Tên ngắn mô tả cell."
    )

    source: str = Field(
        description="Nội dung đầy đủ của cell."
    )

    purpose: str = Field(
        description="Mục đích của cell."
    )

    expected_output: str | None = Field(
        default=None,
        description=(
            "Mô tả kết quả dự kiến khi chạy code cell. "
            "Markdown cell có thể để null."
        ),
    )


class GeneratedSection(BaseModel):
    section_id: str = Field(
        description=(
            "ID của section đang được generate."
        )
    )

    cells: list[NotebookCell] = Field(
        description=(
            "Danh sách các notebook cell "
            "thuộc section này."
        )
    )
