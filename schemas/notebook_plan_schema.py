from pydantic import BaseModel, Field


class NotebookSection(BaseModel):
    section_id: str = Field(
        description="Mã duy nhất của phần notebook."
    )

    title: str = Field(
        description="Tiêu đề phần notebook."
    )

    objective: str = Field(
        description="Mục tiêu của phần này."
    )

    cell_types: list[str] = Field(
        description=(
            "Danh sách loại cell cần tạo, "
            "ví dụ markdown hoặc code."
        )
    )

    tasks: list[str] = Field(
        description="Các công việc cần thực hiện."
    )


class NotebookPlan(BaseModel):
    notebook_title: str = Field(
        description="Tên notebook."
    )

    target_column: str = Field(
        description="Cột target đã xác nhận."
    )

    problem_type: str = Field(
        description=(
            "Loại bài toán regression "
            "hoặc classification."
        )
    )

    objective: str = Field(
        description="Mục tiêu tổng thể của notebook."
    )

    evaluation_metrics: list[str] = Field(
        description="Các metric sẽ sử dụng."
    )

    candidate_models: list[str] = Field(
        description="Các mô hình dự kiến thử nghiệm."
    )

    sections: list[NotebookSection] = Field(
    min_length=8,
    max_length=10,
    description=(
        "Danh sách từ 8 đến 10 section "
        "của notebook theo đúng thứ tự "
        "thực hiện Machine Learning."
    ),
)