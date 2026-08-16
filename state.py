from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class State(TypedDict):
    # ==================================================
    # 1. MESSAGES
    # ==================================================

    # Lịch sử message của người dùng, LLM, system và tool.
    messages: Annotated[
        list[BaseMessage],
        add_messages,
    ]

    # ==================================================
    # 2. DATASET
    # ==================================================

    # Đường dẫn tới dataset.
    dataset_path: str | None

    # Kết quả inspect dataset bằng Python.
    summary: dict | None

    # Kết quả phân tích dataset bằng LLM.
    summary_llm: str | None

    # ==================================================
    # 3. PROBLEM PROPOSAL VÀ HUMAN REVIEW
    # ==================================================

    # Đề xuất bài toán từ LLM.
    problem_proposal: dict | None

    # Target đã được người dùng xác nhận.
    target_column: str | None

    # Loại bài toán đã được xác nhận.
    problem_type: (Literal["regression", "classification"]| None)

    # Trạng thái xác nhận của người dùng.
    approval_status: (Literal["pending", "approved", "rejected"]| None)

    # Phản hồi bổ sung của người dùng.
    user_feedback: str | None

    # Phân tích chi tiết target.
    target_analysis: dict | None

    # ==================================================
    # 4. NOTEBOOK PLAN
    # ==================================================

    # Kế hoạch notebook.
    notebook_plan: dict | None

    # Trạng thái validation của plan.
    plan_validation_status: (Literal["pending", "valid", "invalid"]| None)

    # Danh sách lỗi của plan.
    plan_validation_errors: list[dict] | None

    # Số vòng đã sửa plan.
    fix_plan_attempts: int

    # ==================================================
    # 5. CELL GENERATION
    # ==================================================

    # Danh sách cell do LLM sinh ra.
    notebook_cells: list[dict] | None

    # Trạng thái sinh cells.
    section_generation_status: (Literal["pending", "success", "failed"]| None)

    # Danh sách lỗi phát sinh khi sinh từng section/cell.
    section_generation_errors: list[dict] | None

    # Vị trí section tiếp theo cần được sinh.
    current_section_index: int

    # ID của các section đã được sinh thành công.
    generated_section_ids: list[str]

    # Số lần retry API của section hiện tại.
    section_retry_attempts: int

    # ==================================================
    # 6. CELL VALIDATION VÀ REPAIR
    # ==================================================

    # Trạng thái validation của cells.
    validation_cell_status: (
        Literal["pending", "valid", "invalid"]
        | None
    )

    # Danh sách lỗi của cells.
    validation_cell_errors: list[dict] | None

    # Số vòng đã sửa cells.
    fix_cell_attempts: int

    # ID của những cell đã sửa thành công.
    fixed_cell_ids: list[str] | None

    # Danh sách những cell không sửa được.
    fix_cell_failures: list[dict] | None

    # ==================================================
    # 7. NOTEBOOK BUILD
    # ==================================================

    # Đường dẫn file notebook đầu ra.
    notebook_path: str | None

    # Trạng thái tạo file notebook.
    build_status: (
        Literal["pending", "success", "failed"]
        | None
    )

    # Lỗi riêng xảy ra trong quá trình tạo file notebook.
    build_error: str | None

    # ==================================================
    # 8. GLOBAL ERROR
    # ==================================================

    # Lỗi tổng quát gần nhất của workflow.
    error: str | None
