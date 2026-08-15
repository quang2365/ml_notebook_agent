from typing import TypedDict, Annotated, Literal

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class State(TypedDict):
    ##__________________Messages chung_____________________##
    #message của người dùng/ hệ thông/ Tool/ AI
    messages: Annotated[list[BaseMessage],add_messages]
    ##____________State liên quan đến dataset______________##
    #đường dẫn của dataset
    dataset_path: str | None

    #kết quả tổng quan dataset
    summary: dict | None

    #kết quả tổng quan dataset thông qua llm
    summary_llm: str | None

    problem_proposal: dict | None

    # Target đã được người dùng xác nhận
    target_column: str | None
    ##___________________User Interrupt_____________##
    # Loại bài toán đã xác nhận
    problem_type: (Literal["regression", "classification"]| None)

    # Trạng thái xác nhận
    approval_status: (Literal["pending", "approved", "rejected"]| None)

    # Phản hồi bổ sung từ người dùng
    user_feedback: str | None

    # Phân tích chi tiết target
    target_analysis: dict | None
    ##_________________State liên quan đến plan notebook_______##
    #plan cho việc huấn luyện
    notebook_plan: dict | None
    #trạng thái validate của Plan
    plan_validation_status: Literal["pending","valid","invalid",] | None
    #các lỗi của plan
    plan_validation_errors: list[dict] | None
    #số lần sửa của plan
    plan_fix_attempts: int
    ##__________________State liên quan đến cells______________##
    #lưu trữ các cell cho notebook jupyter
    notebook_cells: list[dict] | None

    #Lỗi xảy ra khi đọc phân tích dataset
    error: str | None

    #tính trạng sinh code 
    generation_cell_status: Literal["pending","success","failed",] | None

    #tình trạng xác thực code
    validation_cell_status: Literal["pending","valid","invalid",] | None

    ##____________________State liên quan đến việc build cell_____________##
    #đường dẫn của output notebook 
    notebook_path: str | None
    #tình trạng của notebook
    build_status: Literal["pending","success","failed",] | None

    #các lỗi của cells sau khi xác thực
    validation_errors: list[dict] | None

    #số lượng lần sửa lỗi
    fix_attempts: int

    #các cell_id chứa lỗi
    fixed_cell_ids: list[str] | None

    fix_failures: list[dict] | None
