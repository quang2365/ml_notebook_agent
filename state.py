from typing import TypedDict, Annotated, Literal

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class State(TypedDict):
    #message của người dùng/ hệ thông/ Tool/ AI
    messages: Annotated[list[BaseMessage],add_messages]

    #đường dẫn của dataset
    dataset_path: str | None

    #kết quả tổng quan dataset
    summary: dict | None

    #kết quả tổng quan dataset thông qua llm
    summary_llm: str | None

    problem_proposal: dict | None

    # Target đã được người dùng xác nhận
    target_column: str | None

    # Loại bài toán đã xác nhận
    problem_type: (
        Literal["regression", "classification"]
        | None
    )

    # Trạng thái xác nhận
    approval_status: (
        Literal["pending", "approved", "rejected"]
        | None
    )

    # Phản hồi bổ sung từ người dùng
    user_feedback: str | None

    # Phân tích chi tiết target
    target_analysis: dict | None

    #plan cho việc huấn luyện
    notebook_plan: dict | None

    #lưu trữ các cell cho notebook jupyter
    notebook_cells: list[dict] | None 
    
    #Lỗi xảy ra khi đọc phân tích dataset
    error: str | None

    validation_status: Literal[
    "pending",
    "valid",
    "invalid",
    ] | None

    validation_errors: list[dict] | None

