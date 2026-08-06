from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import interrupt
from state import State

allow_problem_types = ["regression", "classification"]

def review_problem_node(state: State) -> dict:
    proposal = state.get("problem_proposal")
    summary = state.get("summary")
    decisions = interrupt(
        {
            "title": "Xác nhận bài toán Machine Learning",
            "proposal": proposal,
            "instructions": {
                "approve": {
                    "action": "approve",
                },
                "edit": {
                    "action": "edit",
                    "target_column": "Tên target mới",
                    "problem_type": (
                        "regression hoặc classification"
                    ),
                },
                "reject": {
                    "action": "reject",
                    "feedback": "Lý do từ chối",
                },
            },
        }
    )
    decision = decisions.get("action")
    if decision == "approve":
        target_column = proposal.get("target_column")
        problem_type = proposal.get("problem_type")
        return {
            "target_column": target_column,
            "problem_type": problem_type,
            "approval_status": "approved",
            "messages": [HumanMessage(content=f"bài toán tôi xác nhận là {problem_type}, với cột target là {target_column}"),AIMessage(content=f"Người dùng đã xác nhận bài toán: target là `{target_column}`, loại bài toán là `{problem_type}`")],
        }
    if decision == "edit":
        target_column = decisions.get("target_column")
        problem_type = decisions.get("problem_type")

        dataset_columns = summary.get("column_names", [])

        if target_column not in dataset_columns:
            error_message = f"Cột target `{target_column}` không tồn tại trong dataset."
            return{
                "approval_status": "rejected",
                "error": error_message,
                "messages": [AIMessage(content=error_message)]
            }
        if problem_type not in allow_problem_types:
            error_message = f"Loại bài toán `{problem_type}` không hợp lệ. Vui lòng chọn một trong các loại bài toán hợp lệ: {', '.join(allow_problem_types)}."
            return{
                "approval_status": "rejected",
                "error": error_message,
                "messages": [AIMessage(content=error_message)]
            }
        return {
            "target_column": target_column,
            "problem_type": problem_type,
            "approval_status": "approved",
            "messages": [HumanMessage(content=f"bài toán tôi xác nhận là {problem_type}, với cột target là {target_column}"),AIMessage(content=f"Người dùng đã xác nhận bài toán: target là `{target_column}`, loại bài toán là `{problem_type}`")],
        }
    if decision == "reject":
        return {
            "approval_status": "rejected",
            "messages": [HumanMessage(content="Tôi từ chối đề xuất bài toán"),AIMessage(content="Người dùng đã từ chối đề xuất bài toán")]
        }