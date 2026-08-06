from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import interrupt
from state import State

allow_problem_types = ["regression", "classification"]

def review_problem_node(state: State) -> dict:
    proposal = state.get("problem_proposal")
    summary = state.get("summary")
    decision = interrupt({
        "title":"Xác nhận bài toán Machine Learning",
        "proposal": proposal,
        "alow_action":["approve","edit","reject"]})
    decision = decision.get("action")
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
        target_column = proposal.get("target_column")
        problem_type = proposal.get("problem_type")

        dataset_columns = summary.get("columns", [])

        if target_column not in dataset_columns:
            error_message = f"Cột target `{target_column}` không tồn tại trong dataset."
            return{
                "approval_status": "rejected",
                "error": error_message,
                "messages": [AIMessage(content=error_message)]
            }