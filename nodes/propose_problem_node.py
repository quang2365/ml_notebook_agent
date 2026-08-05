from langchain_core.messages import AIMessage

from state import State

def propose_problem_node(state: State) -> dict:
    dataset_summary = state.get("summary")

    target_candidates = (dataset_summary.get("target_candidates"))
    if not target_candidates:
        problem_type = "không thể đề xuất taret candidates, chờ người dùng nhập"
    else: 
        best_candidate = target_candidates[0]
        target_column = best_candidate.get("column")
        problem_type = best_candidate.get("suggested_problem_type","unknown")
        reasons = best_candidate.get("reasons")
        candidate_score = best_candidate.get("score")
        unique_count =  best_candidate.get("unique_count")
    proposal = {
        "target_column": target_column,
        "problem_type": problem_type,
        "candidate_score": candidate_score,
        "unique_count": unique_count,
        "reasons": reasons,
        "requires_user_confirmation": True,
    }

    reasons_text = "\n".join(
        f"- {reason}"
        for reason in reasons
    )

    message = (
        "## Đề xuất bài toán Machine Learning\n\n"
        f"- **Target đề xuất:** `{target_column}`\n"
        f"- **Loại bài toán:** `{problem_type}`\n"
        f"- **Số giá trị khác nhau của target:** "
        f"{unique_count:,}\n\n"
        "### Lý do đề xuất\n\n"
        f"{reasons_text}\n\n"
        "> Đây chỉ là đề xuất tự động và cần được "
        "người dùng xác nhận."
    )

    return {
        "problem_proposal": proposal,
        "approval_status": "pending",
        "error": None,
        "messages": [
            AIMessage(content=message)
        ],
    }