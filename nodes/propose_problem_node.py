from langchain_core.messages import AIMessage

from state import State

def propose_problem_node(state: State) -> dict:
    dataset_summary = state.get("summary") or {}
    target_candidates = (
        dataset_summary.get("target_candidates")
        or []
    )
    if not target_candidates:

        error_message = (
            "No target candidate was found in the dataset."
        )
        return {
            "problem_proposal": None,
            "approval_status": "rejected",
            "error": error_message,
            "messages": [AIMessage(content=error_message)],
        }

    best_candidate = target_candidates[0]
    target_column = best_candidate.get("column")
    problem_type = best_candidate.get(
        "suggested_problem_type",
        "unknown",
    )
    reasons = best_candidate.get("reasons") or []
    candidate_score = best_candidate.get("score")
    unique_count = best_candidate.get("unique_count")
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
        "## Machine Learning Problem Proposal\n\n"
        f"- **Proposed target:** `{target_column}`\n"
        f"- **Problem type:** `{problem_type}`\n"
        f"- **Number of unique target values:** "
        f"{unique_count if unique_count is not None else 'unknown'}\n\n"
        "### Proposal reasons\n\n"
        f"{reasons_text}\n\n"
        "> This is just an automatic suggestion and needs to be "
        "confirmed by the user."
    )

    return {
        "problem_proposal": proposal,
        "approval_status": "pending",
        "error": None,
        "messages": [
            AIMessage(content=message)
        ],
    }
