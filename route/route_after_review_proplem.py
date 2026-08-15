from state import State
def route_after_review_proplem(state: State) -> str:
    if state["approval_status"] == "approved":
        return "approved"

    return "rejected"
