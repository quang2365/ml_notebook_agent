from state import State
def route_after_review(state: State) -> str:
    if state["approval_status"] == "approved":
        return "approved"

    return "rejected"
