from state import State
MAX_SECTION_RETRY_ATTEMPTS = 3

def route_after_section_generation(state: State) -> str:
    attempts = state.get("section_retry_attempts")
    status = state.get("section_generation_status")
    if status == "failed":
        if attempts < MAX_SECTION_RETRY_ATTEMPTS:
            return "retry"
        return "failed"
    sections = ((state.get("notebook_plan") or {}).get("sections") or [])

    current_index = state.get("current_section_index", 0)

    if current_index < len(sections):
        return "continue"

    return "complete"
