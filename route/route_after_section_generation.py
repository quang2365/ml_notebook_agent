from state import State


def route_after_section_generation(state: State) -> str:
    """Continue section generation, validate completed cells, or stop."""
    if state.get("generation_cell_status") == "failed":
        return "failed"

    sections = (
        (state.get("notebook_plan") or {}).get("sections")
        or []
    )
    current_index = state.get("current_section_index", 0)

    if current_index < len(sections):
        return "continue"

    return "complete"
