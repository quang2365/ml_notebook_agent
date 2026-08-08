from state import State
def route_after_generation(
    state: State,
) -> str:

    status = state.get(
        "generation_status"
    )

    if status == "success":
        return "success"

    return "failed"