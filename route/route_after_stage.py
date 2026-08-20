from state import State


def route_after_stage(state: State) -> str:
    return "failed" if state.get("error") else "success"
