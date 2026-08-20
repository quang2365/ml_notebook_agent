from state import State


MAX_EXECUTION_FIX_ATTEMPTS = 4


def route_after_execution(state: State,) -> str:
    status = state.get("execution_status")
    attempts = state.get("execution_fix_attempts",0,)
    execution_error = (state.get("execution_error")or {})

    if status == "success":
        return "success"

    if (status == "failed"and attempts < MAX_EXECUTION_FIX_ATTEMPTS and execution_error.get("cell_id")):
        return "fix"

    return "failed"