from state import State


MAX_PLAN_FIX_ATTEMPTS = 3


def route_after_plan_validation(
    state: State,
) -> str:

    status = state.get(
        "plan_validation_status"
    )

    attempts = state.get(
        "fix_plan_attempts",
        0,
    )

    if status == "valid":
        return "valid"

    if (
        status == "invalid"
        and attempts < MAX_PLAN_FIX_ATTEMPTS
    ):
        return "fix"

    return "failed"
