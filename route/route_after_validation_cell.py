MAX_FIX_ATTEMPTS = 3

from state import State
def route_after_validation_cell(
    state: State,
) -> str:

    status = state.get(
        "validation_status"
    )

    attempts = state.get(
        "fix_attempts",
        0,
    )

    cells = state.get(
        "notebook_cells"
    )

    # Không có cells thì fixer
    # không có gì để sửa
    if not cells:
        return "failed"

    if status == "valid":
        return "valid"

    if (
        status == "invalid"
        and attempts < MAX_FIX_ATTEMPTS
    ):
        return "fix"

    return "failed"