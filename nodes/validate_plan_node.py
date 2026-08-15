from langchain_core.messages import AIMessage

from state import State
from validators.plan_validator import (
    validate_notebook_plan,
)


def validate_plan_node(state: State) -> dict:
    plan = state.get("notebook_plan")

    if not plan:
        error = "Không có notebook plan để kiểm tra."

        return {
            "plan_validation_status": "invalid",
            "plan_validation_errors": [
                {
                    "error_type": "missing_plan",
                    "location": "notebook_plan",
                    "message": error,
                }
            ],
            "error": error,
            "messages": [
                AIMessage(content=error)
            ],
        }

    errors = validate_notebook_plan(
        plan=plan,
        expected_target=state.get("target_column"),
        expected_problem_type=state.get(
            "problem_type"
        ),
    )

    if errors:
        return {
            "plan_validation_status": "invalid",
            "plan_validation_errors": errors,
            "error": (
                f"Notebook plan có {len(errors)} lỗi."
            ),
            "messages": [
                AIMessage(
                    content=(
                        "Notebook plan chưa hợp lệ. "
                        f"Phát hiện {len(errors)} lỗi."
                    )
                )
            ],
        }

    return {
        "plan_validation_status": "valid",
        "plan_validation_errors": [],
        "error": None,
        "messages": [
            AIMessage(
                content="Notebook plan hợp lệ."
            )
        ],
    }