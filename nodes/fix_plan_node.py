import json

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)

from model.model import llm
from model.structured_output import build_structured_llm, invoke_structured
from schemas.notebook_plan_schema import NotebookPlan
from state import State


fix_plan_llm = build_structured_llm(llm,
    NotebookPlan,
)


def fix_plan_node(state: State) -> dict:
    old_plan = state.get("notebook_plan")
    errors = (
        state.get("plan_validation_errors")
        or []
    )
    attempts = state.get(
        "fix_plan_attempts",
        0,
    )

    if not old_plan:
        return {
            "plan_validation_status": "invalid",
            "fix_plan_attempts": attempts + 1,
            "error": "No old plan to fix.",
        }

    if not errors:
        return {
            "plan_validation_status": "invalid",
            "fix_plan_attempts": attempts + 1,
            "error": "No plan errors to fix.",
        }

    prompt = f"""
OLD NOTEBOOK PLAN:

{json.dumps(
    old_plan,
    ensure_ascii=False,
    indent=2,
)}

ALL VALIDATION ERRORS:

{json.dumps(
    errors,
    ensure_ascii=False,
    indent=2,
)}

Fix the notebook plan based on all the errors above.

Rules:
- Keep target_column unchanged.
- Keep problem_type unchanged.
- Only fix the part causing errors.
- Have 8 to 10 sections.
- section_id must be sequential.
- Each section must have 1 to 5 tasks.
- Return the entire plan after fixing.
"""

    try:
        fixed_plan = invoke_structured(fix_plan_llm, llm, NotebookPlan,
            [
                SystemMessage(
                    content=(
                        "You are a plan-fixing agent "
                        "Machine Learning Notebook."
                    )
                ),
                HumanMessage(content=prompt),
            ]
        )

        return {
            "notebook_plan": fixed_plan.model_dump(),
            "plan_validation_status": "pending",
            "plan_validation_errors": None,
            "fix_plan_attempts": attempts + 1,
            "error": None,
            "messages": [
                AIMessage(
                    content="Notebook plan has been fixed."
                )
            ],
        }

    except Exception as exc:
        return {
            "plan_validation_status": "invalid",
            "fix_plan_attempts": attempts + 1,
            "error": f"Unable to fix plan: {exc}",
            "messages": [
                AIMessage(
                    content=f"Unable to fix plan: {exc}"
                )
            ],
        }
