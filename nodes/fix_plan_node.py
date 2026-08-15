import json

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)

from model.model import llm
from schemas.notebook_plan_schema import NotebookPlan
from state import State


fix_plan_llm = llm.with_structured_output(
    NotebookPlan
)


def fix_plan_node(state: State) -> dict:
    old_plan = state.get("notebook_plan")
    errors = (
        state.get("plan_validation_errors")
        or []
    )
    attempts = state.get(
        "plan_fix_attempts",
        0,
    )

    if not old_plan:
        return {
            "plan_validation_status": "invalid",
            "plan_fix_attempts": attempts + 1,
            "error": "Không có plan cũ để sửa.",
        }

    if not errors:
        return {
            "plan_validation_status": "invalid",
            "plan_fix_attempts": attempts + 1,
            "error": "Không có lỗi plan để sửa.",
        }

    prompt = f"""
NOTEBOOK PLAN CŨ:

{json.dumps(
    old_plan,
    ensure_ascii=False,
    indent=2,
)}

TOÀN BỘ LỖI VALIDATION:

{json.dumps(
    errors,
    ensure_ascii=False,
    indent=2,
)}

Hãy sửa notebook plan dựa trên toàn bộ lỗi trên.

Quy tắc:
- Giữ nguyên target_column.
- Giữ nguyên problem_type.
- Chỉ sửa phần gây lỗi.
- Có từ 8 đến 10 sections.
- section_id phải tuần tự.
- Mỗi section có từ 1 đến 5 tasks.
- Trả về toàn bộ plan sau khi sửa.
"""

    try:
        fixed_plan = fix_plan_llm.invoke(
            [
                SystemMessage(
                    content=(
                        "Bạn là agent sửa kế hoạch "
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
            "plan_fix_attempts": attempts + 1,
            "error": None,
            "messages": [
                AIMessage(
                    content="Đã sửa notebook plan."
                )
            ],
        }

    except Exception as exc:
        return {
            "plan_validation_status": "invalid",
            "plan_fix_attempts": attempts + 1,
            "error": f"Không thể sửa plan: {exc}",
            "messages": [
                AIMessage(
                    content=f"Không thể sửa plan: {exc}"
                )
            ],
        }