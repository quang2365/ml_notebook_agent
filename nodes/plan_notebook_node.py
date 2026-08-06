import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from model.model import llm
from schemas.notebook_plan_schema import NotebookPlan
from state import State


structured_llm = llm.with_structured_output(
    NotebookPlan
)


def plan_notebook_node(state: State) -> dict:
    summary = state.get("summary")
    target_analysis = state.get("target_analysis")
    target_column = state.get("target_column")
    problem_type = state.get("problem_type")

    if not summary:
        return {
            "notebook_plan": None,
            "error": "Không tìm thấy summary của dataset.",
            "messages": [
                AIMessage(
                    content="Không thể lập kế hoạch notebook."
                )
            ],
        }

    if not target_analysis:
        return {
            "notebook_plan": None,
            "error": "Target chưa được phân tích.",
            "messages": [
                AIMessage(
                    content="Không thể lập kế hoạch notebook."
                )
            ],
        }

    system_prompt = """
Bạn là chuyên gia Machine Learning và thiết kế notebook.

Nhiệm vụ của bạn là tạo một kế hoạch notebook Machine Learning
có cấu trúc rõ ràng, phù hợp với dataset và bài toán đã xác nhận.

Yêu cầu:
- Không viết code hoàn chỉnh.
- Chỉ lập kế hoạch các phần và nhiệm vụ.
- Các phần phải theo đúng thứ tự thực hiện.
- Có tiền xử lý dữ liệu.
- Có chia train/test.
- Có baseline model.
- Có ít nhất hai mô hình để so sánh.
- Có metric phù hợp với loại bài toán.
- Có phần đánh giá và kết luận.
"""

    analysis_context = (
        summary.get("analysis_context")
        or summary
    )

    user_prompt = f"""
Hãy lập kế hoạch notebook cho bài toán sau.

Target:
{target_column}

Loại bài toán:
{problem_type}

Thông tin dataset:
{json.dumps(
    analysis_context,
    ensure_ascii=False,
    default=str,
)}

Phân tích target:
{json.dumps(
    target_analysis,
    ensure_ascii=False,
    default=str,
)}
"""

    try:
        plan = structured_llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )

        plan_dict = plan.model_dump()

        return {
            "notebook_plan": plan_dict,
            "error": None,
            "messages": [
                AIMessage(
                    content=build_plan_message(plan_dict)
                )
            ],
        }

    except Exception as exc:
        error_message = (
            f"Không thể lập kế hoạch notebook: {exc}"
        )

        return {
            "notebook_plan": None,
            "error": error_message,
            "messages": [
                AIMessage(content=error_message)
            ],
        }


def build_plan_message(plan: dict) -> str:
    lines = [
        "# Kế hoạch Notebook",
        "",
        f"## {plan.get('notebook_title')}",
        "",
        f"**Mục tiêu:** {plan.get('objective')}",
        "",
        f"**Target:** `{plan.get('target_column')}`",
        "",
        f"**Loại bài toán:** `{plan.get('problem_type')}`",
        "",
        "## Các mô hình dự kiến",
        "",
    ]

    for model_name in plan.get(
        "candidate_models",
        [],
    ):
        lines.append(f"- {model_name}")

    lines.extend(
        [
            "",
            "## Metric đánh giá",
            "",
        ]
    )

    for metric in plan.get(
        "evaluation_metrics",
        [],
    ):
        lines.append(f"- {metric}")

    lines.extend(
        [
            "",
            "## Các phần trong notebook",
            "",
        ]
    )

    for index, section in enumerate(
        plan.get("sections", []),
        start=1,
    ):
        lines.extend(
            [
                f"### {index}. {section.get('title')}",
                "",
                section.get("objective", ""),
                "",
            ]
        )

        for task in section.get("tasks", []):
            lines.append(f"- {task}")

        lines.append("")

    return "\n".join(lines)