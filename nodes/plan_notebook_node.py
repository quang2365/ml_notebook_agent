import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from validators.plan_validator import validate_notebook_plan

from model.model import llm
from schemas.notebook_plan_schema import NotebookPlan
from state import State


structured_llm = llm.with_structured_output(
    NotebookPlan,
    method="function_calling",
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
        Bạn là một Senior Data Scientist chuyên thiết kế
        kế hoạch Jupyter Notebook cho bài toán Machine Learning.

        Nhiệm vụ của bạn là lập KẾ HOẠCH notebook.
        KHÔNG viết code hoàn chỉnh.

        Notebook phải ngắn gọn, có cấu trúc rõ ràng
        và có thể được thực thi tuần tự từ trên xuống dưới.

        QUY TẮC BẮT BUỘC:

        1. Notebook phải có từ 8 đến 10 sections.

        2. section_id phải tuần tự:
        section_1
        section_2
        section_3
        ...

        3. Không được tạo section thừa hoặc chia một
        nhiệm vụ nhỏ thành quá nhiều section.

        4. Thứ tự pipeline Machine Learning phải hợp lý.

        5. Dataset phải được load trước khi sử dụng.

        6. EDA cơ bản phải xảy ra trước modeling.

        7. Phải tách features X và target y.

        8. Train/test split phải xảy ra TRƯỚC mọi bước
        preprocessing có học tham số từ dữ liệu như:
        - imputation
        - scaling
        - encoding
        - feature selection
        - dimensionality reduction
        - learned transformation

        9. Preprocessor chỉ được fit trên training set.

        10. Không được fit preprocessing trên toàn bộ
            dataset trước khi train/test split.

        11. Feature engineering dựa trên công thức cố định
            có thể được mô tả riêng, nhưng mọi transformation
            học tham số phải fit trên train.

        12. Phải có một baseline model.

        13. Phải có ít nhất hai candidate models để
            so sánh khi phù hợp với bài toán.

        14. Các metric phải phù hợp với problem_type.

        15. Phải có phần đánh giá và so sánh model.

        16. Phải có phần kết luận.

        17. Mỗi section chỉ nên chứa từ 1 đến 5 tasks.

        18. Không tạo metric hoặc kết quả giả.

        19. Không thực thi code.

        20. Không tự thay đổi target_column hoặc
            problem_type đã được xác nhận.
        CẤU TRÚC GỢI Ý:

        section_1:
        - Setup môi trường và import thư viện.

        section_2:
        - Load dataset và kiểm tra cấu trúc cơ bản.

        section_3:
        - Exploratory Data Analysis.

        section_4:
        - Xác định X/y và train/test split.

        section_5:
        - Preprocessing và feature engineering.

        section_6:
        - Baseline model.

        section_7:
        - Candidate model thứ nhất.

        section_8:
        - Candidate model thứ hai.

        section_9:
        - Đánh giá, so sánh và phân tích model.

        section_10:
        - Kết luận và hướng phát triển.

        Có thể gộp các section khi phù hợp,
        nhưng tổng số section phải từ 8 đến 10.
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
                "plan_validation_status": "pending",
                "plan_validation_errors": None,
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
