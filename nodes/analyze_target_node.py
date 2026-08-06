from langchain_core.messages import AIMessage

from state import State
from tools.dataset_tools import inspect_dataset


def analyze_target_node(state: State) -> dict:
    dataset_path = state["dataset_path"]
    target_column = state["target_column"]
    problem_type = state["problem_type"]

    try:
        dataset_summary = inspect_dataset.invoke(
            {
                "file_path": dataset_path,
                "target_column": target_column,
                "preview_rows": 5,
                "max_categories": 10,
                "max_profile_columns": 50,
                "max_correlation_pairs": 15,
            }
        )

        target_analysis = dataset_summary.get(
            "target_analysis"
        )

        if not target_analysis:
            raise ValueError(
                "Tool không trả về target_analysis."
            )

        unique_count = target_analysis.get(
            "unique_count"
        )

        missing_count = target_analysis.get(
            "missing_count"
        )

        message_lines = [
            "## Phân tích target đã xác nhận",
            "",
            f"- **Target:** `{target_column}`",
            f"- **Loại bài toán:** `{problem_type}`",
            (
                "- **Số giá trị khác nhau:** "
                f"{unique_count:,}"
            ),
            (
                "- **Số giá trị thiếu:** "
                f"{missing_count:,}"
            ),
        ]

        if problem_type == "regression":
            distribution = (
                target_analysis.get("distribution")
                or {}
            )

            message_lines.extend(
                [
                    "",
                    "### Phân phối target",
                    "",
                    (
                        "- **Trung bình:** "
                        f"{distribution.get('mean')}"
                    ),
                    (
                        "- **Trung vị:** "
                        f"{distribution.get('median')}"
                    ),
                    (
                        "- **Độ lệch chuẩn:** "
                        f"{distribution.get('std')}"
                    ),
                    (
                        "- **Skewness:** "
                        f"{distribution.get('skewness')}"
                    ),
                ]
            )

        elif problem_type == "classification":
            class_distribution = (
                target_analysis.get(
                    "class_distribution"
                )
                or []
            )

            imbalance_ratio = (
                target_analysis.get(
                    "imbalance_ratio"
                )
            )

            message_lines.extend(
                [
                    "",
                    "### Phân bố lớp",
                    "",
                    (
                        "- **Số lớp:** "
                        f"{unique_count}"
                    ),
                    (
                        "- **Tỷ lệ mất cân bằng:** "
                        f"{imbalance_ratio}"
                    ),
                ]
            )

            for item in class_distribution:
                class_name = item.get("class")
                count = item.get("count")
                percentage = item.get("percentage")

                message_lines.append(
                    f"- `{class_name}`: "
                    f"{count:,} mẫu "
                    f"({percentage}%)"
                )

        message_lines.extend(
            [
                "",
                (
                    "Target đã sẵn sàng cho bước "
                    "lập kế hoạch notebook."
                ),
            ]
        )

        return {
            "summary": dataset_summary,
            "target_analysis": target_analysis,
            "error": None,
            "messages": [
                AIMessage(
                    content="\n".join(message_lines)
                )
            ],
        }

    except Exception as exc:
        error_message = str(exc)

        return {
            "target_analysis": None,
            "error": error_message,
            "messages": [
                AIMessage(
                    content=(
                        "Không thể phân tích target: "
                        f"{error_message}"
                    )
                )
            ],
        }