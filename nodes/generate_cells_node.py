import json

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)

from model.model import llm
from schemas.notebook_cell_schema import GeneratedNotebook
from state import State


structured_cell_llm = llm.with_structured_output(
    GeneratedNotebook
)


def generate_cells_node(state: State) -> dict:
    notebook_plan = state.get("notebook_plan")
    dataset_path = state.get("dataset_path")
    target_column = state.get("target_column")
    problem_type = state.get("problem_type")
    summary = state.get("summary")

    if not notebook_plan:
        error_message = (
            "Không tìm thấy kế hoạch notebook."
        )

        return {
            "notebook_cells": None,
            "error": error_message,
            "messages": [
                AIMessage(content=error_message)
            ],
        }

    if not dataset_path:
        error_message = (
            "Không tìm thấy đường dẫn dataset."
        )

        return {
            "notebook_cells": None,
            "error": error_message,
            "messages": [
                AIMessage(content=error_message)
            ],
        }

    analysis_context = {}

    if summary:
        analysis_context = (
            summary.get("analysis_context")
            or {}
        )

    system_prompt = """
Bạn là chuyên gia Python, Machine Learning và thiết kế
Jupyter Notebook.

Nhiệm vụ của bạn là chuyển kế hoạch notebook thành danh sách
các Markdown cell và Python code cell hoàn chỉnh.

Quy tắc bắt buộc:

1. Cell phải được sắp xếp đúng thứ tự thực thi.
2. Mỗi section nên có ít nhất một Markdown cell mô tả.
3. Python code phải hợp lệ và có thể chạy tuần tự.
4. Không đặt code trong dấu ```python.
5. Không sử dụng biến chưa được khai báo ở cell trước.
6. Phải sử dụng đúng đường dẫn dataset được cung cấp.
7. Phải sử dụng đúng target đã xác nhận.
8. Không thực thi code trong quá trình tạo cell.
9. Không tự tạo kết quả đánh giá giả.
10. Không đưa output giả vào source của cell.
11. Dùng random_state=42 khi phù hợp.
12. Tránh data leakage:
    - chia train/test trước khi fit preprocessing;
    - preprocessing phải fit trên tập train.
13. Với dữ liệu numeric và categorical, ưu tiên Pipeline
    và ColumnTransformer.
14. Phải có baseline model trước các mô hình chính.
15. Metric phải phù hợp với loại bài toán.
"""

    user_prompt = f"""
Hãy sinh các cell notebook dựa trên thông tin dưới đây.

Đường dẫn dataset:
{dataset_path}

Target:
{target_column}

Loại bài toán:
{problem_type}

Kế hoạch notebook:
{json.dumps(
    notebook_plan,
    ensure_ascii=False,
    default=str,
    indent=2,
)}

Bối cảnh dataset:
{json.dumps(
    analysis_context,
    ensure_ascii=False,
    default=str,
    indent=2,
)}
"""

    try:
        generated_notebook = structured_cell_llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )

        generated_dict = (
            generated_notebook.model_dump()
        )

        cells = generated_dict["cells"]

        validation_error = validate_cells(
            cells=cells,
            dataset_path=dataset_path,
            target_column=target_column,
        )

        if validation_error:
            return {
                "error": validation_error,
                "messages": [
                    AIMessage(
                        content=validation_error
                    )
                ],
            }

        return {
            "notebook_cells": cells,
            "messages": [
                AIMessage(
                    content=build_cells_summary(
                        notebook_title=generated_dict[
                            "notebook_title"
                        ],
                        cells=cells,
                    )
                )
            ],
        }

    except Exception as exc:
        error_message = (
            "Không thể tạo notebook cells: "
            f"{exc}"
        )

        return {
            "notebook_cells": None,
            "error": error_message,
            "messages": [
                AIMessage(content=error_message)
            ],
        }


def validate_cells(
    cells: list[dict],
    dataset_path: str,
    target_column: str,
) -> str | None:
    if not cells:
        return "LLM không tạo cell nào."

    cell_ids: set[str] = set()

    for index, cell in enumerate(cells):
        cell_id = cell.get("cell_id")
        cell_type = cell.get("cell_type")
        source = cell.get("source")

        if not cell_id:
            return (
                f"Cell tại vị trí {index} "
                "không có cell_id."
            )

        if cell_id in cell_ids:
            return (
                f"Cell ID `{cell_id}` bị trùng."
            )

        cell_ids.add(cell_id)

        if cell_type not in {
            "markdown",
            "code",
        }:
            return (
                f"Cell `{cell_id}` có loại "
                f"`{cell_type}` không hợp lệ."
            )

        if not isinstance(source, str):
            return (
                f"Source của cell `{cell_id}` "
                "không phải chuỗi."
            )

        if cell_type == "code":
            if "```python" in source:
                return (
                    f"Cell `{cell_id}` chứa "
                    "Markdown code fence."
                )

    all_code = "\n".join(
        cell["source"]
        for cell in cells
        if cell["cell_type"] == "code"
    )

    if dataset_path not in all_code:
        return (
            "Các code cell chưa sử dụng "
            "đúng đường dẫn dataset."
        )

    if target_column not in all_code:
        return (
            "Các code cell chưa sử dụng "
            "target đã xác nhận."
        )

    return None


def build_cells_summary(
    notebook_title: str,
    cells: list[dict],
) -> str:
    markdown_count = sum(
        cell["cell_type"] == "markdown"
        for cell in cells
    )

    code_count = sum(
        cell["cell_type"] == "code"
        for cell in cells
    )

    lines = [
        "# Đã tạo Notebook Cells",
        "",
        f"- **Notebook:** {notebook_title}",
        f"- **Tổng số cell:** {len(cells)}",
        f"- **Markdown cell:** {markdown_count}",
        f"- **Code cell:** {code_count}",
        "",
        "## Danh sách cell",
        "",
    ]

    for index, cell in enumerate(
        cells,
        start=1,
    ):
        lines.append(
            f"{index}. `{cell['cell_type']}` — "
            f"{cell['title']}"
        )

    return "\n".join(lines)