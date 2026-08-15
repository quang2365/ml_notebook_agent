import json
import time
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)

from model.model import llm
from schemas.notebook_cell_schema import  GeneratedSection
from state import State



section_llm = llm.with_structured_output(
    GeneratedSection
)
SECTION_SYSTEM_PROMPT = """
Bạn là chuyên gia Python, Machine Learning
và thiết kế Jupyter Notebook.

Nhiệm vụ của bạn là sinh các notebook cell
CHO DUY NHẤT MỘT SECTION được cung cấp.

QUY TẮC BẮT BUỘC:

1. Chỉ sinh cell thuộc section hiện tại.
2. Không sinh nội dung của section khác.
3. Mỗi section nên có từ 2 đến 5 cell.
4. Nên có ít nhất một Markdown cell mô tả section.
5. Python code phải hợp lệ và có thể chạy tuần tự.
6. Không đặt Python code trong Markdown code fence.
7. Không sử dụng biến chưa được khai báo ở các bước trước.
8. Không thay đổi dataset path.
9. Không thay đổi target column.
10. Không thực thi code.
11. Không tạo output hoặc metric giả.
12. Dùng random_state=42 khi phù hợp.
13. Tránh data leakage.
14. Preprocessing phải fit trên train set khi phù hợp.
15. section_id của mọi cell phải đúng với section_id được cung cấp.
16. cell_id phải duy nhất và nên bắt đầu bằng section_id.
17. Phải tuân thủ tuyệt đối VARIABLE CONTRACT được cung cấp.
18. Không tự tạo tên thay thế cho các biến chuẩn
    nếu contract đã quy định tên biến.
19. Chỉ sử dụng biến được tạo ở section hiện tại
    hoặc các section đứng trước.
20. Không giả định tồn tại biến chưa được tạo.
21. Các model nên sử dụng sklearn Pipeline chứa
    preprocessor khi phù hợp để tránh data leakage.
22. Không fit preprocessor riêng trên test set.
23. Không fit preprocessing có học tham số trên
    toàn bộ dataset trước train/test split.
"""
NOTEBOOK_VARIABLE_CONTRACT = """
        VARIABLE CONTRACT CHUNG CHO TOÀN NOTEBOOK:

        Các section phải sử dụng nhất quán các biến sau.

        1. Dataset:

        df
        - DataFrame gốc được đọc từ dataset_path.
        - Không đổi tên thành data, dataset hoặc housing_df.

        target_column
        - Tên cột target đã được người dùng xác nhận.

        2. Features và target:

        X
        - Toàn bộ features trước train/test split.

        y
        - Target.

        3. Train/test split:

        X_train
        X_test
        y_train
        y_test

        Không tự tạo các tên thay thế như:
        X_train_data
        train_X
        features_train
        y_training

        4. Preprocessing:

        preprocessor
        - Đối tượng preprocessing chính.
        - Có thể là ColumnTransformer hoặc Pipeline.

        Không fit preprocessor trên toàn bộ X trước train/test split.

        Ưu tiên đưa preprocessor trực tiếp vào
        sklearn Pipeline cùng model thay vì tạo các biến:

        X_train_processed
        X_test_processed
        X_train_scaled
        X_test_scaled

        trừ khi section plan thực sự yêu cầu.

        5. Quản lý model:

        trained_models = {}
        predictions = {}
        model_results = []

        Mỗi model sau khi train phải được lưu vào:

        trained_models["model_name"]

        Prediction tương ứng:

        predictions["model_name"]

        Metric của model phải được thêm vào:

        model_results.append({
            "model": "...",
            ...
        })

        6. Random state:

        RANDOM_STATE = 42

        Mọi model hoặc train/test split hỗ trợ random_state
        phải sử dụng RANDOM_STATE.

        7. Một section không được sử dụng biến của
        section sau nó.

        8. Không tự đổi tên các biến đã quy định ở trên.
        """
MAX_SECTION_RETRIES = 3
RATE_LIMIT_BASE_DELAY = 15
NORMAL_RETRY_DELAY = 3
def generate_cells_node(
    state: State,
) -> dict:

    notebook_plan = state.get(
        "notebook_plan"
    )

    dataset_path = state.get(
        "dataset_path"
    )

    target_column = state.get(
        "target_column"
    )

    problem_type = state.get(
        "problem_type"
    )

    summary = (
        state.get("summary")
        or {}
    )

    target_analysis = (
        state.get("target_analysis")
        or {}
    )

    # =====================================
    # 1. KIỂM TRA INPUT
    # =====================================

    if not notebook_plan:
        error_message = (
            "Không tìm thấy kế hoạch notebook."
        )

        return {
            "notebook_cells": None,
            "generation_cell_status": "failed",
            "error": error_message,
            "messages": [
                AIMessage(
                    content=error_message
                )
            ],
        }

    if not dataset_path:
        error_message = (
            "Không tìm thấy đường dẫn dataset."
        )

        return {
            "notebook_cells": None,
            "generation_cell_status": "failed",
            "error": error_message,
            "messages": [
                AIMessage(
                    content=error_message
                )
            ],
        }   
    sections = (
        notebook_plan.get("sections")
        or []
    )

    if not sections:
        error_message = (
            "Notebook plan không có section."
        )

        return {
            "notebook_cells": None,
            "generation_cell_status": "failed",
            "error": error_message,
            "messages": [
                AIMessage(
                    content=error_message
                )
        ],
    }

    # =====================================
    # 2. TẠO CONTEXT NHỎ GỌN
    # =====================================

    dataset_context = {
        "column_names":
            summary.get("column_names"),

        "numeric_columns":
            summary.get(
                "numeric_columns"
            ),

        "categorical_columns":
            summary.get(
                "categorical_columns"
            ),

        "possible_id_columns":
            summary.get(
                "possible_id_columns"
            ),

        "target_analysis":
            target_analysis,
    }

    # =====================================
    # 3. NƠI CHỨA TOÀN BỘ CELLS
    # =====================================

    all_cells: list[dict] = []

    # =====================================
    # 4. GENERATE TỪNG SECTION
    # =====================================

    for index, section in enumerate(
        sections,
        start=1,
    ):

        section_id = (
            section.get("section_id")
            or f"section_{index}"
        )

        print(
            f"\nGenerating "
            f"{section_id} "
            f"({index}/{len(sections)})..."
        )

        result = generate_one_section(
            section_id=section_id,
            section=section,
            dataset_path=dataset_path,
            target_column=target_column,
            problem_type=problem_type,
            dataset_context=dataset_context,
        )

        # =================================
        # SECTION THẤT BẠI
        # =================================

        if result["status"] == "failed":

            error_message = (
                "Không thể tạo notebook cells "
                f"cho section `{section_id}`: "
                f"{result['error']}"
            )

            return {
                "notebook_cells": None,
                "generation_cell_status": "failed",
                "error": error_message,
                "messages": [
                    AIMessage(
                        content=error_message
                    )
                ],
            }

        # =================================
        # SECTION THÀNH CÔNG
        # =================================

        section_cells = (
            result.get("cells")
            or []
        )

        all_cells.extend(
            section_cells
        )

        print(
            f"{section_id}: "
            f"{len(section_cells)} cells"
        )   
        if index < len(sections):
            print(
                "Chờ 5 giây trước "
                "section tiếp theo..."
            )
            time.sleep(5)

    # =====================================
    # 5. KIỂM TRA SAU KHI MERGE
    # =====================================

    validation_error = validate_cells(
        cells=all_cells,
        dataset_path=dataset_path,
        target_column=target_column,
    )

    if validation_error:

        return {
            "notebook_cells": all_cells,
            "generation_cell_status": "success",
            "error": validation_error,
            "messages": [
                AIMessage(
                    content=validation_error
                )
            ],
        }

    # =====================================
    # 6. HOÀN TẤT
    # =====================================

    return {
    "notebook_cells": all_cells,
    "generation_cell_status": "success",
    "error": None,
    "messages": [
        AIMessage(
            content=build_cells_summary(
                notebook_title=(
                    notebook_plan.get(
                        "notebook_title"
                    )
                    or "Machine Learning Notebook"
                ),
                cells=all_cells,
            )
        )
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
def generate_one_section(
    section_id: str,
    section: dict,
    dataset_path: str,
    target_column: str,
    problem_type: str,
    dataset_context: dict,
) -> dict:

    user_prompt = f"""
        Hãy tạo các notebook cell cho section sau.

        SECTION ID:
        {section_id}

        SECTION PLAN:
        {json.dumps(
            section,
            ensure_ascii=False,
            default=str,
            indent=2,
        )}

        DATASET PATH:
        {dataset_path}

        TARGET COLUMN:
        {target_column}

        PROBLEM TYPE:
        {problem_type}

        DATASET CONTEXT:
        {json.dumps(
            dataset_context,
            ensure_ascii=False,
            default=str,
            indent=2,
        )}

        VARIABLE CONTRACT:
        {NOTEBOOK_VARIABLE_CONTRACT}
        """
    last_error = None

    for attempt in range(
        1,
        MAX_SECTION_RETRIES + 1,
    ):

        try:
            generated_section = (
                section_llm.invoke(
                    [
                        SystemMessage(
                            content=(
                                SECTION_SYSTEM_PROMPT
                            )
                        ),
                        HumanMessage(
                            content=user_prompt
                        ),
                    ]
                )
            )

            generated_dict = (
                generated_section.model_dump()
            )

            # Kiểm tra section ID
            returned_section_id = (
                generated_dict.get(
                    "section_id"
                )
            )

            if returned_section_id != section_id:
                raise ValueError(
                    "LLM trả sai section_id: "
                    f"{returned_section_id}. "
                    f"Expected: {section_id}"
                )

            cells = (generated_dict.get("cells")or [])

# Không có cell -> section thất bại
            if not cells:
                raise ValueError("LLM không tạo cell nào "f"cho section `{section_id}`.")

            # Kiểm tra các cell trong section
            seen_cell_ids = set()

            for cell in cells:
                cell_id = cell.get("cell_id")
                cell_section_id = (cell.get("section_id"))
                if not cell_id:
                    raise ValueError(f"Section `{section_id}` ""có cell không có cell_id."
                    )
                if (cell_section_id!= section_id):
                    raise ValueError(
                        f"Cell `{cell_id}` "
                        "có section_id "
                        f"`{cell_section_id}` "
                        "nhưng expected là "
                        f"`{section_id}`."
                    )

                if cell_id in seen_cell_ids:
                    raise ValueError(
                        f"cell_id `{cell_id}` "
                        "bị trùng trong "
                        f"section `{section_id}`."
                    )

                seen_cell_ids.add(cell_id)


            return {
                "status": "success",
                "section_id": section_id,
                "cells": cells,
                "error": None,
            }

        except Exception as exc:
            last_error = exc

            print(
                f"[{section_id}] "
                f"attempt "
                f"{attempt}/"
                f"{MAX_SECTION_RETRIES} "
                f"failed: {exc}"
            )

            if attempt < MAX_SECTION_RETRIES:
                error_text = str(exc)

                if (
                    "429" in error_text
                    or "Too Many Requests" in error_text
                ):
                    wait_time = (
                        RATE_LIMIT_BASE_DELAY
                        * (2 ** (attempt - 1))
                    )

                    print(
                        f"[{section_id}] "
                        "NVIDIA rate limit. "
                        f"Chờ {wait_time}s..."
                    )

                    time.sleep(wait_time)

                else:
                    time.sleep(
                        NORMAL_RETRY_DELAY
                    )

    return {
        "status": "failed",
        "section_id": section_id,
        "cells": [],
        "error": str(last_error),
    }