"""Shared LLM utilities for generating one notebook section."""

import json
import time

from langchain_core.messages import HumanMessage, SystemMessage

from model.model import llm
from schemas.notebook_cell_schema import GeneratedSection
from state import State


section_llm = llm.with_structured_output(
    GeneratedSection,
    method="function_calling",  #AI: tránh response_format=json_schema
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


def build_dataset_context(state: State) -> dict:
    """Build the compact dataset context shared by every section."""
    summary = state.get("summary") or {}
    target_analysis = state.get("target_analysis") or {}

    return {
        "column_names": summary.get("column_names"),
        "numeric_columns": summary.get("numeric_columns"),
        "categorical_columns": summary.get("categorical_columns"),
        "possible_id_columns": summary.get("possible_id_columns"),
        "target_analysis": target_analysis,
    }


def generate_one_section(
    section_id: str,
    section: dict,
    dataset_path: str,
    target_column: str,
    problem_type: str,
    dataset_context: dict,
) -> dict:
    """Generate and validate one section, retrying transient failures."""
    user_prompt = f"""
Hãy tạo các notebook cell cho section sau.

SECTION ID:
{section_id}

SECTION PLAN:
{json.dumps(section, ensure_ascii=False, default=str, indent=2)}

DATASET PATH:
{dataset_path}

TARGET COLUMN:
{target_column}

PROBLEM TYPE:
{problem_type}

DATASET CONTEXT:
{json.dumps(dataset_context, ensure_ascii=False, default=str, indent=2)}

VARIABLE CONTRACT:
{NOTEBOOK_VARIABLE_CONTRACT}
"""
    last_error = None

    for attempt in range(1, MAX_SECTION_RETRIES + 1):
        try:
            generated_section = section_llm.invoke(
                [
                    SystemMessage(content=SECTION_SYSTEM_PROMPT),
                    HumanMessage(content=user_prompt),
                ]
            )
            generated_dict = generated_section.model_dump()
            returned_section_id = generated_dict.get("section_id")

            if returned_section_id != section_id:
                raise ValueError(
                    "LLM trả sai section_id: "
                    f"{returned_section_id}. Expected: {section_id}"
                )

            cells = generated_dict.get("cells") or []
            if not cells:
                raise ValueError(
                    f"LLM không tạo cell nào cho section `{section_id}`."
                )

            seen_cell_ids: set[str] = set()
            for cell in cells:
                cell_id = cell.get("cell_id")
                cell_section_id = cell.get("section_id")

                if not cell_id:
                    raise ValueError(
                        f"Section `{section_id}` có cell không có cell_id."
                    )

                if cell_section_id != section_id:
                    raise ValueError(
                        f"Cell `{cell_id}` có section_id "
                        f"`{cell_section_id}` nhưng expected là `{section_id}`."
                    )

                if cell_id in seen_cell_ids:
                    raise ValueError(
                        f"cell_id `{cell_id}` bị trùng trong "
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
                f"[{section_id}] attempt {attempt}/"
                f"{MAX_SECTION_RETRIES} failed: {exc}"
            )

            if attempt < MAX_SECTION_RETRIES:
                error_text = str(exc)
                if "429" in error_text or "Too Many Requests" in error_text:
                    wait_time = RATE_LIMIT_BASE_DELAY * (2 ** (attempt - 1))
                    print(
                        f"[{section_id}] NVIDIA rate limit. "
                        f"Chờ {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    time.sleep(NORMAL_RETRY_DELAY)

    return {
        "status": "failed",
        "section_id": section_id,
        "cells": [],
        "error": str(last_error),
    }
