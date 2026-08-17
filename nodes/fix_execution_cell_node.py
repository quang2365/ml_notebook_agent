import ast
import json
from copy import deepcopy

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)

from model.model import llm
from schemas.fixed_cell_schema import FixedCell
from state import State
from validators.dependency_validator import (
    BUILTIN_NAMES,
    CellDependencyAnalyzer,
    validate_dependencies,
)



# Số vòng tổng thể sẽ được kiểm soát bởi route.
runtime_fix_llm = llm.with_structured_output(
    FixedCell,
    method="function_calling",
)


RUNTIME_FIX_SYSTEM_PROMPT = """
Bạn là chuyên gia sửa lỗi runtime trong Python Machine Learning
Notebook.

Notebook đã vượt qua:

1. Kiểm tra cấu trúc.
2. Kiểm tra cú pháp.
3. Kiểm tra dependency tĩnh.
4. Đánh giá ngữ nghĩa pipeline.

Tuy nhiên, notebook đã phát sinh lỗi khi thực thi thực tế.

Nhiệm vụ của bạn là sửa duy nhất CURRENT CELL dựa trên:

- Runtime exception.
- Traceback.
- Source của current cell.
- Các code cell đã chạy trước đó.
- Những biến đã được định nghĩa.
- Target column và problem type.

CÁC LỖI CÓ THỂ XUẤT HIỆN:

1. TypeError.
2. ValueError.
3. KeyError.
4. IndexError.
5. AttributeError.
6. NameError.
7. ImportError.
8. FileNotFoundError.
9. Lỗi pandas DataFrame và Series.
10. Lỗi NumPy array.
11. Lỗi scikit-learn pipeline.
12. Không tương thích shape hoặc dtype.
13. Lỗi metric và model prediction.
14. Lỗi feature engineering.
15. Lỗi truy cập transformer đã fit.

QUY TẮC BẮT BUỘC:

1. Chỉ sửa CURRENT CELL.

2. Không sửa hoặc viết lại toàn bộ notebook.

3. Không tham chiếu đến cell đứng sau CURRENT CELL.

4. Không thay đổi:

   - dataset path;
   - target column;
   - problem type;
   - train/test split một cách không cần thiết.

5. Không tạo dữ liệu, metric, prediction hoặc model giả.

6. Không tạo biến bằng None hoặc giá trị giả chỉ để lỗi biến mất.

7. Chỉ sử dụng:

   - biến trong AVAILABLE NAMES;
   - object trong PREVIOUS CODE CELLS;
   - biến được định nghĩa hợp lệ trong CURRENT CELL.

8. Nếu lỗi do tên biến không nhất quán, ưu tiên sử dụng biến
   đã tồn tại thay vì tạo biến mới.

9. Nếu lỗi liên quan đến dictionary, phải sử dụng đúng key
   đã được tạo trước đó.

10. Nếu lỗi liên quan đến DataFrame và NumPy:

    - Kiểm tra kiểu dữ liệu thực tế đi qua từng transformer.
    - Không truy cập ndarray bằng tên cột.
    - FunctionTransformer dùng X["column"] phải nhận DataFrame.
    - SimpleImputer, StandardScaler và ColumnTransformer có thể
      trả về NumPy array.
    - Feature engineering dùng tên cột nên chạy trước transformer
      làm mất thông tin tên cột.

11. Nếu lỗi liên quan đến preprocessing:

    - Chỉ fit trên training data.
    - Không fit_transform trên test data.
    - Train và test phải dùng cùng fitted preprocessor.
    - Không tạo data leakage.

12. Nếu lỗi liên quan đến model:

    - Model phải được fit trước khi predict.
    - Sử dụng đúng features và target.
    - Giữ tên model nhất quán giữa trained_models, predictions
      và model_results.

13. Nếu lỗi liên quan đến metric:

    - Sử dụng y_true và y_pred đúng thứ tự.
    - Metric phải phù hợp với problem type.
    - Không sử dụng kết quả của model cuối cho mọi model.
    - RMSE có thể tính bằng:
      np.sqrt(mean_squared_error(y_true, y_pred)).

14. Nếu lỗi do thư viện chưa được cài đặt, không chèn lệnh pip
    install vào notebook. Không sửa import thành thư viện không
    tương đương chỉ để tránh lỗi.

15. Thực hiện thay đổi nhỏ nhất có thể.

16. Giữ nguyên cell_id.

17. Source trả về phải là Python thuần.

18. Không sử dụng Markdown code fence.
"""


def fix_execution_cell_node(
    state: State,
) -> dict:
    """
    Sửa cell gây lỗi trong quá trình thực thi notebook.
    """

    cells = state.get("notebook_cells") or []

    execution_error = (
        state.get("execution_error")
        or {}
    )

    attempts = state.get(
        "execution_fix_attempts",
        0,
    )

    new_attempts = attempts + 1

    if not cells:
        message = (
            "Không có notebook_cells để sửa lỗi runtime."
        )

        return {
            "execution_status": "failed",
            "execution_fix_attempts": new_attempts,
            "error": message,
            "messages": [
                AIMessage(content=message)
            ],
        }

    cell_id = execution_error.get("cell_id")

    if not cell_id:
        message = (
            "Không xác định được cell_id gây lỗi runtime."
        )

        return {
            "execution_status": "failed",
            "execution_fix_attempts": new_attempts,
            "error": message,
            "messages": [
                AIMessage(content=message)
            ],
        }

    cell_index = find_cell_index(
        cells=cells,
        cell_id=cell_id,
    )

    if cell_index is None:
        message = (
            f"Không tìm thấy cell runtime `{cell_id}` "
            "trong notebook_cells."
        )

        return {
            "execution_status": "failed",
            "execution_fix_attempts": new_attempts,
            "error": message,
            "messages": [
                AIMessage(content=message)
            ],
        }

    current_cell = cells[cell_index]

    if current_cell.get("cell_type") != "code":
        message = (
            f"Cell `{cell_id}` không phải code cell."
        )

        return {
            "execution_status": "failed",
            "execution_fix_attempts": new_attempts,
            "error": message,
            "messages": [
                AIMessage(content=message)
            ],
        }

    previous_code_cells = [
        {
            "cell_id": cell.get("cell_id"),
            "section_id": cell.get("section_id"),
            "title": cell.get("title"),
            "purpose": cell.get("purpose"),
            "source": normalize_source(
                cell.get("source")
            ),
        }
        for cell in cells[:cell_index]
        if cell.get("cell_type") == "code"
    ]

    available_names = collect_available_names(
        previous_code_cells
    )

    prompt_context = {
        "target_column": state.get(
            "target_column"
        ),
        "problem_type": state.get(
            "problem_type"
        ),
        "available_names": available_names,
        "previous_code_cells": (
            previous_code_cells
        ),
        "current_cell": {
            "cell_id": current_cell.get(
                "cell_id"
            ),
            "section_id": current_cell.get(
                "section_id"
            ),
            "title": current_cell.get(
                "title"
            ),
            "purpose": current_cell.get(
                "purpose"
            ),
            "source": normalize_source(
                current_cell.get("source")
            ),
        },
        "runtime_error": {
            "error_type": execution_error.get(
                "error_type"
            ),
            "exception_name": (
                execution_error.get(
                    "exception_name"
                )
            ),
            "message": execution_error.get(
                "message"
            ),
            "traceback": execution_error.get(
                "traceback"
            ),
        },
    }

    try:
        result = runtime_fix_llm.invoke(
            [
                SystemMessage(
                    content=(
                        RUNTIME_FIX_SYSTEM_PROMPT
                    )
                ),
                HumanMessage(
                    content=(
                        "Hãy sửa CURRENT CELL dựa trên "
                        "runtime context sau:\n\n"
                        + json.dumps(
                            prompt_context,
                            ensure_ascii=False,
                            default=str,
                        )
                    )
                ),
            ]
        )

        result_dict = result.model_dump()

        returned_cell_id = result_dict.get(
            "cell_id"
        )

        if returned_cell_id != cell_id:
            raise ValueError(
                "LLM thay đổi cell_id: "
                f"{cell_id} -> {returned_cell_id}"
            )

        fixed_source = remove_code_fence(
            result_dict.get("source") or ""
        )


        compile(
            fixed_source,
            filename=cell_id,
            mode="exec",
        )

        updated_cells = deepcopy(cells)
        updated_cells[cell_index]["source"] = (
            fixed_source
        )


        dependency_errors = validate_dependencies(
            updated_cells
        )

        current_cell_errors = [
            error
            for error in dependency_errors
            if error.get("cell_id") == cell_id
        ]

        if current_cell_errors:
            raise ValueError(
                "Bản sửa runtime tạo lỗi dependency: "
                f"{current_cell_errors}"
            )

        return {
            "notebook_cells": updated_cells,


            "validation_cell_status": "pending",
            "validation_cell_errors": None,
            "pipeline_review_status": "pending",
            "pipeline_review_errors": None,


            "build_status": "pending",
            "build_error": None,


            # khi notebook được build và chạy lại.
            "execution_status": "pending",
            "execution_error": None,
            "execution_fix_attempts": new_attempts,

            "fixed_cell_ids": [cell_id],
            "error": None,
            "messages": [
                AIMessage(
                    content=(
                        f"Đã sửa runtime cell `{cell_id}`. "
                        "Cell sẽ được validate, review, "
                        "build và thực thi lại."
                    )
                )
            ],
        }

    except Exception as exc:
        message = (
            f"Không thể sửa runtime cell "
            f"`{cell_id}`: {exc}"
        )

        old_failures = (
            state.get("fix_cell_failures")
            or []
        )

        return {
            "execution_status": "failed",
            "execution_fix_attempts": new_attempts,
            "fix_cell_failures": [
                *old_failures,
                {
                    "cell_id": cell_id,
                    "error_type": (
                        "runtime_fix_failed"
                    ),
                    "exception_type": (
                        type(exc).__name__
                    ),
                    "message": str(exc),
                },
            ],
            "error": message,
            "messages": [
                AIMessage(content=message)
            ],
        }


def find_cell_index(
    cells: list[dict],
    cell_id: str,
) -> int | None:
    """Tìm vị trí cell nội bộ bằng cell_id."""

    for index, cell in enumerate(cells):
        if cell.get("cell_id") == cell_id:
            return index

    return None


def normalize_source(
    source: str | list | None,
) -> str:
    """Chuẩn hóa source thành chuỗi Python."""

    if source is None:
        return ""

    if isinstance(source, list):
        return "".join(source)

    return str(source)


def remove_code_fence(source: str) -> str:
    """Loại bỏ Markdown fence nếu LLM vẫn trả về."""

    source = source.strip()

    if source.startswith("```python"):
        source = source[len("```python"):]

    elif source.startswith("```"):
        source = source[3:]

    if source.endswith("```"):
        source = source[:-3]

    return source.strip()


def collect_available_names(
    previous_code_cells: list[dict],
) -> list[str]:
    """
    Thu thập các biến, hàm và import đã được định nghĩa
    trong những cell đứng trước.
    """

    available_names = set(BUILTIN_NAMES)

    for cell in previous_code_cells:
        source = normalize_source(
            cell.get("source")
        )

        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue

        analyzer = CellDependencyAnalyzer()
        analyzer.visit(tree)

        available_names.update(
            analyzer.defined_names
        )

    return sorted(
        name
        for name in available_names
        if name not in BUILTIN_NAMES
    )