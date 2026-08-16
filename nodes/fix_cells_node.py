from copy import deepcopy

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)

from model.model import llm
from schemas.fixed_cell_schema import FixedCell
from state import State


MAX_CELL_FIX_RETRIES = 3


fix_llm = llm.with_structured_output(
    FixedCell
)


SYSTEM_PROMPT = """
Bạn là chuyên gia Python và Machine Learning.

Nhiệm vụ của bạn là sửa MỘT notebook cell
dựa trên validation errors được cung cấp.

Bạn có thể nhận các loại lỗi:

1. syntax_error
2. undefined_variable

QUY TẮC BẮT BUỘC:

1. Chỉ sửa những vấn đề liên quan đến lỗi
   được cung cấp.

2. Không viết lại toàn bộ notebook.

3. Không thay đổi target column.

4. Không thay đổi dataset path.

5. Không tạo metric hoặc kết quả giả.

6. Không tạo biến tùy tiện chỉ để làm lỗi biến mất.

7. Với undefined_variable:

   - Kiểm tra AVAILABLE NAMES.
   - Kiểm tra PREVIOUS CODE CONTEXT.
   - Nếu biến lỗi chỉ là tên không nhất quán,
     ưu tiên sử dụng biến đã tồn tại.
   - Nếu thực sự cần định nghĩa biến mới,
     chỉ tạo khi logic Machine Learning yêu cầu.

8. Phải tuân thủ VARIABLE CONTRACT.

9. Không tạo data leakage.

10. Preprocessing học tham số chỉ được fit
    trên training data.

11. Giữ thay đổi nhỏ nhất có thể.

12. Source trả về phải là Python thuần,
    không dùng Markdown code fence.
"""

def fix_cells_node(
    state: State,
) -> dict:

    cells = state.get("notebook_cells") or []
    validation_cell_errors = (
        state.get("validation_cell_errors")
        or []
    )

    fix_cell_attempts = state.get(
        "fix_cell_attempts",
        0,
    )

    if not cells:
        message = (
            "Không có notebook_cells để sửa."
        )

        return {
            "error": message,
            "fix_cell_attempts": fix_cell_attempts + 1,
            "messages": [
                AIMessage(content=message)
            ],
        }

    if not validation_cell_errors:
        message = (
            "Không có validation_cell_errors để sửa."
        )
        new_fix_attempts = fix_cell_attempts + 1
        return {
            "validation_cell_status": "invalid",
            "validation_cell_errors": [],
            "fix_cell_attempts": new_fix_attempts,
            "fixed_cell_ids": [],
            "fix_cell_failures": [],
            "error": message,
            "messages": [
                AIMessage(content=message)
            ],
        }

    # Không mutate trực tiếp state cũ
    updated_cells = deepcopy(cells)

    # cell_id -> cell
    cell_map = {
        cell.get("cell_id"): cell
        for cell in updated_cells
        if cell.get("cell_id")
    }

    # Gom lỗi theo cell_id
    errors_by_cell = group_errors_by_cell(
        validation_cell_errors
    )

    fixed_cell_ids = []
    failed_cell_ids = []
    fix_cell_failures = []
    for cell_id, errors in errors_by_cell.items():

        cell = cell_map.get(cell_id)

        if not cell:
            failed_cell_ids.append(cell_id)
            continue

        # Markdown không cần sửa
        if cell.get("cell_type") != "code":
            continue

        original_source = cell.get(
            "source",
            "",
        )

        try:
            fixed_source = fix_single_cell(
                cell=cell,
                errors=errors,
            )

            # Kiểm tra syntax ngay sau khi LLM sửa
            compile(
                fixed_source,
                filename=cell_id,
                mode="exec",
            )

            # Chỉ thay source
            cell["source"] = fixed_source

            fixed_cell_ids.append(
                cell_id
            )

        except Exception as exc:
            failed_cell_ids.append(
                cell_id
            )

            fix_cell_failures.append(
                {
                    "cell_id": cell_id,
                    "title": cell.get("title"),
                    "exception_type": (
                        type(exc).__name__
                    ),
                    "message": str(exc),
                }
            )

    new_fix_attempts = fix_cell_attempts + 1

    message = build_fix_message(
        fix_attempt=new_fix_attempts,
        fixed_cell_ids=fixed_cell_ids,
        failed_cell_ids=failed_cell_ids,
    )

    return {
        "notebook_cells": updated_cells,

        # Validator tiếp theo sẽ xác định lại
        "validation__cell_status": "pending",
        "validation_cell_errors": None,

        "fix_cell_attempts": new_fix_attempts,
        "fixed_cell_ids": fixed_cell_ids,
        "fix_cell_failures": fix_cell_failures,
        "error": (
            None
            if not failed_cell_ids
            else (
                "Một số cell chưa sửa "
                "thành công."
            )
        ),

        "messages": [
            AIMessage(content=message)
        ],
    }
def group_errors_by_cell(
    errors: list[dict],
) -> dict[str, list[dict]]:

    grouped = {}

    for error in errors:
        cell_id = error.get("cell_id")

        if not cell_id:
            continue

        if cell_id not in grouped:
            grouped[cell_id] = []

        grouped[cell_id].append(
            error
        )

    return grouped
def fix_single_cell(
    cell: dict,
    errors: list[dict],
) -> str:

    cell_id = cell["cell_id"]
    source = cell["source"]

    current_source = source
    current_errors = errors
    last_syntax_error = None
    for attempt in range(
        1,
        MAX_CELL_FIX_RETRIES + 1,
    ):

        error_text = format_errors(
            current_errors
        )

        prompt = f"""
CELL ID:
{cell_id}

CELL TITLE:
{cell.get("title")}

PURPOSE:
{cell.get("purpose")}

PYTHON SOURCE:
<source>
{current_source}
</source>

VALIDATION ERRORS:
{error_text}

Hãy sửa toàn bộ lỗi cú pháp trong cell này.

Chỉ thay đổi những phần cần thiết.

Đặc biệt:
- Không để newline thật nằm giữa dấu nháy đơn hoặc nháy kép.
- Đóng đầy đủ dấu ngoặc.
- Đóng đầy đủ string.
- Thêm dấu phẩy bị thiếu khi cần.
- Không thêm ```python.
"""

        result = fix_llm.invoke(
            [
                SystemMessage(
                    content=SYSTEM_PROMPT
                ),
                HumanMessage(
                    content=prompt
                ),
            ]
        )

        result_dict = (
            result.model_dump()
        )

        # LLM không được thay ID
        if (
            result_dict["cell_id"]
            != cell_id
        ):
            raise ValueError(
                "LLM thay đổi cell_id: "
                f"{cell_id} -> "
                f"{result_dict['cell_id']}"
            )

        candidate_source = (
            result_dict["source"]
        )

        # Không chấp nhận Markdown fence
        candidate_source = (
            remove_code_fence(
                candidate_source
            )
        )

        try:
            compile(
                candidate_source,
                filename=cell_id,
                mode="exec",
            )

            return candidate_source

        except SyntaxError as exc:
            last_syntax_error = exc
            # Lần retry kế tiếp sẽ dùng
            # chính lỗi mới này
            current_source = (
                candidate_source
            )

            current_errors = [
                {
                    "cell_id": cell_id,
                    "line": exc.lineno,
                    "offset": exc.offset,
                    "message": exc.msg,
                    "error_line": (
                        exc.text.strip()
                        if exc.text
                        else None
                    ),
                }
            ]

    if last_syntax_error:
        raise RuntimeError(
            f"Cell `{cell_id}` vẫn lỗi sau "
            f"{MAX_CELL_FIX_RETRIES} lần. "
            f"Line {last_syntax_error.lineno}: "
            f"{last_syntax_error.msg}. "
            f"Code: "
            f"{last_syntax_error.text!r}"
        )

    raise RuntimeError(
        f"Không thể sửa cell `{cell_id}`."
    )
def format_errors(
    errors: list[dict],
) -> str:

    lines = []

    for error in errors:

        lines.append(
            f"Type: "
            f"{error.get('error_type')}"
        )

        lines.append(
            f"Line: "
            f"{error.get('line')}"
        )

        lines.append(
            f"Message: "
            f"{error.get('message')}"
        )

        error_line = error.get(
            "error_line"
        )

        if error_line:
            lines.append(
                f"Problematic line: "
                f"{error_line}"
            )

        lines.append("")

    return "\n".join(lines)
def remove_code_fence(
    source: str,
) -> str:

    source = source.strip()

    if source.startswith("```python"):
        source = source[
            len("```python"):
        ]

    elif source.startswith("```"):
        source = source[3:]

    if source.endswith("```"):
        source = source[:-3]

    return source.strip()
def build_fix_message(
    fix_attempt: int,
    fixed_cell_ids: list[str],
    failed_cell_ids: list[str],
) -> str:

    lines = [
        "# Notebook Cell Repair",
        "",
        f"**Fix round:** {fix_attempt}",
        "",
        (
            f"**Sửa thành công:** "
            f"{len(fixed_cell_ids)} cell"
        ),
        (
            f"**Chưa sửa được:** "
            f"{len(failed_cell_ids)} cell"
        ),
    ]

    if fixed_cell_ids:
        lines.extend(
            [
                "",
                "## Cells đã sửa",
            ]
        )

        for cell_id in fixed_cell_ids:
            lines.append(
                f"- `{cell_id}`"
            )

    if failed_cell_ids:
        lines.extend(
            [
                "",
                "## Cells chưa sửa được",
            ]
        )

        for cell_id in failed_cell_ids:
            lines.append(
                f"- `{cell_id}`"
            )

    lines.extend(
        [
            "",
            (
                "Các cell sẽ được đưa trở lại "
                "`validate_cells`."
            ),
        ]
    )

    return "\n".join(lines)

def build_previous_code_context(
    cells: list[dict],
    target_index: int,
    max_cells: int = 5,
) -> str:

    previous_cells = []

    for cell in cells[:target_index]:
        if (
            cell.get("cell_type")
            == "code"
        ):
            previous_cells.append(
                cell
            )

    previous_cells = (
        previous_cells[-max_cells:]
    )

    blocks = []

    for cell in previous_cells:
        blocks.append(
            f"""
CELL ID:
{cell.get("cell_id")}

SOURCE:
{cell.get("source")}
"""
        )

    return "\n".join(
        blocks
    )
