from langchain_core.messages import AIMessage

from tools.section_generation import (
    build_dataset_context,
    generate_one_section,
)
from state import State


def generate_section_node(
    state: State,
) -> dict:
    """
    Sinh cells cho đúng một section.

    Tác dụng:
    - Mỗi lần node chỉ gọi LLM cho một section.
    - Nối cells mới vào notebook_cells hiện có.
    - Tăng current_section_index sau khi thành công.
    - Cho phép LangGraph checkpoint sau từng section.
    """

    # ==================================================
    # 1. ĐỌC NOTEBOOK PLAN
    # ==================================================

    notebook_plan = (
        state.get("notebook_plan")
        or {}
    )

    sections = (
        notebook_plan.get("sections")
        or []
    )

    # Không có section thì không thể generate.
    if not sections:
        error_message = (
            "Không thể generate section: "
            "notebook plan không có section."
        )

        return {
            "generation_cell_status": "failed",
            "generation_cell_errors": [
                {
                    "error_type": "missing_sections",
                    "section_id": None,
                    "message": error_message,
                }
            ],
            "error": error_message,
            "messages": [
                AIMessage(
                    content=error_message
                )
            ],
        }

    # ==================================================
    # 2. XÁC ĐỊNH SECTION HIỆN TẠI
    # ==================================================

    # Index bắt đầu từ 0:
    # 0 = section đầu tiên
    # 1 = section thứ hai
    # ...
    current_index = state.get(
        "current_section_index",
        0,
    )

    # Index âm là state không hợp lệ.
    if current_index < 0:
        error_message = (
            "current_section_index không thể "
            "nhỏ hơn 0."
        )

        return {
            "generation_cell_status": "failed",
            "generation_cell_errors": [
                {
                    "error_type": (
                        "invalid_section_index"
                    ),
                    "section_id": None,
                    "message": error_message,
                }
            ],
            "error": error_message,
        }

    # Nếu index đã bằng số section thì toàn bộ
    # section đã được sinh xong.
    if current_index >= len(sections):
        return {
            "generation_cell_status": "success",
            "section_retry_attempts": 0, #giai thich
            "error": None,
            "messages": [
                AIMessage(
                    content=(
                        "Tất cả notebook sections "
                        "đã được sinh."
                    )
                )
            ],
        }

    # Lấy đúng một section theo index hiện tại.
    section = sections[current_index]

    # Nếu plan thiếu section_id, tạo ID dự phòng.
    section_id = (
        section.get("section_id")
        or f"section_{current_index + 1}"
    )

    # ==================================================
    # 3. KIỂM TRA INPUT CẦN CHO LLM
    # ==================================================

    dataset_path = state.get("dataset_path")

    target_column = state.get("target_column")

    problem_type = state.get("problem_type")

    if not dataset_path:
        return build_generation_failure(
            state=state,
            section_id=section_id,
            error_type="missing_dataset_path",
            message=(
                "Không có dataset_path để "
                f"generate `{section_id}`."
            ),
        )

    if not target_column:
        return build_generation_failure(
            state=state,
            section_id=section_id,
            error_type="missing_target_column",
            message=(
                "Không có target_column để "
                f"generate `{section_id}`."
            ),
        )

    if not problem_type:
        return build_generation_failure(
            state=state,
            section_id=section_id,
            error_type="missing_problem_type",
            message=(
                "Không có problem_type để "
                f"generate `{section_id}`."
            ),
        )

    # ==================================================
    # 4. TẠO DATASET CONTEXT
    # ==================================================

    # Tái sử dụng hàm đã tách ở bước trước.
    # Nhờ vậy mỗi section nhận cùng cấu trúc context.
    dataset_context = build_dataset_context(
        state
    )

    # ==================================================
    # 5. GENERATE ĐÚNG MỘT SECTION
    # ==================================================

    result = generate_one_section(
        section_id=section_id,
        section=section,
        dataset_path=dataset_path,
        target_column=target_column,
        problem_type=problem_type,
        dataset_context=dataset_context,
    )

    # generate_one_section đã tự retry API.
    # Nếu vẫn thất bại, giữ nguyên cells cũ và index cũ.
    if result.get("status") == "failed":
        return build_generation_failure(
            state=state,
            section_id=section_id,
            error_type=(
                "section_generation_failed"
            ),
            message=(
                result.get("error")
                or (
                    "Không thể generate "
                    f"`{section_id}`."
                )
            ),
        )

    # ==================================================
    # 6. NỐI CELLS MỚI VÀO STATE
    # ==================================================

    old_cells = (
        state.get("notebook_cells")
        or []
    )

    new_cells = (
        result.get("cells")
        or []
    )

    # Không cho phép section thành công nhưng không có cell.
    if not new_cells:
        return build_generation_failure(
            state=state,
            section_id=section_id,
            error_type="empty_section",
            message=(
                f"`{section_id}` không tạo cell nào."
            ),
        )

    # Tạo list mới thay vì mutate list trong State.
    updated_cells = [
        *old_cells,
        *new_cells,
    ]

    old_generated_ids = (
        state.get("generated_section_ids")
        or []
    )

    # Tạo danh sách mới để tránh mutate state cũ.
    updated_generated_ids = [
        *old_generated_ids,
        section_id,
    ]#giai thich

    # Chuyển con trỏ sang section tiếp theo.
    next_index = current_index + 1

    # Nếu vừa sinh section cuối, trạng thái là success.
    # Nếu vẫn còn section, trạng thái là pending.
    generation_status = (
        "success"
        if next_index >= len(sections)
        else "pending"
    )

    return {
        # Giữ cells cũ và nối thêm cells section mới.
        "notebook_cells": updated_cells,

        # Lưu section đã hoàn thành.
        "generated_section_ids": (
            updated_generated_ids
        ),

        # Lần sau node sẽ lấy section tiếp theo.
        "current_section_index": next_index,

        # Section thành công nên reset retry về 0.
        "section_retry_attempts": 0,

        # Pending nếu còn section, success nếu đã hết.
        "generation_cell_status": (
            generation_status
        ),

        "error": None,

        "messages": [
            AIMessage(
                content=(
                    f"Đã generate `{section_id}` "
                    f"với {len(new_cells)} cells. "
                    f"Tiến trình: "
                    f"{next_index}/{len(sections)}."
                )
            )
        ],
    }


def build_generation_failure(
    state: State,
    section_id: str | None,
    error_type: str,
    message: str,
) -> dict:
    """
    Tạo kết quả lỗi thống nhất cho generation.

    Tác dụng:
    - Không lặp code xử lý lỗi.
    - Giữ lại cells đã sinh ở section trước.
    - Giữ current_section_index để có thể retry.
    - Tăng số lần section hiện tại thất bại.
    """

    old_errors = (
        state.get("generation_cell_errors")
        or []
    )

    retry_attempts = (
        state.get(
            "section_retry_attempts",
            0,
        )
        + 1
    )

    new_error = {
        "error_type": error_type,
        "section_id": section_id,
        "message": message,
        "attempt": retry_attempts,
    }

    return {
        # Trả lại cells cũ để thể hiện rõ
        # chúng không bị xóa khi section mới lỗi.
        "notebook_cells": (
            state.get("notebook_cells")
            or []
        ),

        # Không tăng index khi section chưa thành công.
        "current_section_index": state.get(
            "current_section_index",
            0,
        ),

        # Không thêm section lỗi vào danh sách hoàn thành.
        "generated_section_ids": (
            state.get("generated_section_ids")
            or []
        ),

        "generation_cell_status": "failed",

        "generation_cell_errors": [
            *old_errors,
            new_error,
        ],

        "section_retry_attempts": retry_attempts,

        "error": message,

        "messages": [
            AIMessage(content=message)
        ],
    }
