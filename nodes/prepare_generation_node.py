from langchain_core.messages import AIMessage

from state import State


def prepare_generation_node(state:State) ->dict:
    notebook_plan = (
        state.get("notebook_plan")
        or {}
    )

    sections = (
        notebook_plan.get("sections")
        or []
    )

    if not sections:
        error_message = (
            "Không thể chuẩn bị generation: "
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

    return {
        "notebook_cells": [],

        "generation_cell_status": "pending",

        "generation_cell_errors": [],

        "current_section_index": 0,

        "generated_section_ids": [],

        "section_retry_attempts": 0,

        # Xóa lỗi tổng quát cũ.
        "error": None,

        "messages": [
            AIMessage(
                content=(
                    "Đã chuẩn bị generation cho "
                    f"{len(sections)} sections."
                )
            )
        ],
    }