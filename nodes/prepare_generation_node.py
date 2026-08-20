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
            "Unable to prepare generation: "
            "notebook plan has no section."
        )

        return {
            "section_generation_status": "failed",

            "section_generation_errors": [
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

        "section_generation_status": "pending",

        "section_generation_errors": [],

        "current_section_index": 0,

        "generated_section_ids": [],

        "section_retry_attempts": 0,

        # Remove the old generic error.
        "error": None,

        "messages": [
            AIMessage(
                content=(
                    "Prepared generation for "
                    f"{len(sections)} sections."
                )
            )
        ],
    }
