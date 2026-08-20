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
    Generate cells for exactly one section.

    Effects:
    - Each time the node calls the LLM for only one section.
    - Append new cells to the existing notebook_cells.
    - Increment current_section_index after success.
    - Enable a LangGraph checkpoint after each section.
    """

    # ==================================================
    # 1. READ NOTEBOOK PLAN
    # ==================================================

    notebook_plan = (
        state.get("notebook_plan")
        or {}
    )

    sections = (
        notebook_plan.get("sections")
        or []
    )

    # Without a section, generation is not possible.
    if not sections:
        error_message = (
            "Cannot generate section: "
            "the notebook plan has no sections."
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

    # ==================================================
    # 2. DETERMINE CURRENT SECTION
    # ==================================================

    # Index starts at 0:
    # 0 = first section
    # 1 = second section
    # ...
    current_index = state.get(
        "current_section_index",
        0,
    )

    # A negative index is an invalid state.
    if current_index < 0:
        error_message = (
            "current_section_index cannot "
            "be less than 0."
        )

        return {
            "section_generation_status": "failed",
            "section_generation_errors": [
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

    # If the index is already equal to the number of sections, all
    # sections have been generated.
    if current_index >= len(sections):
        return {
            "section_generation_status": "success",
            "section_retry_attempts": 0, #giai thich
            "error": None,
            "messages": [
                AIMessage(
                    content=(
                        "All notebook sections "
                        "have been generated."
                    )
                )
            ],
        }

    # Get exactly one section based on the current index.
    section = sections[current_index]

    # If the plan lacks section_id, create a fallback ID.
    section_id = (
        section.get("section_id")
        or f"section_{current_index + 1}"
    )

    # ==================================================
    # 3. CHECK INPUT REQUIRED FOR LLM
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
                "No dataset_path to "
                f"generate `{section_id}`."
            ),
        )

    if not target_column:
        return build_generation_failure(
            state=state,
            section_id=section_id,
            error_type="missing_target_column",
            message=(
                "No target_column to "
                f"generate `{section_id}`."
            ),
        )

    if not problem_type:
        return build_generation_failure(
            state=state,
            section_id=section_id,
            error_type="missing_problem_type",
            message=(
                "No problem_type to "
                f"generate `{section_id}`."
            ),
        )

    # ==================================================
    # 4. CREATE DATASET CONTEXT
    # ==================================================

    # Reuse the function extracted in the previous step.
    # Thus, each section receives the same context structure.
    dataset_context = build_dataset_context(
        state
    )


    # Markdown, output, and message are not included in the context to save tokens.
    previous_code_cells = [
        {
            "cell_id": cell.get("cell_id"),
            "section_id": cell.get("section_id"),
            "title": cell.get("title"),
            "source": cell.get("source"),
        }
        for cell in (
            state.get("notebook_cells")
            or []
        )
        if cell.get("cell_type") == "code"
    ]

    # ==================================================
    # 5. GENERATE EXACTLY ONE SECTION
    # ==================================================

    result = generate_one_section(
        section_id=section_id,
        section=section,
        dataset_path=dataset_path,
        target_column=target_column,
        problem_type=problem_type,
        dataset_context=dataset_context,
        previous_code_cells=previous_code_cells,
    )

    # generate_one_section automatically retries the API.
    # If it still fails, keep the old cells and old index.
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
                    "Cannot generate "
                    f"`{section_id}`."
                )
            ),
        )

    # ==================================================
    # 6. APPEND NEW CELLS TO STATE
    # ==================================================

    old_cells = (
        state.get("notebook_cells")
        or []
    )

    new_cells = (
        result.get("cells")
        or []
    )

    # Do not allow a section to succeed if it produces no cells.
    if not new_cells:
        return build_generation_failure(
            state=state,
            section_id=section_id,
            error_type="empty_section",
            message=(
                f"`{section_id}` did not create any cells."
            ),
        )

    # Create a new list instead of mutating the list in State.
    updated_cells = [
        *old_cells,
        *new_cells,
    ]

    old_generated_ids = (
        state.get("generated_section_ids")
        or []
    )

    # Create a new list to avoid mutating the old state.
    updated_generated_ids = [
        *old_generated_ids,
        section_id,
    ]#giai thich

    # Move the pointer to the next section.
    next_index = current_index + 1

    # If the last section was just generated, the status is success.
    # If sections remain, the status is pending.
    generation_status = (
        "success"
        if next_index >= len(sections)
        else "pending"
    )

    return {
        # Keep the old cells and append the new section's cells.
        "notebook_cells": updated_cells,

        # Save the completed section.
        "generated_section_ids": (
            updated_generated_ids
        ),

        # Next time, the node will get the next section.
        "current_section_index": next_index,

        # The section succeeded, so reset retry to 0.
        "section_retry_attempts": 0,

        # Pending if sections remain, success if none remain.
        "section_generation_status": (
            generation_status
        ),

        "error": None,

        "messages": [
            AIMessage(
                content=(
                    f"Generated `{section_id}` "
                    f"with {len(new_cells)} cells. "
                    f"Progress: "
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
    Create a unified error result for generation.

    Effects:
    - No repeated error-handling code.
    - Retain cells generated in the previous section.
    - Keep current_section_index to allow retry.
    - Increment the current section's failure count.
    """

    old_errors = (
        state.get("section_generation_errors")
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
        # Return the old cells to make it clear
        # that they are not deleted when the new section fails.
        "notebook_cells": (
            state.get("notebook_cells")
            or []
        ),

        # Do not increment the index when the section has not yet succeeded.
        "current_section_index": state.get(
            "current_section_index",
            0,
        ),

        # Do not add the failed section to the completed list.
        "generated_section_ids": (
            state.get("generated_section_ids")
            or []
        ),

        "section_generation_status": "failed",

        "section_generation_errors": [
            *old_errors,
            new_error,
        ],

        "section_retry_attempts": retry_attempts,

        "error": message,

        "messages": [
            AIMessage(content=message)
        ],
    }
