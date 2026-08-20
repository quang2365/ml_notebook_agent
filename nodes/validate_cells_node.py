from langchain_core.messages import AIMessage
from validators.dependency_validator import validate_dependencies
from validators.cell_validator import validate_cells
from state import State




def validate_cells_node(state: State) -> dict:
    cells = state.get("notebook_cells")
    print(
        "\n========== ENTER VALIDATE_CELLS =========="
    )
    if not cells:
        error_message = (
            "No notebook_cells are available for validation."
        )

        return {
            "validation_cell_status": "invalid",
            "validation_cell_errors": [
                {
                    "error_type": "missing_cells",
                    "message": error_message,
                }
            ],
            "error": error_message,
            "messages": [
                AIMessage(content=error_message)
            ],
        }

    validation_cell_errors = validate_cells(cells)
    dependency_errors = (validate_dependencies(cells)
    )

    validation_cell_errors.extend(
        dependency_errors
    )
    # ==========================
    # ERRORS FOUND
    # ==========================
    if validation_cell_errors:
        message = build_validation_message(
            validation_cell_errors
        )

        return {
            "validation_cell_status": "invalid",
            "validation_cell_errors": validation_cell_errors,
            "error": (
                f"Detected "
                f"{len(validation_cell_errors)} errors "
                "trong notebook cells."
            ),
            "messages": [
                AIMessage(content=message)
            ],
        }

    # ==========================
    # NO ERRORS FOUND
    # ==========================
    code_count = sum(
        cell.get("cell_type") == "code"
        for cell in cells
    )

    markdown_count = sum(
        cell.get("cell_type") == "markdown"
        for cell in cells
    )

    return {
        "validation_cell_status": "valid",
        "validation_cell_errors": [],
        "error": None,
        "messages": [
            AIMessage(
                content=(
                    "Notebook cells passed "
                    "structural and syntax validation.\n\n"
                    f"- Total cells: {len(cells)}\n"
                    f"- Code cells: {code_count}\n"
                    f"- Markdown cells: {markdown_count}\n"
                    "- Syntax errors: 0"
                )
            )
        ],
    }

def build_validation_message(
    errors: list[dict],
) -> str:

    lines = [
        "# Notebook Cell Validation",
        "",
        f"Detected **{len(errors)} errors**.",
        "",
    ]

    for index, error in enumerate(
        errors,
        start=1,
    ):
        lines.append(
            f"## {index}. "
            f"{error.get('cell_id', 'unknown')}"
        )

        lines.append(
            f"- Type: "
            f"`{error.get('error_type')}`"
        )

        if error.get("line"):
            lines.append(
                f"- Line: {error['line']}"
            )

        lines.append(
            f"- Message: "
            f"{error.get('message')}"
        )

        if error.get("error_line"):
            lines.extend(
                [
                    "",
                    "```python",
                    error["error_line"],
                    "```",
                ]
            )

        lines.append("")

    return "\n".join(lines)
