from langchain_core.messages import AIMessage

from state import State


REQUIRED_CELL_FIELDS = {
    "cell_id",
    "section_id",
    "cell_type",
    "title",
    "source",
}


def validate_cells_node(state: State) -> dict:
    cells = state.get("notebook_cells")

    if not cells:
        error_message = (
            "Không có notebook_cells để kiểm tra."
        )

        return {
            "validation_status": "invalid",
            "validation_errors": [
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

    validation_errors = validate_cells(cells)

    # ==========================
    # CÓ LỖI
    # ==========================
    if validation_errors:
        message = build_validation_message(
            validation_errors
        )

        return {
            "validation_status": "invalid",
            "validation_errors": validation_errors,
            "error": (
                f"Phát hiện "
                f"{len(validation_errors)} lỗi "
                "trong notebook cells."
            ),
            "messages": [
                AIMessage(content=message)
            ],
        }

    # ==========================
    # KHÔNG CÓ LỖI
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
        "validation_status": "valid",
        "validation_errors": [],
        "error": None,
        "messages": [
            AIMessage(
                content=(
                    "Notebook cells đã vượt qua "
                    "kiểm tra cấu trúc và cú pháp.\n\n"
                    f"- Tổng cell: {len(cells)}\n"
                    f"- Code cell: {code_count}\n"
                    f"- Markdown cell: {markdown_count}\n"
                    "- Syntax errors: 0"
                )
            )
        ],
    }


def validate_cells(
    cells: list[dict],
) -> list[dict]:

    errors: list[dict] = []

    seen_ids: set[str] = set()

    for index, cell in enumerate(cells):
        # ==========================
        # 1. Kiểm tra cell có phải dict
        # ==========================
        if not isinstance(cell, dict):
            errors.append(
                {
                    "cell_index": index,
                    "error_type": "invalid_cell",
                    "message": (
                        "Cell không phải dictionary."
                    ),
                }
            )
            continue

        cell_id = cell.get("cell_id")

        # ==========================
        # 2. Kiểm tra field bắt buộc
        # ==========================
        missing_fields = (
            REQUIRED_CELL_FIELDS
            - set(cell.keys())
        )

        if missing_fields:
            errors.append(
                {
                    "cell_index": index,
                    "cell_id": cell_id,
                    "error_type": "missing_fields",
                    "missing_fields": sorted(
                        missing_fields
                    ),
                    "message": (
                        "Cell thiếu field bắt buộc: "
                        f"{sorted(missing_fields)}"
                    ),
                }
            )

        # ==========================
        # 3. Kiểm tra cell_id
        # ==========================
        if not cell_id:
            errors.append(
                {
                    "cell_index": index,
                    "error_type": "missing_cell_id",
                    "message": (
                        "Cell không có cell_id."
                    ),
                }
            )

        elif cell_id in seen_ids:
            errors.append(
                {
                    "cell_index": index,
                    "cell_id": cell_id,
                    "error_type": "duplicate_cell_id",
                    "message": (
                        f"cell_id `{cell_id}` "
                        "bị trùng."
                    ),
                }
            )

        else:
            seen_ids.add(cell_id)

        # ==========================
        # 4. Kiểm tra cell_type
        # ==========================
        cell_type = cell.get("cell_type")

        if cell_type not in {
            "markdown",
            "code",
        }:
            errors.append(
                {
                    "cell_index": index,
                    "cell_id": cell_id,
                    "error_type": "invalid_cell_type",
                    "message": (
                        f"cell_type `{cell_type}` "
                        "không hợp lệ."
                    ),
                }
            )
            continue

        # ==========================
        # 5. Kiểm tra source
        # ==========================
        source = cell.get("source")

        if not isinstance(source, str):
            errors.append(
                {
                    "cell_index": index,
                    "cell_id": cell_id,
                    "error_type": "invalid_source",
                    "message": (
                        "source phải là string."
                    ),
                }
            )
            continue

        if not source.strip():
            errors.append(
                {
                    "cell_index": index,
                    "cell_id": cell_id,
                    "error_type": "empty_source",
                    "message": (
                        "Cell có source rỗng."
                    ),
                }
            )
            continue

        # ==========================
        # 6. Markdown không cần compile
        # ==========================
        if cell_type == "markdown":
            continue

        # ==========================
        # 7. Code fence không nên tồn tại
        # ==========================
        if "```python" in source:
            errors.append(
                {
                    "cell_index": index,
                    "cell_id": cell_id,
                    "error_type": "code_fence",
                    "message": (
                        "Code cell chứa "
                        "Markdown code fence."
                    ),
                }
            )

        # ==========================
        # 8. Kiểm tra Python syntax
        # ==========================
        try:
            compile(
                source,
                filename=(
                    cell_id
                    or f"cell_{index}"
                ),
                mode="exec",
            )

        except SyntaxError as exc:
            errors.append(
                {
                    "cell_index": index,
                    "cell_id": cell_id,
                    "title": cell.get("title"),
                    "error_type": "syntax_error",
                    "line": exc.lineno,
                    "offset": exc.offset,
                    "message": exc.msg,
                    "error_line": (
                        exc.text.strip()
                        if exc.text
                        else None
                    ),
                }
            )

    return errors


def build_validation_message(
    errors: list[dict],
) -> str:

    lines = [
        "# Notebook Cell Validation",
        "",
        f"Phát hiện **{len(errors)} lỗi**.",
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