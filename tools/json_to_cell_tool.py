"""Convert JSON produced by the agent into Jupyter notebook cells."""

from __future__ import annotations

import json
from typing import Any


ALLOWED_CELL_TYPES = {"code", "markdown"}


def _source_lines(source: str) -> list[str]:
    """Return source in the format expected by nbformat/Jupyter."""
    if not source:
        return []

    lines = source.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    return lines


def json_object_to_cell(cell_data: dict[str, Any]) -> dict[str, Any]:
    """Convert one agent cell object to a Jupyter cell dictionary."""
    if not isinstance(cell_data, dict):
        raise TypeError("Cell JSON must be an object.")

    cell_type = cell_data.get("cell_type")
    if cell_type not in ALLOWED_CELL_TYPES:
        raise ValueError(
            "cell_type must be either 'code' or 'markdown'."
        )

    source = cell_data.get("source", "")
    if not isinstance(source, str):
        raise TypeError("Cell source must be a string.")

    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": _source_lines(source),
    }

    if cell_type == "code":
        cell.update({
            "execution_count": None,
            "outputs": [],
        })

    return cell


def json_string_to_cells(json_string: str) -> list[dict[str, Any]]:
    """Convert a JSON string containing one cell or a list of cells."""
    try:
        payload = json.loads(json_string)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc.msg}") from exc

    if isinstance(payload, dict) and "cells" in payload:
        payload = payload["cells"]
    elif isinstance(payload, dict):
        payload = [payload]

    if not isinstance(payload, list):
        raise TypeError("JSON must contain a cell object or a list of cells.")

    return [json_object_to_cell(item) for item in payload]


def notebook_cell_to_json(cell: dict[str, Any]) -> str:
    """Serialize one Jupyter cell to readable JSON."""
    return json.dumps(cell, ensure_ascii=False, indent=2)
