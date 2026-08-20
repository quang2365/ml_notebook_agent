
import json
import re
from datetime import datetime
from pathlib import Path
from state import State
from langchain_core.messages import AIMessage
from tools.json_to_cell_tool import json_object_to_cell

def build_notebook_path(state: State) -> str:
    """Create a unique output path from the dataset, target, and timestamp."""
    dataset_path = state.get("dataset_path") or "dataset"
    dataset_name = Path(dataset_path).stem or "dataset"
    target = state.get("target_column") or "notebook"
    safe_name = re.sub(
        r"[^A-Za-z0-9_-]+",
        "_",
        f"{dataset_name}_{target}",
    ).strip("_") or "notebook"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"output/{safe_name}_{timestamp}.ipynb"

def notebook_builder(state: State) -> dict:
    notebook_cell = state.get("notebook_cells")
    notebook_path = state.get("notebook_path") or build_notebook_path(state)
    if not notebook_cell:
        error = "empty, no notebook cell"
        return {

            "messages": [AIMessage(content=error)],
            "error": error,
            "build_error": error,
            "build_status": "failed",
        }
    cells = [
        json_object_to_cell(item)
        for item in  notebook_cell
    ]
    write_cells_to_ipynb(cells,notebook_path)
    return {

        "notebook_path": notebook_path,
        "build_status": "success",
        "build_error": None,
        "error": None,
        "messages": [
            AIMessage(
                content=f"Created notebook at `{notebook_path}`."
            )
        ],
    }


def write_cells_to_ipynb(cells: list[dict],notebook_path: str,) -> None:
    path = Path(notebook_path)

    path.parent.mkdir(parents=True,exist_ok=True,)

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            notebook,
            file,
            ensure_ascii=False,
            indent=2,
        )
