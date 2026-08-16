
import json
from pathlib import Path
from state import State
from langchain_core.messages import AIMessage
from tools.json_to_cell_tool import json_object_to_cell


def notebook_builder(state: State) -> dict:
    notebook_cell = state.get("notebook_cells")
    notebook_path = state.get("notebook_path") or "output/test.ipynb"  #AI
    if not notebook_cell:
        error = "rỗng, chưa có cell notebook"
        return {

            "messages": [AIMessage(content=error)],
            "error": error,
            "build_error": error,  #AI
            "build_status": "failed",  #AI
        }
    cells = [
        json_object_to_cell(item)
        for item in  notebook_cell
    ]
    write_cells_to_ipynb(cells,notebook_path)
    return {
        "notebook_cells": cells,
        "notebook_path": notebook_path,
        "build_status": "success",  #AI
        "build_error": None,  #AI
        "error": None,  #AI
        "messages": [  #AI
            AIMessage(
                content=f"Đã tạo notebook tại `{notebook_path}`."
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
