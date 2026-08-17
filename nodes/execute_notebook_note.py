from pathlib import Path

import nbformat
from langchain_core.messages import AIMessage
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

from state import State


def execute_notebook_node(
    state: State,
) -> dict:
    notebook_path = state.get("notebook_path")

    if not notebook_path:
        message = ("Không có notebook_path để thực thi.")
        return {
            "execution_status": "failed",
            "execution_error": {
                "error_type": "missing_notebook_path",
                "message": message,
            },
            "error": message,
            "messages": [
                AIMessage(content=message)
            ],
        }
    path = Path(notebook_path)

    if not path.exists():
        message = ( f"Không tìm thấy notebook: {path}")
        return {
            "execution_status": "failed",
            "execution_error": {
                "error_type": "notebook_not_found",
                "message": message,
            },
            "error": message,
            "messages": [
                AIMessage(content=message)
            ],
        }

    try:
        with path.open("r", encoding="utf-8",) as file:
            notebook = nbformat.read(file,as_version=4,)

        client = NotebookClient(notebook, timeout=300,kernel_name="python3")

        #AI: Dataset path trong State được tính từ thư mục gốc ứng dụng,
        # không phải từ thư mục output chứa notebook.
        execution_cwd = Path.cwd().resolve()
        client.execute(cwd=str(execution_cwd))

        # Ghi cả output sau khi chạy vào notebook.
        with path.open("w",encoding="utf-8") as file:
            nbformat.write(
                notebook,
                file,
            )

        return {
            "execution_status": "success",
            "execution_error": None,
            "error": None,
            "messages": [
                AIMessage(
                    content=(
                        "Notebook đã thực thi thành công."
                    )
                )
            ],
        }

    except CellExecutionError as exc:
        message = str(exc)

        return {
            "execution_status": "failed",
            "execution_error": {
                "error_type": "cell_execution_error",
                "message": message,
            },
            "error": message,
            "messages": [
                AIMessage(content=message)
            ],
        }

    except Exception as exc:
        message = str(exc)

        return {
            "execution_status": "failed",
            "execution_error": {
                "error_type": type(exc).__name__,
                "message": message,
            },
            "error": message,
            "messages": [
                AIMessage(content=message)
            ],
        }
