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

        failed_cells = extract_failed_cell(notebook) or {}
        failed_cell_id = failed_cells.get("cell_id")
        exception_name = failed_cells.get("exception_name")
        exception_value = failed_cells.get("exception_value")
        traceback_text = failed_cells.get("traceback")
        return {
            "execution_status": "failed",
            "execution_error": {
                "error_type": "cell_execution_error",
                "cell_id": failed_cell_id,
                "exception_name": exception_name,
                "exception_value": exception_value,
                "message": message,
                "traceback": traceback_text or message,
                "source": failed_cells.get("source"),
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
def extract_failed_cell(
    notebook,
) -> dict | None:
    """
    Tìm cell phát sinh runtime error trong notebook đã chạy.

    nbclient ghi error vào outputs của cell trước khi ném
    CellExecutionError.
    """


    for cell in reversed(notebook.cells):
        if cell.get("cell_type") != "code":
            continue

        outputs = cell.get("outputs") or []

        for output in reversed(outputs):
            if output.get("output_type") != "error":
                continue

            metadata = cell.get("metadata") or {}

            agent_metadata = (
                metadata.get("agent")
                or {}
            )
            failed_cell_id = (
                agent_metadata.get("cell_id")
                or cell.get("id")
            )

            traceback_lines = (
                output.get("traceback")
                or []
            )

            return {
                "cell_id": failed_cell_id,
                "exception_name": output.get(
                    "ename"
                ),
                "exception_value": output.get(
                    "evalue"
                ),
                "traceback": "\n".join(
                    traceback_lines
                ),
                "source": normalize_source(
                    cell.get("source")
                ),
            }

    return None


def normalize_source(
    source: str | list | None,
) -> str:
    if source is None:
        return ""

    if isinstance(source, list):
        return "".join(source)

    return str(source)
