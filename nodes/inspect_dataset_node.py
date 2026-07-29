from langchain_core.messages import AIMessage

from state import State
from tools.dataset_tools import inspect_dataset


def inspect_dataset_note(state:State)-> dict:
    dataset_path = state.get("dataset_path")
    if not dataset_path:
        return{
            "error":"state chưa nhận đường dẫn dataset",
            "messages": [AIMessage(content="Tôi chưa nhận được đường dẫn")]
        }
    try:
        summary = inspect_dataset.invoke(
            {
                "file_path":dataset_path
            })
        message = (f"tên của file là {summary['file_name']}"
                   f"số dòng: {summary['rows']}"
                   f"số cột: {summary['columns']}")
        return {
            "messages": [AIMessage(content=f"đây là bản tóm tắt của dataset {message}")],
            "summary": summary,
            "dataset_path": dataset_path
        }
    except Exception as ect:
        error_messages = str(ect)
        return{
            "error":error_messages,
            "messages": [AIMessage(content=f"không thể đọc file lỗi là do: {error_messages}")]
        }