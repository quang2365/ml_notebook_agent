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
        message = ("## Đã đọc dataset thành công\n\n"
            f"- **Tên file:** {summary['file_name']}\n"
            f"- **Số dòng:** {summary['rows']:,}\n"
            f"- **Số cột:** {summary['columns']}\n"
            f"- **Cột số:** "
            f"{len(summary['numeric_columns'])}\n"
            f"- **Cột phân loại:** "
            f"{len(summary['categorical_columns'])}\n"
            f"- **Giá trị thiếu:** {summary['total_missing_values']:,}\n"
            f"- **Dòng trùng lặp:** "
            f"{summary['duplicate_rows']:,}")
        return {
            "messages": [AIMessage(content=f"đây là bản tóm tắt của dataset \n {message}")],
            "summary": summary,
            "dataset_path": dataset_path
        }
    except Exception as ect:
        error_messages = str(ect)
        return{
            "error":error_messages,
            "messages": [AIMessage(content=f"không thể đọc file lỗi là do: {error_messages}")]
        }