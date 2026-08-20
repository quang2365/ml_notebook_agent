from langchain_core.messages import AIMessage

from state import State
from tools.dataset_tools import inspect_dataset


def inspect_dataset_note(state:State)-> dict:
    dataset_path = state.get("dataset_path")
    if not dataset_path:
        return{
            "error":"state has not received the dataset path",
            "messages": [AIMessage(content="I have not received the path")]
        }
    try:
        summary = inspect_dataset.invoke(
            {
                "file_path":dataset_path
            })
        message = (
            "## Dataset read successfully\n\n"
f"- **File name:** {summary['file_name']}\n"
f"- **Size:** "
f"{summary['rows']:,} rows × "
f"{summary['columns']} columns\n"
f"- **Numeric columns:** "
            f"{len(summary['numeric_columns'])}\n"
f"- **Categorical columns:** "
            f"{len(summary['categorical_columns'])}\n"
f"- **Missing values:** "
            f"{summary['total_missing_values']:,} "
            f"({summary['total_missing_percentage']}%)\n"
f"- **Duplicate rows:** "
            f"{summary['duplicate_rows']:,}\n"
f"- **Potential ID columns:** "
            f"{len(summary['possible_id_columns'])}\n"
f"- **Columns with highly skewed distribution:** "
            f"{len(summary['analysis_context']['data_quality']['quality_warnings'])}"
)
        return {
            "messages": [AIMessage(content=f"this is the summary of the dataset \n {message}")],
            "summary": summary,
            "dataset_path": dataset_path
        }
    except Exception as ect:
        error_messages = str(ect)
        return{
            "error":error_messages,
            "messages": [AIMessage(content=f"Could not read the file; the error is due to: {error_messages}")]
        }
