import json
from langchain_core.messages import HumanMessage
from graph import build_graph
from IPython.display import Markdown,display
from rich.console import Console
from rich.markdown import Markdown
def print_result(result: dict) -> None:
    """
    In kết quả graph theo định dạng dễ đọc.
    """

    printable_result = {
        "dataset_path": result.get("dataset_path"),
        "dataset_summary": result.get("summary"),
        "error": result.get("error"),
        "llm_summary": result.get("summary_llm"),
        "messages": [
            {
                "type": message.type,
                "content": message.content,
            }
            for message in result.get("messages", [])
        ],
    }

    print(
        json.dumps(
            printable_result,
            indent=4,
            ensure_ascii=False,
        )
    )


def main() -> None:
    graph = build_graph()

    initial_state = {
        "messages": [
            HumanMessage(
                content="Hãy phân tích dataset này."
            )
        ],
        "dataset_path": "./data/housing.csv",
        "summary": None,
        "error": None,
    }

    result = graph.invoke(initial_state)
    console = Console()
    console.print(Markdown(result["messages"][-1].content))


if __name__ == "__main__":
    main()