import json
from langchain_core.messages import HumanMessage
from graph import build_graph
from IPython.display import Markdown,display
from rich.console import Console
from rich.markdown import Markdown

def main() -> None:
    graph = build_graph()
    config = {"configuration": {"thread_id":"test_01"}}
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

    result = graph.invoke(initial_state,config)
    console = Console()
    console.print(result["messages"][-1].content)
    interrupts = result.get("__interrupt__", [])
    if interrupts:
        payload = interrupts[0].value
        console.print(payload)


if __name__ == "__main__":
    main()