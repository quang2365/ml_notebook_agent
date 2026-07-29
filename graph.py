from langgraph.graph import END,START, StateGraph

from nodes.inspect_dataset_node import inspect_dataset_note
from state import State

def build_graph():
    builder = StateGraph(State)

    builder.add_node("inspect_data",inspect_dataset_note)

    builder.add_edge(START,"inspect_data")

    builder.add_edge("inspect_data",END)

    graph = builder.compile()

    return graph