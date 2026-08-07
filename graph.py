from langgraph.graph import END,START, StateGraph
from langgraph.checkpoint.memory  import InMemorySaver

from nodes.inspect_dataset_node import inspect_dataset_note
from nodes.analyze_dataset_node import analyze_dataset_note
from nodes.propose_problem_node import propose_problem_node
from nodes.review_problem_node import review_problem_node
from nodes.analyze_target_node import analyze_target_node
from nodes.plan_notebook_node import plan_notebook_node
from nodes.generate_cells_node import generate_cells_node
from nodes.validate_cells_node import validate_cells_node
from state import State


def route_after_review(state: State) -> str:
    if state["approval_status"] == "approved":
        return "approved"

    return "rejected"

def build_graph():
    builder = StateGraph(State)

    builder.add_node("inspect_data",inspect_dataset_note)
    builder.add_node("analyze_data",analyze_dataset_note)
    builder.add_node("propose_problem",propose_problem_node)
    builder.add_node("review_problem",review_problem_node)
    builder.add_node("analyze_target",analyze_target_node)
    builder.add_node("plan_notebook",plan_notebook_node)
    builder.add_node("generate_cells",generate_cells_node)
    builder.add_node("validate_cells",validate_cells_node)
    builder.add_edge(START,"inspect_data")
    builder.add_edge("inspect_data","analyze_data")
    builder.add_edge("analyze_data","propose_problem")
    builder.add_edge("propose_problem","review_problem")
    builder.add_conditional_edges("review_problem",route_after_review,{"approved":"analyze_target","rejected":END})
    builder.add_edge("analyze_target","plan_notebook")
    builder.add_edge("plan_notebook","generate_cells")
    builder.add_edge("generate_cells","validate_cells")
    builder.add_edge("validate_cells",END)
    graph = builder.compile(checkpointer=InMemorySaver())

    return graph